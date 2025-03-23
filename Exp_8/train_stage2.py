import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
# python train_stage2.py --config configs/train/train_stage2.yaml

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    # Setup optimizer:
    # opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

    # ===== 【VAE test】 =====
    parameters_to_optimize = list(cldm.vae.parameters())
    opt = torch.optim.AdamW(parameters_to_optimize, lr=cfg.train.learning_rate)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training:
    cldm.train().to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []

    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt, rgb = batch
            # gt shape: torch.Size([16, 512, 512, 3])
            # lq shape: torch.Size([16, 512, 512, 3])
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
            # gt shape: torch.Size([16, 3, 512, 512])
            # lq shape: torch.Size([16, 3, 512, 512])

            # 将解码器的输出与原始输入比较。目的是更新vae.decoder的参数
            z_lq = pure_cldm.vae_encode(lq * 2 - 1)
            reconstructed = pure_cldm.vae_decode(z_lq)         
            loss = torch.nn.functional.mse_loss(reconstructed, lq)   

            opt.zero_grad()
            accelerator.backward(loss)

            # 计算 VAE 参数梯度的范数
            vae_grad_norm = 0
            for name, param in pure_cldm.vae.named_parameters():
                if param.grad is not None:
                    vae_grad_norm += param.grad.data.norm(2).item() ** 2
            vae_grad_norm = vae_grad_norm ** 0.5

            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}, VAE Grad Norm: {vae_grad_norm:.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)
                    writer.add_scalar("grad/vae_grad_norm", vae_grad_norm, global_step)  # 记录 VAE 梯度范数到 TensorBoard

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # 保存vae
                    checkpoint_vae = pure_cldm.vae.state_dict()        
                    ckpt_path_vae = f"{ckpt_dir}/{global_step:07d}_vae.pt"
                    torch.save(checkpoint_vae, ckpt_path_vae)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                log_reconstructed = reconstructed[:N]       # 重建图像
                
                cldm.eval()
                with torch.no_grad():
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", (log_lq + 1) / 2),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                            (
                                "image/reconstructed_lq",
                                (log_reconstructed + 1) / 2,        
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)