import os
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torchvision.utils import save_image
from einops import rearrange  
from PIL import Image
from accelerate import Accelerator
from diffbir.model import ControlLDM
from diffbir.utils.common import instantiate_from_config, to
from diffbir.dataset.codeformer import CodeformerDataset
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
"""
python inference_VAE.py \
--config configs/train/train_stage2.yaml \
--vae_ckpt experiment/experiment_VAE_test_1/stage2/checkpoints/0030000_vae.pt \
--input_dir datasets/ZZCX_2_1/test/LQ \
--output_dir results/3.20_vae_inference/gt_restructured_1 

"""

def main(args):

    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

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

    # if args.vae_ckpt:
    #     vae_sd = torch.load(args.vae_ckpt, map_location="cpu")
    #     if "state_dict" in vae_sd:
    #         vae_sd = vae_sd["state_dict"]
    #     vae_sd = {k.replace("module.", ""): v for k, v in vae_sd.items()}
    #     cldm.vae.load_state_dict(vae_sd, strict=True)
    #     print(f"Loaded VAE from {args.vae_ckpt}")
    # else:
    #     print("No VAE checkpoint provided, using initialized weights")

    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

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
    # cldm.eval().to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    os.makedirs(args.output_dir, exist_ok=True)

    for i, batch in enumerate(loader):
        # 加载数据（使用训练时的加载逻辑）
        to(batch, device)
        batch = batch_transform(batch)
        gt, lq, prompt, rgb = batch

        gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

        with torch.no_grad():
            # lq = lq * 2 - 1
            # z_0 = pure_cldm.vae_encode(lq)
            z_0 = pure_cldm.vae_encode(gt)
            decoded = pure_cldm.vae_decode(z_0)

            # （根据训练时的处理方式调整）
            decoded = (decoded + 1) / 2

        print()
        # 按原名保存图片
        index = i * cfg.train.batch_size
        for j in range(decoded.shape[0]):
            image_file = dataset.image_files_HQ[index + j]
            file_name = os.path.basename(image_file["image_path"])

            try:
                save_image(decoded[j],
                           os.path.join(args.output_dir, file_name),
                           normalize=False)
                print(f"Processed {file_name} -> {os.path.join(args.output_dir, file_name)}")
            except Exception as e:
                print(f"Error saving {file_name}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="训练配置文件路径")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE模型权重路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录")
    args = parser.parse_args()

    main(args)