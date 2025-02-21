import os
from argparse import ArgumentParser
import copy
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from torchvision.utils import save_image

from diffbir.model import SwinIR
from diffbir.utils.common import instantiate_from_config, to
from torchvision import transforms

def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)

    # 加载 SwinIR 模型
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    swinir.eval().to(device)
    print(f"load SwinIR from {cfg.train.swinir_path}")

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
    print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # 定义保存图片的文件夹
    output_folder = "datasets/ZZCX_2_1/train/swinir_LQ"
    os.makedirs(output_folder, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    postprocess = transforms.Compose([
        transforms.ToPILImage()
    ])

    for batch_idx, batch in enumerate(tqdm(loader)):
        to(batch, device)
        batch = batch_transform(batch)
        _, lq, _, _ = batch  # 只需要低质量图像 lq
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

        with torch.no_grad():
            clean = swinir(lq)

        # 保存处理后的图片
        for i in range(clean.shape[0]):
            image_idx = batch_idx * cfg.train.batch_size + i
            save_image(clean[i], os.path.join(output_folder, f"image_{image_idx}.png"))

        # 后处理输出结果
        # output_image = postprocess(output_tensor.squeeze(0).clamp(0, 1))

    print("All images have been processed and saved.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)