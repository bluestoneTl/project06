import os
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torchvision.utils import save_image
from einops import rearrange  # 新增导入，用于维度重排
from PIL import Image
from accelerate import Accelerator
from diffbir.model import ControlLDM
from diffbir.utils.common import instantiate_from_config, to
from diffbir.dataset.codeformer import CodeformerDataset

"""
python inference_VAE.py \
--config configs/train/train_stage2.yaml \
--vae_ckpt experiment/experiment_VAE_test_1/stage2/checkpoints/0020000_vae.pt \
--input_dir datasets/ZZCX_2_1/test/LQ \
--output_dir results/img
"""

def main(args):

    accelerator = Accelerator()
    device = accelerator.device

    cfg = OmegaConf.load(args.config)

    cldm = instantiate_from_config(cfg.model.cldm)

    if args.vae_ckpt:
        vae_sd = torch.load(args.vae_ckpt, map_location="cpu")
        if "state_dict" in vae_sd:
            vae_sd = vae_sd["state_dict"]
        vae_sd = {k.replace("module.", ""): v for k, v in vae_sd.items()}
        cldm.vae.load_state_dict(vae_sd, strict=True)
        print(f"Loaded VAE from {args.vae_ckpt}")
    else:
        print("No VAE checkpoint provided, using initialized weights")

    cldm.vae.eval().to(device)
    pure_vae = accelerator.unwrap_model(cldm.vae)

    # 初始化数据加载器（使用训练时的配置）
    dataset = instantiate_from_config(cfg.dataset.train)

    os.makedirs(args.output_dir, exist_ok=True)

    for index in range(len(dataset)):
        # 加载数据（使用训练时的加载逻辑）
        gt, lq, prompt, rgb = dataset[index]

        # 转换为模型输入格式，将 (512, 512, 3) 变为 (1, 3, 512, 512)。跟训练匹配
        gt_tensor = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():

            posterior = pure_vae.encode(gt_tensor)
            z = posterior.mode()  
            decoded = pure_vae.decode(z)

            # （根据训练时的处理方式调整）
            decoded = (decoded + 1) / 2  

        # 获取原始文件名
        image_file = dataset.image_files_HQ[index]
        file_name = os.path.basename(image_file["image_path"])

        # 保存结果，按原名保存
        save_image(decoded[0], 
                   os.path.join(args.output_dir, file_name),
                   normalize=False)

        print(f"Processed {file_name} -> {os.path.join(args.output_dir, file_name)}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="训练配置文件路径")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE模型权重路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录")
    args = parser.parse_args()

    main(args)