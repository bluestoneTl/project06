import torch
from torchvision import transforms
from PIL import Image
from diffbir.model.swinir import SwinIR
import os

# 配置模型参数
img_size = 64
patch_size = 1
in_chans = 3
embed_dim = 180
depths = [6, 6, 6, 6, 6, 6, 6, 6]
num_heads = [6, 6, 6, 6, 6, 6, 6, 6]
window_size = 8
mlp_ratio = 2
sf = 8
img_range = 1.0
upsampler = "nearest+conv"
resi_connection = "1conv"
unshuffle = True
unshuffle_scale = 8

swinir = SwinIR(
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_chans,
    embed_dim=embed_dim,
    depths=depths,
    num_heads=num_heads,
    window_size=window_size,
    mlp_ratio=mlp_ratio,
    sf=sf,
    img_range=img_range,
    upsampler=upsampler,
    resi_connection=resi_connection,
    unshuffle=unshuffle,
    unshuffle_scale=unshuffle_scale
)

swinir_path = "weights/my_swinir.pt" 
sd = torch.load(swinir_path, map_location="cpu")
if "state_dict" in sd:
    sd = sd["state_dict"]
sd = {
    (k[len("module."):] if k.startswith("module.") else k): v
    for k, v in sd.items()
}
swinir.load_state_dict(sd, strict=True)
swinir.eval()

preprocess = transforms.Compose([
    transforms.ToTensor()
])
postprocess = transforms.Compose([
    transforms.ToPILImage()
])

# 定义输入和输出文件夹路径
input_folder = "datasets/ZZCX_2_1/test/LQ_mini"  
output_folder = "datasets/ZZCX_2_1/test/swinir_LQ"  
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 读取图片
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGB')

        # 预处理图片
        input_tensor = preprocess(image).unsqueeze(0)  # 添加批次维度

        # 使用 SwinIR 模型处理图片
        with torch.no_grad():
            output_tensor = swinir(input_tensor)

        # 后处理输出结果
        output_image = postprocess(output_tensor.squeeze(0).clamp(0, 1))

        # 保存处理后的图片
        output_path = os.path.join(output_folder, filename)
        output_image.save(output_path)
        print(f"Processed and saved {output_path}")

print("All images have been processed and saved.")