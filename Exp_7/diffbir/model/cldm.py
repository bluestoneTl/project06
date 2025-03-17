from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from .controlnet import ControlledUnetModel, ControlNet, ControlNet_LCA, ControlNet_RCA
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor
    ):
        super().__init__()
        self.controlnet_LCA = ControlNet_LCA(**controlnet_cfg)
        self.controlnet_RCA = ControlNet_RCA(**controlnet_cfg)
        unet_cfg['controlnet_LCA'] = self.controlnet_LCA                # 这里传入controlnet到unet
        unet_cfg['controlnet_RCA'] = self.controlnet_RCA
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13

        self.conv = nn.Conv2d(
            in_channels=6,  # 输入通道数，因为输入张量的通道数为 6
            out_channels=3,  # 输出通道数，将通道数从 6 降低到 3
            kernel_size=1,  # 使用 1x1 卷积核，不改变特征图的大小
            stride=1,  # 步长为 1，保持尺寸不变
            padding=0  # 不使用填充，因为 1x1 卷积核且步长为 1 时不需要填充
        )

    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:       # 【test】
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_lca_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet_LCA.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    @torch.no_grad()
    def load_controlnet_lca_from_unet(self) -> Tuple[Set[str]]:
        # 初始化LCA模块
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet_LCA.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet_LCA.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    @torch.no_grad()
    def load_controlnet_rca_from_unet(self) -> Tuple[Set[str]]:
        # 仅初始化RCA模块
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet_RCA.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet_RCA.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        # condition: torch.Tensor,  # 新增条件特征作为输入
        cond_RGB: torch.Tensor,  # 新增条件RGB特征作为输入
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),    # prompt
            # cond_img shape: torch.Size([16, 3, 512, 512])    # 输入的cond_img的形状
            c_img=self.vae_encode(
                cond_img * 2 - 1,   # 像素值范围从 [0, 1] 转换为 [-1, 1]
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ), 
            c_rgb=cond_RGB,               # 【融合RGB图像方法二】               
            # c_img shape: torch.Size([16, 4, 64, 64])   # 原本输出的结果
        )


    def forward(self, x_noisy, t, cond, is_first_stage):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        c_rgb = cond["c_rgb"]  # 【融合RGB图像方法一/方法二】
        
        # control = self.controlnet(x=x_noisy, hint=c_img, rgb=c_rgb, timesteps=t, context=c_txt)  # 【融合RGB图像方法二】
        # control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=c_txt,
            # control=control,
            hint=c_img,
            rgb=c_rgb,
            only_mid_control=False,
            is_first_stage=is_first_stage,
        )
        return eps

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype
        # convert unet blocks to dtype
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # convert controlnet blocks and zero-convs to dtype
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # GroupNorm32 only works with float32
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)
