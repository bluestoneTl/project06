import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from typing import Optional, Any

from .distributions import DiagonalGaussianDistribution
from .config import Config, AttnMode

from .CFW_utils import make_layer, ResidualDenseBlock, RRDB, ResBlock

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        print(f"building AttnBlock (vanilla) with {in_channels} in_channels")

        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        print(
            f"building MemoryEfficientAttnBlock (xformers) with {in_channels} in_channels"
        )
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


class SDPAttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        print(f"building SDPAttnBlock (sdp) with {in_channels} in_channels")
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in [
        "vanilla",
        "sdp",
        "xformers",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "sdp":
        return SDPAttnBlock(in_channels)
    elif attn_type == "xformers":
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()


class Encoder(nn.Module):
    def __init__(           # 参数在train_stage2.yaml 中的 vae_cfg:
        self,
        *,
        ch,     # 128
        out_ch,  # 3
        ch_mult=(1, 2, 4, 8),  # (1, 2, 4, 4)
        num_res_blocks,     # 2
        attn_resolutions,   # []
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,    # 3
        resolution,     # 256
        z_channels,     # 4
        double_z=True,
        use_linear_attn=False,
        **ignore_kwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,           
            2 * z_channels if double_z else z_channels,     # 2 * 4 = 8
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, return_fea=False):
        # timestep embedding
        temb = None
        # x.shape: torch.Size([16, 3, 512, 512])
        # downsampling
        hs = [self.conv_in(x)]           # nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)，具体在上面342行
        
        fea_list = []       # 【CFW输入Fe】

        for i_level in range(self.num_resolutions):     # num_resolutions = 4，有4层下采样层级
            for i_block in range(self.num_res_blocks):  # num_res_blocks = 2，每层有2个残差块
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if return_fea:  # 【CFW输入Fe】
                if i_level==1 or i_level==2:
                    fea_list.append(h)

            if i_level != self.num_resolutions - 1:     # 最后一层不进行下采样
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if return_fea:  # 【CFW输入Fe】
            return h, fea_list
        else:
            return h

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        **ignorekwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
class Decoder_Mix(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        num_fuse_block=2, fusion_w=1.0,     # 【CFW新增】
        **ignorekwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        self.fusion_w = fusion_w     # 【CFW新增】

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            if i_level != self.num_resolutions-1:   # 【CFW新增】
                if i_level != 0:
                    fuse_layer = Fuse_sft_block_RRDB(in_ch=block_out, out_ch=block_out, num_block=num_fuse_block)
                    setattr(self, 'fusion_layer_{}'.format(i_level), fuse_layer)

            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z, enc_fea):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != self.num_resolutions-1 and i_level != 0:      # 【CFW新增】
                cur_fuse_layer = getattr(self, 'fusion_layer_{}'.format(i_level))
                h = cur_fuse_layer(enc_fea[i_level-1], h, self.fusion_w)    # Fe Fd W       CFW模块
            
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
class Fuse_sft_block_RRDB(nn.Module):           #　【CFW模块】
    def __init__(self, in_ch, out_ch, num_block=1, num_grow_ch=32):
        super().__init__()
        self.encode_enc_1 = ResBlock(2*in_ch, in_ch)
        self.encode_enc_2 = make_layer(RRDB, num_block, num_feat=in_ch, num_grow_ch=num_grow_ch)
        self.encode_enc_3 = ResBlock(in_ch, out_ch)

    def forward(self, enc_feat, dec_feat, w=1):     # Fe Fd W
        enc_feat = self.encode_enc_1(torch.cat([enc_feat, dec_feat], dim=1))
        enc_feat = self.encode_enc_2(enc_feat)
        enc_feat = self.encode_enc_3(enc_feat)
        residual = w * enc_feat
        out = dec_feat + residual
        return out
    
class AutoencoderKL(nn.Module):

    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)     # z_channels=4, embed_dim=4
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):            # 外面的调用操作是，z = encoder(image).mode() * self.scale_factor
        # x shape: torch.Size([16, 3, 512, 512])
        h = self.encoder(x)         
        # h shape: torch.Size([16, 8, 64, 64])
        moments = self.quant_conv(h)    # 具体代码在上面569行
        # moments shape: torch.Size([16, 8, 64, 64])
        posterior = DiagonalGaussianDistribution(moments)           
        # posterior.mode()后 shape: torch.Size([16, 4, 64, 64])
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
class AutoencoderKLResi(nn.Module):     # 【CFW新增】

    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.fusion_w = 1.0
    
    def encode(self, x):
        h, enc_fea = self.encoder(x, return_fea=True)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, enc_fea

    def decode(self, z, enc_fea):
        z = self.post_quant_conv(z) 
        dec = self.decoder(z, enc_fea)
        return dec
    
    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea = self.encode(input)
        dec = self.decode(latent, enc_fea)
        return dec, posterior

# # 对于 AutoencoderKL 类 的 encode 方法的解释
#     def encode(self, x):
#         # x shape: torch.Size([16, 3, 512, 512])
#         h = self.encoder(x)
#         # 将输入 x 传入 encoder 中进行处理。
#         # encoder 是 Encoder 类的实例，首先进入 Encoder 的 __init__ 方法进行初始化，根据传入的 ddconfig 配置参数确定模型结构。
#         # 1. 初始卷积层：
#         #    - 输入 x 经过 self.conv_in 卷积层，根据 ddconfig 中 in_channels 为 3，ch 为 128，
#         #    - 该卷积层是 nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)，
#         #    - 输出的特征图通道数变为 128，空间尺寸不变，此时形状为 [16, 128, 512, 512]。
#         # 2. 下采样层级：
#         #    - 共有 self.num_resolutions 个下采样层级，根据 ddconfig 中 ch_mult 列表长度确定为 4 层。
#         #    - 在每个层级中，有 self.num_res_blocks 个残差块，这里为 2 个。
#         #    - 每个层级的残差块处理输入特征图时，假设不改变特征图形状（在满足一定条件下）。
#         #    - 每个层级处理完后，若不是最后一层，会进行下采样操作，下采样使空间尺寸减半，通道数根据 ch_mult 变化。
#         #    - 经过 4 个下采样层级后，特征图形状变为 [16, 512, 64, 64]（通道数变为 128 * 4 = 512，空间尺寸经过多次下采样变为 64x64）。
#         # 3. 中间模块：
#         #    - 中间模块包含 2 个残差块和 1 个注意力模块（但由于配置中不使用注意力模块，注意力模块不产生实际影响）。
#         #    - 经过中间模块处理后，特征图形状仍为 [16, 512, 64, 64]。
#         # 4. 输出模块：
#         #    - 先经过归一化层和激活函数，形状不变。
#         #    - 再经过 self.conv_out 卷积层，根据 ddconfig 中 z_channels 为 4，double_z 为 true，
#         #    - 该卷积层是 nn.Conv2d(in_channels=512, out_channels=2 * 4 = 8, kernel_size=3, stride=1, padding=1)，
#         #    - 输出特征图形状变为 [16, 8, 64, 64]。
#         # 最终 h 的形状为 [16, 8, 64, 64]。
#         moments = self.quant_conv(h)
#         # 将 encoder 输出的 h 传入 self.quant_conv 卷积层进行处理。
#         # self.quant_conv 是 nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)，
#         # 已知 ddconfig["z_channels"] 为 4，embed_dim 为 4，所以该卷积层是 nn.Conv2d(8, 8, 1)，
#         # 处理后特征图的形状不变，moments 形状仍为 [16, 8, 64, 64]。
#         posterior = DiagonalGaussianDistribution(moments)
#         # 使用 DiagonalGaussianDistribution 类对 moments 进行处理，得到一个对角高斯分布对象 posterior。
#         # 在 DiagonalGaussianDistribution 的 __init__ 方法中，将 moments 按通道维度分成均值和对数方差两部分，
#         # 并对对数方差进行了范围限制，同时计算了标准差和方差。
#         # 调用 posterior.mode() 方法后，会返回均值，此时形状为 [16, 4, 64, 64]（因为均值是按通道维度分成两部分后的一半）。
#         return posterior