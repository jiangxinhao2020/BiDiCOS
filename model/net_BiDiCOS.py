import os
import torch
import warnings
from functools import partial
from collections import OrderedDict
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
from denoisingdiffusionpytorch.denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention
from torch.distributions.uniform import Uniform
import numpy as np
from timm.models.layers import DropPath
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)

            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        # time_embed

        self.time_embed = nn.ModuleList()
        for i in range(0, len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, timesteps, cond_img):
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0]))
        time_token = time_token.unsqueeze(dim=1)

        B = cond_img.shape[0]
        outs = []
        x, H, W = self.patch_embed1(cond_img)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 2
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 3
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, timesteps, cond_img):
        x = self.forward_features(timesteps, cond_img)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))



class Upsample1(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

class Addition(nn.Module):
    def __init__(self):
        super(Addition, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(256, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(256, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(32*2, 32, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3_r,x3_d):
        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        # print(SCA_ca.shape)
        # print(SCA_d_ca.shape)
        SCA_concat = torch.cat((SCA_ca, SCA_d_ca), 2)
        # print(SCA_concat.shape)
        SCA_concat = F.softmax(SCA_concat, dim=2)
        # print(SCA_concat.shape)
        SCA_split = torch.chunk(SCA_concat,2,dim=2)
        # print(SCA_split.shape)
        SCA_ca = SCA_split[0]
        # print(SCA_ca.shape)
        SCA_d_ca = SCA_split[1]
        # print(SCA_d_ca.shape)
        SCA_3_o = x3_r * SCA_ca.expand_as(x3_r)
        SCA_3d_o = x3_d * SCA_d_ca.expand_as(x3_d)

        Co_ca3 = SCA_3_o + SCA_3d_o

        return  Co_ca3

class ASFF_4(nn.Module):
    def __init__(self, inter_dim=256):
        super(ASFF_4, self).__init__()

        self.inter_dim = 256


        self.weight_level_0 = BasicConv(self.inter_dim, 128, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, 128, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, 128, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, 128, 1, 1)

        self.weight_levels = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 1, 1)
        self.skip_add = nn.quantized.FloatFunctional()
        self.addition = Addition()

    def forward(self, input0, input1):

        input00 = self.weight_levels(input0)
        input11 = self.weight_levels(input1)

        levels_weight_v = torch.cat((input00, input11), 1)
        levels_weight = F.softmax(levels_weight_v, dim=1)
        fused_out_reduced_1 = input0 * levels_weight[:, 0:1, :, :]
        fused_out_reduced_2 = input1 * levels_weight[:, 1:2, :, :]
        fused_out_reduced = self.addition(fused_out_reduced_1,fused_out_reduced_2)
        out = self.conv(fused_out_reduced)

        return out



class BlockBody(nn.Module):
    def __init__(self, channels=[128, 128, 128, 128]):
        super(BlockBody, self).__init__()

        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.asff_scalezero31 = ASFF_4(inter_dim=channels[0])
        self.asff_scalezero32 = ASFF_4(inter_dim=channels[0])
        self.asff_scalezero33 = ASFF_4(inter_dim=channels[0])
        self.asff_scalezero34 = ASFF_4(inter_dim=channels[0])
        self.asff_scalezero35 = ASFF_4(inter_dim=channels[0])
        self.asff_scalezero36 = ASFF_4(inter_dim=channels[0])
        self.asff_scalezero37 = ASFF_4(inter_dim=channels[0])

        self.blocks_concat10 = nn.Sequential(
            BasicConv(256 , 256, 3),
        )

    def forward(self, x):
        x0,x1, x2, x3 = x

        x01 = self.asff_scalezero31(x0, self.upsample_scaleone3_2(x1))

        x23 = self.asff_scalezero33(x2, self.upsample_scaleone3_2(x3))

        x0123 = self.asff_scalezero37(x01, self.upsample_scaletwo3_4(x23))

        scalezero = self.blocks_concat10(x0123)

        return scalezero

class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 out_channels=256):
        super(AFPN, self).__init__()

        self.fp16_enabled = False

        self.conv0 = BasicConv(in_channels[0], 256, 1)
        self.conv1 = BasicConv(in_channels[1], 256, 1)
        self.conv2 = BasicConv(in_channels[2], 256, 1)
        self.conv3 = BasicConv(in_channels[3], 256, 1)

        self.body = nn.Sequential(
            BlockBody([256, 256, 256, 256])
        )

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0 = self.body([ x0,x1, x2, x3])


        return out0


class BFFM(nn.Module):
    def __init__(self):
        super(BFFM, self).__init__()

        self.conv21 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,padding=7//2,bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1f = nn.Conv2d(32, 32, kernel_size=1)

    def forward(self, x3_r, x3_d):
        x000 = x3_r + x3_d
        x001 = torch.cat([x3_r, x000], dim=1)
        x002 = torch.cat([x3_d, x000], dim=1)

        max_out1,_ = torch.max(x001,dim=1,keepdim=True)
        avg_out1 = torch.mean(x001, dim=1, keepdim=True)
        x1 = torch.cat([max_out1,avg_out1],dim=1)
        out1 = self.sigmoid(self.conv21(x1))

        max_out2, _ = torch.max(x002, dim=1, keepdim=True)
        avg_out2 = torch.mean(x002, dim=1, keepdim=True)
        x2 = torch.cat([max_out2, avg_out2], dim=1)
        out2 = self.sigmoid(self.conv21(x2))

        out = torch.cat([out1, out2], dim=1)
        out = F.softmax(out, dim=1)

        x3_r = x3_r + x3_r * out[:, 0:1, :, :]
        x3_d = x3_d + x3_d * out[:, 1:2, :, :]

        return x3_r, x3_d

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x




class Decoder1(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder1, self).__init__()
        self.num_classes = class_num
        embedding_dim = dim
        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )
        resnet_block = partial(ResnetBlock, groups=8)
        self.down1 = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.down2 = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )
        self.up1 = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample1(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample1(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )
        self.up2 = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample1(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample1(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )
        self.pred1 = nn.Sequential(
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )
        self.pred2 = nn.Sequential(
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

        self.afpn1 = AFPN()
        self.afpn2 = AFPN()
        self.fusion = BFFM()


    def forward(self, inputs, timesteps,x,y):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        ##############################################
        _y = [y]
        for blk1 in self.down1:
            if isinstance(blk1, ResnetBlock):
                y = blk1(y, t)
                _y.append(y)
            else:
                y = blk1(y)
        _x = [x]
        for blk2 in self.down2:
            if isinstance(blk2, ResnetBlock):
                x = blk2(x, t)
                _x.append(x)
            else:
                x = blk2(x)
        ############## MLP decoder on C1-C4 ###########
        out1 = self.afpn1(inputs)
        out2 = self.afpn2(inputs)

        # fusion x_feat and x then transposed conv
        # main decoder
        y = torch.cat([out1, y], dim=1)
        for blk1 in self.up1:
            if isinstance(blk1, ResnetBlock):
                y += _y.pop()
                y = blk1(y, t)
            else:
                y = blk1(y)
        x = torch.cat([out2, x], dim=1)
        for blk2 in self.up2:
            if isinstance(blk2, ResnetBlock):
                x += _x.pop()
                x = blk2(x, t)
            else:
                x = blk2(x)

        x,y = self.fusion(x, y)
        pred_depth = self.pred1(y)
        pred_gt = self.pred2(x)
        return pred_gt, pred_depth

class net(nn.Module):
    def __init__(self, class_num=2, mask_chans=0, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        self.backbone = pvt_v2_b2(in_chans=3, mask_chans=mask_chans)
        self.decode_head1 = Decoder1(dims=[64, 128, 320, 512], dim=256, class_num=class_num, mask_chans=mask_chans)
        self._init_weights()  # load pretrain

    def forward(self, x, y, timesteps, cond_img):
        # Feature Extraction
        features = self.backbone(timesteps, cond_img)

        gt_pred, depth_pred = self.decode_head1(features, timesteps, x, y)

        return gt_pred, depth_pred

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0',
            'pvt_v2_b1',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b4_m',
            'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        pretrained_dict = torch.load(self._download_weights('pvt_v2_b2')) #for save mem
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x,y, timesteps, cond_img):
        return self.forward(x, y,timesteps, cond_img)

    def extract_features(self, cond_img):
        # do nothing
        return cond_img


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass

