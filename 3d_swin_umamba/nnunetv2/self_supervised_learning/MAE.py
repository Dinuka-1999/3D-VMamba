import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from nnunetv2.paths import pretrained_models
import os
join = os.path.join


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from nnunetv2.utilities.pos_embed import get_3d_sincos_pos_embed
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


# class LocalGAT3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#         self.attn = nn.Linear(out_channels * 2, 1)
#         self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
        
#     def window_partition(self, x, window_size=7):
#         """
#         Args:   
#             x: (B, C, D, H, W)
#             window_size (int): window size
#         Returns:
#             windows: (num_windows*B, C, window_size, window_size, window_size)
#         """
#         # x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
#         B, C, D, H, W = x.shape
#         x = x.view(B, C,
#                    D // window_size, window_size,
#                    H // window_size, window_size,
#                    W // window_size, window_size)
#         x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1, C, window_size, window_size, window_size)
#         #(num_windows*B, C, window_size, window_size, window_size)
#         return x

#     def window_reverse(self,windows, window_size, D, H, W):
#         """
#         Args:
#             windows: (num_windows*B, C, window_size, window_size, window_size)
#             window_size (int): Window size
#             D (int): Depth of image
#             H (int): Height of image
#             W (int): Width of image
#         Returns:
#             x: (B, D, H, W, C)
#         """
#         nD, nH, nW = D // window_size, H // window_size, W // window_size
#         B = windows.shape[0] // (nD * nH * nW)
#         x = windows.view(B, nD, nH, nW, -1, window_size, window_size, window_size)
#         x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
#         return x.view(B, D, H, W, -1)

#     def forward(self, x):
#         x = x.permute(0, 4, 1, 2, 3).contiguous() #(B, C, D, H, W)
#         h = self.proj(x)
#         window_size = math.gcd(h.shape[2], h.shape[3], h.shape[4]) 

#         windows = self.window_partition(h, window_size=window_size)  # D*H*W*C

#         neighbors = self.extract_3d_patches(windows) # D*H*W*C*9
#         h_center = windows.unsqueeze(2).expand_as(neighbors) # D*H*W*C*9

#         concat = torch.cat([h_center, neighbors], dim=1).permute(0,3,4,5,2,1).contiguous() # D*H*W*9*C*2
#         e = self.attn(concat) # D*H*W*9*1
#         alpha = torch.softmax(self.leakyRelu(e), dim=4).permute(0, 5, 4, 1, 2, 3) # D*H*W*9*1 -> B,1,9,WS,WS,WS
#         out = (alpha * neighbors).sum(dim=2) #B,C, WS,WS,WS
#         out = self.window_reverse(out, window_size=windows.shape[2], D=x.shape[2], H=x.shape[3], W=x.shape[4])
#         return out
    
#     def extract_3d_patches(self, x, padding=1):
#         """
#         Extract only the 8 diagonal neighbors + center.

#         Input:
#             x: (B, C, D, H, W)

#         Output:
#             patches: (B, C, 9, D, H, W)
#         """
#         B, C, D, H, W = x.shape

#         # Pad once
#         x = F.pad(
#             x,
#             (padding, padding,
#             padding, padding,
#             padding, padding)
#         )

#         # Relative offsets:
#         offsets = [
#             (-1, -1, -1),
#             (-1, -1,  1),
#             (-1,  1, -1),
#             (-1,  1,  1),
#             ( 0,  0,  0),   # center
#             ( 1, -1, -1),
#             ( 1, -1,  1),
#             ( 1,  1, -1),
#             ( 1,  1,  1),
#         ]

#         neighbors = []

#         for dz, dy, dx in offsets:
#             patch = x[
#                 :,
#                 :,
#                 1 + dz : 1 + dz + D,
#                 1 + dy : 1 + dy + H,
#                 1 + dx : 1 + dx + W,
#             ]
#             neighbors.append(patch)
        
#         # Stack along neighbor dimension
#         patches = torch.stack(neighbors, dim=2)
#         return patches


class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        self.output_dim = embed_dim
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 4, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    

class PatchMerging3D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, strides= [2,2,2]):
        super().__init__()
        self.dim = dim
        self.reduction_8 = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.reduction_4 = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.reduction_2 = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
        self.stride = strides

    def forward(self, x):
        B, D, H, W, C = x.shape
        
        SHAPE_FIX = [-1, -1, -1]
        if (W % self.stride[2] != 0) or (H % self.stride[1] != 0) or (D % self.stride[0] != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = D // self.stride[0]
            SHAPE_FIX[1] = H // self.stride[1]
            SHAPE_FIX[2] = W // self.stride[2]

        # [2,2,1], [1,1,2], [1,2,1], [2,1,1]
        if self.stride == [2,2,2]:
            x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
            x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
            x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
            x3 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
            x4 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
            x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
            x6 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
            x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C

            if SHAPE_FIX[0] > 0:
                x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x4 = x4[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x5 = x5[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x6 = x6[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x7 = x7[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :SHAPE_FIX[2], :]


            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
            x = x.view(B, D//2, H//2, W//2, 8 * C)  # B D/2*H/2*W/2 8*C

            x = self.norm(x)
            x = self.reduction_8(x)

        elif self.stride == [1,2,2]:
            x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
            x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
            x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
            x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C

            if SHAPE_FIX[0] > 0:
                x0 = x0[:, :, :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x1 = x1[:, :, :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x2 = x2[:, :, :SHAPE_FIX[1], :SHAPE_FIX[2], :]
                x3 = x3[:, :, :SHAPE_FIX[1], :SHAPE_FIX[2], :]

            x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C
            x = x.view(B, D, H//2, W//2, 4 * C)  # B D H/2 W/2 4*C
            x = self.norm(x)
            x = self.reduction_4(x)

        elif self.stride == [2,1,2]:
            x0 = x[:, 0::2, :, 0::2, :]  # B D/2 H W/2 C
            x1 = x[:, 1::2, :, 0::2, :]  # B D/2 H W/2 C
            x2 = x[:, 0::2, :, 1::2, :]  # B D/2 H W/2 C
            x3 = x[:, 1::2, :, 1::2, :]  # B D/2 H W/2 C

            if SHAPE_FIX[0] > 0:
                x0 = x0[:, :SHAPE_FIX[0], :, :SHAPE_FIX[2], :]
                x1 = x1[:, :SHAPE_FIX[0], :, :SHAPE_FIX[2], :]
                x2 = x2[:, :SHAPE_FIX[0], :, :SHAPE_FIX[2], :]
                x3 = x3[:, :SHAPE_FIX[0], :, :SHAPE_FIX[2], :]

            x = torch.cat([x0, x1, x2, x3], -1)  # B D/2 H W/2 4*C
            x = x.view(B, D//2, H, W//2, 4 * C)  # B D/2 H W/2 4*C
            x = self.norm(x)
            x = self.reduction_4(x)
        
        elif self.stride == [2,2,1]:
            x0 = x[:, 0::2, 0::2, :, :]  # B D/2 H/2 W C
            x1 = x[:, 1::2, 0::2, :, :]  # B D/2 H/2 W C
            x2 = x[:, 0::2, 1::2, :, :]  # B D/2 H/2 W C
            x3 = x[:, 1::2, 1::2, :, :]  # B D/2 H/2 W C

            if SHAPE_FIX[0] > 0:
                x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :, :]
                x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :, :]
                x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :, :]
                x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :, :]

            x = torch.cat([x0, x1, x2, x3], -1)  # B D/2 H/2 W 4*C
            x = x.view(B, D//2, H//2, W, 4 * C)  # B D/2 H/2 W 4*C
            x = self.norm(x)
            x = self.reduction_4(x)
        
        elif self.stride == [1,1,2]:
            x0 = x[:, :, :, 0::2, :]  # B D H W/2 C
            x1 = x[:, :, :, 1::2, :]  # B D H W/2 C
            if SHAPE_FIX[0] > 0:
                x0 = x0[:, :, :, :SHAPE_FIX[2], :]
                x1 = x1[:, :, :, :SHAPE_FIX[2], :]
            x = torch.cat([x0, x1], -1)  # B D H W/2 2*C
            x = x.view(B, D, H, W//2, 2 * C)  # B D H W/2 2*C
            x = self.norm(x)
            x = self.reduction_2(x)
        
        elif self.stride == [1,2,1]:
            x0 = x[:, :, 0::2, :, :]  # B D H/2 W C
            x1 = x[:, :, 1::2, :, :]  # B D H/2 W C
            if SHAPE_FIX[0] > 0:
                x0 = x0[:, :, :SHAPE_FIX[1], :, :]
                x1 = x1[:, :, :SHAPE_FIX[1], :, :]
            x = torch.cat([x0, x1], -1)  # B D H/2 W 2*C
            x = x.view(B, D, H//2, W, 2 * C)  # B D H/2 W 2*C
            x = self.norm(x)
            x = self.reduction_2(x)

        elif self.stride == [2,1,1]:
            x0 = x[:, 0::2, :, :, :]  # B D/2 H W C
            x1 = x[:, 1::2, :, :, :]  # B D/2 H W C
            if SHAPE_FIX[0] > 0:
                x0 = x0[:, :SHAPE_FIX[0], :, :, :]
                x1 = x1[:, :SHAPE_FIX[0], :, :, :]
            x = torch.cat([x0, x1], -1)  # B D/2 H W 2*C
            x = x.view(B, D//2, H, W, 2 * C)  # B D/2 H W 2*C
            x = self.norm(x)
            x = self.reduction_2(x)

        return x


class SS3D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model #96
        self.d_state = d_state #
        self.d_conv = d_conv if isinstance(d_conv, (list, tuple)) else [d_conv, d_conv, d_conv]
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=6, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        )

        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=6, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=6, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=6, merge=True) # (K=6, D, N)
        self.Ds = self.D_init(self.d_inner, copies=6, merge=True) # (K=6, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, D, H, W = x.shape 
        L = H * W * D 
        K = 6 
        
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        x_hw = x.view(B, -1, L)
        x_hd = x.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, L)
        # x_hd = torch.transpose(torch.transpose(x, dim0=2, dim1=3).contiguous(), dim0=3, dim1=4).contiguous().view(B, -1, L)
        x_wd = torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L)

        x_hwhdwd = torch.stack([x_hw, x_hd, x_wd], dim=1).view(B, 3, -1, L) # (b, 3, d_inner, l)
        
        xs = torch.cat([x_hwhdwd, torch.flip(x_hwhdwd, dims=[-1])], dim=1) # (b, k, d, l) (B, K, d_inner, L)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ##### K* (dt_rank + d_state * 2)*L
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) ##### K * d_inner * L

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None, 
            delta_bias=dt_projs_bias, 
            delta_softplus=True, 
            return_last_state=False, 
        ).view(B, K, -1, L)  # (b, k, d_inner, l) 
        assert out_y.dtype == torch.float 
    
        inv_y = torch.flip(out_y[:, 3:6], dims=[-1]).view(B, 3, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        hd_y = torch.transpose(torch.transpose(out_y[:,1].view(B,-1,H,W,D), dim0=3, dim1=4).contiguous().view(B,-1,H,D,W),\
                               dim0=2,dim1=3).contiguous().view(B, -1, L)
        wd_y = torch.transpose(out_y[:, 2].view(B, -1, W, H, D), dim0=2, dim1=4).contiguous().view(B, -1, L)
        
        # invhd_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invhd_y = torch.transpose(torch.transpose(inv_y[:,1].view(B,-1,H,W,D), dim0=3, dim1=4).contiguous().view(B,-1,H,D,W),\
                               dim0=2,dim1=3).contiguous().view(B, -1, L)
        
        invwd_y = torch.transpose(inv_y[:, 2].view(B, -1, W, H, D), dim0=2, dim1=4).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], hd_y, invhd_y, wd_y, invwd_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, D, H, W, C = x.shape
        
        xz = self.in_proj(x) #####
        x, z = xz.chunk(2, dim=-1) # (b, d, h, w, d_inner)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x)) #####
        y1, y2, y3, y4, y5, y6 = self.forward_core(x) ######
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 + y5 + y6
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, D, H, W, -1) ######
        y = self.out_norm(y) #######
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class VSSLayer(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, #96
        depth, # 2
        attn_drop=0.,
        drop_path=0., #. 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    

class VSSMEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, strides = [2,2,2], depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.strides = strides
        self.dims = dims
        self.patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None) ## 3D convereted
        self.pos_embd = nn.Parameter(torch.zeros(1, 64, 64, 64, self.embed_dim), requires_grad=False)

        # WASTED absolute position embedding ======================
        # self.ape = False
        # if self.ape:
        #     self.patches_resolution = self.patch_embed.patches_resolution
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)
        # self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.gnn_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # gat_layer = LocalGAT3D(in_channels=dims[i_layer], out_channels=dims[i_layer])
            # self.gnn_layers.append(gat_layer)
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging3D(dim=dims[i_layer], norm_layer=norm_layer, strides=strides[i_layer])) # i_layer = 0,1,2
        self.initialise_weights()

    def initialise_weights(self):
        pos_embed = get_3d_sincos_pos_embed(64, 64, 64, self.pos_embd.shape[-1])
        self.pos_embd.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv3D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def random_masking(self,x, mask_ratio):
        B,D,H,W,C = x.shape
        L = D*H*W
        keep_mask = int((1-mask_ratio) * L)

        x_flat = x.view(B, L, C)

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        restore_index = torch.argsort(ids_shuffle, dim=1)
        keep_ids = ids_shuffle[:, :keep_mask]

        # mask = torch.zeros(B, L, device=x.device, dtype=x.dtype)
        # mask.scatter_(1, keep_ids, 1.0)

        # x_flat = x_flat * (1.0 - mask.unsqueeze(-1))

        # x_masked = x_flat.view(B, D, H, W, C)
        # mask = mask.view(B, D, H, W)

        # return x_masked, mask

        x_masked = torch.gather(x_flat, dim=1, index=keep_ids.unsqueeze(-1).repeat(1, 1, C)).view(B, D//2, H//2, W//2, C)    
        mask = torch.ones([B, L], device=x.device)
        mask[:, :keep_mask] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=restore_index).view(B, D, H, W)

        return x_masked, mask, restore_index


    def forward(self, x):
        # x_ret = []
        # x_ret.append(x)

        x = self.patch_embed(x)
        x = x + self.pos_embd
        x, mask, restore_index = self.random_masking(x, mask_ratio=0.5)
        # x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            # x = self.gnn_layers[s](x)
            x = layer(x)
            # x_ret.append(x.permute(0, 4, 1, 2, 3))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x, mask, restore_index

class SSL_decoder(nn.Module):
    def __init__(self, in_channels=[768, 384, 192, 96], **kwargs):
        super().__init__()
        self.in_channels = in_channels[0]
        self.output_channels = 1
        self.de_mask_tokens = nn.Parameter(torch.zeros(1, 1, 96)) # (1, 1, C)
        self.patch_embed = nn.Linear(in_channels[0], in_channels[1]) # 768 -> 384 (4,4,4)
        self.up_conv1 = nn.ConvTranspose3d(in_channels[1], in_channels[2], kernel_size=2, stride=2) # 384 -> 192 (8,8,8)
        self.norm1 = nn.BatchNorm3d(in_channels[2])
        self.act = nn.ReLU()
        self.up_conv2 = nn.ConvTranspose3d(in_channels[2], in_channels[3], kernel_size=4, stride=4) # 192 -> 96 (32,32,32)
        self.norm2 = nn.BatchNorm3d(in_channels[3])
        self.up_conv3 = nn.ConvTranspose3d(in_channels[3], self.output_channels, kernel_size=2, stride=2) # 96 -> 1 (128,128,128)
        self.norm3 = nn.BatchNorm3d(self.output_channels)
        self.final_conv = nn.Conv3d(self.output_channels, self.output_channels, kernel_size=3, padding='same') # 1 -> 1 (128,128,128)

        torch.nn.init.normal_(self.de_mask_tokens, std=.02)

    def de_masking(self, x, restore_index):
        B, C, D, H, W = x.shape
        L = D * H * W
        x_flat = x.view(B, L, C)
        masked_tokens = self.de_mask_tokens.repeat(B, restore_index.shape[1] + 1 -L, 1)
        x = torch.cat([x_flat, masked_tokens], dim=1)
        x = torch.gather(x, dim=1, index=restore_index.unsqueeze(-1).repeat(1, 1, C)).view(B, D*2, H*2, W*2, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def forward(self, x, restore_index):
        x = self.patch_embed(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (b, c, d, h, w)
        x = self.act(self.norm1(self.up_conv1(x)))
        x = self.act(self.norm2(self.up_conv2(x)))
        x = self.de_masking(x, restore_index)
        x = self.act(self.norm3(self.up_conv3(x)))
        x = self.final_conv(x)
        return x
    
class SSL(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        feat_size=[48, 96, 192, 384, 768],
        strides=[2,2,2,2],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        res_block: bool = True,
        spatial_dims=3,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.strides = strides
        self.spatial_dims = spatial_dims

        self.vssm_encoder = VSSMEncoder(patch_size=strides[0], in_chans=in_chans, strides=strides[1:])
        self.ssl_decoder = SSL_decoder()
    
    def build_loss(self, imgs, pred, mask):

        loss = (pred - imgs) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, x_in): 
        #x = (128,128,128,1)
        vss_outs, mask, restore_index = self.vssm_encoder(x_in) 
        decoder_out = self.ssl_decoder(vss_outs, restore_index)
        return decoder_out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True
