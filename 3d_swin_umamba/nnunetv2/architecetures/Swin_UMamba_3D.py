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
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


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
    def compute_conv_feature_map_size(self, input_size):
        # assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
        #                                                                         "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
        #                                                                         "Give input_size=(x, y(, z))!"
        return np.prod([self.output_dim, *[i//j for i,j in zip(input_size, self.proj.kernel_size)]], dtype=np.int64)
    

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
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
        self.stride = strides

    def forward(self, x):
        B, D, H, W, C = x.shape
        
        SHAPE_FIX = [-1, -1, -1]
        if (W % 2 != 0) or (H % 2 != 0) or (D % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = D // 2
            SHAPE_FIX[1] = H // 2
            SHAPE_FIX[2] = W // 2

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
        x = self.reduction(x)

        return x
    
    def compute_conv_feature_map_size(self, input_size):
        return np.prod([self.dim * 2, input_size[0]//self.stride[0], input_size[1]//self.stride[1], input_size[2]//self.stride[2]], dtype=np.int64)


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
        x_hd = torch.transpose(torch.transpose(x, dim0=2, dim1=3).contiguous(), dim0=3, dim1=4).contiguous().view(B, -1, L)
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
    
    def compute_conv_feature_map_size(self, input_size):
        output  = np.int64(0)
        output += np.prod([self.d_inner * 2, *[i for i in input_size]], dtype=np.int64) # in_proj
        conv_output_size = [j+2*((self.d_conv[i]-1)//2)-self.d_conv[i]+1 for i,j in enumerate(input_size)]
        output += np.prod([self.d_inner, *conv_output_size], dtype=np.int64) # conv2d
        output += np.prod([4,self.dt_rank + self.d_state * 2+self.d_inner*2, *conv_output_size], dtype=np.int64) #forward_core
        output +=np.prod([self.d_model, *conv_output_size], dtype=np.int64) # out_proj
        return output
     

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
    def compute_conv_feature_map_size(self, input_size):
        return self.self_attention.compute_conv_feature_map_size(input_size)


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
    
    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for blk in self.blocks:
            output += blk.compute_conv_feature_map_size(input_size)
        if self.downsample is not None:
            output += self.downsample.compute_conv_feature_map_size(input_size)
        return output
    

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

        # WASTED absolute position embedding ======================
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
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

    def forward(self, x):
        x_ret = []
        x_ret.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 4, 1, 2, 3))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret
    
    def compute_conv_feature_map_size(self, input_size):
        input_sizes = []
        input_sizes.append(input_size)
        output = self.patch_embed.compute_conv_feature_map_size(input_size)
        input_size = [i//j for i,j in zip(input_size, self.patch_size)]
        input_sizes.append(input_size)

        for s in range(self.num_layers):
            output += self.layers[s].compute_conv_feature_map_size(input_size)
            if s < len(self.downsamples):
                output += self.downsamples[s].compute_conv_feature_map_size(input_size)
                input_size = [i//j for i,j in zip(input_size, self.strides[s])]
                input_sizes.append(input_size)
        return output, input_sizes
    
class SwinUMamba3D(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        feat_size=[48, 96, 192, 384, 768],
        strides=[2,2,2,2,2],
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
        self.stem = nn.Sequential(
              nn.Conv3d(in_chans, feat_size[0], kernel_size=7, stride=strides[0], padding=3),
              nn.InstanceNorm3d(feat_size[0], eps=1e-5, affine=True),
        )
        self.spatial_dims = spatial_dims
        self.vssm_encoder = VSSMEncoder(patch_size=strides[1], in_chans=feat_size[0], strides=strides[2:])
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder6 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # deep supervision support
        self.deep_supervision = deep_supervision
        self.out_layers = nn.ModuleList()
        for i in range(4):
            self.out_layers.append(UnetOutBlock(
                spatial_dims=spatial_dims, 
                in_channels=self.feat_size[i], 
                out_channels=self.out_chans
            ))

    def forward(self, x_in): 
        x1 = self.stem(x_in) 
        vss_outs = self.vssm_encoder(x1) 
        enc1 = self.encoder1(x_in) 
        enc2 = self.encoder2(vss_outs[0]) 
        enc3 = self.encoder3(vss_outs[1]) 
        enc4 = self.encoder4(vss_outs[2]) 
        enc5 = self.encoder5(vss_outs[3]) 
        enc_hidden = vss_outs[4] 
        dec4 = self.decoder6(enc_hidden, enc5) 
        dec3 = self.decoder5(dec4, enc4) 
        dec2 = self.decoder4(dec3, enc3) 
        dec1 = self.decoder3(dec2, enc2) 
        dec0 = self.decoder2(dec1, enc1) 
        dec_out = self.decoder1(dec0) 

        if self.deep_supervision:
            feat_out = [dec_out, dec1, dec2, dec3]
            out = []
            for i in range(4):
                pred = self.out_layers[i](feat_out[i])
                out.append(pred)
        else:
            out = self.out_layers[0](dec_out)

        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True

    def compute_conv_feature_map_size(self, input_size):
        output= np.prod([self.feat_size[0], *[i//2 for i in input_size]], dtype=np.int64) #stem
        output += np.prod([self.feat_size[0],*[i for i in input_size]], dtype=np.int64)*3 # encoder1
        input_size_0= [i for i in input_size]
        input_size = [i//2 for i in input_size] # after stem
        vssm_output, vssm_output_sizes = self.vssm_encoder.compute_conv_feature_map_size(input_size) 
        output += vssm_output

        for r in range(1,len(self.feat_size)):
            output+=np.prod([self.feat_size[r],*[i for i in vssm_output_sizes[r-1]]], dtype=np.int64)*3 # encoder2,3,4,5

        #decoders 
        for r in range(4):
            output+=np.prod([self.feat_size[4-r],*[i for i in vssm_output_sizes[3-r]]], dtype=np.int64)*4 # decoder6,5,4,3

        output+= np.prod([self.feat_size[0],*input_size_0], dtype=np.int64)*4 + np.prod([self.feat_size[0],*input_size_0], dtype=np.int64)*3# decoder2, out_layer

        if self.deep_supervision:
            for r in range(3):
                output+= np.prod([self.out_chans, *[i for i in vssm_output_sizes[r]]], dtype=np.int64)
        output+= np.prod([self.out_chans, *input_size_0], dtype=np.int64) # out_layer
        return output

def get_swin_umamba_3D_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True
):
    
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = SwinUMamba3D(
        in_chans=num_input_channels,
        out_chans=label_manager.num_segmentation_heads,
        feat_size=[48, 96, 192, 384, 768],
        deep_supervision=deep_supervision,
        hidden_size=768,
    )

    return model 