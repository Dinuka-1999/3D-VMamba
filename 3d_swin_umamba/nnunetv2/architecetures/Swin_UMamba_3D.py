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
    
class SwinUMamba3D(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        feat_size=[48, 96, 192, 384, 768],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        res_block: bool = True,
        spatial_dims=2,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.stem = nn.Sequential(
              nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
              nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )
        self.spatial_dims = spatial_dims
        
    def compute_conv_feature_map_size(self, input_size):
        output = np.prod([*self.feat_size, *[i for i in input_size]], dtype=np.int64) # stem
        return output