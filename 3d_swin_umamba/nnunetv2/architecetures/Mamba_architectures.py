import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from nnunetv2.architecetures.UMamba_2D import UMamba2D
from nnunetv2.architecetures.UMamba_3D import UMamba3D
from nnunetv2.architecetures.Swin_UMamba_2D import SwinUMamba2D
from nnunetv2.architecetures.Swin_UMamba_3D import SwinUMamba3D
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class UMamba(object):
   def __new__(self,
               input_size: Tuple[int, ...], 
               input_channels: int,
               n_stages: int,
               features_per_stage: Union[int, List[int], Tuple[int, ...]],
               conv_op: Type[_ConvNd],
               kernel_sizes: Union[int, List[int], Tuple[int, ...]],
               strides: Union[int, List[int], Tuple[int, ...]],
               n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
               num_classes: int,  
               n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
               conv_bias: bool = False,
               norm_op: Union[None, Type[nn.Module]] = None,
               norm_op_kwargs: dict = None,
               dropout_op: Union[None, Type[_DropoutNd]] = None,
               dropout_op_kwargs: dict = None,
               nonlin: Union[None, Type[torch.nn.Module]] = None,
               nonlin_kwargs: dict = None,
               deep_supervision: bool = False,
               stem_channels: int = None
               ):
   
      """conv_op can be used to determine 2d or 3d"""

      if conv_op in [nn.Conv2d, nn.ConvTranspose2d]:
         return UMamba2D(input_size, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                        strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision,
                        stem_channels)
      elif conv_op in [nn.Conv3d, nn.ConvTranspose3d]:
         return UMamba3D(input_size, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
                        strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision,
                        stem_channels)
      else:
         raise ValueError(f"conv_op must be one of {nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d}, "
                          f"but got {conv_op} instead!")

   """How is this accessed? 
    when we call the "static_VRAM_usage" function , it find this module and initialise with the parameters
    might need to modify the arhictrecture_kwargs in the planner for mamba """
   
class SwinUMamba(object):
   def __new__(self,
               input_channels: int,
               features_per_stage: Union[int, List[int], Tuple[int, ...]],
               conv_op: Type[_ConvNd],
               num_classes: int,  
               deep_supervision: bool = False
               ):
   
      """conv_op can be used to determine 2d or 3d"""

      if conv_op in [nn.Conv2d, nn.ConvTranspose2d]:
         return SwinUMamba2D(in_chans=input_channels, 
                             out_chans=num_classes,
                             feat_size=features_per_stage,
                             deep_supervision=deep_supervision,
                             hidden_size= 768)
      
      elif conv_op in [nn.Conv3d, nn.ConvTranspose3d]:
         return SwinUMamba3D(in_chans=input_channels, 
                             out_chans=num_classes,
                             feat_size=features_per_stage,
                             deep_supervision=deep_supervision,
                             hidden_size= 768)
      else:
         raise ValueError(f"conv_op must be one of {nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d}, "
                          f"but got {conv_op} instead!")