import numpy as np
from torch import nn

class UMamba(nn.Module):
    def __init__(self, input_channels, output_channels, base_num_features, num_pool_per_axis, num_conv_per_stage,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d, dropout_op=nn.Dropout3d,
                 nonlin_op=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, upscale_logits=False,
                 convolutional_pooling=False, convolutional_upsampling=False):
        super(UMamba, self).__init__()
        print("Using UMamba as the UNet architecture!")