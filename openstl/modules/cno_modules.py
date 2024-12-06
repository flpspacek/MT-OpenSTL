'''
- This code is based on code from: https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/main/CNO2d_vanilla_torch_version/CNO2d.py
'''


import os

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNO3D_LReLu(nn.Module):
    '''
    - LReLu activation function for 3d tensors, with tricubic interpolation
    '''
    def __init__(self,
                in_size: tuple[int, int, int],
                out_size: tuple[int, int, int]
                ) -> None:
        super(CNO3D_LReLu, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    @staticmethod
    def _cubic_interpolation(x: torch.Tensor, scale: Optional[tuple[float, float, float]]=None, size: Optional[tuple[int, int, int]]=None) -> torch.Tensor:
        if scale is None and size is None:
            raise ValueError('Either scale or size cannot be None')
        
        batch, channel, temporal, spatial_1, spatial_2 = x.shape

        # Spatial interpolation
        size_spatial = size[1:] if size is not None else None
        scale_spatial = scale[1:] if scale is not None else None
        x = x.view(batch * channel, 1, temporal, spatial_1, spatial_2)
        x = x.permute(0, 2, 1, 3, 4)
        #ic(x.shape)
        x_interpolated = F.interpolate(x.reshape(-1, 1, spatial_1, spatial_2), size=size_spatial, scale_factor=scale_spatial, mode='bicubic', antialias=True)
        _, _, new_spatial_1, new_spatial_2 = x_interpolated.shape

        # Temporal interpolation
        size_temporal = (1,size[0]) if size is not None else None
        scale_temporal = (1,scale[0]) if scale is not None else None
        x = x_interpolated.view(batch * channel, temporal, 1, new_spatial_1, new_spatial_2)
        x = x.permute(0, 3, 4, 1, 2)
        x_interpolated = F.interpolate(x.reshape(-1, 1, 1, temporal), size=size_temporal, scale_factor=scale_temporal, mode='bicubic', antialias=True)
        
        # Reconstruction
        _, _, _, new_temporal = x_interpolated.shape
        x = x_interpolated.view(batch * channel, new_spatial_1, new_spatial_2, new_temporal, 1)
        x = x.permute(0, 3, 1, 2, 4)

        # Restore batch and channel dimensions
        x = x.view(batch, channel, new_temporal, new_spatial_1, new_spatial_2)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #ic(x.shape)
        x = CNO3D_LReLu._cubic_interpolation(x, size = (2 * self.in_size[0], 2 * self.in_size[1], 2 * self.in_size[2]))
        x = self.act(x)
        x = CNO3D_LReLu._cubic_interpolation(x, size = self.out_size)

        return x


class CNOBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                in_size,
                out_size,
                use_bn = True
                ):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size  = in_size
        self.out_size = out_size
        self.convolution = torch.nn.Conv3d(in_channels   = self.in_channels,
                                            out_channels = self.out_channels,
                                            kernel_size  = 3,
                                            padding      = 1)

        if use_bn:
            self.batch_norm  = nn.BatchNorm3d(self.out_channels)
        else:
            self.batch_norm  = nn.Identity()
        self.act = CNO3D_LReLu(in_size = self.in_size, out_size = self.out_size)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)

        return self.act(x)
    

class LiftProjectBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                size,
                latent_dim = 64
                ):
        super(LiftProjectBlock, self).__init__()

        self.inter_CNOBlock = CNOBlock(in_channels       = in_channels,
                                        out_channels     = latent_dim,
                                        in_size          = size,
                                        out_size         = size,
                                        use_bn           = False)

        self.convolution = torch.nn.Conv3d(in_channels   = latent_dim,
                                            out_channels = out_channels,
                                            kernel_size  = 3,
                                            padding      = 1)


    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self,
                channels,
                size,
                use_bn = True
                ):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size     = size

        self.convolution1 = torch.nn.Conv3d(in_channels = self.channels,
                                            out_channels= self.channels,
                                            kernel_size = 3,
                                            padding     = 1)
        self.convolution2 = torch.nn.Conv3d(in_channels = self.channels,
                                            out_channels= self.channels,
                                            kernel_size = 3,
                                            padding     = 1)

        if use_bn:
            self.batch_norm1  = nn.BatchNorm3d(self.channels)
            self.batch_norm2  = nn.BatchNorm3d(self.channels)

        else:
            self.batch_norm1  = nn.Identity()
            self.batch_norm2  = nn.Identity()

        self.act           = CNO3D_LReLu(in_size  = self.size,
                                        out_size = self.size)
        
    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)

        return x + out
    
class ResNet(nn.Module):
    def __init__(self,
                channels,
                size,
                num_blocks,
                use_bn = True
                ):
        super(ResNet, self).__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock(channels = channels,
                                                size = size,
                                                use_bn = use_bn))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)

        return x
