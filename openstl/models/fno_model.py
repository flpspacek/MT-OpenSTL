import random
from typing import Union, Optional
from math import ceil

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from openstl.modules.fno_modules import FNOBlock, ChannelMLP

class FNO_Model(nn.Module):
    """
    - Assuming non-zero lifting/projection_channel_ratio
    """

    def __init__(self,
                 model_type: str,
                 n_modes: Union[int, tuple[int, ...]],
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 ndim: int,
                 lifting_channel_ratio: int=2,
                 projection_channel_ratio: int=2,
                 n_layers: int=4,
                 activation: nn.Module=F.gelu,
                 channel_mlp_expansion: float=0.5,
                 channel_mlp_dropout: float=0.0,
                 **kwargs) -> None:
        super(FNO_Model, self).__init__()

        if model_type.lower() not in ['basic', 'mlp', 'skip']:
            raise ValueError('Invalid model type! The model_type parameter has to be one of: "basic", "mlp", "skip".')

        if isinstance(n_modes, tuple):
            assert len(n_modes) == ndim, "Invalid number of modes!"
            self.n_modes = n_modes
        elif isinstance(n_modes, int):
            self.n_modes = tuple([n_modes for _ in range(ndim)])
        else:
            self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.activation = activation
        self.model_type = model_type.lower()
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout

        #ic(self.n_modes)

        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * self.hidden_channels
        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        # Lift to higher dimension
        self.lifting = ChannelMLP(in_channels=self.in_channels, out_channels=self.hidden_channels, hidden_channels=self.lifting_channels, n_layers=2, activation=self.activation)
        # n_layers of intergral operators and activation functions
        self.fourier_layers = nn.ModuleList([FNOBlock(model_type=self.model_type, n_modes=self.n_modes, hidden_channels=hidden_channels, n_layers=self.n_layers, activation=self.activation, channel_mlp_expansion=self.channel_mlp_expansion, channel_mlp_dropout=self.channel_mlp_dropout) for _ in range(n_layers)])
        # Back to target dimension
        self.projection = ChannelMLP(in_channels=self.hidden_channels, out_channels=self.out_channels, hidden_channels=self.projection_channels, n_layers=2, activation=self.activation)

    def forward(self, x: torch.Tensor, output_shape: Optional[tuple[int, ...]] = None) -> torch.Tensor:
        """
        - Expecting out_shape >= in_shape
        """
        if output_shape is None:
            out_shapes = self.n_layers * [None]
        # Calculate output shape for each layer so it gradually gets closer to the desired output shape
        else:
            in_shape = x.shape[2:]
            out_in_ratio = tuple(map(lambda i, j: i / j, output_shape, in_shape))
            res_scaling_coefs = [np.power(a, (1./self.n_layers)) for a in out_in_ratio]
            out_shapes = [tuple([round((in_shape[i] * np.power(coef, layer)) / 2.) * 2 for i, coef in enumerate(res_scaling_coefs)]) for layer in range(1, self.n_layers)]                
            
            # Last shape have to match the final output shape
            out_shapes.append(output_shape)

        #ic(output_shape)
        #ic(out_shapes)
        x = self.lifting(x)

        for layer_idx in range(self.n_layers):
            x = self.fourier_layers[layer_idx](x, layer_idx, out_shapes[layer_idx])
            #ic(x.shape)
        
        x = self.projection(x)

        return x

