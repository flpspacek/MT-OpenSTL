from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from openstl.models.cno_model import CNO_Model


class CNOLSTMCell(nn.Module):
    '''
    - Linear projection messes up the discretization invariance
    '''
    def __init__(self, cno_block_args: dict, in_channels: int, num_hidden: int, n_layers: int=1) -> None:
        super(CNOLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.n_layers = n_layers
        self.CNO_block = CNO_Model(**cno_block_args)
        self.LP = nn.Conv2d(in_channels+num_hidden, num_hidden, 1, 1, padding=0, bias=False)
        self.tanh_gate = F.tanh
        self.sigmoid_gate = F.sigmoid

    def forward(self, x_t, h_t, c_t) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_t is None:
            B, C, H, W = x_t.shape
            h_t = torch.zeros(B, self.num_hidden, H, W).to(x_t.device)
        if c_t is None:
            B, C, H, W = x_t.shape
            c_t = torch.zeros(B, self.num_hidden, H, W).to(x_t.device)

        # Add the hidden state as another channel
        x = torch.concat((x_t, h_t), dim=1)

        # Do the linear transformation
        x = self.LP(x)

        # CNO block
        cno_out = self.CNO_block(x)

        # Pass the output of CNO block through tanh and add it to the last cell state    
        cell = self.tanh_gate(cno_out)
        
        # Pass the output of CNO block through sigmoid
        F_t = self.sigmoid_gate(cno_out)

        # Multiply tanh of new cell state with sigmoid of CNO block output to obtain new hidden state
        c_t1 = F_t * (c_t + cell)
        h_t1 = F_t * self.tanh_gate(c_t1)

        # Return new states

        return c_t1, h_t1
    