from typing import Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from openstl.modules.cnolstm_modules import CNOLSTMCell


class CNOLSTM_B_Model(nn.Module):
        def __init__(self, in_T: int, in_channels: int, num_hidden: int, n_layers: int, cno_block_args: dict, **kwargs) -> None:
            super(CNOLSTM_B_Model, self).__init__()
            self.in_T = in_T
            self.in_chanels = in_channels
            self.n_layers = n_layers
            self.cells = nn.ModuleList([CNOLSTMCell(cno_block_args, in_channels=(in_channels if n == 0 else num_hidden), num_hidden=num_hidden) for n in range(n_layers)])
            self.h_to_out = nn.Conv2d(num_hidden, in_channels, 1, 1, padding=0, bias=False)

        def forward(self, x, teacher_forcing_prob: float=0.5) -> torch.Tensor:
            total_T = x.shape[2] 
            next_frames = []
            # Initialize last predicted frame
            last_frame = x[:, :, self.in_T - 1, :, :]

            c_ts, h_ts = [None for _ in range(self.n_layers)], [None for _ in range(self.n_layers)]
            # Known input 
            for i in range(self.in_T - 1):
                # Deep layers
                for layer, cnolstm in enumerate(self.cells):
                    x_next = x[:, :, i, :, :] if layer == 0 else h_t
                    c_t, h_t = cnolstm(x_next, c_ts[layer], h_ts[layer])
                    c_ts[layer] = c_t
                    h_ts[layer] = h_t
                # Use hidden state from the last layer to reconstruct the output
                next_frames.append(self.h_to_out(h_t))
            # Predicted input
            for i in range(total_T - self.in_T):
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
                # Use the GT frame instead of the predicted one
                if self.training and use_teacher_forcing:
                    last_frame = x[:, :, self.in_T - 1 + i, :, :]
                # Deep layers
                for layer, cnolstm in enumerate(self.cells):
                    x_next = last_frame if layer == 0 else h_t
                    c_t, h_t = cnolstm(x_next, c_ts[layer], h_ts[layer])
                    c_ts[layer] = c_t
                    h_ts[layer] = h_t
                # Use hidden state from the last layer to reconstruct the output
                last_frame = self.h_to_out(h_t)
                next_frames.append(last_frame)

            next_frames = torch.stack(next_frames, dim=0).permute(1,2,0,3,4).contiguous()

            return next_frames
