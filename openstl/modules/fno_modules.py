import itertools

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing import Union

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, k: int=32, delta: float=0.1) -> None:
        super(LinearSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.delta = delta
        self.k = k

        # Learnable linear projections for Query, Key, Value
        self.W_f = nn.Conv1d(embed_dim, embed_dim, 1, bias=False)
        self.W_g = nn.Conv1d(embed_dim, embed_dim, 1, bias=False)
        self.W_h = nn.Conv1d(embed_dim, embed_dim, 1, bias=False)
        
        # Lazy initialization of projection matrices
        self.register_buffer('P1', None)
        self.register_buffer('P2', None)

    def _initialize_projection_matrices(self, n) -> None:
            """
            Initialize projection matrices P1 and P2 if not already initialized
            Args:
                n: Input sequence length (T*S1*S2 for 3D)
            """
            if self.P1 is None or self.P2 is None:
                # Generate random matrix R with Gaussian distribution
                R = torch.randn(self.k, n) / (self.k ** 0.5)
                
                # Create projection matrices P1 and P2
                self.P1 = self.delta * R
                self.P2 = torch.exp(-self.delta * R)

    def forward(self, x) -> torch.Tensor:
        """
        x: (batch_size, seq_len, embed_dim)
        Returns: (batch_size, seq_len, embed_dim)
        """
        reshaped = False
        shape = x.shape
        N = torch.prod(torch.tensor(x.shape[2:]))

        # Reshape if needed
        if x.ndim > 3:
            x = x.view((shape[0], shape[1], -1)) # batch_size, channels (embed_dim), N (temp*spat1*spat2)
            reshaped = True

        # Initialize projection matrices if needed
        self._initialize_projection_matrices(N)

        Q = self.W_f(x)  # (B, C, N)
        K = self.W_g(x)  # (B, C, N)
        V = self.W_h(x)  # (B, C, N)

        # Project K and V using P1 and P2
        # Move projection matrices to the same device as input
        P1 = self.P1.to(x.device)
        P2 = self.P2.to(x.device)
        
        # Projected key and value: [B, C, k]
        K_projected = torch.matmul(K, P1.t())  
        V_projected = torch.matmul(V, P2.t())
        
        scale = torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        attention_scores = torch.matmul(Q.permute((0, 2, 1)), K_projected) / scale  # [B, N, k]
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute output
        x = torch.matmul(attention_weights, V_projected.transpose(-2, -1))  # [B, N, C]
 
        if reshaped:
            x = x.view((shape[0], self.embed_dim, *shape[2:])) # (B, C, T, S1, S2)

        return x


class SpectralConv(nn.Module):
    """
    - Expecting only real-valued inputs
    - Expecting in_channels == out_channels
    - No trainable bias parameter
    - Using all n_modes
    - Precision: ['half', 'mixed', 'full']
    """

    def __init__(self, channels: int, n_modes: Union[int, tuple[int, ...]], precision='full') -> None:
        super(SpectralConv, self).__init__()

        # Precision
        self.fno_block_precision = precision

        # Setup modes for n dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        # Find  the shape of weight matric
        R_shape = (channels, *self.n_modes)
        # Initialize the weight matrix
        init_std = (1 / (channels))**0.5
        self.R = torch.normal(0, init_std, R_shape, dtype=torch.cfloat)
        self.R = nn.Parameter(self.R)


    def _contract(self, x: torch.Tensor, R: torch.Tensor):
        """
        - Assuming in_channels == out_channels
        """
        return x * R

    @staticmethod
    def _resample(x: torch.Tensor, res_scale: float, axis: list[int], output_shape: tuple[int, ...]) -> torch.Tensor:
        """
        - Allows super-resolution
        - A module for generic n-dimentional interpolation (Fourier resampling)
        """
        res_scale = [res_scale]*len(axis)

        if len(axis) == 1:
            return F.interpolate(x, size=output_shape[0], mode='linear', align_corners=True)
        if len(axis) == 2:
            return F.interpolate(x, size=output_shape, mode='bicubic', align_corners=True)
        
        X = torch.fft.rfftn(x.float(), norm='forward', dim=axis)
    
        new_fft_size = list(output_shape)
        new_fft_size[-1] = new_fft_size[-1]//2 + 1 # Redundant last coefficient
        new_fft_size_c = [min(i,j) for (i,j) in zip(new_fft_size, X.shape[-len(axis):])]
        out_fft = torch.zeros([x.shape[0], x.shape[1], *new_fft_size], device=x.device, dtype=torch.cfloat)

        mode_indexing = [((None, m//2), (-m//2, None)) for m in new_fft_size_c[:-1]] + [((None, new_fft_size_c[-1]), )]
        for _, boundaries in enumerate(itertools.product(*mode_indexing)):

            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            out_fft[idx_tuple] = X[idx_tuple]
        y = torch.fft.irfftn(out_fft, s= output_shape, norm='forward', dim=axis)

        return y


    def transform(self, x: torch.Tensor, output_shape: Optional[tuple[int, ...]] = None) -> torch.Tensor:
        if output_shape is None:
            return x
        
        in_shape = list(x.shape[2:])

        if in_shape == output_shape:
            return x
        
        return self._resample(x, 1.0, list(range(2, x.ndim)), output_shape=output_shape)

    def forward(self, x: torch.Tensor, output_shape: Optional[tuple[int, ...]] = None) -> torch.Tensor:
        """
        - Expected shape: (batch, ...)
        """
        _, _, *mode_sizes = x.shape
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        # Compute Fourier coefficients
        x = torch.fft.rfftn(x, norm="forward", dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        # Get relevant fourier modes
        slices_x = [slice(None), slice(None)]
        slices_x += [slice(0, mode) for mode in self.n_modes]
        rel_modes = x[slices_x]

        # Multiply relevant Fourier modes
        out_fft = self._contract(rel_modes, self.R)

        # Change number mode_sizes to match the output shape
        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        # Return to physical space
        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm="forward")
        return x
    
    @property
    def n_modes(self) -> list[int]:
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes) -> None:
        if isinstance(n_modes, int): # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space 
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off information from the last mode
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes


class FNOBlock(nn.Module):
    """
    - Encapsules single FNO layer
    """

    def __init__(self,
                 model_type: str,
                 n_modes: Union[int, tuple[int, ...]],
                 hidden_channels: int,
                 n_layers: int,
                 activation: nn.Module=F.gelu,
                 channel_mlp_expansion: float=0.5,
                 channel_mlp_dropout: float=0.0,
                 use_self_attention: bool=False,
                 precision: str='mixed') -> None:
        
        super(FNOBlock, self).__init__()

        self.model_type = model_type
        self.n_layers = n_layers
        self.activation = activation

        self.spec_conv = SpectralConv(channels=hidden_channels, n_modes=n_modes, precision=precision)
        self.W = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1)
        if model_type in ['mlp', 'skip']:
            self.channel_mlp = ChannelMLP(in_channels=hidden_channels, out_channels=hidden_channels, hidden_channels=round(hidden_channels*channel_mlp_expansion), dropout=channel_mlp_dropout)
        else:
            self.channel_mlp = None

        if use_self_attention:
            self.lin_self_attention = LinearSelfAttention(hidden_channels)
        else:
            self.lin_self_attention = None
    
    def _skip_connection(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(shape[0], shape[1], -1)
        x = self.W(x)

        x = x.view(shape[0], shape[1], *shape[2:])

        return x
    
    def forward(self, x: torch.Tensor, idx: int, output_shape: Optional[tuple[int, ...]] = None) -> torch.Tensor:
        # Skip connection with optional super-resolution
        x_fno_skip = self._skip_connection(x)
        x_fno_skip = self.spec_conv.transform(x_fno_skip, output_shape)

        # Spectral convolution
        x_fno = self.spec_conv(x, output_shape)

        if self.channel_mlp is not None:
            x_fno = self.channel_mlp(x_fno)

        if self.lin_self_attention is not None:
            x_fno = self.lin_self_attention(x_fno)

        # Aggregation
        if self.model_type == 'skip':
            x = self.spec_conv.transform(x, output_shape) + (x_fno_skip + x_fno) # + +
        else:
            x = x_fno_skip + x_fno

        if idx != (self.n_layers - 1):
            x = self.activation(x)

        return x


class ChannelMLP(nn.Module):
    """
    - Used as a Lift/Project block
    - Used as an optional MLP block after spectral convolution 
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int=None, n_layers: int=2, activation: nn.Module = F.gelu, dropout: float = 0.0) -> None:
        super(ChannelMLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.activation = activation

        self.nn = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0 and i == (self.n_layers - 1):
                self.nn.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.nn.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (self.n_layers - 1):
                self.nn.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.nn.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))
        
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = False
        shape = x.shape

        if x.ndim > 3:
            x = x.reshape((shape[0], shape[1], -1))
            reshaped = True

        for i, layer in enumerate(self.nn):
            x = layer(x)
            if i < self.n_layers - 1:
                x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        if reshaped:
            x = x.reshape((shape[0], self.out_channels, *shape[2:]))

        return x
