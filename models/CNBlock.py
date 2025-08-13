import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Depth_wise_Conv1d import DepthwiseConv1d
from .MLP import MLP

class CNBlock(nn.Module):
    def __init__(self, dim:int, kernel_size:int, ratio:float):
        super().__init__()

        self.hdim = int(ratio * dim)

        self.attn = DepthwiseConv1d(
            in_channels = dim,
            kernel_size = kernel_size,
            stride      = 1,
            padding     = kernel_size-1,
            dilation    = 1,
            bias        = True
        )
        self.mlp  = MLP(dim=dim, ratio=ratio)

        self.post_ln      = nn.LayerNorm(dim, eps=1e-8)
        self.attention_ln = nn.LayerNorm(dim, eps=1e-8)

    def forward(self, x):
        h = self.attn(self.post_ln(x).permute(0, 2, 1)).permute(0, 2, 1)
        h += x
        h = self.mlp(self.attention_ln(h)) + h
        return h