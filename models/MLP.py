import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, dim:int, ratio:float):
        super().__init__()

        self.hdim = int(ratio * dim)

        self.mlp_0 = nn.Parameter(
            torch.randn(
                self.hdim*2, dim
            )
        )

        self.mlp_1 = nn.Parameter(
            torch.randn(
                dim, self.hdim
            )
        )

    def forward(self, x):
        gate, x = F.linear(x, self.mlp_0).chunk(2, dim=-1)
        gating = F.silu(gate) * x
        h = F.linear(gating, self.mlp_1)
        return h