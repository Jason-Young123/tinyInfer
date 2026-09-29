import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    def forward(self, x):
        if x.shape[-1] % 2 != 0:
            raise ValueError("SiluAndMul expects an even last dimension")
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up



