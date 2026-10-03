import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-1, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter( # 每个token共享同一组hidden维度缩放参数
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float() # 均方根用fp32计算, 确保数值稳定性
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(input_dtype)






