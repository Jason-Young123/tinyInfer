import torch
import torch.nn.functional as F
from torch import nn


class LinearBase(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool=False, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features   # 输入shape的最后一维大小
        self.out_features = out_features # 输出shape的最后一维大小
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x): # y = x@W.T + b
        return F.linear(x, self.weight, self.bias)


# [q, k, v] = x @ [Wq, Wk, Wv] = [x @ Wq, x @ Wk, x @ Wv]
class QKVParallelLinear(LinearBase):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, bias=False, dtype=None, device=None):
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        out_features = self.q_size + 2 * self.kv_size
        super().__init__( # 把q/k/v拼接为一个大矩阵进行一次线性变换, 再切割回原尺寸
            hidden_size, 
            out_features, 
            bias = bias,
            dtype = dtype,
            device = device
        )
    
    def split_qkv(self, x):
        q, k, v = torch.split(
            x,
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        return q, k, v


# SwiGLU, 需要先升维, 然后用激活函数element-wise处理元素, 然后降维; 涉及到W_gate, W_up和W_down三个矩阵
class MergedGateUpLinear(LinearBase):
    def __init__(self, hidden_size, intermediate_size, **kwargs): # kwargs代表其他的所有关键字参数
        self.intermediate_size = intermediate_size
        super().__init__(hidden_size, 2 * intermediate_size, **kwargs) # 输出为合并后的

    def split_gate_up(self, x): # 将up-scale后的输出拆分为gate和up
        return x.chunk(2, dim=-1) # 按最后一维平分x





