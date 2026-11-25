import torch
from torch import nn
import torch.nn.functional as F
from tinyinfer.utils.extension import load_extension


# 单卡 baseline 直接使用 PyTorch Embedding
"""
token_id: shape = [num_flat_tokens]
   ↓
查表
   ↓
hidden vector: shape = [num_flat_tokens, hidden_dim]
"""
class VocabEmbedding(nn.Embedding): # 初始化需提供hidden_size和token_id列表, 输出即为token向量
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids)



# 把 hidden state 投影到 vocabulary logits
"""
hidden vector: shape = [num_flat_tokens, hidden_dim]
    ↓
Linear
    ↓
vocabulary logits: shape = [num_flat_tokens, vocab_size], 即每个向量对应各个token的可能性份数
(后续配合sampler即可得到token_id)
"""
# [M, hidden_size] @ [vocab_size, hidden_size] -> [M, vocab_size]
class LMHeadBase(nn.Module):
    def __init__(self, hidden_size:int, vocab_size:int, bias:bool=False, dtype=None, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # 注意量化之后weight_packed和weight_scale的存储类型都是uint8
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=dtype, device=device), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(vocab_size, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, X):
        if X.shape[0] <= 32: # 小尺寸用自定义kernel, 暂不支持bias
            if self.bias is not None:
                raise RuntimeError("LMHeadBase Temporarily Not Support Bias")
            result = load_extension().lm_head_forward(X, self.weight)
        else: # 大尺寸fallback至F.linear
            result = F.linear(X, self.weight, self.bias)

        return result



# [M, hidden_size] @ [vocab_size, hidden_size] -> [M, vocab_size]
class LMHeadMXFP4(nn.Module):
    def __init__(self, hidden_size:int, vocab_size:int, bias:bool=False, dtype=None, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        assert hidden_size % 32 == 0
        # 注意量化之后weight_packed和weight_scale的存储类型都是uint8
        self.weight_packed = nn.Parameter(torch.empty(vocab_size, hidden_size // 2, dtype=torch.uint8, device=device), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(vocab_size, hidden_size // 32, dtype=torch.uint8, device = device), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(vocab_size, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, X):
        if self.bias is not None:
            raise RuntimeError("LMHeadMXFP4 Temporarily Not Support Bias")
        result = load_extension().lm_head_forward_mxfp4(X, self.weight_packed, self.weight_scale)
        return result