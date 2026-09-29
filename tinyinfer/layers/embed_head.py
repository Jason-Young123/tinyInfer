import torch
from torch import nn


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
class LMHead(nn.Linear):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            vocab_size,
            bias=bias,
            **kwargs,
        )




