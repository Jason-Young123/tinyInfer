from dataclasses import dataclass
import torch


# 解决的问题: ModelRunner 准备好了一批请求之后，如何把“这一轮推理的动态信息”
#（prefill/decode模式、变长序列长度、KV cache位置等）传递给模型 forward 和 Attention kernel?
@dataclass(slots=True) # dataclass无需手动写__init__构造函数; slots=True可以防止误加成员变量, 从而维护类结构静态稳定
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None # cumsum of seqlen_q, 用于存放各个seq的边界信息
    cu_seqlens_k: torch.Tensor | None = None # cumsum of seqlen_k, 用于存放各个seq的边界信息
    max_seqlen_q: int = 0                    # max seqlen_q, 用于fA中拼batch; 注意seqlen_q代表这轮prefill中实际需要新计算的token数目
    max_seqlen_k: int = 0                    # max seqlen_k, 用于fA中拼batch; seqlen_k代表这轮prefill中attention需要涉及的完整token数目
    slot_mapping: torch.Tensor | None = None    
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_CONTEXT = Context()


def set_context(**kwargs):
    global _CONTEXT
    _CONTEXT = Context(**kwargs)


def get_context() -> Context:
    return _CONTEXT


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
