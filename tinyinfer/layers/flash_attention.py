import torch
from tinyinfer.utils.extension import load_extension


#_FLASH_LOGGED = False

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_start_pos: int,
    scale: float,
    is_causal:bool # 是否启用causal mask
) -> torch.Tensor:
    if q.device.type != "cuda":
        raise ValueError("FlashAttention requires CUDA tensors")
    
    #global _FLASH_LOGGED
    #if not _FLASH_LOGGED:
    #    print("[tinyInfer] FlashAttention CUDA backend is active")
    #    _FLASH_LOGGED = True

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    return load_extension().flash_attention_forward( # 调用由注册flash_attention_bind.cpp注册的函数名
        q,
        k,
        v,
        int(query_start_pos),
        float(scale),
        is_causal
    )




