from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_extension():
    try:
        from tinyinfer import _C
    except ImportError as exc:
        raise RuntimeError(
            "tinyinfer._C is not built; run: "
            "MAX_JOBS=4 python -m pip install -e . --no-build-isolation"
        ) from exc
    return _C


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

    return _load_extension().flash_attention_forward(
        q,
        k,
        v,
        int(query_start_pos),
        float(scale),
        is_causal
    )




