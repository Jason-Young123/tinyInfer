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


# q_len=1 专用的 Paged Flash-Decoding Python wrapper
def flash_decoding(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    kvcache_block_size: int,
    scale: float,
) -> torch.Tensor:
    if q.device.type != "cuda":
        raise ValueError("Flash-Decoding requires CUDA tensors")
    if q.ndim != 3 or q.shape[0] != 1:
        raise ValueError("Flash-Decoding requires q shape [1, Hq, D]")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "K/V cache must be [num_blocks, block_size, Hkv, D]"
        )
    if int(context_len) <= 0:
        raise ValueError("context_len must be positive")

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_table = block_table.contiguous()

    return _load_extension().flash_decoding_forward( # 定义于bind.cpp
        q,
        k_cache,
        v_cache,
        block_table,
        int(context_len),
        int(kvcache_block_size),
        float(scale),
    )



