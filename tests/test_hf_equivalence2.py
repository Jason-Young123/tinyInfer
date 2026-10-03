import os
import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.context import reset_context, set_context
from tinyinfer.utils.loader import load_weights


@pytest.fixture(scope="module")
def local_model_path():
    path = os.getenv("TINYINFER_MODEL")
    if not path:
        pytest.skip("TINYINFER_MODEL is not set")
    return path


def _set_single_prefill_context(num_tokens: int, block_size: int):
    num_blocks = math.ceil(num_tokens / block_size)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    set_context(
        q_lens=torch.tensor([num_tokens], dtype=torch.int32),
        context_lens=torch.tensor([num_tokens], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32),
        max_seqlen_q=num_tokens,
        slot_mapping=slot_mapping,
        block_tables=block_table,
        is_prefill=torch.tensor([True]),
        is_sample=torch.tensor([True]),
    )


@pytest.mark.model
def test_cold_prefill_logits_match_hf():
    local_model_path = "/home/jason/huggingface/Qwen3-0.6B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    cfg = hf_model.config
    cfg.kvcache_block_size = 16
    head_dim = int(
        getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    )

    tiny_model = Qwen3ForCausalLM(cfg, kv_cache=None).to(device, dtype=dtype)
    load_weights(tiny_model, local_model_path, strict=True)

    prompt = "Explain KV cache in one sentence."
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    positions = torch.arange(len(input_ids), dtype=torch.long, device=device)

    num_blocks = math.ceil(len(input_ids) / cfg.kvcache_block_size)
    cache = PagedKVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=max(4, num_blocks),
        block_size=cfg.kvcache_block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    tiny_model.set_kv_cache(cache)
    tiny_model.eval()

    _set_single_prefill_context(len(input_ids), cfg.kvcache_block_size)
    # 把 context tensors 移到模型 device。
    from tinyinfer.utils.context import get_context
    ctx = get_context()
    set_context(
        q_lens=ctx.q_lens.to(device),
        context_lens=ctx.context_lens.to(device),
        cu_seqlens_q=ctx.cu_seqlens_q.to(device),
        max_seqlen_q=ctx.max_seqlen_q,
        slot_mapping=ctx.slot_mapping.to(device),
        block_tables=ctx.block_tables.to(device),
        is_prefill=ctx.is_prefill.to(device),
        is_sample=ctx.is_sample.to(device),
    )

    try:
        with torch.no_grad():
            hf_logits = hf_model(ids.unsqueeze(0)).logits[:, -1, :]

            tiny_hidden, _ = tiny_model(ids, positions)
            tiny_logits = tiny_model.compute_logits(tiny_hidden[-1:])
    finally:
        reset_context()

    diff = (tiny_logits.float() - hf_logits.float()).abs()
    print("max_abs_error =", diff.max().item())
    print("mean_abs_error =", diff.mean().item())

    # CPU FP32 可以更严；BF16 baseline 先用较宽阈值，观察后再收紧。
    if dtype == torch.float32:
        torch.testing.assert_close(
            tiny_logits.float(),
            hf_logits.float(),
            atol=1e-4,
            rtol=1e-4,
        )
    else:
        torch.testing.assert_close(
            tiny_logits.float(),
            hf_logits.float(),
            atol=5e-2,
            rtol=5e-2,
        )
