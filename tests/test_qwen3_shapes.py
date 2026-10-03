from types import SimpleNamespace

import torch

from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.context import reset_context, set_context


def make_fake_qwen3_config():
    return SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        kvcache_block_size=4,
    )


def test_qwen3_mixed_shape_contract_cpu():
    torch.manual_seed(0)
    cfg = make_fake_qwen3_config()

    cache = PagedKVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=8,
        block_size=cfg.kvcache_block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    model = Qwen3ForCausalLM(cfg, cache).float().eval()

    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                param.fill_(1.0)


    # 两条 request：第一条 q_len=1，第二条 q_len=3。
    input_ids = torch.tensor([7, 11, 12, 13], dtype=torch.long)
    positions = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    # seq0 使用 physical block 0；seq1 使用 physical block 1。
    set_context(
        q_lens=torch.tensor([1, 3], dtype=torch.int32),
        context_lens=torch.tensor([1, 3], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 4], dtype=torch.int32),
        max_seqlen_q=3,
        slot_mapping=torch.tensor([0, 4, 5, 6], dtype=torch.long),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
        is_prefill=torch.tensor([False, True]),
        is_sample=torch.tensor([True, True]),
    )

    try:
        with torch.no_grad():
            hidden = model(input_ids, positions)
            logits = model.compute_logits(hidden[[0, 3]])
    finally:
        reset_context()

    assert hidden.shape == (4, cfg.hidden_size)
    assert logits.shape == (2, cfg.vocab_size)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(logits).all()








