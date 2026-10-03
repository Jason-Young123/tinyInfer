import os
import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.context import reset_context, set_context
from tinyinfer.utils.loader import load_weights

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def _compare(a, b, tag):
    diff_ab = (a.float() - b.float()).abs()
    print(tag, ":", diff_ab.max().item(), diff_ab.mean().item())




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
        attn_implementation="sdpa"
    ).to(device).eval()

    #print(
    #"HF attention implementation:",
    #hf_model.config._attn_implementation)

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
            # HuggingFace
            hf_out = hf_model(ids.unsqueeze(0), output_hidden_states=True,)
            hf_all_hidden_states = hf_out.hidden_states

            # tinyInfer
            tiny_final_hidden_states, tiny_all_hidden_states = tiny_model(
                ids,
                positions,
                output_hidden_states=True,
            )

    finally:
        reset_context()

    # 补全后面的diff检查逻辑
    print("num HF hidden states   =", len(hf_all_hidden_states))
    print("num tiny hidden states =", len(tiny_all_hidden_states))
    print()

    for i, (hf_hidden, tiny_hidden) in enumerate(
        zip(hf_all_hidden_states, tiny_all_hidden_states)
    ):
        # HF 带 batch 维：
        # [1, T, hidden_size]
        #
        # tinyInfer 是 flat token：
        # [T, hidden_size]
        hf_hidden = hf_hidden.squeeze(0)

        diff = (
            tiny_hidden.float() - hf_hidden.float()).abs()

        print(
            f"hidden[{i:02d}] "
            f"shape={tuple(tiny_hidden.shape)} "
            f"max_abs_error={diff.max().item():.8f} "
            f"mean_abs_error={diff.mean().item():.8f}"
        )
    

    
    # 这两个embed确保是完全相同的
    hf_embed = hf_out.hidden_states[0].squeeze(0)
    tiny_embed = tiny_all_hidden_states[0]
    
    # 查验decoder layer0
    hf_layer0 = hf_model.model.layers[0]
    tiny_layer0 = tiny_model.model.layers[0]

    # 查验decoder layer0 的 layernorm
    hf_norm = hf_layer0.input_layernorm(hf_embed)
    tiny_norm = tiny_layer0.input_layernorm(tiny_embed)
    diff = (hf_norm.float() - tiny_norm.float()).abs()
    print(
        "input_layernorm:",
        diff.max().item(),
        diff.mean().item()
    )

    # 查验decoder layer0 的 q/k/v
    hf_q = hf_layer0.self_attn.q_proj(hf_norm)
    hf_k = hf_layer0.self_attn.k_proj(hf_norm)
    hf_v = hf_layer0.self_attn.v_proj(hf_norm)
    tiny_qkv = tiny_layer0.self_attn.qkv_proj(tiny_norm)
    tiny_q, tiny_k, tiny_v = (tiny_layer0.self_attn.qkv_proj.split_qkv(tiny_qkv))
    diff_q = (hf_q.float() - tiny_q.float()).abs()
    diff_k = (hf_k.float() - tiny_k.float()).abs()
    diff_v = (hf_v.float() - tiny_v.float()).abs()
    print("q:", diff_q.max().item(), diff_q.mean().item())
    print("k:", diff_k.max().item(), diff_k.mean().item())
    print("v:", diff_v.max().item(), diff_v.mean().item())

    # 查验 q/k_norm
    hf_q = hf_q.reshape(-1, cfg.num_attention_heads, cfg.head_dim)
    hf_k = hf_k.reshape(-1, cfg.num_key_value_heads, cfg.head_dim)
    tiny_q = tiny_q.reshape(-1, cfg.num_attention_heads, cfg.head_dim)
    tiny_k = tiny_k.reshape(-1, cfg.num_key_value_heads, cfg.head_dim)
    hf_qn = hf_layer0.self_attn.q_norm(hf_q)
    hf_kn = hf_layer0.self_attn.k_norm(hf_k)
    tiny_qn = tiny_layer0.self_attn.q_norm(tiny_q)
    tiny_kn = tiny_layer0.self_attn.k_norm(tiny_k)
    _compare(hf_qn, tiny_qn, "qn")
    _compare(hf_kn, tiny_kn, "kn")

    # 查验ROPE
    hf_qn_hf = hf_qn.transpose(0,1).unsqueeze(0)
    hf_kn_hf = hf_kn.transpose(0,1).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(hf_qn_hf, position_ids=positions.unsqueeze(0))
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_qn_hf, hf_kn_hf, cos, sin)
    hf_q_rope = hf_q_rope.squeeze(0).transpose(0,1)
    hf_k_rope = hf_k_rope.squeeze(0).transpose(0,1)

    tiny_q_rope,tiny_k_rope = tiny_layer0.self_attn.rotary_emb(tiny_qn, tiny_kn, positions)
    _compare(hf_q_rope.float(), tiny_q_rope.float(), "q_rope")
    _compare(hf_k_rope.float(), tiny_k_rope.float(), "k_rope")
  

    # 查验attention
    #hf_attn_out = hf_layer0.self_attn.attn(hf_q_rope, hf_k_rope, hf_v)
    #tiny_attn_out = tiny_layer0.self_attn.attn(tiny_q_rope, tiny_k_rope, tiny_v)
    #_compare(hf_attn_out, tiny_attn_out, "attn_out")
    








