import os
import torch
import pytest

from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config

local_model_path = "/home/jason/huggingface/Qwen3-0.6B"

# 辅助函数
def make_llm(model_path: str, **overrides):
    values = dict(
        model=model_path,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        max_model_len=512,
        kvcache_block_size=16,
        gpu_memory_utilization=0.80,
    )
    values.update(overrides)
    return LLM(Config(**values))


def build_prompt(question: str, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. "
                "Answer with one short sentence. "
                "Use only factual language."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )




"""
def test_greedy_generation_is_deterministic():
    llm = make_llm(local_model_path)
    params = SamplingParams(greedy=True, max_tokens=8)
    prompt = "The capital of France is"

    out1 = llm.generate([prompt], params)[0]#["token_ids"]
    out2 = llm.generate([prompt], params)[0]#["token_ids"]

    out1_token_ids = out1["token_ids"]
    out2_token_ids = out2["token_ids"]
    out1_text = out1["text"]
    out2_text = out2["text"]

    print("out1_token_ids:", out1_token_ids)
    print("out2_token_ids:", out2_token_ids)
    print("out1_text:", out1_text)
    print("out2_text:", out2_text)
"""

"""
def test_prefix_cache_does_not_change_output():
    llm = make_llm(local_model_path)
    params = SamplingParams(greedy=True, max_tokens=64)
    tokenizer = llm.tokenizer

    # 注意：两个 prompt 必须共享完整 prefix, 这样 prefix cache 才有意义; 因此Messages中的system content要长一点(作为shared context)
    prompt_a = build_prompt("What is a KV cache?", tokenizer)
    prompt_b = build_prompt("What is grouped-query attention?", tokenizer)

    # 第一次请求：建立 shared prefix 的 persistent cache
    out_a = llm.generate([prompt_a], params)[0]
    print("first output:")
    print(out_a["text"])
    print("cached tokens after first request:", out_a["num_cached_tokens"])

    # 第二次请求：应该命中前面的 system + user template prefix
    out_b = llm.generate([prompt_b], params)[0]
    print("\nsecond output:")
    print(out_b["text"])
    print("num_cached_tokens:", out_b["num_cached_tokens"])
"""

"""
def test_eos_stops_generation():
    llm = make_llm(local_model_path)
    tokenizer = llm.tokenizer
    params = SamplingParams(greedy=True, max_tokens=128)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer briefly.",
        },
        {
            "role": "user",
            "content": "Say hello and nothing else.",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    output = llm.generate([prompt], params)[0]
    token_ids = output["token_ids"]
    print("generated token ids:")
    print(token_ids)
    print("\ngenerated text:")
    print(output["text"])

    eos_id = tokenizer.eos_token_id
    print("\neos_token_id:", eos_id)
    print("contains eos:", eos_id in token_ids)

    if eos_id in token_ids:
        eos_pos = token_ids.index(eos_id)
        print("eos position:", eos_pos)
        print(
            "tokens after eos:",
            token_ids[eos_pos + 1:]
        )

        # EOS 后不应该继续生成
        assert len(token_ids[eos_pos + 1:]) == 0

    else:
        pytest.fail(
            "Model did not generate EOS token, "
            "cannot verify EOS stopping."
        )
"""

"""
def test_single_request_generation():
    llm = make_llm(
        local_model_path,
        max_num_seqs=1,
        max_num_batched_tokens=32,
    )

    params = SamplingParams(
        greedy=True,
        max_tokens=32,
    )

    prompts = [
        build_prompt(
            "Explain paged KV cache briefly.",
            llm.tokenizer,
        ),
        build_prompt(
            "What is RoPE?",
            llm.tokenizer,
        ),
        build_prompt(
            "Why does GQA reduce KV memory?",
            llm.tokenizer,
        ),
    ]

    outputs = []

    for prompt in prompts:
        out = llm.generate([prompt], params)[0]
        outputs.append(out["token_ids"])

        print("prompt:", prompt)
        print("tokens:", out["token_ids"])
        print("text:", out["text"])
        print("cached:", out["num_cached_tokens"])
        print()

    assert len(outputs) == 3
"""





def test_mixed_batch_generation():
    llm = make_llm(
        local_model_path,
        max_num_seqs=8,
        max_num_batched_tokens=16, # 注意这里至少要设置为block_size, 否则连一个完整的cache block都无法生成, 必然cache命中率为0
    )

    params = SamplingParams(
        greedy=True,
        max_tokens=32,
    )

    prompts = [
        build_prompt(
            "Explain paged KV cache briefly.",
            llm.tokenizer,
        ),
        build_prompt(
            "What is RoPE?",
            llm.tokenizer,
        ),
        build_prompt(
            "Why does GQA reduce KV memory?",
            llm.tokenizer,
        ),
    ]

    outputs = llm.generate(
        prompts,
        params,
    )

    token_ids = [
        item["token_ids"]
        for item in outputs
    ]

    for i, item in enumerate(outputs):
        print(f"request {i}")
        print("tokens:", item["token_ids"])
        print("text:", item["text"])
        print("cached:", item["num_cached_tokens"])
        print()

    # 三个请求都应该完成
    assert len(outputs) == len(prompts)

    # 每个请求应该至少生成一个 token
    for ids in token_ids:
        assert len(ids) > 0





