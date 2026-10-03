import os

import pytest
from transformers import AutoConfig

from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.loader import load_weights



def test_loader_loads_every_required_parameter():
    local_qwen3_path = "/home/jason/huggingface/Qwen3-0.6B"

    hf_config = AutoConfig.from_pretrained(
        local_qwen3_path, 
        trust_remote_code=True,
    )
    hf_config.kvcache_block_size = 16

    # loader accounting 本身不需要真正 KV storage，因此先 kv_cache=None。
    model = Qwen3ForCausalLM(hf_config, kv_cache=None)
    report = load_weights(
        model,
        local_qwen3_path,
        strict=True,
    )

    assert report.missing == []
    assert report.unexpected == []

    print("loaded parameters:", len(report.loaded))
    print("packed shards:", len(report.packed))
    print("skipped tied weights:", report.skipped_tied)

    print("loaded parameters in detail:", report.loaded)
    print("packed shards in detail:", report.packed)


