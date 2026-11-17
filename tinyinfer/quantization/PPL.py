
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from tinyinfer.config import Config
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.context import reset_context, set_context
from tinyinfer.utils.loader import load_weights
from tinyinfer.utils.profiler import TimeProfiler


DEFAULT_MODEL = Path.home() / "huggingface" / "Qwen3-4B"


def resolve_quantization(model_path: Path, requested: str) -> str | None:
    if requested == "none":
        return None
    if requested == "mxfp4":
        return "mxfp4"

    metadata_path = model_path / "quantization_config.json"
    if not metadata_path.is_file():
        return None
    quantization = json.loads(metadata_path.read_text(encoding="utf-8")).get("quantization")
    if quantization not in (None, "mxfp4"):
        raise ValueError(f"Unsupported quantization: {quantization}")
    return quantization


@torch.inference_mode()
def evaluate_ppl(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    block_tables: torch.Tensor,
    max_length: int,
    stride: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Score every next-token target once, using a sliding context window."""
    total_tokens = input_ids.numel()
    if total_tokens < 2:
        raise ValueError("The corpus must tokenize to at least 2 tokens.")
    if max_length < 2 or not 1 <= stride <= max_length - 1 or chunk_size <= 0:
        raise ValueError("Require max_length >= 2, 1 <= stride < max_length, chunk_size > 0.")

    total_nll = torch.zeros((), dtype=torch.float64, device=device)
    scored_tokens = 0
    target_start = 1

    while target_start < total_tokens:
        target_end = min(target_start + stride, total_tokens)
        window_start = max(0, target_end - max_length)
        window_input = input_ids[window_start : target_end - 1]
        score_input_start = target_start - 1 - window_start

        for chunk_start in range(0, window_input.numel(), chunk_size):
            chunk_end = min(chunk_start + chunk_size, window_input.numel())
            q_len = chunk_end - chunk_start
            positions = torch.arange(chunk_start, chunk_end, device=device, dtype=torch.long)

            try:
                set_context(
                    context_lens=[chunk_end],
                    cu_seqlens_q=[0, q_len],
                    slot_mapping=positions,
                    block_tables=block_tables,
                )
                hidden, _ = model(
                    window_input[chunk_start:chunk_end],
                    positions,
                    output_hidden_states=False,
                )
            finally:
                reset_context()

            # Earlier tokens in this window populate KV but are not scored again.
            score_start = max(chunk_start, score_input_start)
            if score_start < chunk_end:
                logits = model.compute_logits(hidden[score_start - chunk_start :]).float()
                labels = input_ids[
                    window_start + score_start + 1 :
                    window_start + chunk_end + 1
                ]
                if logits.shape[0] != labels.numel():
                    raise RuntimeError("Logits/labels length mismatch.")

                total_nll += F.cross_entropy(logits, labels, reduction="sum").double()
                scored_tokens += labels.numel()

            del hidden

        target_start = target_end

    torch.cuda.synchronize(device)
    mean_nll = (total_nll / scored_tokens).item()
    return {
        "corpus_tokens": int(total_tokens),
        "scored_tokens": int(scored_tokens),
        "total_nll": total_nll.item(),
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure fixed-corpus PPL with tinyInfer.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Defaults to --model; use one shared tokenizer for BF16/MXFP4.")
    parser.add_argument("--text-file", type=Path, required=True, help="Fixed UTF-8 corpus.")
    parser.add_argument("--quantization", choices=("auto", "none", "mxfp4"), default="auto")
    parser.add_argument("--attention-backend", choices=("torch", "flash", "flashinfer"), default="flash")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    text_path = args.text_file.expanduser().resolve()
    tokenizer_path = (args.tokenizer or args.model).expanduser().resolve()

    for path in (model_path, tokenizer_path, text_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not model_path.is_dir() or not tokenizer_path.is_dir() or not text_path.is_file():
        raise ValueError("--model/--tokenizer must be directories and --text-file must be a file.")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This tinyInfer PPL path requires a CUDA device.")
    if args.block_size <= 0 or args.max_length < 2:
        raise ValueError("block_size must be positive and max_length must be >= 2.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16.")

    quantization = resolve_quantization(model_path, args.quantization)
    config = Config(
        model=str(model_path),
        max_model_len=args.max_length,
        kvcache_block_size=args.block_size,
        attention_backend=args.attention_backend,
        quantization=quantization,
    )
    config.load_hf_config()
    hf_config = config.hf_config
    hf_config.kvcache_block_size = args.block_size

    max_length = min(args.max_length, int(hf_config.max_position_embeddings))
    if not 1 <= args.stride <= max_length - 1:
        raise ValueError(f"stride must be in [1, {max_length - 1}].")

    # Runtime profiler events are not needed for an accuracy evaluation.
    TimeProfiler.PROFILER = False

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    text = text_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty corpus: {text_path}")
    input_ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if input_ids.numel() < 2:
        raise ValueError("The tokenized corpus must contain at least 2 tokens.")

    print(f"[PPL] model={model_path}")
    print(f"[PPL] quantization={quantization or 'none'} | dtype={args.dtype} | backend={args.attention_backend}")
    print(f"[PPL] tokenizer={tokenizer_path}")
    print(f"[PPL] corpus={text_path} | tokens={input_ids.numel()}")
    print(f"[PPL] max_length={max_length} | stride={args.stride} | chunk_size={args.chunk_size}")

    model = Qwen3ForCausalLM(
        hf_config, kv_cache=None, backend=args.attention_backend, quantization=quantization
    ).to(device=device, dtype=dtype)
    report = load_weights(model, model_path, strict=True)
    model.eval()
    print(f"[PPL] loaded parameters: {len(report.loaded)}")

    head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))
    num_blocks = math.ceil(max_length / args.block_size)
    kv_cache = PagedKVCache(
        num_layers=int(hf_config.num_hidden_layers),
        num_blocks=num_blocks,
        block_size=args.block_size,
        num_kv_heads=int(hf_config.num_key_value_heads),
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    model.set_kv_cache(kv_cache)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

    metrics = evaluate_ppl(
        model, input_ids, block_tables, max_length,
        args.stride, args.chunk_size, device,
    )

    print("\n========== tinyInfer PPL ==========")
    print(f"Corpus tokens : {metrics['corpus_tokens']}")
    print(f"Scored tokens : {metrics['scored_tokens']}")
    print(f"Total NLL     : {metrics['total_nll']:.6f}")
    print(f"Mean NLL      : {metrics['mean_nll']:.6f}")
    print(f"Perplexity    : {metrics['ppl']:.6f}")
    print("===================================")


if __name__ == "__main__":
    main()
