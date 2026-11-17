from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tinyinfer.config import Config
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.context import reset_context, set_context
from tinyinfer.utils.loader import load_weights
from tinyinfer.utils.profiler import TimeProfiler


DEFAULT_MODEL = Path.home() / "huggingface" / "Qwen3-4B"
DEFAULT_QUANTIZED_MODEL = Path.home() / "huggingface" / "Qwen3-4B-MXFP4"


@torch.inference_mode()
def evaluate_top1(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    block_tables: torch.Tensor,
    max_length: int,
    stride: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the next-token argmax at every position, stored on CPU."""
    total_tokens = input_ids.numel()
    if total_tokens < 2:
        raise ValueError("The corpus must tokenize to at least 2 tokens.")
    if max_length < 2 or not 1 <= stride < max_length or chunk_size <= 0:
        raise ValueError("Require max_length >= 2, 1 <= stride < max_length, chunk_size > 0.")

    predictions = []
    target_start = 1

    while target_start < total_tokens:
        target_end = min(target_start + stride, total_tokens)
        window_start = max(0, target_end - max_length)
        window_input = input_ids[window_start : target_end - 1]
        score_input_start = target_start - 1 - window_start

        for chunk_start in range(0, window_input.numel(), chunk_size):
            chunk_end = min(chunk_start + chunk_size, window_input.numel())
            positions = torch.arange(chunk_start, chunk_end, device=device, dtype=torch.long)

            try:
                set_context(
                    context_lens=[chunk_end],
                    cu_seqlens_q=[0, chunk_end - chunk_start],
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

            score_start = max(chunk_start, score_input_start)
            if score_start < chunk_end:
                logits = model.compute_logits(hidden[score_start - chunk_start :])
                predictions.append(logits.argmax(dim=-1).to(device="cpu", dtype=torch.int32))
                del logits
            del hidden

        target_start = target_end

    torch.cuda.synchronize(device)
    predictions = torch.cat(predictions)
    if predictions.numel() != total_tokens - 1:
        raise RuntimeError(f"Prediction count mismatch: expected {total_tokens - 1}, got {predictions.numel()}")
    return predictions


@torch.inference_mode()
def run_model(
    model_path: Path,
    quantization: str | None,
    hf_config,
    input_ids: torch.Tensor,
    args,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
) -> torch.Tensor:
    """Load one model, evaluate the corpus, and release its GPU memory."""
    model = kv_cache = block_tables = None
    try:
        model = Qwen3ForCausalLM(
            hf_config, kv_cache=None, backend=args.attention_backend, quantization=quantization
        ).to(device=device, dtype=dtype)
        report = load_weights(model, model_path, strict=True)
        model.eval()
        print(f"[TOP1] loaded {model_path.name}: {len(report.loaded)} parameters")

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

        return evaluate_top1(
            model, input_ids, block_tables, max_length,
            args.stride, args.chunk_size, device,
        )
    finally:
        reset_context()
        del model, kv_cache, block_tables
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Measure BF16 vs MXFP4 Top-1 Agreement with tinyInfer.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="BF16 reference model.")
    parser.add_argument("--quantized-model", type=Path, default=DEFAULT_QUANTIZED_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=None, help="Defaults to --model.")
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--quantization", choices=("none", "mxfp4"), default="mxfp4")
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
    quantized_model_path = args.quantized_model.expanduser().resolve()
    tokenizer_path = (args.tokenizer or args.model).expanduser().resolve()
    text_path = args.text_file.expanduser().resolve()

    for path in (model_path, quantized_model_path, tokenizer_path, text_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not all(path.is_dir() for path in (model_path, quantized_model_path, tokenizer_path)) or not text_path.is_file():
        raise ValueError("--model/--quantized-model/--tokenizer must be directories and --text-file must be a file.")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This tinyInfer Top-1 Agreement path requires a CUDA device.")
    if args.block_size <= 0 or args.max_length < 2 or args.chunk_size <= 0:
        raise ValueError("block_size/chunk_size must be positive and max_length must be >= 2.")

    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16.")

    config = Config(
        model=str(model_path),
        max_model_len=args.max_length,
        kvcache_block_size=args.block_size,
        attention_backend=args.attention_backend,
    )
    config.load_hf_config()
    hf_config = config.hf_config
    hf_config.kvcache_block_size = args.block_size
    max_length = min(args.max_length, int(hf_config.max_position_embeddings))
    if not 1 <= args.stride < max_length:
        raise ValueError(f"stride must be in [1, {max_length - 1}].")

    TimeProfiler.PROFILER = False
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    text = text_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty corpus: {text_path}")
    input_ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if input_ids.numel() < 2:
        raise ValueError("The tokenized corpus must contain at least 2 tokens.")

    print(f"[TOP1] corpus={text_path} | tokens={input_ids.numel()}")
    print(f"[TOP1] max_length={max_length} | stride={args.stride} | chunk_size={args.chunk_size}")
    print(f"[TOP1] backend={args.attention_backend} | dtype={args.dtype} | quantization={args.quantization}")

    pred_bf16 = run_model(model_path, None, hf_config, input_ids, args, device, dtype, max_length)
    pred_mxfp4 = run_model(
        quantized_model_path, args.quantization, hf_config, input_ids, args, device, dtype, max_length
    )
    if pred_bf16.shape != pred_mxfp4.shape:
        raise RuntimeError(f"Prediction shape mismatch: BF16={pred_bf16.shape}, MXFP4={pred_mxfp4.shape}")

    agree = int((pred_bf16 == pred_mxfp4).sum().item())
    total = pred_bf16.numel()
    print("\n========== tinyInfer Top-1 Agreement ==========")
    print(f"Corpus          : {text_path.name}")
    print(f"Compared tokens : {total}")
    print(f"Top-1 Agreement : {agree / total * 100:.4f}%")
    print("================================================")


if __name__ == "__main__":
    main()
