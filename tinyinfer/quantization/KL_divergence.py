
from __future__ import annotations

import argparse
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
DEFAULT_QUANTIZED_MODEL = Path.home() / "huggingface" / "Qwen3-4B-MXFP4"


def forward_chunk(model, input_chunk, positions, chunk_end, block_tables):
    q_len = input_chunk.numel()

    try:
        set_context(
            context_lens=[chunk_end],
            cu_seqlens_q=[0, q_len],
            slot_mapping=positions,
            block_tables=block_tables,
        )
        hidden, _ = model(
            input_chunk,
            positions,
            output_hidden_states=False,
        )
    finally:
        reset_context()

    return hidden


@torch.inference_mode()
def evaluate_kl(
    teacher: Qwen3ForCausalLM,
    student: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    teacher_blocks: torch.Tensor,
    student_blocks: torch.Tensor,
    max_length: int,
    stride: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, float | int]:

    total_tokens = input_ids.numel()
    if total_tokens < 2:
        raise ValueError("The corpus must tokenize to at least 2 tokens.")
    if max_length < 2 or not 1 <= stride <= max_length - 1 or chunk_size <= 0:
        raise ValueError("Require max_length >= 2, 1 <= stride < max_length, chunk_size > 0.")

    total_kl = torch.zeros((), dtype=torch.float64, device=device)
    compared_tokens = 0
    target_start = 1

    while target_start < total_tokens:
        target_end = min(target_start + stride, total_tokens)
        window_start = max(0, target_end - max_length)

        window_input = input_ids[window_start : target_end - 1]
        score_input_start = target_start - 1 - window_start

        for chunk_start in range(0, window_input.numel(), chunk_size):
            chunk_end = min(chunk_start + chunk_size, window_input.numel())
            q_len = chunk_end - chunk_start

            positions = torch.arange(
                chunk_start, chunk_end,
                device=device,
                dtype=torch.long,
            )
            input_chunk = window_input[chunk_start:chunk_end]

            # Both models receive identical tokens and positions.
            teacher_hidden = forward_chunk(
                teacher, input_chunk, positions, chunk_end, teacher_blocks
            )
            student_hidden = forward_chunk(
                student, input_chunk, positions, chunk_end, student_blocks
            )

            # Only score positions not covered by previous windows.
            score_start = max(chunk_start, score_input_start)

            if score_start < chunk_end:
                offset = score_start - chunk_start

                teacher_logits = teacher.compute_logits(
                    teacher_hidden[offset:]
                ).float()

                student_logits = student.compute_logits(
                    student_hidden[offset:]
                ).float()

                # BF16 is the reference distribution P.
                log_p = F.log_softmax(teacher_logits, dim=-1)

                # MXFP4 is the approximated distribution Q.
                log_q = F.log_softmax(student_logits, dim=-1)

                # KL(P || Q), one value per prediction position.
                kl = F.kl_div(
                    log_q,
                    log_p,
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)

                total_kl += kl.double().sum()
                compared_tokens += kl.numel()

            del teacher_hidden, student_hidden

        target_start = target_end

    torch.cuda.synchronize(device)

    mean_kl = (total_kl / compared_tokens).item()

    return {
        "corpus_tokens": int(total_tokens),
        "compared_tokens": int(compared_tokens),
        "total_kl": total_kl.item(),
        "mean_kl": mean_kl,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure BF16 vs MXFP4 KL Divergence with tinyInfer."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="BF16 reference model.")
    parser.add_argument("--quantized-model", type=Path, default=DEFAULT_QUANTIZED_MODEL,
                        help="MXFP4 model.")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Defaults to --model; both models use the same tokenizer.")
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--attention-backend",
                        choices=("torch", "flash", "flashinfer"), default="flash")
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
    quantized_path = args.quantized_model.expanduser().resolve()
    tokenizer_path = (args.tokenizer or args.model).expanduser().resolve()
    text_path = args.text_file.expanduser().resolve()

    for path in (model_path, quantized_path, tokenizer_path, text_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if not all(path.is_dir() for path in (model_path, quantized_path, tokenizer_path)):
        raise ValueError("--model, --quantized-model and --tokenizer must be directories.")
    if not text_path.is_file():
        raise ValueError("--text-file must be a file.")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This tinyInfer KL path requires a CUDA device.")
    if args.block_size <= 0 or args.max_length < 2:
        raise ValueError("block_size must be positive and max_length must be >= 2.")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16.")

    # Load shared model configuration.
    config = Config(
        model=str(model_path),
        max_model_len=args.max_length,
        kvcache_block_size=args.block_size,
        attention_backend=args.attention_backend,
        quantization=None,
    )
    config.load_hf_config()

    hf_config = config.hf_config
    hf_config.kvcache_block_size = args.block_size

    max_length = min(args.max_length, int(hf_config.max_position_embeddings))
    if not 1 <= args.stride <= max_length - 1:
        raise ValueError(f"stride must be in [1, {max_length - 1}].")

    # Runtime profiler is unnecessary for accuracy evaluation.
    TimeProfiler.PROFILER = False

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
    )

    text = text_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty corpus: {text_path}")

    input_ids = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0].to(device=device, dtype=torch.long)

    if input_ids.numel() < 2:
        raise ValueError("The tokenized corpus must contain at least 2 tokens.")

    print(f"[KL] corpus={text_path} | tokens={input_ids.numel()}")
    print(f"[KL] max_length={max_length} | stride={args.stride} | chunk_size={args.chunk_size}")
    print(f"[KL] backend={args.attention_backend} | dtype={args.dtype}")
    print(f"[KL] teacher={model_path}")
    print(f"[KL] student={quantized_path}")

    # Load BF16 teacher.
    teacher = Qwen3ForCausalLM(
        hf_config,
        kv_cache=None,
        backend=args.attention_backend,
        quantization=None,
    ).to(device=device, dtype=dtype)

    teacher_report = load_weights(teacher, model_path, strict=True)
    teacher.eval()
    print(f"[KL] loaded BF16: {len(teacher_report.loaded)} parameters")

    # Load MXFP4 student.
    student = Qwen3ForCausalLM(
        hf_config,
        kv_cache=None,
        backend=args.attention_backend,
        quantization="mxfp4",
    ).to(device=device, dtype=dtype)

    student_report = load_weights(student, quantized_path, strict=True)
    student.eval()
    print(f"[KL] loaded MXFP4: {len(student_report.loaded)} parameters")

    # Each model maintains an independent KV cache.
    head_dim = int(
        getattr(hf_config, "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads)
    )
    num_blocks = math.ceil(max_length / args.block_size)

    def create_kv_cache():
        return PagedKVCache(
            num_layers=int(hf_config.num_hidden_layers),
            num_blocks=num_blocks,
            block_size=args.block_size,
            num_kv_heads=int(hf_config.num_key_value_heads),
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    teacher.set_kv_cache(create_kv_cache())
    student.set_kv_cache(create_kv_cache())

    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)

    metrics = evaluate_kl(
        teacher,
        student,
        input_ids,
        block_tables,
        block_tables,
        max_length,
        args.stride,
        args.chunk_size,
        device,
    )

    print("\n========== tinyInfer KL Divergence ==========")
    print(f"Corpus tokens  : {metrics['corpus_tokens']}")
    print(f"Compared tokens: {metrics['compared_tokens']}")
    print(f"Total KL       : {metrics['total_kl']:.8f}")
    print(f"Mean KL        : {metrics['mean_kl']:.8f} nats/token")
    print("=============================================")


if __name__ == "__main__":
    main()
