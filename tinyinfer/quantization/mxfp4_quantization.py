#!/usr/bin/env python3
"""
Offline MXFP4 weight-only quantization for dense Qwen3 checkpoints.

Quantizes:
  - model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
  - lm_head.weight

Preserves:
  - model.embed_tokens.weight in its original dtype (FP16/BF16/FP32)
  - all other non-quantized tensors and auxiliary tokenizer/config files

If config.json has tie_word_embeddings=true and the source checkpoint stores
only one of the tied tensors, this script reconstructs an independent
model.embed_tokens.weight and quantizes the same source tensor for lm_head.

Dependencies: torch, safetensors.
CPU-only; no transformers or GPU required.

The output is a custom tinyInfer checkpoint. Vanilla Transformers cannot load
the packed MXFP4 tensors without a custom model/loader.
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# OCP FP4 E2M1 positive representable values, indexed by 3-bit magnitude code.
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MIDPOINTS = tuple((a + b) / 2.0 for a, b in zip(FP4_VALUES, FP4_VALUES[1:]))

MLP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

EMBED_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"
LM_HEAD_PACKED_KEY = "lm_head.weight_packed"
LM_HEAD_SCALE_KEY = "lm_head.weight_scale"

COPY_SUFFIXES = {".json", ".txt", ".model", ".tiktoken", ".jinja", ".vocab", ".py"}
FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def quantize_chunk(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [rows,K] floating-point weights into FP4 [rows,K/2] and E8M0 [rows,K/32]."""
    if weight.ndim != 2 or weight.shape[1] % 32:
        raise ValueError(f"expected [rows,K] with K multiple of 32; got {tuple(weight.shape)}")

    x = weight.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(x).all()):
        raise ValueError("non-finite weights (NaN/Inf) are not supported")

    rows, k = x.shape
    groups = x.reshape(rows, k // 32, 32)
    amax = groups.abs().amax(dim=-1)

    # For all-zero groups choose scale=1, code=0; avoid log2(0).
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    exponent = torch.ceil(torch.log2(safe_amax / 6.0)).to(torch.int32)
    exponent = torch.where(amax > 0, exponent, torch.zeros_like(exponent))

    if bool(((exponent < -127) | (exponent > 127)).any()):
        raise ValueError("E8M0 exponent out of representable range [-127,127]")

    scale = torch.ldexp(torch.ones_like(amax), exponent)
    scaled = groups / scale.unsqueeze(-1)
    magnitude = scaled.abs().clamp(max=6.0)

    # E2M1 round-to-nearest-even at the seven representable-value midpoints.
    mag_code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for lower_code, midpoint in enumerate(FP4_MIDPOINTS):
        upper_code = lower_code + 1
        choose_upper = (magnitude > midpoint) | (
            (magnitude == midpoint) & (upper_code % 2 == 0)
        )
        mag_code += choose_upper.to(torch.uint8)

    code = mag_code | (torch.signbit(scaled).to(torch.uint8) << 3)
    flat = code.reshape(rows, k)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)

    # E8M0 stores the biased exponent directly; 255 is reserved for NaN.
    scale_bits = (exponent + 127).to(torch.uint8)
    return packed.contiguous(), scale_bits.contiguous()


def dequantize_chunk(packed: torch.Tensor, scale_bits: torch.Tensor) -> torch.Tensor:
    """Reference decode to float32, shape [rows,K]."""
    if packed.ndim != 2 or scale_bits.ndim != 2 or packed.shape[0] != scale_bits.shape[0]:
        raise ValueError("packed/scale shape mismatch")

    rows, half_k = packed.shape
    k = half_k * 2
    if k % 32 or scale_bits.shape[1] != k // 32:
        raise ValueError("scale shape mismatch")
    if bool((scale_bits == 255).any()):
        raise ValueError("E8M0 NaN scale detected (255)")

    codes = torch.empty((rows, k), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4

    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=packed.device)
    values = lut[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)

    exponent = scale_bits.to(torch.int32) - 127
    scales = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    return (values.reshape(rows, k // 32, 32) * scales.unsqueeze(-1)).reshape(rows, k)


def quantize_matrix(
    tensor: torch.Tensor,
    name: str,
    chunk_rows: int,
    verify: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    """Quantize a complete [out,in] matrix in row chunks to limit peak CPU memory."""
    if tensor.ndim != 2 or tensor.shape[1] % 32:
        raise ValueError(f"{name}: require 2D [out,in] and in % 32 == 0; got {tuple(tensor.shape)}")
    if tensor.dtype not in FLOAT_DTYPES:
        raise ValueError(f"{name}: unsupported dtype {tensor.dtype}")

    packed_parts = []
    scale_parts = []
    mse_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    total = 0

    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = tensor[start:start + chunk_rows]
        packed, scale = quantize_chunk(chunk)
        packed_parts.append(packed)
        scale_parts.append(scale)

        if verify:
            ref = chunk.float()
            decoded = dequantize_chunk(packed, scale)
            diff = decoded - ref
            mse_sum += float(diff.square().sum(dtype=torch.float64))
            abs_sum += float(diff.abs().sum(dtype=torch.float64))
            max_abs = max(max_abs, float(diff.abs().max()))
            total += ref.numel()

    packed_tensor = torch.cat(packed_parts, dim=0)
    scale_tensor = torch.cat(scale_parts, dim=0)

    error = None
    if verify:
        if total == 0:
            raise ValueError(f"{name}: cannot verify an empty tensor")
        error = {
            "tensor": name,
            "mae": abs_sum / total,
            "rmse": math.sqrt(mse_sum / total),
            "max_abs": max_abs,
        }

    return packed_tensor, scale_tensor, error


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Original HF Qwen3 checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="New checkpoint directory; must be empty or absent")
    parser.add_argument("--chunk-rows", type=int, default=128, help="Rows processed at once; limits peak CPU RAM")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode quantized chunks and measure quantization error; use --no-verify to disable",
    )
    return parser.parse_args()


def check_config(source: Path) -> tuple[dict, int]:
    path = source / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing config.json: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError(f"Expected model_type=qwen3, got {config.get('model_type')!r}")
    if config.get("num_experts") or config.get("num_local_experts"):
        raise ValueError("MoE model not supported by this dense-Qwen3 quantizer")

    n_layers = int(config["num_hidden_layers"])
    if "hidden_size" not in config or "vocab_size" not in config:
        raise ValueError("config.json must contain hidden_size and vocab_size")

    hidden_size = int(config["hidden_size"])
    vocab_size = int(config["vocab_size"])
    if hidden_size <= 0 or hidden_size % 32:
        raise ValueError(f"config hidden_size must be positive and divisible by 32; got {hidden_size}")
    if vocab_size <= 0:
        raise ValueError(f"config vocab_size must be positive; got {vocab_size}")

    return config, n_layers


def input_shards(source: Path) -> list[Path]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        if not names:
            raise ValueError("empty weight_map")
        shards = [source / name for name in names]
        missing = [str(path) for path in shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing source shards: {missing}")
        return shards

    standalone = source / "model.safetensors"
    if standalone.is_file():
        return [standalone]

    shards = sorted(source.glob("model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError("No HF safetensors checkpoint found")
    return shards


def scan_checkpoint_keys(shards: list[Path]) -> dict[str, Path]:
    """Read only Safetensors metadata to locate keys without loading tensor data."""
    key_to_shard = {}
    for path in shards:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in key_to_shard:
                    raise ValueError(f"duplicate tensor key across shards: {key}")
                key_to_shard[key] = path
    return key_to_shard


def resolve_embedding_and_lm_head_keys(
    config: dict,
    key_to_shard: dict[str, Path],
) -> tuple[str, str, bool]:
    """
    Return (embedding_source_key, lm_head_source_key, source_tied).

    If tied, one canonical source tensor is used for both outputs. Prefer
    model.embed_tokens.weight when present; otherwise use lm_head.weight.
    If untied, both independent source tensors are required.
    """
    source_tied = bool(config.get("tie_word_embeddings", False))
    has_embed = EMBED_KEY in key_to_shard
    has_lm_head = LM_HEAD_KEY in key_to_shard

    if source_tied:
        if has_embed:
            canonical_key = EMBED_KEY
        elif has_lm_head:
            canonical_key = LM_HEAD_KEY
        else:
            raise ValueError(
                "tie_word_embeddings=true, but neither model.embed_tokens.weight "
                "nor lm_head.weight exists in the source checkpoint"
            )
        return canonical_key, canonical_key, True

    if not has_embed:
        raise ValueError("tie_word_embeddings=false, but model.embed_tokens.weight is missing")
    if not has_lm_head:
        raise ValueError("tie_word_embeddings=false, but lm_head.weight is missing")

    return EMBED_KEY, LM_HEAD_KEY, False


def copy_auxiliary(source: Path, dest: Path) -> None:
    for item in source.iterdir():
        if not item.is_file():
            continue
        if item.name in {"model.safetensors.index.json", "quantization_config.json"}:
            continue
        if item.suffix == ".safetensors" or item.suffix not in COPY_SUFFIXES:
            continue
        shutil.copy2(item, dest / item.name)


def validate_embedding_tensor(tensor: torch.Tensor, name: str, vocab_size: int, hidden_size: int) -> None:
    if tensor.ndim != 2 or tuple(tensor.shape) != (vocab_size, hidden_size):
        raise ValueError(
            f"{name}: expected shape {(vocab_size, hidden_size)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype not in FLOAT_DTYPES:
        raise ValueError(f"{name}: unsupported dtype {tensor.dtype}")


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    dest = args.output.expanduser().resolve()

    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError("Input/output must be separate, non-nested directories")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {dest}")

    config, n_layers = check_config(source)
    hidden_size = int(config["hidden_size"])
    vocab_size = int(config["vocab_size"])
    source_index_exists = (source / "model.safetensors.index.json").exists()
    shards = input_shards(source)
    key_to_shard = scan_checkpoint_keys(shards)

    embed_source_key, lm_head_source_key, source_tied = resolve_embedding_and_lm_head_keys(
        config, key_to_shard
    )

    print(
        f"Source: {source}\nDestination: {dest}\n"
        f"Layers: {n_layers}, source shards: {len(shards)}\n"
        f"Source tie_word_embeddings: {source_tied}\n"
        f"Embedding source key: {embed_source_key}\n"
        f"LM Head quantization source key: {lm_head_source_key}",
        flush=True,
    )

    if source_tied and EMBED_KEY in key_to_shard and LM_HEAD_KEY in key_to_shard:
        print(
            "NOTE: both tied keys exist; model.embed_tokens.weight is treated as "
            "the canonical shared source. The duplicate lm_head.weight is omitted.",
            flush=True,
        )

    dest.mkdir(parents=True, exist_ok=True)
    copy_auxiliary(source, dest)

    # Decouple the model-level parameter relationship in the output checkpoint.
    config["tie_word_embeddings"] = False
    (dest / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    out_map: dict[str, str] = {}
    counts: dict[int, set[str]] = {}
    errors = []

    num_mlp_quantized = 0
    mlp_original_bytes = 0
    mlp_compressed_bytes = 0

    lm_head_quantized = False
    lm_head_original_bytes = 0
    lm_head_compressed_bytes = 0
    embedding_original_bytes = 0
    output_tensor_bytes = 0

    def quantize_lm_head_tensor(tensor: torch.Tensor, source_key: str, output_tensors: dict) -> None:
        nonlocal lm_head_quantized, lm_head_original_bytes, lm_head_compressed_bytes

        validate_embedding_tensor(tensor, source_key, vocab_size, hidden_size)
        packed, scale, error = quantize_matrix(tensor, LM_HEAD_KEY, args.chunk_rows, args.verify)

        output_tensors[LM_HEAD_PACKED_KEY] = packed
        output_tensors[LM_HEAD_SCALE_KEY] = scale

        lm_head_original_bytes += tensor.numel() * tensor.element_size()
        lm_head_compressed_bytes += packed.numel() + scale.numel()
        lm_head_quantized = True

        if error is not None:
            errors.append(error)
            print(
                f"  {LM_HEAD_KEY}: {tuple(tensor.shape)} | "
                f"MAE={error['mae']:.6g} RMSE={error['rmse']:.6g} "
                f"MAX={error['max_abs']:.6g}",
                flush=True,
            )
        else:
            print(f"  {LM_HEAD_KEY}: {tuple(tensor.shape)} | MXFP4", flush=True)

    for shard_index, path in enumerate(shards, start=1):
        output_tensors: dict[str, torch.Tensor] = {}
        print(f"[{shard_index}/{len(shards)}] {path.name}", flush=True)

        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                tensor = reader.get_tensor(key)

                # The canonical source tensor may represent both tied parameters.
                if key == embed_source_key:
                    validate_embedding_tensor(tensor, key, vocab_size, hidden_size)
                    output_tensors[EMBED_KEY] = tensor.contiguous()
                    embedding_original_bytes += tensor.numel() * tensor.element_size()
                    print(
                        f"  {EMBED_KEY}: {tuple(tensor.shape)} | preserved {tensor.dtype}",
                        flush=True,
                    )

                    if lm_head_source_key == key:
                        quantize_lm_head_tensor(tensor, key, output_tensors)
                    continue

                # For untied models, LM Head has its own source tensor.
                if key == lm_head_source_key:
                    quantize_lm_head_tensor(tensor, key, output_tensors)
                    continue

                # If tied and both keys were serialized, omit the non-canonical duplicate.
                if source_tied and key == LM_HEAD_KEY and lm_head_source_key != LM_HEAD_KEY:
                    continue

                match = MLP_PATTERN.fullmatch(key)
                if match is None:
                    output_tensors[key] = tensor.contiguous()
                    continue

                layer, projection = int(match.group(1)), match.group(2)
                if layer < 0 or layer >= n_layers:
                    raise ValueError(f"invalid MLP layer index: {key}")
                if projection in counts.setdefault(layer, set()):
                    raise ValueError(f"duplicate MLP weight: {key}")

                packed_tensor, scale_tensor, error = quantize_matrix(
                    tensor, key, args.chunk_rows, args.verify
                )

                prefix = key[:-len(".weight")]
                packed_key = prefix + ".weight_packed"
                scale_key = prefix + ".weight_scale"
                output_tensors[packed_key] = packed_tensor
                output_tensors[scale_key] = scale_tensor

                counts[layer].add(projection)
                num_mlp_quantized += 1
                mlp_original_bytes += tensor.numel() * tensor.element_size()
                mlp_compressed_bytes += packed_tensor.numel() + scale_tensor.numel()

                if error is not None:
                    errors.append(error)
                    print(
                        f"  {key}: {tuple(tensor.shape)} | "
                        f"MAE={error['mae']:.6g} RMSE={error['rmse']:.6g} "
                        f"MAX={error['max_abs']:.6g}",
                        flush=True,
                    )
                else:
                    print(f"  {key}: {tuple(tensor.shape)} | MXFP4", flush=True)

        # A shard can become empty when it contained only a duplicate tied lm_head key.
        # Do not emit empty shard files; the output index below maps only real tensors.
        if not output_tensors:
            print(f"  No output tensors in {path.name}; skipping empty shard.", flush=True)
            continue

        out_path = dest / path.name
        save_file(output_tensors, str(out_path), metadata={"format": "pt"})

        for new_key, output_tensor in output_tensors.items():
            if new_key in out_map:
                raise ValueError(f"duplicate output key: {new_key}")
            out_map[new_key] = path.name
            output_tensor_bytes += output_tensor.numel() * output_tensor.element_size()

        del output_tensors

    incomplete = {
        layer: sorted(set(MLP_PROJECTIONS) - counts.get(layer, set()))
        for layer in range(n_layers)
        if counts.get(layer, set()) != set(MLP_PROJECTIONS)
    }
    if incomplete:
        raise ValueError(f"Incomplete MLP projections; refusing to finalize checkpoint: {incomplete}")

    if not embedding_original_bytes:
        raise ValueError("Embedding tensor was not preserved")
    if not lm_head_quantized:
        raise ValueError("LM Head weight was not quantized")
    if LM_HEAD_KEY in out_map:
        raise ValueError("Unexpected floating-point lm_head.weight remains in output checkpoint")
    if EMBED_KEY not in out_map:
        raise ValueError("model.embed_tokens.weight is missing from output checkpoint")

    # Rebuild the Safetensors index when the source was indexed or output is sharded.
    if source_index_exists or len(set(out_map.values())) > 1:
        index = {
            "metadata": {"total_size": output_tensor_bytes},
            "weight_map": out_map,
        }
        (dest / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    quant_config = {
        "format": "tinyinfer_mxfp4_weight_only_v1",
        "quantization": "mxfp4",
        "target_modules": [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "lm_head",
        ],
        "block_size": 32,
        "axis": "input_features_last_dim",
        "value_format": "fp4_e2m1",
        "scale_format": "e8m0",
        "scale_selection": "ceil_log2_amax_over_6",
        "rounding": "nearest_ties_to_even",
        "packed_dtype": "uint8",
        "packed_nibble_order": "first_element_low_nibble",
        "scale_dtype": "uint8_raw_e8m0_exponent",
        "scale_encoding": "stored_byte = unbiased_exponent + 127",
        "weight_key_suffixes": ["weight_packed", "weight_scale"],
        "source_model_type": config["model_type"],
        "source_num_hidden_layers": n_layers,
        "source_tie_word_embeddings": source_tied,
        "tie_word_embeddings": False,
        "embedding_weight_key": EMBED_KEY,
        "lm_head_source_key": lm_head_source_key,
        "lm_head_packed_key": LM_HEAD_PACKED_KEY,
        "lm_head_scale_key": LM_HEAD_SCALE_KEY,
        "original_mlp_weight_bytes": mlp_original_bytes,
        "quantized_mlp_weight_bytes": mlp_compressed_bytes,
        "original_lm_head_weight_bytes": lm_head_original_bytes,
        "quantized_lm_head_weight_bytes": lm_head_compressed_bytes,
        "embedding_weight_bytes_preserved": embedding_original_bytes,
        "note": (
            "Custom tinyInfer checkpoint. Embedding remains in source floating-point dtype; "
            "LM Head is independently stored as MXFP4. Standard Transformers loading "
            "requires a custom model/loader."
        ),
    }
    (dest / "quantization_config.json").write_text(
        json.dumps(quant_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "num_quantized_mlp_matrices": num_mlp_quantized,
        "num_quantized_lm_head_matrices": 1,
        "num_layers": n_layers,
        "source_tie_word_embeddings": source_tied,
        "output_tie_word_embeddings": False,
        "mlp_original_bytes": mlp_original_bytes,
        "mlp_quantized_bytes": mlp_compressed_bytes,
        "mlp_compression_ratio": (
            mlp_original_bytes / mlp_compressed_bytes if mlp_compressed_bytes else None
        ),
        "lm_head_original_bytes": lm_head_original_bytes,
        "lm_head_quantized_bytes": lm_head_compressed_bytes,
        "lm_head_compression_ratio": (
            lm_head_original_bytes / lm_head_compressed_bytes if lm_head_compressed_bytes else None
        ),
        "embedding_weight_bytes_preserved": embedding_original_bytes,
        "tensor_errors": errors,
    }
    (dest / "mxfp4_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    total_original = mlp_original_bytes + lm_head_original_bytes
    total_compressed = mlp_compressed_bytes + lm_head_compressed_bytes
    total_ratio = total_original / total_compressed if total_compressed else float("inf")

    print(
        f"\nDONE: {num_mlp_quantized} MLP matrices + 1 LM Head matrix\n"
        f"MLP: {mlp_original_bytes / 2**20:.2f} MiB -> {mlp_compressed_bytes / 2**20:.2f} MiB\n"
        f"LM Head: {lm_head_original_bytes / 2**20:.2f} MiB -> {lm_head_compressed_bytes / 2**20:.2f} MiB\n"
        f"Combined quantized weights: {total_original / 2**20:.2f} MiB -> "
        f"{total_compressed / 2**20:.2f} MiB ({total_ratio:.3f}x)\n"
        f"Embedding preserved: {embedding_original_bytes / 2**20:.2f} MiB\n"
        f"Output: {dest}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
