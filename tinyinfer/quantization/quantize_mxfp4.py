#!/usr/bin/env python3
"""Offline MXFP4 weight-only quantization of Qwen3 MLP checkpoints.

Dependencies: torch, safetensors. CPU-only; no transformers/GPU needed.
Output is a *custom* tinyInfer checkpoint, not directly loadable by vanilla HF.
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
COPY_SUFFIXES = {".json", ".txt", ".model", ".tiktoken", ".jinja", ".vocab", ".py"}


def quantize_chunk(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [rows, K] BF16/FP16/FP32 into FP4 [rows,K/2] and E8M0 [rows,K/32].

    E8M0 scale selection: ceil(log2(max(abs(x))/6)), i.e. no saturation for
    finite normal-range model weights. E2M1 rounding is nearest, ties to even.
    Stored scale is the raw E8M0 biased exponent (uint8), NOT float scale.
    """
    if weight.ndim != 2 or weight.shape[1] % 32:
        raise ValueError(f"expected [rows,K], K multiple of 32; got {tuple(weight.shape)}")
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

    # E2M1 RNE: at midpoint choose code whose least-significant bit is even.
    # Value table is monotonically increasing, so comparing 7 midpoints suffices.
    mag_code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for lower_code, midpoint in enumerate(FP4_MIDPOINTS):
        upper_code = lower_code + 1
        choose_upper = (magnitude > midpoint) | ((magnitude == midpoint) & (upper_code % 2 == 0))
        mag_code += choose_upper.to(torch.uint8)
    code = mag_code | (torch.signbit(scaled).to(torch.uint8) << 3)
    flat = code.reshape(rows, k)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)
    scale_bits = (exponent + 127).to(torch.uint8)  # 255 reserved for E8M0 NaN.
    return packed.contiguous(), scale_bits.contiguous()


def dequantize_chunk(packed: torch.Tensor, scale_bits: torch.Tensor) -> torch.Tensor:
    """Reference decode to float32, shape [rows, K]."""
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Original HF Qwen3 directory")
    parser.add_argument("--output", required=True, type=Path, help="New checkpoint directory (must be empty)")
    parser.add_argument("--chunk-rows", type=int, default=128, help="Rows processed at once (limits peak CPU RAM)")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True, help="Decode and measure quantization error")
    return parser.parse_args()


def check_config(source: Path) -> tuple[dict, int]:
    path = source / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing config.json: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError(f"Expected model_type=qwen3, got {config.get('model_type')!r}")
    if config.get("num_experts") or config.get("num_local_experts"):
        raise ValueError("MoE model not supported by this dense-Qwen3 MLP quantizer")
    n_layers = int(config["num_hidden_layers"])
    return config, n_layers


def input_shards(source: Path) -> list[Path]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        if not names:
            raise ValueError("empty weight_map")
        shards = [source / x for x in names]
        missing = [str(x) for x in shards if not x.is_file()]
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


def copy_auxiliary(source: Path, dest: Path) -> None:
    for item in source.iterdir():
        if not item.is_file() or item.name in {"model.safetensors.index.json", "quantization_config.json"}:
            continue
        if item.name.endswith(".safetensors") or item.suffix not in COPY_SUFFIXES:
            continue
        shutil.copy2(item, dest / item.name)


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
    shards = input_shards(source)
    print(f"Source: {source}\nDestination: {dest}\nLayers: {n_layers}, shards: {len(shards)}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    copy_auxiliary(source, dest)
    out_map: dict[str, str] = {}
    seen: set[str] = set()
    counts: dict[int, set[str]] = {}
    num_quantized = 0
    original_bytes = 0
    compressed_bytes = 0
    errors = []

    for shard_index, path in enumerate(shards, start=1):
        output_tensors: dict[str, torch.Tensor] = {}
        print(f"[{shard_index}/{len(shards)}] {path.name}", flush=True)
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in seen:
                    raise ValueError(f"duplicate tensor key across shards: {key}")
                seen.add(key)
                tensor = reader.get_tensor(key)
                match = MLP_PATTERN.fullmatch(key)
                if match is None:
                    output_tensors[key] = tensor.contiguous()
                    continue
                layer, projection = int(match.group(1)), match.group(2)
                if layer < 0 or layer >= n_layers or projection in counts.setdefault(layer, set()):
                    raise ValueError(f"duplicate/invalid MLP weight: {key}")
                if tensor.ndim != 2 or tensor.shape[1] % 32:
                    raise ValueError(f"{key}: require 2D [out,in] and in % 32 == 0; got {tuple(tensor.shape)}")
                if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    raise ValueError(f"{key}: unsupported dtype {tensor.dtype}")
                packed_parts, scale_parts = [], []
                mse_sum, abs_sum, max_abs, total = 0.0, 0.0, 0.0, 0
                for start in range(0, tensor.shape[0], args.chunk_rows):
                    chunk = tensor[start:start + args.chunk_rows]
                    packed, scale = quantize_chunk(chunk)
                    packed_parts.append(packed)
                    scale_parts.append(scale)
                    if args.verify:
                        ref = chunk.float()
                        decoded = dequantize_chunk(packed, scale)
                        diff = decoded - ref
                        mse_sum += float(diff.square().sum(dtype=torch.float64))
                        abs_sum += float(diff.abs().sum(dtype=torch.float64))
                        max_abs = max(max_abs, float(diff.abs().max()))
                        total += ref.numel()
                packed_tensor = torch.cat(packed_parts, dim=0)
                scale_tensor = torch.cat(scale_parts, dim=0)
                output_tensors[key[:-len(".weight")] + ".weight_packed"] = packed_tensor
                output_tensors[key[:-len(".weight")] + ".weight_scale"] = scale_tensor
                counts[layer].add(projection)
                num_quantized += 1
                original_bytes += tensor.numel() * tensor.element_size()
                compressed_bytes += packed_tensor.numel() + scale_tensor.numel()
                if args.verify:
                    error = {"tensor": key, "mae": abs_sum / total, "rmse": math.sqrt(mse_sum / total), "max_abs": max_abs}
                    errors.append(error)
                    print(f"  {key}: {tuple(tensor.shape)} | MAE={error['mae']:.6g} RMSE={error['rmse']:.6g}", flush=True)
                else:
                    print(f"  {key}: {tuple(tensor.shape)}", flush=True)
        out_path = dest / path.name
        save_file(output_tensors, str(out_path), metadata={"format": "pt"})
        for new_key in output_tensors:
            if new_key in out_map:
                raise ValueError(f"duplicate output key: {new_key}")
            out_map[new_key] = path.name
        del output_tensors

    incomplete = {layer: sorted(set(MLP_PROJECTIONS) - counts.get(layer, set())) for layer in range(n_layers)
                  if counts.get(layer, set()) != set(MLP_PROJECTIONS)}
    if incomplete:
        raise ValueError(f"Incomplete MLP projections; refusing to finalize checkpoint: {incomplete}")
    if len(shards) > 1:
        weight_bytes = sum((dest / path.name).stat().st_size for path in shards)
        (dest / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": weight_bytes}, "weight_map": out_map}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    quant_config = {
        "format": "tinyinfer_mxfp4_weight_only_v1",
        "quantization": "mxfp4",
        "target_modules": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
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
        "original_mlp_weight_bytes": original_bytes,
        "quantized_mlp_weight_bytes": compressed_bytes,
        "note": "Custom checkpoint; standard transformers.from_pretrained does not support this layout without a custom loader",
    }
    (dest / "quantization_config.json").write_text(json.dumps(quant_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "num_quantized_matrices": num_quantized,
        "num_layers": n_layers,
        "mlp_original_bytes": original_bytes,
        "mlp_quantized_bytes": compressed_bytes,
        "mlp_compression_ratio": original_bytes / compressed_bytes,
        "tensor_errors": errors,
    }
    (dest / "mxfp4_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nDONE: {num_quantized} matrices, {original_bytes / 2**20:.2f} MiB -> {compressed_bytes / 2**20:.2f} MiB"
          f" ({original_bytes / compressed_bytes:.3f}x). Output: {dest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
