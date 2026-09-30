from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(slots=True)
class LoadReport: # 加载模型权重之后的report
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    packed: list[str] = field(default_factory=list)
    skipped_tied: list[str] = field(default_factory=list)



# 按文件名顺序遍历一个本地 HF safetensors checkpoint;
# 返回一个generator, 即若干 ("weight_name", tensor值) pair
def iter_safetensor_weights(model_path: str | Path):
    model_path = Path(model_path).expanduser().resolve()
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"no .safetensors files found under {model_path}"
        )
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


# param: tinyInfer中的某个参数, 比如xxLayer.weight; tensor: 来自safetensor的具体数值; name: 该参数名称, 注意这里已经确保这个name在tinyInfer和Qwen3中相同
def _copy_checked(param: torch.nn.Parameter, tensor: torch.Tensor, name: str):
    if tuple(param.shape) != tuple(tensor.shape):
        raise RuntimeError(
            f"shape mismatch for {name}: "
            f"parameter={tuple(param.shape)} checkpoint={tuple(tensor.shape)}"
        )
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype)) # 注意, copy_仅替换tinyInfer中某个权重参数的数据, 而不是改变参数本身




# 把 HF Qwen3 checkpoint 映射到 tinyInfer packed 参数布局
def load_weights(model, model_path: str | Path, strict: bool = True) -> LoadReport:
    params = dict(model.named_parameters()) # tinyInfer的所有参数
    loaded_params: set[str] = set()
    report = LoadReport()

    # packed 参数的 shard 名 -> (tinyInfer 子模块名, shard kind)
    packed_attn_suffix = {
        "q_proj.weight": "q",
        "k_proj.weight": "k",
        "v_proj.weight": "v",
    }
    packed_mlp_suffix = {
        "gate_proj.weight": "gate",
        "up_proj.weight": "up",
    }

    for ckpt_name, tensor in iter_safetensor_weights(model_path):
        # 1. packed QKV
        matched = False
        for suffix, shard in packed_attn_suffix.items():
            if ckpt_name.endswith("self_attn." + suffix):
                prefix = ckpt_name[: -len(suffix)] # 去掉suffix
                target = prefix + "qkv_proj.weight" # suffix统一为qkv_proj.weight
                if target not in params: # tinyInfer中没有这个qualified param
                    report.unexpected.append(ckpt_name)
                    matched = True
                    break

                param = params[target]
                module_path = target.rsplit(".weight", 1)[0]
                module = model.get_submodule(module_path)
                q_size = module.q_size
                kv_size = module.kv_size

                # 按照q/k/v放入不同的切片范围
                if shard == "q":
                    begin, end = 0, q_size
                elif shard == "k":
                    begin, end = q_size, q_size + kv_size
                else:
                    begin, end = q_size + kv_size, q_size + 2 * kv_size

                view = param.data[begin:end]
                if tuple(view.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        f"packed QKV shape mismatch for {ckpt_name}: "
                        f"target slice={tuple(view.shape)} checkpoint={tuple(tensor.shape)}"
                    )
                view.copy_(tensor.to(device=param.device, dtype=param.dtype))

                # 一个 packed 参数只有 q/k/v 三片都读完才算完整；
                # 这里用 packed 记录 shard，最后单独根据 shard 计数确认。
                report.packed.append(f"{target}:{shard}")
                matched = True
                break
        if matched:
            continue

        # 2. packed Gate/Up
        for suffix, shard in packed_mlp_suffix.items():
            if ckpt_name.endswith("mlp." + suffix):
                prefix = ckpt_name[: -len(suffix)]
                target = prefix + "gate_up_proj.weight"
                if target not in params:
                    report.unexpected.append(ckpt_name)
                    matched = True
                    break

                param = params[target]
                module_path = target.rsplit(".weight", 1)[0]
                module = model.get_submodule(module_path)
                size = module.intermediate_size

                begin, end = (0, size) if shard == "gate" else (size, 2 * size)
                view = param.data[begin:end]
                if tuple(view.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        f"packed Gate/Up shape mismatch for {ckpt_name}: "
                        f"target slice={tuple(view.shape)} checkpoint={tuple(tensor.shape)}"
                    )
                view.copy_(tensor.to(device=param.device, dtype=param.dtype))
                report.packed.append(f"{target}:{shard}")
                matched = True
                break
        if matched:
            continue

        # 3. tied lm_head：checkpoint 有 lm_head，但模型侧因为 tying 没有独立参数
        if ckpt_name == "lm_head.weight" and ckpt_name not in params:
            report.skipped_tied.append(ckpt_name)
            continue

        # 4. 普通 1:1 参数, 即在Qwen和tinyInfer中param的qualified name完全相同; 包括embed_weight
        if ckpt_name in params:
            _copy_checked(params[ckpt_name], tensor, ckpt_name)
            loaded_params.add(ckpt_name)
            report.loaded.append(ckpt_name)
        else:
            report.unexpected.append(ckpt_name)

    # 5. packed 参数完整性 accounting
    packed_shards: dict[str, set[str]] = {}
    for entry in report.packed:
        target, shard = entry.rsplit(":", 1)
        packed_shards.setdefault(target, set()).add(shard)

    for target, shards in packed_shards.items():
        required = {"q", "k", "v"} if target.endswith("qkv_proj.weight") else {"gate", "up"}
        if shards == required:
            loaded_params.add(target)
            report.loaded.append(target)
        elif strict:
            raise RuntimeError(
                f"incomplete packed parameter {target}: got {sorted(shards)}, "
                f"expected {sorted(required)}"
            )

    # tied embeddings 时 named_parameters() 通常只暴露一份 Parameter；
    # 若仍有 lm_head.weight 且它与 embedding 共用 storage，则视为已加载。
    if "lm_head.weight" in params and "model.embed_tokens.weight" in loaded_params:
        if params["lm_head.weight"] is params.get("model.embed_tokens.weight"):
            loaded_params.add("lm_head.weight")
            report.skipped_tied.append("lm_head.weight")

    report.missing = sorted(set(params) - loaded_params)

    if strict:
        if report.unexpected:
            raise RuntimeError(
                "unexpected checkpoint weights: "
                f"{report.unexpected[:20]}"
            )
        if report.missing:
            raise RuntimeError(
                "unloaded model parameters: "
                f"{report.missing[:20]}"
            )

    return report








