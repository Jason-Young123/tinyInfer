from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open


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
                print(name)


if __name__ == "__main__":
    iter_safetensor_weights("/home/jason/huggingface/Qwen3-0.6B")


