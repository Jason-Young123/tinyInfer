import argparse
import uvicorn
import torch

from tinyinfer import LLM
from tinyinfer.config import Config
from tinyinfer.runtime import DynamicEngineService
from tinyinfer.ui import create_app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/jason/huggingface/Qwen3-0.6B",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        model=args.model,
        max_num_batched_tokens=512,
        max_num_seqs=8,
        max_model_len=16384, # max_tokens = 8192
        kvcache_block_size=16,
        gpu_memory_utilization=0.80,
    )
    engine = LLM(config, device="cuda", dtype=torch.bfloat16) # 显式注明device和dtype
    service = DynamicEngineService(engine, snapshot_interval_s=0.25)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
