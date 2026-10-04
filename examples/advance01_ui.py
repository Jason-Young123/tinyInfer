import argparse
import uvicorn

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
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        model=args.model,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        max_model_len=2048,
        kvcache_block_size=16,
        gpu_memory_utilization=0.80,
    )
    engine = LLM(config)
    service = DynamicEngineService(engine)
    app = create_app(
        service,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        greedy=args.greedy,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


