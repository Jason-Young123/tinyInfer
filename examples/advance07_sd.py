import argparse
from pathlib import Path

import torch
import uvicorn

from tinyinfer import LLM
from tinyinfer.config import Config
from tinyinfer.runtime import DynamicEngineService
from tinyinfer.ui import create_app


DEFAULT_MODEL_DIR = Path.home() / "huggingface"
DEFAULT_TARGET_MODEL = DEFAULT_MODEL_DIR / "Qwen3-4B"
DEFAULT_DRAFT_MODEL = DEFAULT_MODEL_DIR / "Qwen3-0.6B"


def parse_args():
    parser = argparse.ArgumentParser(description="Run tinyInfer Speculative Decoding UI.")
    parser.add_argument("--speculative", action="store_true", help="Enable speculative decoding.")
    parser.add_argument("--model", default=str(DEFAULT_TARGET_MODEL), help="Target model path.")
    parser.add_argument("--draft-model", default=str(DEFAULT_DRAFT_MODEL), help="Draft model path.")
    parser.add_argument("--max-draft-tokens", type=int, default=5, help="Global K, valid range: 1..32.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run tinyInfer SD.")

    config = Config(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.block_size,
        speculative=args.speculative,
        draft_model=args.draft_model if args.speculative else None,
        max_draft_tokens=args.max_draft_tokens,
    )

    print("[tinyInfer] Starting tinyInfer UI")
    print(f"[tinyInfer] Target: {config.model}")
    print(f"[tinyInfer] Mode  : {'Speculative Decoding' if args.speculative else 'Target-only'}")
    if args.speculative:
        print(f"[tinyInfer] Draft : {config.draft_model}")
        print(f"[tinyInfer] K     : {config.max_draft_tokens}")

    engine = LLM(config, device="cuda", dtype=torch.bfloat16)
    service = DynamicEngineService(engine, snapshot_interval_s=0.25)
    app = create_app(service)

    ui_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"[tinyInfer] UI    : http://{ui_host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
