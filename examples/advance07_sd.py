import argparse
from pathlib import Path
import threading

import torch
import uvicorn

from tinyinfer import LLM
from tinyinfer.config import Config
from tinyinfer.runtime import DynamicEngineService
from tinyinfer.ui.app import create_app, run_profiler_gui
from tinyinfer.utils.profiler import TimeProfiler


DEFAULT_MODEL_DIR = Path.home() / "huggingface"
DEFAULT_TARGET_MODEL = DEFAULT_MODEL_DIR / "Qwen3-4B"
DEFAULT_DRAFT_MODEL = DEFAULT_MODEL_DIR / "Qwen3-0.6B"


def parse_args():
    parser = argparse.ArgumentParser(description="Run tinyInfer SD UI.")
    parser.add_argument("--speculative", action="store_true", help="Enable speculative decoding.")
    parser.add_argument("--model", default=str(DEFAULT_TARGET_MODEL), help="Target model path.")
    parser.add_argument("--attention-backend", type=str, choices=["torch", "flash"], default="torch", help="Attention backend: torch or flash")
    parser.add_argument("--draft-model", default=str(DEFAULT_DRAFT_MODEL), help="Draft model path.")
    parser.add_argument("--max-draft-tokens", type=int, default=5, help="Global K, valid range: 1..32.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def _serve_with_optional_profiler_gui(app, host: str, port: int) -> None:
    # Uvicorn owns a background thread. Its asyncio loop never calls Matplotlib.
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port)
    )
    server_thread = threading.Thread(
        target=server.run,
        name="tinyinfer-uvicorn",
        daemon=False,
    )
    server_thread.start()

    try:
        with TimeProfiler._registry_lock:
            has_profilers = bool(TimeProfiler.profilers)

        # GTK/Matplotlib GUI must run on the process main thread.
        if TimeProfiler.PROFILER and has_profilers:
            run_profiler_gui(
                refresh_interval=0.2,
                should_stop=server_thread.is_alive,
            )

        # Keep the process alive after profiler windows are closed.
        # Ctrl+C is handled here on the main thread.
        while server_thread.is_alive():
            server_thread.join(timeout=0.2)

    except KeyboardInterrupt:
        print("\n[tinyInfer] Shutdown requested", flush=True)

    finally:
        # Uvicorn's signal handlers are not installed from its background thread.
        # Request graceful shutdown explicitly, then wait for lifespan cleanup.
        server.should_exit = True
        server_thread.join()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run tinyInfer.")

    config = Config(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.block_size,
        attention_backend=args.attention_backend,
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

    # Model construction and its existing memory profiling happen before
    # the Uvicorn and GUI threads/event loops are started.
    engine = LLM(config, device="cuda", dtype=torch.bfloat16)
    service = DynamicEngineService(engine, snapshot_interval_s=0.25)
    app = create_app(service)

    ui_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"[tinyInfer] UI    : http://{ui_host}:{args.port}")

    _serve_with_optional_profiler_gui(app, args.host, args.port)


if __name__ == "__main__":
    main()
