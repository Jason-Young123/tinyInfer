#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models to a local directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/jason/huggingface/",
        help="Root directory used to store downloaded models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Hugging Face model repo IDs to download.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision / branch / commit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    for repo_id in args.models:
        model_name = repo_id.rsplit("/", 1)[-1]
        local_dir = output_root / model_name

        print(f"\n[Download] {repo_id}")
        print(f"[Target]   {local_dir}")

        snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            local_dir=str(local_dir),
        )

        print(f"[Done]     {repo_id}")

    print("\nAll downloads completed.")


if __name__ == "__main__":
    main()
