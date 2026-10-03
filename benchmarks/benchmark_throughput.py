import argparse
import time

from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config

from benchmarks.workloads import build_workloads


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="/home/jason/huggingface/Qwen3-0.6B") # 模型路径
    p.add_argument("--max-num-batched-tokens", type=int, default=256) # 一批次中最多可以同时前向传播的token数目(可以来自不同seq)
    p.add_argument("--max-num-seqs", type=int, default=16) # 一批次中最多可以同时前向传播的seq请求数目
    p.add_argument("--max-model-len", type=int, default=512) # 每一个seq请求最多包含多少token(prompt + 生成的token)
    p.add_argument("--kvcache-block-size", type=int, default=16) # paged KV cache block大小(单位:tokens)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8) # gpu利用率上限

    p.add_argument("--workload", choices=["short", "long_prefill", "shared_prefix"], default="short") # 可以选择workload = short/long_prefill/shared_prefix
    p.add_argument("--max-tokens", type=int, default=128) # 每个seq请求最多可以decode生成多少token

    return p.parse_args()


def main():
    args = parse_args()

    llm = LLM(
        Config(
            model=args.model,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            kvcache_block_size=args.kvcache_block_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    )

    WORKLOADS = build_workloads(llm.tokenizer)
    specs = WORKLOADS[args.workload]

    prompts = [x.prompt for x in specs]
    params = [SamplingParams(greedy=True, max_tokens=args.max_tokens) for x in specs]

    prompt_tokens_total = sum(
        len(llm.tokenizer.encode(p, add_special_tokens=False))
        for p in prompts
    )

    begin = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall_time = time.perf_counter() - begin

    output_tokens_total = sum(len(x["token_ids"]) for x in outputs)
    cached_prompt_tokens = sum(x.get("num_cached_tokens", 0) for x in outputs)
    actual_computed_prompt_tokens = prompt_tokens_total - cached_prompt_tokens


    print("Overall performance statistics: ")
    print("--------------------------------------------------------------------------")
    print(f"wall_time_s={wall_time:.6f}")
    print(f"requests_total={len(outputs)}")
    print(f"prompt_tokens_total={prompt_tokens_total}")
    print(f"cached_prompt_tokens={cached_prompt_tokens}")
    print(f"actual_computed_prompt_tokens={actual_computed_prompt_tokens}")
    print(f"output_tokens_total={output_tokens_total}")
    print(f"output_tokens_per_s={output_tokens_total / wall_time:.3f}")
    print(
        "computed_plus_output_tokens_per_s="
        f"{(actual_computed_prompt_tokens + output_tokens_total) / wall_time:.3f}"
    )
    print(f"requests_per_s={len(outputs) / wall_time:.3f}")
    print("--------------------------------------------------------------------------\n")


    print("Request-wise performance statistics: ")
    for i, output in enumerate(outputs):
        print("--------------------------------------------------------------------------")
        print(f"seq[{i}]")
        stats = output["statistics"]
        # token statistics
        print(f"prompt_tokens={stats['prompt_tokens']}")
        print(f"output_tokens={stats['output_tokens']}")
        print(f"decode_tokens={stats['decode_tokens']}")
        print(f"total_tokens={stats['total_tokens']}")
        print(f"cached_tokens={stats['cached_tokens']}")

         # latency
        print(f"queue_time_s={stats['queue_time']:.6f}")
        print(f"prefill_time_s={stats['prefill_time']:.6f}")
        print(f"decode_time_s={stats['decode_time']:.6f}")
        print(f"ttft_s={stats['ttft']:.6f}")
        print(f"service_time_s={stats['service_time']:.6f}")
        print(f"e2e_latency_s={stats['e2e_latency']:.6f}")
        print(f"tpot_s={stats['tpot']:.6f}" if stats["tpot"] is not None else "tpot_s=None")
        print(f"mean_itl_s={stats['mean_itl']:.6f}" if stats["mean_itl"] is not None else "mean_itl_s=None")
        print(f"min_itl_s={stats['min_itl']:.6f}" if stats["min_itl"] is not None else "min_itl_s=None")
        print(f"max_itl_s={stats['max_itl']:.6f}" if stats["max_itl"] is not None else "max_itl_s=None")

        # throughput
        print(
            f"decode_throughput_tokens_per_s={stats['decode_throughput']:.3f}"
            if stats["decode_throughput"] is not None
            else "decode_throughput_tokens_per_s=None"
        )
        print(
            f"request_throughput_tokens_per_s={stats['request_throughput']:.3f}"
            if stats["request_throughput"] is not None
            else "request_throughput_tokens_per_s=None"
        )
        print("--------------------------------------------------------------------------")
    



if __name__ == "__main__":
    main()
