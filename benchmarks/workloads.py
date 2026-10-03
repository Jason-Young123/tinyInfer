from dataclasses import dataclass

from tinyinfer.utils.prompt import build_prompt


@dataclass(slots=True)
class RequestSpec:
    prompt: str
    max_tokens: int


# ============================================================
# Workload source text
# ============================================================

# short:
#   prompt ≈ 32~64 tokens
#   output = 128
#
# system + user + chat-template special tokens 合起来控制在这个量级。
_SHORT_QUESTIONS = [
    (
        "Explain KV cache in one concise paragraph, including what is "
        "stored and why it reduces autoregressive decoding computation."
    ),
    (
        "Explain grouped-query attention in one concise paragraph and "
        "state why it reduces KV-cache memory compared with multi-head attention."
    ),
    (
        "Explain rotary positional embedding in one concise paragraph and "
        "state how it injects token position information into attention."
    ),
    (
        "Explain paged KV cache in one concise paragraph and describe why "
        "fixed-size blocks help manage GPU memory efficiently."
    ),
    (
        "Explain chunked prefill in one concise paragraph and state why it "
        "can improve scheduling fairness for long prompts."
    ),
    (
        "Explain prefix caching in one concise paragraph and describe when "
        "previously computed KV-cache blocks can be reused."
    ),
    (
        "Explain prefill and decode in one concise paragraph and summarize "
        "the main computational difference between the two stages."
    ),
    (
        "Explain causal self-attention in one concise paragraph and state "
        "why each token may only attend to itself and earlier tokens."
    ),
    (
        "Explain continuous batching in one concise paragraph and describe "
        "how it differs from processing a fixed static batch."
    ),
    (
        "Explain weight tying in one concise paragraph and state why an "
        "embedding matrix can also be reused as the language-model output head."
    ),
]

# long_prefill:
#   prompt ≈ 2K~4K tokens
#   output = 32
#
# 使用固定技术文本重复构造 deterministic 长上下文。
# 不使用随机字符串，保证不同 benchmark run 的输入完全一致。
_LONG_CONTEXT_UNIT = (
    "Transformer inference processes token representations through repeated "
    "attention and feed-forward layers. During autoregressive decoding, "
    "previously computed key and value tensors can be retained in a KV cache "
    "so that earlier tokens do not need to recompute their attention states. "
    "Paged KV cache organizes these tensors into fixed-size physical blocks, "
    "allowing logical sequences to use non-contiguous memory while reducing "
    "external fragmentation and improving memory management efficiency. "
)

# 实际 token 数与 tokenizer 有关。
# 对 Qwen3 tokenizer，这个重复次数应落在约 2K~4K token 区间；
# benchmark 中最好额外做 token-length assertion。
_LONG_CONTEXT = _LONG_CONTEXT_UNIT * 35

_LONG_QUESTION = (
    _LONG_CONTEXT
    + "\n\nSummarize the technical discussion above in one concise paragraph."
)


# shared_prefix:
#   大 system prompt
#   +
#   不同 user question
#   output = 32
#
# 三个 request 应共享大量完整 KV-cache blocks。
_SHARED_SYSTEM_UNIT = (
    "You are a concise technical assistant specializing in transformer "
    "inference systems. Use precise terminology and factual statements. "
    "Focus on model execution, attention, KV-cache organization, scheduling, "
    "memory management, and inference efficiency. Do not use bullet points. "
)

# 足够长，使 shared prefix 跨越多个 block。
_SHARED_SYSTEM = _SHARED_SYSTEM_UNIT * 40

_SHARED_QUESTIONS = [
    "What is rotary positional embedding and how is it applied to queries and keys?",
    "What is grouped-query attention and why can it reduce KV-cache memory usage?",
    "What is chunked prefill and why is it useful for inference scheduling?",
]


# ============================================================
# Build workloads after tokenizer is available
# ============================================================

def build_workloads(tokenizer) -> dict[str, list[RequestSpec]]:
    short = [
        RequestSpec(
            prompt=build_prompt(
                question=question,
                tokenizer=tokenizer,
            ),
            max_tokens=32,
        )
        for question in _SHORT_QUESTIONS
    ]

    long_prefill = [
        RequestSpec(
            prompt=build_prompt(
                question=_LONG_QUESTION,
                tokenizer=tokenizer,
            ),
            max_tokens=32,
        )
    ]

    shared_prefix = [
        RequestSpec(
            prompt=build_prompt(
                question=question,
                tokenizer=tokenizer,
                system_prompt=_SHARED_SYSTEM,
            ),
            max_tokens=32,
        )
        for question in _SHARED_QUESTIONS
    ]

    return {
        "short": short,
        "long_prefill": long_prefill,
        "shared_prefix": shared_prefix,
    }




def print_workload_lengths(workloads, tokenizer):
    for name, requests in workloads.items():
        print(f"\n{name}:")

        for i, request in enumerate(requests):
            num_tokens = len(
                tokenizer.encode(
                    request.prompt,
                    add_special_tokens=False,
                )
            )

            print(
                f"  request {i}: "
                f"prompt_tokens={num_tokens}, "
                f"max_tokens={request.max_tokens}"
            )