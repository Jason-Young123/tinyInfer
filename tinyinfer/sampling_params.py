from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    topk: int = 100
    max_tokens: int = 64        # 最多允许生成的token数
    ignore_eos: bool = False
    greedy: bool = False

    def __post_init__(self):
        if self.temperature <= 0.0:
            raise ValueError("Fatal: Sampling temperature not positive.")
        if self.topk < 1:
            raise ValueError("Fatal: Sampling topk less than 1.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")



