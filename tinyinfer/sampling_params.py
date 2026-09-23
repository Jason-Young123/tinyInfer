from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64        # 最多允许生成的token数
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature <= 0.0:
            raise ValueError(
                "tinyInfer Day01 sampler only supports temperature > 0; "
                "greedy mode will be added later explicitly."
            )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")



