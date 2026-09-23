from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Config:
    model: str | None = None
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 64                  # 最多同时允许多少条请求同时推理 
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    kvcache_block_size: int = 256

    hf_config: Any | None = None
    eos_token_id: int = -1
    num_kvcache_blocks: int = 0

    def validate_runtime_fields(self):
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")



