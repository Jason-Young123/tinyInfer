from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Config:
    model: str | None = None
    max_num_batched_tokens: int = 4096      # 一轮调度中, 所有 Sequence 加起来, 最多允许送进模型计算多少个 token
    max_num_seqs: int = 64                  # 一轮调度中, 最多同时允许多少条请求同时推理 
    max_model_len: int = 4096               # 单条 Sequence 最多允许多长的总上下文; 实际单条sequence上下文长度上限 = min(max_model_len, prompt + sampling_params.max_tokens)
    gpu_memory_utilization: float = 0.90    # 允许 tinyInfer 最多使用 GPU 显存的大约比例(用于KV Cache blocks计算)
    tensor_parallel_size: int = 1           # 单卡默认TP=1, 后期会在单卡上模拟TP = 2/4
    enforce_eager: bool = True              # 是否强制使用普通 eager execution; 如果为False可以启用CUDA Graph
    kvcache_block_size: int = 256           # 一个 KV Cache block 容纳多少个 token 的 K/V

    hf_config: Any | None = None
    eos_token_id: int = -1
    num_kvcache_blocks: int = 0             # 运行时根据 GPU 剩余显存计算出来的KV Cache block总数量

    def validate_runtime_fields(self):
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")



