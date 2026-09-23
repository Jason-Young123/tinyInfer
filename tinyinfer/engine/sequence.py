from enum import Enum, auto
from itertools import count

from tinyinfer.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()   # 已进入引擎，等待 Scheduler 调度
    RUNNING = auto()   # 正在参与 prefill / decode
    FINISHED = auto()  # 已完成，不再参与调度


class Sequence:
    counter = count()   # 给每个 Sequence 分配全局递增 seq_id
    block_size = 256    # 一个 KV block 可容纳的 token 数

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.sampling_params = sampling_params #每个 seq 可以有不同的采样参数

        # token_ids 始终保存：prompt tokens + 已生成 tokens
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)

        # KV / 调度相关状态
        self.num_cached_tokens = 0       # 已经存在有效 KV Cache 的 token 数
        self.num_scheduled_tokens = 0    # 本轮 Scheduler 准备计算的 token 数; 和后续 chunked prefill 配合
        self.is_prefill = True           # True: prefill；False: decode
        self.block_table: list[int] = [] # logical KV block -> physical block id

    @property
    def last_token(self) -> int:
        """返回当前最后一个 token，decode 阶段通常只需要它作为新输入。"""
        return self.token_ids[-1]

    @property
    def num_tokens(self) -> int:
        """当前总 token 数 = prompt + completion。"""
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        """已经生成的 token 数。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        """当前请求是否已经结束。"""
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int):
        """把模型新生成的 token 追加到当前 Sequence。"""
        self.token_ids.append(int(token_id))

    def should_stop(self, eos_token_id: int) -> bool:
        """达到 max_tokens，或生成 EOS 时停止。"""
        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            return True

        if (
            not self.sampling_params.ignore_eos
            and eos_token_id >= 0
            and self.last_token == eos_token_id
        ):
            return True

        return False

