from enum import Enum, auto
from itertools import count
from dataclasses import dataclass
import time

from tinyinfer.sampling_params import SamplingParams


# 计算单条请求内部 ITL 的分位数。
def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


class SequenceStatus(Enum):
    WAITING = auto()   # 已进入引擎，等待 Scheduler 调度
    RUNNING = auto()   # 正在参与 prefill / decode
    FINISHED = auto()  # 已完成，不再参与调度


class Sequence:
    counter = count()   # 给每个 Sequence 分配全局递增 seq_id
    block_size = 256    # 一个 KV block 可容纳的 token 数; 注意这里的block_size为逻辑大小(token数目)而非物理大小(Byte)

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.sampling_params = sampling_params #每个 seq 可以有不同的采样参数

        # token_ids 始终保存：prompt tokens + 已生成 tokens
        self.token_ids = list(token_ids)
        # 静态参数, 不像 num_tokens 一样是动态property参数
        self.num_prompt_tokens = len(token_ids) # 创建Sequence对象时已经确定, 后续token_ids可能在逐渐变长但num_prompts_tokens不会再改变

        # KV / 调度相关状态
        # self.num_prefix_cached_tokens = 0       # 已经存在 KV cache、无需本轮重新计算的 prefix token 数
        # self.num_scheduled_tokens = 0    # scheduler 决定本轮要真正送进模型的 token 数; 和后续 chunked prefill 配合
        # self.is_prefill = True           # True: prefill；False: decode
        self.num_cached_tokens = 0              # 本请求 admission 时从 Prefix Cache 复用了多少 token
        self.num_computed_tokens = 0            # 运行时真状态：从位置 0 开始，有多少 token 的 KV 已经可用
        self.num_scheduled_tokens = 0           # 这一轮要新增计算多少 token
        self.block_table: list[int] = []        # logical KV block -> physical block id
        self.last_block_hash: int = 0           # 最近一个满block对应的hash code

        # performance
        self.arrival_time = time.perf_counter() # 创建seq的时间戳; seq对象创建时即记录arrival_time
        self.first_scheduled_time: float | None = None # 第一次被调度的时间戳
        self.first_token_time: float | None = None # TTFT
        self.finished_time: float | None = None
        self.output_token_times: list[float] = []

    @property
    def last_token(self) -> int:
        """返回当前最后一个 token，decode 阶段通常只需要它作为新输入。"""
        return self.token_ids[-1]

    @property
    def num_tokens(self) -> int: # 有property修饰的成员函数可以直接像访问成员变量一样访问
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

    @property
    def num_blocks(self) -> int: # 对于当前的length需要多少个物理block
        n = self.num_tokens
        return (n + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self) -> int:
        rem = self.num_tokens % self.block_size
        return rem if rem else self.block_size

    @property # prompt的prefill是否已经完成
    def prompt_computed(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property # prompt还剩多少没有完成
    def num_prompt_tokens_remaining(self) -> int:
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property # 是否还需要继续decode, 满足两个条件: 1. prefill已经完成 + 2. 还有tokens没进KV cache
    def needs_decode(self) -> bool:
        return self.prompt_computed and self.num_computed_tokens < self.num_tokens



    # logic_block_id -> 该逻辑block内所有token
    def block_token_ids(self, logical_idx: int) -> list[int]:
        begin = logical_idx * self.block_size
        end = min(begin + self.block_size, self.num_tokens)
        return self.token_ids[begin:end]

    def mark_scheduled(self, n: int): # 把本轮要计算KV cache的token数设为n
        if n <= 0:
            raise ValueError("scheduled token count must be positive")
        if self.num_computed_tokens + n > self.num_tokens: # 比如decode阶段通常每轮针对1个token计算KV cache, 如果为2则不满足条件
            raise ValueError("cannot schedule beyond available token ids")
        self.num_scheduled_tokens = n
        if self.first_scheduled_time is None: # 首次被调度
            self.first_scheduled_time = time.perf_counter()

    def append_token(self, token_id: int):
        """把模型新生成的 token 追加到当前 Sequence。"""
        self.token_ids.append(int(token_id))

        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.output_token_times.append(now) # 同步计时

    def mark_finished(self) -> None:
        self.status = SequenceStatus.FINISHED
        self.finished_time = time.perf_counter()
        #print("only for test: ", self.finished_time)


    def should_stop(self, eos_token_id: int, max_model_len: int) -> bool:
        """达到 completion 上限、模型上下文上限，或生成 EOS 时停止。"""
        # case1: 用户指定的最大生成长度
        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            return True

        # case2: 模型允许的最大总上下文长度：
        # prompt tokens + completion tokens
        if self.num_tokens >= max_model_len:
            return True

        # case3: 遇到 EOS
        if (
            not self.sampling_params.ignore_eos
            and eos_token_id >= 0
            and self.last_token == eos_token_id
        ):
            return True

        return False


    # 性能统计: 单位统一为s;
    def statistics(self) -> dict[str, int | float | None]:
        # token statistics
        prompt_tokens = self.num_prompt_tokens
        output_tokens = self.num_completion_tokens
        decode_tokens = max(0, output_tokens - 1)
        total_tokens = self.num_tokens
        cached_tokens = self.num_cached_tokens

        # arrival -> first scheduled
        queue_time = None
        if self.first_scheduled_time is not None:
            queue_time = self.first_scheduled_time - self.arrival_time

        # first scheduled -> first output token
        prefill_time = None
        if self.first_scheduled_time is not None and self.first_token_time is not None:
            prefill_time = self.first_token_time - self.first_scheduled_time

        # first output token -> finished
        decode_time = None
        if self.first_token_time is not None and self.finished_time is not None:
            decode_time = self.finished_time - self.first_token_time

        # arrival -> first output token
        ttft = None
        if self.first_token_time is not None:
            ttft = self.first_token_time - self.arrival_time

        # first scheduled -> finished
        service_time = None
        if self.first_scheduled_time is not None and self.finished_time is not None:
            service_time = self.finished_time - self.first_scheduled_time

        # arrival -> finished
        e2e_latency = None
        if self.finished_time is not None:
            e2e_latency = self.finished_time - self.arrival_time

        # average latency per decode token
        tpot = None
        if decode_time is not None and decode_tokens > 0:
            tpot = decode_time / decode_tokens

        # adjacent output-token latency
        itls = [
            self.output_token_times[i] - self.output_token_times[i - 1]
            for i in range(1, len(self.output_token_times))
        ]
        mean_itl = sum(itls) / len(itls) if itls else None
        min_itl = min(itls) if itls else None
        max_itl = max(itls) if itls else None
        # User panel 显示上一条请求内部的 TPOT/ITL 分位数。
        tpot_p50 = _percentile(itls, 0.50)
        tpot_p95 = _percentile(itls, 0.95)
        tpot_p99 = _percentile(itls, 0.99)

        # steady-state decode throughput
        decode_throughput = None
        if decode_time is not None and decode_time > 0 and decode_tokens > 0:
            decode_throughput = decode_tokens / decode_time

        # whole-request output throughput
        request_throughput = None
        if e2e_latency is not None and e2e_latency > 0:
            request_throughput = output_tokens / e2e_latency

        return {
            # token statistics
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens, # 输出总token
            "decode_tokens": decode_tokens, # decode输出总token, = output_tokens - 1
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,

            # latency
            "queue_time": queue_time,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "ttft": ttft,
            "service_time": service_time,
            "e2e_latency": e2e_latency,
            "tpot": tpot,
            "tpot_p50": tpot_p50,
            "tpot_p95": tpot_p95,
            "tpot_p99": tpot_p99,
            "mean_itl": mean_itl,
            "min_itl": min_itl,
            "max_itl": max_itl,

            # throughput
            "decode_throughput": decode_throughput,
            "request_throughput": request_throughput,
        }





# 最小调度单元, 一个处于chunked prefill/decode阶段的seq
@dataclass(slots=True)
class ScheduledItem:
    seq: Sequence
    start_pos: int
    num_tokens: int # 这一轮对于该seq而言需要送入模型前向传播的token数
    is_prefill: bool
    # 这一条 Sequence 在本轮 forward 完成后是否应该从它最后一个 query 的 logits 采样新 token; 对于chunked prefill阶段为false
    sample_after: bool 


# 调度单元集合, 针对一批次推理
@dataclass(slots=True)
class SchedulerOutput:
    items: list[ScheduledItem]

    @property # 这一批一共要处理多少个token
    def num_scheduled_tokens(self) -> int:
        return sum(item.num_tokens for item in self.items)

    @property # 这一批有多少个seq
    def seqs(self) -> list[Sequence]:
        return [item.seq for item in self.items]





