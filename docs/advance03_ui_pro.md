# tinyInfer Advance 03：UI Pro

这一讲直接基于你当前上传的 `tinyInfer-master(4)` 继续修改，不另起一套 UI 或 runtime。目标只有三项：

1. KV cache block map 改为三态：`empty / cached / active`。
2. 左侧 User 区域与右侧 Statistics 区域解耦；左侧独立上下滚动，User 扩展到 5 个。
3. User Panel 与全局 Performance Statistics 都增加 TPOT 的 `p50 / p95 / p99`。

本讲所有需要修改的文件都给出**修改后的完整内容**，复制覆盖即可；新增或修改处统一用 `[MOD]` 注释标记。没有出现在本讲里的文件保持当前版本不变。

---

## 0. 本次目录变化

```text
tinyInfer/
├── tinyinfer/
│   ├── engine/
│   │   ├── sequence.py          # [MOD] 单请求 TPOT p50/p95/p99
│   │   └── llm_engine.py        # [MOD] KV block 三态 snapshot
│   ├── runtime/
│   │   └── dynamic_engine.py    # [MOD] 全局 request-level TPOT 分位数
│   └── ui/
│       └── static/
│           ├── index.html       # [MOD] 三态图例 + 全局 TPOT 分位区
│           ├── style.css        # [MOD] 左侧独立滚动 + 三态配色
│           └── app.js           # [MOD] 5 User + 三态刷新 + 分位显示
└── docs/
    └── advance03_ui_pro.md
```

这里有一个统计口径要先固定：

- **User Panel 的 TPOT p50/p95/p99**：针对该 User **上一条已完成请求内部**的相邻输出 token latency（ITL）计算。
- **Performance Statistics 的 TPOT p50/p95/p99**：针对服务器启动以来**所有已完成请求的 request-level TPOT**计算。

因此两者不是同一个集合上的 percentile，这一点是刻意区分的。

KV block 三态则定义为：

```text
empty   : ref_count == 0 && hash is None
cached  : ref_count == 0 && hash is not None
active  : ref_count > 0
```

其中 `cached` 表示曾经被使用且完整 block 的 prefix metadata 仍保留，可被后续请求重新激活；`active` 无论当前 block 是否已经形成 hash，只要 `ref_count > 0` 都视为正在使用。

---

# 1. Step 1：为单条 Sequence 增加 TPOT p50/p95/p99

修改文件：

```text
tinyinfer/engine/sequence.py
```

`Sequence.statistics()` 原本已经保存 `output_token_times`，因此不需要额外在 UI 层猜测 token 间隔。这里直接利用相邻输出 token 时间戳构造 ITL 列表，并计算 p50/p95/p99。这样 User Panel 拿到 `finished` event 时即可直接显示上一条请求的分位指标。

下面是**修改后的完整文件**，直接覆盖当前文件即可：

```python
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass
import time

from tinyinfer.sampling_params import SamplingParams


# [MOD] 计算单条请求内部 ITL 的分位数。
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
        self.arrival_time = time.perf_counter() # 创建seq的时间戳
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
        self.output_token_times.append(now)

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
        # [MOD] User panel 显示上一条请求内部的 TPOT/ITL 分位数。
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
            # [MOD] 单请求内部相邻输出 token latency 的分位数。
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
```

---

# 2. Step 2：Resource snapshot 增加 KV cache 三态

修改文件：

```text
tinyinfer/engine/llm_engine.py
```

这里不修改 `BlockManager` 的分配逻辑，只扩展只读 runtime snapshot。为了避免每 250 ms 发送一整个长度约一万的字符串状态数组，后端只发送两个稀疏 ID 列表：`active_block_ids` 与 `cached_block_ids`；前端把剩余 block 自动解释为 `empty`。

下面是**修改后的完整文件**，直接覆盖当前文件即可：

```python
from dataclasses import dataclass
import torch

from transformers import AutoTokenizer
from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence
from tinyinfer.sampling_params import SamplingParams


@dataclass(slots=True)
class StreamUpdate:
    seq_id: int
    token_id: int
    completion_token_ids: list[int]
    finished: bool
    statistics: dict | None


# 整体调用链条: LLM -> LLMEngine -> 创建seqs列表 -> Scheduler -> ModelRunner
class LLMEngine:
    def __init__(self, config: Config | None = None, device: str = "cuda"):
        self.config = config or Config()
        self.config.validate_runtime_fields()
        self.config.load_hf_config()

        Sequence.block_size = self.config.kvcache_block_size

        # ModelRunner 先初始化，因为它会根据真实模型显存占用计算config.num_kvcache_blocks;
        # 随后 Scheduler/BlockManager 才能使用该值。
        self.model_runner = ModelRunner(self.config, device=device)
        self.scheduler = Scheduler(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        self.config.eos_token_id = (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1)
        self.scheduler.eos_token_id = self.config.eos_token_id


    def add_request(self, token_ids: list[int], params: SamplingParams): # 简化后的创建请求函数, 用一个token list代表
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq)
        return seq.seq_id
 
    def build_chat_prompt(self, text:str, system_prompt:str|None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        if self.tokenizer.chat_template:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        return text

    # 把 UI 字符串转换成真正 chat-template prompt, 再进入原 Scheduler
    def add_text_request(
        self,
        text: str,
        params: SamplingParams,
        system_prompt: str | None = None,
    ) -> int:
        prompt = self.build_chat_prompt(text, system_prompt)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self.add_request(token_ids, params)


    def step_with_updates(self) -> tuple[list, list[StreamUpdate]]:
        # 对内: 走通schedule -> run -> postprocess的完整一轮调度流程
        output = self.scheduler.schedule() # 从waiting池子中挑选本轮进行推理的请求
        if not output.items:
            return [], []
        sampled_tokens = self.model_runner.run(output) # 输出为dict of {seq_id: token_id}, 表示所有需要采样的请求所产生的下一个token id是什么
        self.scheduler.postprocess(output, sampled_tokens) # 将产生的下一个token拼接到对应的seq请求中, 并注册prefix cache; 管理waiting/running list

        # 对外: 把当前一轮真正生成token的请求信息打包送给前端, 进行网页端聊天框刷新
        updates = []
        for item in output.items:
            token_id = sampled_tokens.get(item.seq.seq_id) # 得到产生next token请求的next token id
            if token_id is None: # 如果这一批的某个seq没有生成下一个token则直接跳过, 无需进行前端网页刷新
                continue
            seq = item.seq
            updates.append(
                StreamUpdate(
                    seq_id=seq.seq_id,
                    token_id=token_id,
                    completion_token_ids=list(
                        seq.token_ids[seq.num_prompt_tokens:]
                    ),
                    finished=seq.is_finished,
                    statistics=seq.statistics() if seq.is_finished else None,
                )
            )

        return output.items, updates # 前者是内部信息(这一轮调度了哪些请求, 不论是否生成next token); 后者是对外信息(这一轮哪些请求生成了next token从而需要刷新聊天框)

    def step(self): # 对内的前向传播调度函数, 不对外传递信息
        items, _ = self.step_with_updates()
        return items

    
    # 资源快照只读取 CPU metadata，不做 CUDA synchronize。
    def runtime_resource_snapshot(self) -> dict:
        device = self.model_runner.device
        if device.type != "cuda": # cpu端直接忽略
            return {
                "total_bytes": 0,
                "unused_bytes": 0,
                "reserved_bytes": 0,
                "allocated_kv_bytes": 0,
                "available_kv_bytes": 0,
                "block_bytes": 0,
                "num_blocks": 0,
                "active_block_ids": [],
                "cached_block_ids": [],  # [MOD] ref_count=0 且保留 prefix cache 的 block。
            }

        profile = self.model_runner.resource_profile
        total_bytes = profile.total_bytes
        unused_bytes = profile.unused_bytes
        reserved_bytes = profile.reserved_bytes
        block_bytes = profile.block_bytes
        blocks = self.scheduler.block_manager.blocks
        active_block_ids = [
            block.block_id for block in blocks if block.ref_count > 0
        ]
        # [MOD] 蓝色状态只表示 cached-free；其余非 active block 视为从未分配/空闲。
        cached_block_ids = [
            block.block_id
            for block in blocks
            if block.ref_count == 0 and block.hash is not None
        ]
        allocated_kv_bytes = len(active_block_ids) * block_bytes
        available_kv_bytes = max(0, total_bytes - unused_bytes - reserved_bytes - allocated_kv_bytes)

        return {
            "total_bytes": total_bytes,
            "unused_bytes": unused_bytes,
            "reserved_bytes": reserved_bytes,
            "allocated_kv_bytes": int(allocated_kv_bytes),
            "available_kv_bytes": int(available_kv_bytes),
            "block_bytes": int(block_bytes),
            "num_blocks": len(blocks),
            "active_block_ids": active_block_ids,
            "cached_block_ids": cached_block_ids,  # [MOD]
        }




    # 针对所有seq请求生成/decode完整的token id序列
    def generate_token_ids(
        self,
        prompts: list[list[int]], # 每个请求本身是一个list,因此这里是list的list
        sampling_params: SamplingParams | list[SamplingParams], # 如果只有一个sampling_params说明所有请求共用, 需要进行广播
    ):
        # basic argument check
        if isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompts)
        else:
            params = sampling_params
        if len(params) != len(prompts):
            raise ValueError("prompts and sampling params length mismatch")

        # 创建Sequence列表并加入到Scheduler中由其进行调度
        seqs = []
        for token_ids, p in zip(prompts, params): # seq达到一刻
            seq = Sequence(token_ids, p)
            seqs.append(seq)
            self.scheduler.add(seq)

        # 直到Scheduler调度完成, 所有请求全部推理完毕; 实际上Scheduler会调用ModelRunner进行真正的推理
        while not self.scheduler.is_finished():
            items = self.step()
            if not items and not self.scheduler.is_finished():
                raise RuntimeError("scheduler made no progress while requests remain")

        seqs.sort(key=lambda x: x.seq_id)
        return [
            {
                "token_ids": seq.token_ids[seq.num_prompt_tokens:], # decode阶段生成的token id, 不包含prompt
                "all_token_ids": list(seq.token_ids), # 所有的token_id
                "num_cached_tokens": seq.num_cached_tokens, # prefix cache命中的token数目
                "statistics": seq.statistics(),
            }
            for seq in seqs
        ]


    # 上层最终调用的接口, 输入输出均为str列表
    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        # step1: str -> token_id
        prompt_token_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]

        # step2: 运行直到每个seq请求都推理完毕, 得到完整的token_ids列表
        outputs = self.generate_token_ids(prompt_token_ids, sampling_params)

        # step3: 将输出的token_ids 解码为字符串
        for out in outputs:
            out["text"] = self.tokenizer.decode(out["token_ids"], skip_special_tokens=True) # 动态为dict新建"text"键

        return outputs
```

---

# 3. Step 3：全局 Performance Statistics 增加 TPOT p50/p95/p99

修改文件：

```text
tinyinfer/runtime/dynamic_engine.py
```

现有全局统计只保存 sum/count，因此只能计算平均值。这里额外保存每条已完成请求的 request-level `tpot`，再在 runtime snapshot 中计算 p50/p95/p99。其他指标仍保持原来的平均值逻辑不变。

下面是**修改后的完整文件**，直接覆盖当前文件即可：

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Callable

from tinyinfer.engine.llm_engine import LLMEngine
from tinyinfer.sampling_params import SamplingParams


# [MOD] 系统级 TPOT 分位数按“每条已完成请求的 TPOT”统计。
def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


PERFORMANCE_KEYS = (
    "queue_time",
    "prefill_time",
    "decode_time",
    "ttft",
    "service_time",
    "e2e_latency",
    "tpot",
    "decode_throughput",
    "request_throughput",
)


@dataclass(slots=True)
class SubmitRequest:
    user_id: str
    text: str
    params: SamplingParams


@dataclass(slots=True)
class RuntimeEvent:
    type: str
    user_id: str | None = None
    seq_id: int | None = None
    text: str | None = None
    statistics: dict | None = None
    resource: dict | None = None
    performance: dict | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DynamicEngineService:
    def __init__(
        self,
        engine: LLMEngine,
        system_prompt: str = "You are a helpful, concise assistant.",
        snapshot_interval_s: float = 0.25,
    ):
        self.engine = engine
        self.system_prompt = system_prompt
        self.snapshot_interval_s = snapshot_interval_s

        self._commands: Queue[list[SubmitRequest] | None] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None
        self._event_sink: Callable[[dict], None] | None = None

        self._busy_users: set[str] = set()
        self._busy_lock = Lock()
        self._seq_to_user: dict[int, str] = {}

        self._metric_sums = {key: 0.0 for key in PERFORMANCE_KEYS}
        self._metric_counts = {key: 0 for key in PERFORMANCE_KEYS}
        self._completed_requests = 0
        self._tpot_values: list[float] = []  # [MOD] 保存 request-level TPOT 用于 p50/p95/p99。

        self._last_snapshot_at = 0.0
        self._snapshot_lock = Lock()
        self._last_runtime_snapshot: dict | None = None

    def start(self, event_sink: Callable[[dict], None]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._event_sink = event_sink
        self._stop.clear()
        self._thread = Thread(
            target=self._worker_loop,
            name="tinyinfer-engine",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._commands.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)

    def submit(self, request: SubmitRequest) -> bool:
        return bool(self.submit_many([request]))

    def submit_many(self, requests: list[SubmitRequest]) -> list[str]:
        accepted = []
        batch = []

        with self._busy_lock:
            for request in requests:
                if not request.text.strip():
                    continue
                if request.user_id in self._busy_users:
                    continue
                self._busy_users.add(request.user_id)
                accepted.append(request.user_id)
                batch.append(request)

        if batch:
            self._commands.put(batch)
        return accepted

    def is_user_busy(self, user_id: str) -> bool:
        with self._busy_lock:
            return user_id in self._busy_users

    def current_runtime_snapshot(self) -> dict | None:
        with self._snapshot_lock:
            return deepcopy(self._last_runtime_snapshot)

    def _emit(self, event: RuntimeEvent) -> None:
        if self._event_sink:
            self._event_sink(event.to_dict())

    def _release_user(self, user_id: str) -> None:
        with self._busy_lock:
            self._busy_users.discard(user_id)

    def _admit_batch(self, batch: list[SubmitRequest]) -> None:
        for request in batch:
            try:
                seq_id = self.engine.add_text_request( # 关键逻辑: 进入waiting list
                    request.text,
                    request.params,
                    self.system_prompt,
                )
                self._seq_to_user[seq_id] = request.user_id
                self._emit(
                    RuntimeEvent(
                        type="started",
                        user_id=request.user_id,
                        seq_id=seq_id,
                    )
                )
            except Exception as exc:
                self._release_user(request.user_id)
                self._emit(
                    RuntimeEvent(
                        type="error",
                        user_id=request.user_id,
                        message=str(exc),
                    )
                )

    def _drain_commands(self) -> None:
        while True:
            try:
                batch = self._commands.get_nowait()
            except Empty:
                return
            if batch is None:
                return
            self._admit_batch(batch)

    def _record_statistics(self, statistics: dict | None) -> None:
        if not statistics:
            return
        self._completed_requests += 1
        for key in PERFORMANCE_KEYS:
            value = statistics.get(key)
            if value is None:
                continue
            numeric = float(value)
            self._metric_sums[key] += numeric
            self._metric_counts[key] += 1
            if key == "tpot":
                self._tpot_values.append(numeric)  # [MOD]

    def _performance_snapshot(self) -> dict:
        averages = {}
        for key in PERFORMANCE_KEYS:
            count = self._metric_counts[key]
            averages[key] = (
                self._metric_sums[key] / count if count else None
            )
        return {
            "completed_requests": self._completed_requests,
            "averages": averages,
            # [MOD] 所有已完成 seq 的 request-level TPOT 分布。
            "tpot_percentiles": {
                "p50": _percentile(self._tpot_values, 0.50),
                "p95": _percentile(self._tpot_values, 0.95),
                "p99": _percentile(self._tpot_values, 0.99),
            },
        }

    def _emit_runtime_snapshot(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_snapshot_at < self.snapshot_interval_s:
            return
        self._last_snapshot_at = now

        snapshot = {
            "resource": self.engine.runtime_resource_snapshot(),
            "performance": self._performance_snapshot(),
        }
        with self._snapshot_lock:
            self._last_runtime_snapshot = deepcopy(snapshot)

        self._emit(
            RuntimeEvent(
                type="runtime",
                resource=snapshot["resource"],
                performance=snapshot["performance"],
            )
        )

    def _handle_updates(self, updates) -> bool:
        any_finished = False
        for update in updates:
            user_id = self._seq_to_user.get(update.seq_id)
            if user_id is None:
                continue

            text = self.engine.tokenizer.decode(
                update.completion_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self._emit(
                RuntimeEvent(
                    type="token",
                    user_id=user_id,
                    seq_id=update.seq_id,
                    text=text,
                )
            )

            if update.finished:
                any_finished = True
                self._record_statistics(update.statistics)
                self._emit(
                    RuntimeEvent(
                        type="finished",
                        user_id=user_id,
                        seq_id=update.seq_id,
                        text=text,
                        statistics=update.statistics,
                    )
                )
                self._seq_to_user.pop(update.seq_id, None)
                self._release_user(user_id)
        return any_finished

    def _fail_active_requests(self, message: str) -> None:
        for seq_id, user_id in list(self._seq_to_user.items()):
            self._emit(
                RuntimeEvent(
                    type="error",
                    user_id=user_id,
                    seq_id=seq_id,
                    message=message,
                )
            )
            self._release_user(user_id)
        self._seq_to_user.clear()

    def _worker_loop(self) -> None: # 最核心的循环函数
        self._emit_runtime_snapshot(force=True)

        while not self._stop.is_set():
            if self.engine.scheduler.is_finished():
                try:
                    batch = self._commands.get(timeout=0.1)
                except Empty:
                    self._emit_runtime_snapshot()
                    continue
                if batch is None:
                    continue
                self._admit_batch(batch)

            self._drain_commands()
            if self.engine.scheduler.is_finished():
                continue

            try:
                items, updates = self.engine.step_with_updates()
                if not items and not self.engine.scheduler.is_finished():
                    raise RuntimeError(
                        "scheduler made no progress while requests remain"
                    )
                any_finished = self._handle_updates(updates)
                self._emit_runtime_snapshot(force=any_finished)
            except Exception as exc:
                self._fail_active_requests(str(exc))
                self._emit_runtime_snapshot(force=True)
                return
```

---

# 4. Step 4：Statistics Panel 增加三态图例与全局 TPOT 分位区

修改文件：

```text
tinyinfer/ui/static/index.html
```

HTML 只增加 KV 三态 legend 和全局 TPOT percentile 展示区；左右主体结构保持原样，真正的左右解耦通过 CSS 完成。

下面是**修改后的完整文件**，直接覆盖当前文件即可：

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tinyInfer Console</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="shell">
    <section class="users" id="users"></section>

    <aside class="side">
      <section class="panel statistics-panel">
        <div class="panel-head global-head">
          <div>
            <div class="panel-title">Statistics</div>
            <div class="subtle">Runtime telemetry · refresh ~4 Hz</div>
          </div>
          <div class="live-dot"><span></span>live</div>
        </div>

        <section class="stats-section resource-section">
          <div class="section-title">Resource statistics</div>

          <div class="memory-bar" id="memory-bar" aria-label="GPU memory budget composition">
            <div class="memory-segment unused" id="segment-unused"></div>
            <div class="memory-segment reserved" id="segment-reserved"></div>
            <div class="memory-segment allocated" id="segment-allocated"></div>
            <div class="memory-segment available" id="segment-available"></div>
          </div>

          <div class="memory-legend">
            <div><span class="swatch unused"></span><span>Unused</span><strong id="mem-unused">--</strong></div>
            <div><span class="swatch reserved"></span><span>Reserved</span><strong id="mem-reserved">--</strong></div>
            <div><span class="swatch allocated"></span><span>Allocated</span><strong id="mem-allocated">--</strong></div>
            <div><span class="swatch available"></span><span>Available</span><strong id="mem-available">--</strong></div>
          </div>

          <div class="kv-head">
            <div>
              <div class="section-label">KV cache blocks</div>
              <div class="subtle" id="kv-summary">Waiting for runtime snapshot</div>
            </div>
            <!-- [MOD] KV block map 改为 empty / cached / active 三态。 -->
            <div class="kv-key">
              <span class="swatch block-empty"></span>empty
              <span class="swatch block-cached"></span>cached
              <span class="swatch block-active"></span>active
            </div>
          </div>
          <div class="block-grid" id="block-grid"></div>
          <div class="resource-note">Reserved is captured before KV pool creation. The block map distinguishes never-used empty blocks, cached-free blocks and active blocks.</div>
        </section>

        <section class="stats-section performance-section">
          <div class="section-title-row">
            <div class="section-title">Performance statistics</div>
            <div class="request-count" id="completed-requests">0 completed</div>
          </div>
          <div class="performance-grid" id="performance-grid"></div>
          <!-- [MOD] 全局 TPOT 分位数：对所有已完成请求的 request-level TPOT 取分位数。 -->
          <div class="percentile-head">TPOT percentiles</div>
          <div class="percentile-grid" id="performance-tpot-percentiles">
            <div class="metric-card"><span>P50</span><strong data-tpot-percentile="p50">--</strong></div>
            <div class="metric-card"><span>P95</span><strong data-tpot-percentile="p95">--</strong></div>
            <div class="metric-card"><span>P99</span><strong data-tpot-percentile="p99">--</strong></div>
          </div>
        </section>
      </section>

      <section class="panel controller">
        <div class="panel-title">Controller</div>
        <div class="controller-grid">
          <button id="start-all" class="primary" disabled>Start all ready</button>
          <div class="controller-slot">future</div>
          <div class="controller-slot">future</div>
          <div class="controller-slot">future</div>
        </div>
      </section>
    </aside>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

---

# 5. Step 5：左侧独立滚动、User Panel 加高、KV 三色样式

修改文件：

```text
tinyinfer/ui/static/style.css
```

左侧 `.users` 从三行等分 grid 改成独立纵向滚动容器，每个 User Panel 保持固定最小高度，因此 5 个 User 不会被压扁。右侧 `.side` 继续占满 viewport 高度，不随左侧滚动。输入框与输出区域适当增高。

下面是**修改后的完整文件**，直接覆盖当前文件即可：

```css
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0b0d12;
  color: #edf1f7;
  --panel: rgba(18, 21, 28, .9);
  --panel-soft: #11151d;
  --border: #282e3a;
  --muted: #8c96a8;
  --accent: #dfe7ff;
  --accent-dark: #111827;
  --unused: #565d68;
  --reserved: #c55b3d;
  --allocated: #e59a58;
  --available: #54b985;
  --cached: #6ea8fe; /* [MOD] cached-free KV block */
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(73, 95, 160, .17), transparent 34rem),
    #0b0d12;
}

button, textarea, input { font: inherit; }

.shell {
  width: min(1760px, calc(100vw - 32px));
  height: calc(100vh - 32px);
  margin: 16px auto;
  display: grid;
  grid-template-columns: minmax(0, 2.25fr) minmax(390px, .95fr);
  gap: 14px;
}

.users {
  min-height: 0;
  overflow-y: auto; /* [MOD] 左侧 User 区域独立滚动。 */
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding-right: 4px;
}

.side {
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(0, 1fr) 145px;
  gap: 14px;
}

.panel {
  min-height: 0;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  box-shadow: 0 18px 50px rgba(0, 0, 0, .24);
  backdrop-filter: blur(18px);
}

.user-panel {
  flex: 0 0 auto; /* [MOD] 不再被三等分压缩。 */
  min-height: 610px;
  padding: 16px;
  display: grid;
  grid-template-rows: auto minmax(250px, 1fr) auto;
  gap: 12px;
}

.panel-head,
.section-title-row,
.kv-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.panel-title {
  color: #f7f9fc;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: .02em;
}

.status,
.subtle,
.request-count,
.kv-key,
.resource-note {
  color: var(--muted);
  font-size: 11px;
}

.status.busy { color: #9ab5ff; }

.output {
  overflow-y: auto;
  padding: 12px 14px;
  border: 1px solid #232936;
  border-radius: 13px;
  background: #0e1117;
  color: #dce3ee;
  white-space: pre-wrap;
  line-height: 1.5;
  font-size: 13px;
}

.user-lower {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(310px, .95fr);
  gap: 10px;
  align-items: stretch;
}

.composer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 9px;
  align-items: end;
}

textarea {
  width: 100%;
  min-height: 126px; /* [MOD] 左侧可滚动后适当增高输入框。 */
  max-height: 190px;
  resize: vertical;
  border: 1px solid #2b3240;
  border-radius: 13px;
  outline: none;
  padding: 11px 13px;
  background: #151922;
  color: #f5f7fb;
  transition: border-color .15s ease, box-shadow .15s ease, opacity .15s ease;
}

textarea:focus {
  border-color: #657dc8;
  box-shadow: 0 0 0 3px rgba(101, 125, 200, .14);
}

textarea:read-only { color: #aeb6c5; }

button {
  border: 0;
  border-radius: 11px;
  padding: 10px 15px;
  font-weight: 700;
  cursor: pointer;
  transition: transform .12s ease, opacity .12s ease, background .12s ease;
}

button:not(:disabled):active { transform: translateY(1px); }
button:disabled { cursor: default; opacity: .38; }

.primary {
  background: var(--accent);
  color: var(--accent-dark);
}

.user-tools {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, .85fr);
  gap: 8px;
}

.mini-panel {
  border: 1px solid #252b36;
  border-radius: 12px;
  background: #10141b;
  padding: 9px 10px;
}

.mini-title,
.section-label,
.section-title {
  color: #cfd6e2;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}

.control-row {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr) 42px;
  gap: 7px;
  align-items: center;
  margin-top: 8px;
  color: #aeb7c6;
  font-size: 10px;
}

.control-value {
  text-align: right;
  color: #eef2f8;
  font-variant-numeric: tabular-nums;
}

input[type="range"] {
  width: 100%;
  accent-color: #aabaf0;
}

input:disabled { opacity: .42; }

.toggle-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  color: #aeb7c6;
  font-size: 10px;
}

.switch {
  position: relative;
  width: 34px;
  height: 19px;
  display: inline-block;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider-toggle {
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background: #3a414e;
  transition: .16s ease;
}

.slider-toggle::after {
  content: "";
  position: absolute;
  width: 13px;
  height: 13px;
  left: 3px;
  top: 3px;
  border-radius: 50%;
  background: #e7ebf2;
  transition: .16s ease;
}

.switch input:checked + .slider-toggle { background: #6c84d1; }
.switch input:checked + .slider-toggle::after { transform: translateX(15px); }
.switch input:disabled + .slider-toggle { opacity: .42; }

.user-stat-grid {
  margin-top: 9px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr)); /* [MOD] 6项统计两列展示。 */
  gap: 8px 12px;
}

.user-stat {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 10px;
  color: #949eae;
}

.user-stat strong {
  color: #edf1f7;
  font-variant-numeric: tabular-nums;
}

.statistics-panel {
  padding: 16px;
  overflow-y: auto;
}

.global-head { margin-bottom: 12px; }
.live-dot {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #9aa4b4;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.live-dot span {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--unused);
  box-shadow: 0 0 12px rgba(84, 185, 133, .55);
}

.stats-section {
  border: 1px solid #252b36;
  border-radius: 14px;
  background: rgba(15, 18, 24, .82);
  padding: 12px;
}
.stats-section + .stats-section { margin-top: 10px; }


.memory-bar {
  height: 17px;
  display: flex;
  overflow: hidden;
  border-radius: 999px;
  margin-top: 9px;
  background: #1c212b;
  border: 1px solid #2c3340;
}
.memory-segment { height: 100%; min-width: 0; transition: width .18s ease; }
.unused { background: var(--unused); }
.reserved { background: var(--reserved); }
.allocated { background: var(--allocated); }
.available { background: var(--available); }

.memory-legend {
  margin-top: 10px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px 12px;
}
.memory-legend > div {
  display: grid;
  grid-template-columns: 9px 1fr auto;
  gap: 6px;
  align-items: center;
  color: #9da7b7;
  font-size: 10px;
}
.memory-legend strong { color: #edf1f7; font-variant-numeric: tabular-nums; }

.swatch {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex: 0 0 auto;
}

.kv-head { margin-top: 13px; }
.kv-key { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
.block-empty { background: var(--available); } /* [MOD] */
.block-cached { background: var(--cached); } /* [MOD] */
.block-active { background: var(--allocated); } /* [MOD] */

.block-grid {
  margin-top: 8px;
  max-height: 175px;
  overflow-y: auto;
  display: grid;
  grid-template-columns: repeat(48, minmax(0, 1fr));
  gap: 2px;
  padding: 7px;
  border-radius: 10px;
  border: 1px solid #252b36;
  background: #0c1015;
}

.kv-block {
  aspect-ratio: 1;
  min-width: 0;
  border-radius: 1px;
  opacity: .88;
}
.kv-block.empty { background: var(--available); } /* [MOD] 从未分配 */
.kv-block.cached { background: var(--cached); } /* [MOD] cached-free */
.kv-block.active { background: var(--allocated); opacity: 1; } /* [MOD] active */

.resource-note { margin-top: 7px; line-height: 1.35; }

.performance-grid {
  margin-top: 10px;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 7px;
}

.metric-card {
  min-width: 0;
  padding: 8px;
  border-radius: 10px;
  background: #121720;
  border: 1px solid #232a35;
}
.metric-card span {
  display: block;
  color: #8f99aa;
  font-size: 9px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.metric-card strong {
  display: block;
  margin-top: 4px;
  color: #eef2f8;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.percentile-head {
  margin-top: 12px; /* [MOD] */
  color: #9ba6b6;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}

.percentile-grid {
  margin-top: 7px; /* [MOD] */
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 7px;
}

.controller { padding: 16px; }
.controller-grid {
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
#start-all { grid-column: span 1; }
.controller-slot {
  display: grid;
  place-items: center;
  min-height: 39px;
  border: 1px dashed #29303d;
  border-radius: 11px;
  color: #4e5664;
  font-size: 10px;
}

@media (max-width: 1250px) {
  .user-lower { grid-template-columns: 1fr; }
  .user-tools { grid-template-columns: 1fr 1fr; }
  .output { min-height: 90px; }
}

@media (max-width: 980px) {
  .shell {
    height: auto;
    grid-template-columns: 1fr;
  }
  .users { grid-template-rows: repeat(3, auto); }
  .user-panel { min-height: 430px; }
  .side { grid-template-rows: auto 145px; }
}
```

---

# 6. Step 6：扩展至 5 个 User，并接入三态 KV 与 TPOT 分位显示

修改文件：

```text
tinyinfer/ui/static/app.js
```

前端扩展到 5 个独立 User；KV map 维护 `activeBlocks` 与 `cachedBlocks` 两组集合，只重绘状态发生变化的方块。User 完成后显示该请求自己的 TPOT mean/p50/p95/p99；全局 runtime event 则更新所有已完成请求的 TPOT 分位数。

下面是**修改后的完整文件**，直接覆盖当前文件即可：

```javascript
// [MOD] User panel 从 3 个扩展到 5 个。
const USERS = ["user1", "user2", "user3", "user4", "user5"];
const PERFORMANCE_METRICS = [
  ["queue_time", "Queue", "latency"],
  ["prefill_time", "Prefill", "latency"],
  ["decode_time", "Decode", "latency"],
  ["ttft", "TTFT", "latency"],
  ["service_time", "Service", "latency"],
  ["e2e_latency", "E2E", "latency"],
  ["tpot", "TPOT", "latency"],
  ["decode_throughput", "Decode tok/s", "throughput"],
  ["request_throughput", "Request tok/s", "throughput"],
];

const states = new Map();
const usersRoot = document.querySelector("#users");
const startAll = document.querySelector("#start-all");
const blockGrid = document.querySelector("#block-grid");
const performanceGrid = document.querySelector("#performance-grid");
const ws = new WebSocket(`ws://${location.host}/ws`);

let blockNodes = [];
let activeBlocks = new Set();
let cachedBlocks = new Set(); // [MOD] 独立保存 cached-free block。

function makePanel(userId, index) {
  const panel = document.createElement("section");
  panel.className = "panel user-panel";
  panel.innerHTML = `
    <div class="panel-head">
      <div class="panel-title">User ${index + 1}</div>
      <div class="status">Idle</div>
    </div>
    <div class="output"></div>
    <div class="user-lower">
      <div class="composer">
        <textarea placeholder="Type a prompt..."></textarea>
        <button class="primary start-button" disabled>Start</button>
      </div>
      <div class="user-tools">
        <section class="mini-panel request-controls">
          <div class="mini-title">Controller</div>
          <div class="control-row">
            <span>max tokens</span>
            <input class="max-tokens" type="range" min="32" max="8192" step="32" value="1024">
            <span class="control-value max-value">1024</span>
          </div>
          <div class="control-row">
            <span>temperature</span>
            <input class="temperature" type="range" min="0.1" max="2.0" step="0.1" value="0.8">
            <span class="control-value temp-value">0.8</span>
          </div>
          <div class="toggle-row">
            <span>greedy</span>
            <label class="switch">
              <input class="greedy" type="checkbox" checked>
              <span class="slider-toggle"></span>
            </label>
          </div>
        </section>
        <section class="mini-panel user-statistics">
          <div class="mini-title">Last request</div>
          <div class="user-stat-grid">
            <div class="user-stat"><span>TTFT</span><strong class="stat-ttft">--</strong></div>
            <div class="user-stat"><span>TPOT mean</span><strong class="stat-tpot">--</strong></div>
            <div class="user-stat"><span>TPOT p50</span><strong class="stat-tpot-p50">--</strong></div>
            <div class="user-stat"><span>TPOT p95</span><strong class="stat-tpot-p95">--</strong></div>
            <div class="user-stat"><span>TPOT p99</span><strong class="stat-tpot-p99">--</strong></div>
            <div class="user-stat"><span>E2E</span><strong class="stat-e2e">--</strong></div>
          </div>
        </section>
      </div>
    </div>`;

  const state = {
    userId,
    panel,
    input: panel.querySelector("textarea"),
    button: panel.querySelector(".start-button"),
    output: panel.querySelector(".output"),
    status: panel.querySelector(".status"),
    maxTokens: panel.querySelector(".max-tokens"),
    maxValue: panel.querySelector(".max-value"),
    temperature: panel.querySelector(".temperature"),
    tempValue: panel.querySelector(".temp-value"),
    greedy: panel.querySelector(".greedy"),
    statTTFT: panel.querySelector(".stat-ttft"),
    statTPOT: panel.querySelector(".stat-tpot"),
    statTPOTP50: panel.querySelector(".stat-tpot-p50"), // [MOD]
    statTPOTP95: panel.querySelector(".stat-tpot-p95"), // [MOD]
    statTPOTP99: panel.querySelector(".stat-tpot-p99"), // [MOD]
    statE2E: panel.querySelector(".stat-e2e"),
    busy: false,
    outputBase: "",
  };
  states.set(userId, state);

  state.input.addEventListener("input", refreshControls);
  state.button.addEventListener("click", () => submitOne(state));
  state.maxTokens.addEventListener("input", () => {
    state.maxValue.textContent = state.maxTokens.value;
  });
  state.temperature.addEventListener("input", () => {
    state.tempValue.textContent = Number(state.temperature.value).toFixed(1);
  });

  usersRoot.appendChild(panel);
}

function ready(state) {
  return !state.busy && state.input.value.trim().length > 0;
}

function setBusy(state, busy) {
  state.busy = busy;
  state.input.readOnly = busy;
  state.maxTokens.disabled = busy;
  state.temperature.disabled = busy;
  state.greedy.disabled = busy;
  state.status.textContent = busy ? "Running" : "Idle";
  state.status.classList.toggle("busy", busy);
  refreshControls();
}

function refreshControls() {
  for (const state of states.values()) {
    state.button.disabled = !ready(state) || ws.readyState !== WebSocket.OPEN;
  }
  startAll.disabled =
    ws.readyState !== WebSocket.OPEN ||
    ![...states.values()].some(ready);
}

function requestPayload(state) {
  return {
    user_id: state.userId,
    text: state.input.value,
    max_tokens: Number(state.maxTokens.value),
    temperature: Number(state.temperature.value),
    greedy: state.greedy.checked,
  };
}

function beginLocal(state) {
  state.outputBase = state.output.textContent;
  setBusy(state, true);
}

function submitOne(state) {
  if (!ready(state) || ws.readyState !== WebSocket.OPEN) return;
  const payload = requestPayload(state);
  beginLocal(state);
  ws.send(JSON.stringify({type: "submit", ...payload}));
}

startAll.addEventListener("click", () => {
  if (ws.readyState !== WebSocket.OPEN) return;
  const requests = [];
  for (const state of states.values()) {
    if (!ready(state)) continue;
    requests.push(requestPayload(state));
    beginLocal(state);
  }
  if (requests.length) {
    ws.send(JSON.stringify({type: "submit_many", requests}));
  }
});

function formatLatency(value) {
  if (value == null) return "--";
  return `${(value * 1000).toFixed(value < 0.1 ? 1 : 0)} ms`;
}

function formatThroughput(value) {
  if (value == null) return "--";
  return `${value.toFixed(2)} tok/s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const mib = bytes / (1024 ** 2);
  if (mib < 1024) return `${mib.toFixed(2)} MB`;
  return `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
}

function formatMemoryPart(bytes, total) {
  if (!total) return "--";
  const percent = 100 * bytes / total;
  return `${formatBytes(bytes)} · ${percent.toFixed(1)}%`;
}

function updateUserStatistics(state, statistics) {
  if (!statistics) return;
  state.statTTFT.textContent = formatLatency(statistics.ttft);
  state.statTPOT.textContent = formatLatency(statistics.tpot);
  state.statTPOTP50.textContent = formatLatency(statistics.tpot_p50); // [MOD]
  state.statTPOTP95.textContent = formatLatency(statistics.tpot_p95); // [MOD]
  state.statTPOTP99.textContent = formatLatency(statistics.tpot_p99); // [MOD]
  state.statE2E.textContent = formatLatency(statistics.e2e_latency);
}

function setSegment(id, bytes, total) {
  const node = document.querySelector(id);
  const percent = total > 0 ? Math.max(0, 100 * bytes / total) : 0;
  node.style.width = `${percent}%`;
}

function ensureBlockGrid(numBlocks) {
  if (blockNodes.length === numBlocks) return;
  blockGrid.replaceChildren();
  blockNodes = new Array(numBlocks);
  activeBlocks = new Set();
  cachedBlocks = new Set(); // [MOD]

  const fragment = document.createDocumentFragment();
  for (let blockId = 0; blockId < numBlocks; blockId += 1) {
    const node = document.createElement("div");
    node.className = "kv-block empty"; // [MOD] 默认状态明确为 empty。
    node.title = `block ${blockId} · empty`;
    blockNodes[blockId] = node;
    fragment.appendChild(node);
  }
  blockGrid.appendChild(fragment);
}

// [MOD] 一个 block 只属于 empty / cached / active 三种状态之一。
function paintBlock(blockId, activeSet, cachedSet) {
  const node = blockNodes[blockId];
  if (!node) return;

  node.classList.remove("empty", "cached", "active");
  if (activeSet.has(blockId)) {
    node.classList.add("active");
    node.title = `block ${blockId} · active`;
  } else if (cachedSet.has(blockId)) {
    node.classList.add("cached");
    node.title = `block ${blockId} · cached`;
  } else {
    node.classList.add("empty");
    node.title = `block ${blockId} · empty`;
  }
}

function updateBlockGrid(numBlocks, activeBlockIds, cachedBlockIds) {
  ensureBlockGrid(numBlocks);
  const nextActive = new Set(activeBlockIds ?? []);
  const nextCached = new Set(cachedBlockIds ?? []);

  // [MOD] 只重绘状态发生过变化的 block，避免每次刷新扫描全部 DOM。
  const touched = new Set([
    ...activeBlocks,
    ...cachedBlocks,
    ...nextActive,
    ...nextCached,
  ]);
  for (const blockId of touched) {
    paintBlock(blockId, nextActive, nextCached);
  }

  activeBlocks = nextActive;
  cachedBlocks = nextCached;
}

function updateResource(resource) {
  if (!resource) return;
  const total = resource.total_bytes ?? 0;
  const unused = resource.unused_bytes ?? 0;
  const reserved = resource.reserved_bytes ?? 0;
  const allocated = resource.allocated_kv_bytes ?? 0;
  const available = resource.available_kv_bytes ?? 0;

  document.querySelector("#mem-unused").textContent = formatMemoryPart(unused, total);
  document.querySelector("#mem-reserved").textContent = formatMemoryPart(reserved, total);
  document.querySelector("#mem-allocated").textContent = formatMemoryPart(allocated, total);
  document.querySelector("#mem-available").textContent = formatMemoryPart(available, total);

  setSegment("#segment-unused", unused, total);
  setSegment("#segment-reserved", reserved, total);
  setSegment("#segment-allocated", allocated, total);
  setSegment("#segment-available", available, total);

  const numBlocks = resource.num_blocks ?? 0;
  const activeIds = resource.active_block_ids ?? [];
  const cachedIds = resource.cached_block_ids ?? []; // [MOD]
  document.querySelector("#kv-summary").textContent =
    `${activeIds.length} active · ${cachedIds.length} cached · ${numBlocks} total · ${formatBytes(resource.block_bytes ?? 0)} / block`;
  updateBlockGrid(numBlocks, activeIds, cachedIds); // [MOD]
}

function initPerformanceGrid() {
  const fragment = document.createDocumentFragment();
  for (const [key, label] of PERFORMANCE_METRICS) {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.dataset.metric = key;
    card.innerHTML = `<span>${label}</span><strong>--</strong>`;
    fragment.appendChild(card);
  }
  performanceGrid.appendChild(fragment);
}

function updatePerformance(performance) {
  if (!performance) return;
  const count = performance.completed_requests ?? 0;
  document.querySelector("#completed-requests").textContent = `${count} completed`;
  const averages = performance.averages ?? {};

  for (const [key, , kind] of PERFORMANCE_METRICS) {
    const card = performanceGrid.querySelector(`[data-metric="${key}"] strong`);
    const value = averages[key];
    card.textContent = kind === "throughput"
      ? formatThroughput(value)
      : formatLatency(value);
  }

  // [MOD] 全局 TPOT 分位数是所有已完成请求的 request-level TPOT 分布。
  const percentiles = performance.tpot_percentiles ?? {};
  for (const key of ["p50", "p95", "p99"]) {
    const node = document.querySelector(`[data-tpot-percentile="${key}"]`);
    node.textContent = formatLatency(percentiles[key]);
  }
}

ws.addEventListener("message", (message) => {
  const event = JSON.parse(message.data);

  if (event.type === "runtime") {
    updateResource(event.resource);
    updatePerformance(event.performance);
    return;
  }

  if (event.type === "accepted") {
    const accepted = new Set(event.accepted_users);
    for (const userId of event.requested_users) {
      if (!accepted.has(userId)) {
        const state = states.get(userId);
        if (state) setBusy(state, false);
      }
    }
    return;
  }

  const state = states.get(event.user_id);
  if (!state) return;

  if (event.type === "token") {
    state.output.textContent = state.outputBase + (event.text ?? "");
    state.output.scrollTop = state.output.scrollHeight;
    return;
  }

  if (event.type === "finished") {
    state.output.textContent = state.outputBase + (event.text ?? "") + "\n\n";
    state.output.scrollTop = state.output.scrollHeight;
    updateUserStatistics(state, event.statistics);
    state.input.value = "";
    setBusy(state, false);
    return;
  }

  if (event.type === "error") {
    state.output.textContent = state.outputBase + `[error] ${event.message}\n\n`;
    state.output.scrollTop = state.output.scrollHeight;
    setBusy(state, false);
  }
});

ws.addEventListener("open", refreshControls);
ws.addEventListener("close", () => {
  for (const state of states.values()) state.button.disabled = true;
  startAll.disabled = true;
});

USERS.forEach(makePanel);
initPerformanceGrid();
refreshControls();
```

---

# 7. 检查与启动

本次没有修改 `tinyinfer/ui/app.py`、`examples/advance02_ui_plus.py`、Scheduler 或模型执行路径，因此仍可使用当前 Advance 02 的启动入口。

先做语法检查：

```bash
python -m compileall -q tinyinfer
node --check tinyinfer/ui/static/app.js
```

然后启动：

```bash
python examples/advance02_ui_plus.py \
  --model /home/jason/huggingface/Qwen3-0.6B
```

浏览器继续访问：

```text
http://127.0.0.1:8000/
```

预期现象：

```text
左侧：
User1
User2
User3
User4
User5
↑
独立滚动

右侧：
Statistics 固定
├── Resource statistics
│   └── KV block map
│       ├── green  = empty
│       ├── blue   = cached
│       └── orange = active
└── Performance statistics
    ├── 原有平均指标
    └── TPOT p50 / p95 / p99
```

每个 User 的 `Last request` 会显示：

```text
TTFT
TPOT mean
TPOT p50
TPOT p95
TPOT p99
E2E
```

## 7.1 两种 TPOT percentile 的含义

单 User 的 `TPOT p50/p95/p99` 来自该请求内部：

```text
token1 --ITL1--> token2 --ITL2--> token3 ... 
```

因此它能看出这一条请求 decode 是否稳定。

全局 Performance Statistics 的 TPOT percentile 则来自：

```text
request1.tpot
request2.tpot
...
requestN.tpot
```

它适合后续做大规模 workload 时观察跨请求的尾延迟。

## 7.2 KV 三态的生命周期

一个 block 最典型的状态变化是：

```text
empty
  ↓ 首次分配
active
  ↓ 完整 block 被注册为 prefix cache，随后请求结束
cached
  ↓ 后续 prefix cache hit
active
  ↓ 被释放
cached
  ↓ LRU 再分配并 eviction/reset
active
```

如果一个 partial block 尚未形成 hash，请求结束后它会落回：

```text
empty
```

因此三色图现在能比原来的“active / available”二色图更准确地展示 Prefix Cache 的生命周期。

---

# 8. 本讲修改边界

本讲只扩展 observability 与 UI，不改变：

```text
Scheduler policy
BlockManager allocation / eviction policy
Paged KV physical layout
DynamicEngineService worker loop
ModelRunner execution
WebSocket request protocol
```

因此如果 Advance 02 当前已经能正常推理，这一讲的修改不会改变模型输出路径；它只让你更清楚地看到 KV cache 状态和 TPOT 尾延迟。
