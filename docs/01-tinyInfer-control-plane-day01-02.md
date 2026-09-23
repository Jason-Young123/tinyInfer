# tinyInfer 手把手教程（上）：Day 01–02 —— 从空目录搭出控制面、Sequence 与 Continuous Batching

> 本教程不是“读 nano-vLLM 源码的旁注”，而是要求你像搭 tinyIR 一样，从空目录开始把一个小型 LLM inference runtime 一层层造出来。最终工程名固定为 `tinyInfer`。
>
> 参考对象：GeeeekExplorer/nano-vllm。我们复现它的**组件边界、数据流和关键系统思想**，但不做逐行抄写；先写最小正确版本，再逐步替换成接近真实推理框架的实现。
>
> 本篇覆盖原 9-day 计划中的 Day 01–02。Day 03–06 见后两篇；Day 07–09 的实践内容保持原讲义不变。

---

# 0. 这一次为什么一定要“从零搭”

你已经体验到：直接 clone nano-vLLM 后读 `Scheduler`、`BlockManager`、`ModelRunner`，很容易知道“每一行大概在做什么”，却不容易真正形成下面这种因果关系：

```text
为什么一定要有 Sequence？
    ↓
因为 request 的生命周期跨越很多 decode iteration

为什么 Scheduler 不能只返回一批 request？
    ↓
因为每轮可执行 token 数、KV 可用量、prefill/decode 状态都不同

为什么 ModelRunner 要接收 metadata，而不是字符串？
    ↓
因为 GPU 只认识 tensor；runtime 必须把 request state 编译成 tensor metadata
```

因此 tinyInfer 采用和 tinyIR 一样的教学方式：

```text
先提出一个最小问题
    ↓
只新增解决这个问题所需的文件
    ↓
写最小代码
    ↓
立刻做临时测试
    ↓
确认心智模型
    ↓
再进入下一层
```

最终你不是“复刻一个仓库”，而是能解释为什么这个仓库必须长成这样。

---

# 1. 最终目录树先看一眼，但今天只建一小部分

最终 tinyInfer 会长成：

```text
tinyInfer/
├── pyproject.toml
├── README.md
├── examples/
│   └── basic_generate.py
├── benchmarks/
│   ├── throughput.py
│   └── latency.py
├── tests/
│   ├── test_sequence.py
│   ├── test_scheduler.py
│   ├── test_block_manager.py
│   ├── test_attention_meta.py
│   └── test_sampler.py
└── tinyinfer/
    ├── __init__.py
    ├── config.py
    ├── sampling_params.py
    ├── llm.py
    ├── engine/
    │   ├── sequence.py
    │   ├── scheduler.py
    │   ├── block_manager.py
    │   ├── model_runner.py
    │   └── llm_engine.py
    ├── layers/
    │   ├── activation.py
    │   ├── attention.py
    │   ├── embed_head.py
    │   ├── layernorm.py
    │   ├── linear.py
    │   ├── rotary_embedding.py
    │   └── sampler.py
    ├── models/
    │   └── qwen3.py
    └── utils/
        ├── context.py
        └── loader.py
```

**今天不要一次性创建全部文件。** Day 01 只需要公共 API + 最小 Engine；Day 02 才加入 Sequence/Scheduler。

---

# Day 01：从空目录到第一条 `LLM.generate()` 调用链

# 2. Step 0：创建空仓库

```bash
cd ~/Learning-Inference
mkdir tinyInfer
cd tinyInfer

git init
mkdir -p tinyinfer/engine examples tests notes

touch README.md
```

此时：

```text
tinyInfer/
├── README.md
├── examples/
├── notes/
├── tests/
└── tinyinfer/
    └── engine/
```

先提交：

```bash
git add .
git commit -m "init: create tinyInfer skeleton"
```

这一 commit 很重要：以后所有推理组件都有清晰演化轨迹。

---

# 3. Step 1：先解决“Python 怎么认识这个工程”

创建：

```text
pyproject.toml
```

内容：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tinyinfer"
version = "0.1.0"
description = "A tiny LLM inference runtime built step by step"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4",
    "transformers>=4.51",
    "safetensors",
    "xxhash",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["tinyinfer*"]
```

创建：

```text
tinyinfer/__init__.py
```

暂时只写：

```python
__version__ = "0.1.0"
```

你当前机器不想使用 venv，因此沿用你已经确定的策略：

```bash
python -m pip install --user -U setuptools
python -m pip install --user -e .
```

如果当前 setuptools/build backend 仍然不支持 editable install，不要阻塞学习，直接：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

并把：

```bash
export PYTHONPATH=$HOME/Learning-Inference/tinyInfer:$PYTHONPATH
```

加入 `~/.bashrc`。

## 临时测试 1

```bash
python - <<'PY'
import tinyinfer
print(tinyinfer.__file__)
PY
```

目标：只验证 package import，不碰 GPU。

---

# 4. Step 2：先定义 SamplingParams —— 为什么它应该独立存在？

一次推理请求有两类输入：

```text
模型输入：prompt/token ids
生成策略：temperature / max_tokens / stop 条件
```

如果把它们混在一起，Engine 后面会很难区分：

```text
“这个 token 是模型状态”
还是
“这个字段只是生成策略”
```

创建：

```text
tinyinfer/sampling_params.py
```

```python
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature <= 0.0:
            raise ValueError(
                "tinyInfer Day01 sampler only supports temperature > 0; "
                "greedy mode will be added later explicitly."
            )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
```

为什么先禁止 `temperature=0`？

不是说 greedy decoding 不重要，而是为了把两种语义明确分开：

```text
temperature sampling
    logits / T
    → stochastic sample

greedy decoding
    logits
    → argmax
```

后面 Day 08/09 你会再专门实现 logit-space sampling 与 greedy speculative decoding。

## 临时测试 2

创建：

```text
tests/test_sampling_params.py
```

```python
import pytest
from tinyinfer import SamplingParams


def test_sampling_params():
    p = SamplingParams(temperature=0.6, max_tokens=8)
    assert p.temperature == 0.6
    assert p.max_tokens == 8


def test_reject_zero_temperature():
    with pytest.raises(ValueError):
        SamplingParams(temperature=0.0)
```

先在 `__init__.py` 导出：

```python
from tinyinfer.sampling_params import SamplingParams

__all__ = ["SamplingParams"]
```

运行：

```bash
python -m pip install --user pytest
pytest -q tests/test_sampling_params.py
```

---

# 5. Step 3：定义 Config —— 推理系统为什么需要自己的配置，而不是直接用 HF config？

Hugging Face `config.json` 描述的是**模型结构**：

```text
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
...
```

推理 runtime 还需要另一套配置：

```text
max_num_seqs
max_num_batched_tokens
KV block size
GPU memory utilization
TP size
是否强制 eager
```

两者不是一回事。

创建：

```text
tinyinfer/config.py
```

第一版先允许 `model=None`，这样 Day01/02 可以用 ToyRunner 在 CPU 上验证控制面：

```python
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Config:
    model: str | None = None
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 64
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
```

现在 Config 只是“runtime contract”。Day 05 再让它加载 `AutoConfig`。

---

# 6. Step 4：先不要写真正模型，造一个 ToyModelRunner

这是整个教程非常重要的教学选择。

如果你现在直接碰：

```text
Qwen3
FlashAttention
KV cache
CUDA Graph
TP
```

一旦输出错了，你根本不知道是：

```text
Engine bug？
Scheduler bug？
模型 bug？
attention metadata bug？
CUDA bug？
```

因此先造一个“永远能产生下一个 token”的假执行器。

创建：

```text
tinyinfer/engine/model_runner.py
```

Day01 先写最小版本：

```python
class ToyModelRunner:
    """Control-plane-only runner used in Day01/02.

    It does not run a neural network. For every sequence it simply emits
    `(last_token + 1) % 1000` so that the engine loop can be tested.
    """

    def run(self, sequences, is_prefill: bool):
        next_tokens = []
        for seq in sequences:
            next_tokens.append((seq.last_token + 1) % 1000)
        return next_tokens
```

注意：这里已经故意保留：

```python
run(sequences, is_prefill)
```

而不是：

```python
run(input_ids)
```

因为后面真实 `ModelRunner` 的工作就是：

```text
Sequence objects
    ↓ prepare inputs
GPU tensors + attention metadata
    ↓ model forward
next token ids
```

ToyRunner 是接口占位，而不是随便写的 demo。

---

# 7. Step 5：定义 Sequence —— request 的“运行时实体”

创建：

```text
tinyinfer/engine/sequence.py
```

```python
from enum import Enum, auto
from itertools import count

from tinyinfer.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()
    block_size = 256

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.sampling_params = sampling_params

        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True

        self.block_table: list[int] = []

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int):
        self.token_ids.append(int(token_id))

    def should_stop(self, eos_token_id: int) -> bool:
        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            return True
        if (
            not self.sampling_params.ignore_eos
            and eos_token_id >= 0
            and self.last_token == eos_token_id
        ):
            return True
        return False
```

## 这里先停下来理解 `seq_id`

`Sequence.counter` 是 class-level 计数器：

```text
Sequence(...) → id 0
Sequence(...) → id 1
Sequence(...) → id 2
```

它表示：

```text
“这是第几个被创建的 Sequence 对象”
```

不是：

```text
prompt 在 Python list 里的下标
```

这也解释了你之前在 nano-vLLM 中看到真实 request id 从 4 开始：`ModelRunner` warmup 提前创建了 dummy `Sequence`，把 0~3 消耗掉；你手工 `next(Sequence.counter)` 后又额外消耗一个，于是下一次从 5 开始。到了 Day05，我们会在自己的 tinyInfer 中**主动复现这个现象**，这样你会真正理解它，而不是靠读日志猜。

## 临时测试 3

```text
tests/test_sequence.py
```

```python
from tinyinfer.engine.sequence import Sequence, SequenceStatus
from tinyinfer import SamplingParams


def test_sequence_lifecycle():
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=2))
    assert seq.status is SequenceStatus.WAITING
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 0

    seq.append_token(13)
    assert seq.num_completion_tokens == 1
    assert not seq.should_stop(-1)

    seq.append_token(14)
    assert seq.should_stop(-1)
```

运行：

```bash
pytest -q tests/test_sequence.py
```

---

# 8. Step 6：第一次写 Scheduler，但只做最小状态机

创建：

```text
tinyinfer/engine/scheduler.py
```

Day01 先不碰 KV cache，只实现：

```text
waiting → running → finished
```

```python
from collections import deque

from tinyinfer.config import Config
from tinyinfer.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_id = config.eos_token_id

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

        if not self.running:
            return [], False

        # Day01 先把所有新加入的 request 视为 prefill。
        is_prefill = any(seq.is_prefill for seq in self.running)
        return list(self.running), is_prefill

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        finished = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            seq.is_prefill = False

            if seq.should_stop(self.eos_token_id):
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)

        if finished:
            done_ids = {seq.seq_id for seq in finished}
            self.running = deque(
                seq for seq in self.running if seq.seq_id not in done_ids
            )
```

这个版本还不是高性能 scheduler，但已经有真正的 runtime 状态机。

---

# 9. Step 7：写 LLMEngine —— 整个框架的“循环总控”

创建：

```text
tinyinfer/engine/llm_engine.py
```

先不用 tokenizer；为了让 Day01 可独立测试，Engine 接收 token ids：

```python
from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ToyModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence
from tinyinfer.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.validate_runtime_fields()
        Sequence.block_size = self.config.kvcache_block_size

        self.scheduler = Scheduler(self.config)
        self.model_runner = ToyModelRunner()

    def add_request(self, token_ids: list[int], params: SamplingParams):
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []

        next_tokens = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, next_tokens)
        return seqs

    def generate_token_ids(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        if isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompts)
        else:
            params = sampling_params

        if len(params) != len(prompts):
            raise ValueError("prompts and sampling params length mismatch")

        seqs = []
        for token_ids, p in zip(prompts, params):
            seq = Sequence(token_ids, p)
            seqs.append(seq)
            self.scheduler.add(seq)

        while not self.scheduler.is_finished():
            self.step()

        seqs.sort(key=lambda x: x.seq_id)
        return [
            {
                "token_ids": seq.token_ids[seq.num_prompt_tokens:],
                "all_token_ids": seq.token_ids,
            }
            for seq in seqs
        ]
```

现在最重要的调用链已经出现：

```text
generate_token_ids
    ↓
Sequence
    ↓
Scheduler.add
    ↓
while not finished
    ↓
Scheduler.schedule
    ↓
ModelRunner.run
    ↓
Scheduler.postprocess
```

这就是 nano-vLLM/vLLM runtime 最核心的控制环。

---

# 10. Step 8：对外只暴露 LLM

创建：

```text
tinyinfer/llm.py
```

```python
from tinyinfer.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """Public user-facing API."""

    pass
```

修改：

```text
tinyinfer/__init__.py
```

```python
from tinyinfer.llm import LLM
from tinyinfer.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]
```

## 临时测试 4：第一条完整调用链

创建：

```text
examples/day01_toy_generate.py
```

```python
from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config


llm = LLM(Config(max_num_seqs=2))

outputs = llm.generate_token_ids(
    [[10, 20], [100]],
    SamplingParams(temperature=1.0, max_tokens=3, ignore_eos=True),
)

for i, out in enumerate(outputs):
    print(i, out)
```

运行：

```bash
python examples/day01_toy_generate.py
```

ToyRunner 的规则是“最后一个 token + 1”，所以应该看到类似：

```text
0 ... 21,22,23
1 ... 101,102,103
```

今天你还没有模型，但已经拥有真正的 inference engine loop。

---

# 11. Day01 验收：你必须能自己画出这张图

```text
User API
  LLM
   │
   ▼
LLMEngine.generate_token_ids
   │
   ├── create Sequence
   │
   ├── Scheduler.add
   │
   └── loop
       ├── Scheduler.schedule
       ├── ModelRunner.run
       └── Scheduler.postprocess
```

如果这张图你画不出来，不进入 Day02。

建议 commit：

```bash
git add .
git commit -m "day01: build minimal inference engine loop"
```

---

# Day 02：把 Scheduler 真的变成 Continuous Batching Scheduler

# 12. 今天先问一个问题：为什么 Day01 的 Scheduler 还不够？

Day01 的：

```python
return list(self.running), is_prefill
```

相当粗暴。

真实场景：

```text
A: prompt 800 tokens, output 30
B: prompt 20 tokens,  output 3
C: prompt 300 tokens, output 12
D: prompt 40 tokens,  output 5
E: prompt 500 tokens, output 20
```

如果一次最多 4 seq：

```text
round 0: A B C D
B 先结束
round ?: A C D E
D 结束
round ?: A C E
...
```

这才是 continuous batching：**batch 成员在 iteration 边界动态变化**。

---

# 13. Step 1：先给 Sequence 补“本轮调度 token 数”

`Sequence` 已经有：

```python
num_cached_tokens
num_scheduled_tokens
```

现在明确语义：

```text
num_cached_tokens
    = 已经存在 KV cache、无需本轮重新计算的 prefix token 数

num_scheduled_tokens
    = scheduler 决定本轮要真正送进模型的 token 数
```

Day02 暂时不实现真实 KV，只先使用这两个字段表达 prefill/decode 差异。

在 `sequence.py` 增加：

```python
@property
def num_uncached_tokens(self) -> int:
    return self.num_tokens - self.num_cached_tokens
```

以及：

```python
def mark_scheduled(self, n: int):
    if n <= 0:
        raise ValueError("scheduled token count must be positive")
    self.num_scheduled_tokens = n
```

---

# 14. Step 2：明确 Prefill 和 Decode 的调度单位

Prefill：

```text
prompt = 800 tokens
本轮可能一次调度 800 个 token
```

Decode：

```text
已有历史 token
本轮只需要计算新的最后 1 token
```

因此调度预算不是：

```text
max_num_seqs
```

一个约束，而是至少两个：

```text
max_num_seqs
max_num_batched_tokens
```

这就是为什么真实推理 scheduler 不是简单 queue。

---

# 15. Step 3：重写 `Scheduler.schedule()`

现在先实现一个“prefill 优先 + decode 继续”的 baseline。

替换 `schedule()`：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled: list[Sequence] = []
    token_budget = self.max_num_batched_tokens

    # ------------------------------------------------------------
    # Phase A: admit waiting requests for prefill
    # ------------------------------------------------------------
    while self.waiting and len(self.running) < self.max_num_seqs:
        seq = self.waiting[0]
        prefill_tokens = seq.num_tokens - seq.num_cached_tokens

        if prefill_tokens > token_budget:
            break

        self.waiting.popleft()
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = True
        seq.mark_scheduled(prefill_tokens)
        self.running.append(seq)
        scheduled.append(seq)
        token_budget -= prefill_tokens

    if scheduled:
        return scheduled, True

    # ------------------------------------------------------------
    # Phase B: one-token decode for running requests
    # ------------------------------------------------------------
    for seq in list(self.running):
        if token_budget <= 0:
            break
        seq.is_prefill = False
        seq.mark_scheduled(1)
        scheduled.append(seq)
        token_budget -= 1

    return scheduled, False
```

先不要纠结这个策略是否最优。它最重要的教学价值是：

```text
prefill 与 decode 被显式拆成两种 schedule path
```

Day07 才做 chunked prefill / fairness 优化。

---

# 16. Step 4：修改 `postprocess()` —— prefill 和 decode 的结果语义不同

ToyRunner 当前每个 scheduled sequence 都返回一个 token，但真实 prefill 的模型 forward 只需要：

```text
最后一个 prompt position 的 logits
→ sample 第一个 generated token
```

所以 Day02 可以仍然保持“一轮产生一个新 token”，但要在状态上记录：

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    finished = []

    for seq, token_id in zip(seqs, token_ids):
        # prefill/decode 本轮都确认了原有 token 对应 KV 已经计算完成
        seq.num_cached_tokens = seq.num_tokens

        # sampled token 加入 sequence；这个新 token 的 KV 要等下一轮 decode
        seq.append_token(token_id)
        seq.num_scheduled_tokens = 0
        seq.is_prefill = False

        if seq.should_stop(self.eos_token_id):
            seq.status = SequenceStatus.FINISHED
            finished.append(seq)

    done_ids = {seq.seq_id for seq in finished}
    if done_ids:
        self.running = deque(
            seq for seq in self.running if seq.seq_id not in done_ids
        )
```

注意这个细节：

```text
prefill 完成 prompt KV
↓
sample 出 token X
↓
X 已经成为 sequence 的 token
但 X 自己的 KV 还没写入
↓
下一轮 decode 计算 X 的 K/V
```

这是后面理解 decode slot mapping 的关键。

---

# 17. Step 5：加入可控 trace，而不是到处 print

创建：

```text
tinyinfer/utils.py
```

```python
import os


def trace_enabled(name: str) -> bool:
    return os.getenv(name, "0") == "1"
```

在 `scheduler.py`：

```python
from tinyinfer.utils import trace_enabled
```

在 `schedule()` 开头：

```python
if trace_enabled("TINYINFER_TRACE_SCHED"):
    print(
        "[sched]",
        "waiting=", len(self.waiting),
        "waiting_ids=", [s.seq_id for s in self.waiting],
        "running=", len(self.running),
        "running_ids=", [s.seq_id for s in self.running],
    )
```

这比永久 print 好，因为后面 benchmark 不会被 I/O 污染。

---

# 18. Step 6：构造和你之前完全相同的混合 workload

创建：

```text
examples/day02_scheduler_trace.py
```

先用 token list，不依赖 tokenizer：

```python
from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config


prompts = [
    [1] * 800,
    [2] * 20,
    [3] * 300,
    [4] * 40,
    [5] * 500,
]

params = [
    SamplingParams(temperature=1.0, max_tokens=n, ignore_eos=True)
    for n in [30, 3, 12, 5, 20]
]

llm = LLM(
    Config(
        max_num_seqs=4,
        max_num_batched_tokens=4096,
    )
)

out = llm.generate_token_ids(prompts, params)

for i, x in enumerate(out):
    print(i, len(x["token_ids"]))
```

运行：

```bash
TINYINFER_TRACE_SCHED=1 \
python examples/day02_scheduler_trace.py \
2>&1 | tee notes/day02-scheduler.log
```

你应该观察到类似的状态变化：

```text
waiting=[0,1,2,3,4] running=[]
...
某些短请求先完成
...
running 集合逐渐缩小
```

最后必须：

```text
0 30
1 3
2 12
3 5
4 20
```

---

# 19. Step 7：为什么这里可能不会立刻出现“第 5 个请求补进来”？

因为我们的 baseline 有一个有意保留的限制：

```text
prefill phase 和 decode phase 分开
```

如果 running 中已有 decode request，而 waiting 中新 request 需要 prefill，当前 baseline 不一定混合：

```text
prefill + decode in same iteration
```

这正是后面 Day07 “mixed/chunked prefill + fairness” 的优化空间。

不要一开始就把最复杂策略写进去。否则你不会知道优化到底改了什么。

---

# 20. Step 8：给 Scheduler 写真正的单元测试

创建：

```text
tests/test_scheduler.py
```

```python
from tinyinfer import SamplingParams
from tinyinfer.config import Config
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence, SequenceStatus


def make_seq(n_prompt, n_out):
    return Sequence(
        [1] * n_prompt,
        SamplingParams(temperature=1.0, max_tokens=n_out, ignore_eos=True),
    )


def test_max_num_seqs():
    sched = Scheduler(Config(max_num_seqs=2, max_num_batched_tokens=1000))
    seqs = [make_seq(10, 2) for _ in range(3)]
    for s in seqs:
        sched.add(s)

    picked, is_prefill = sched.schedule()
    assert is_prefill
    assert len(picked) == 2
    assert len(sched.running) == 2
    assert len(sched.waiting) == 1
    assert all(s.status is SequenceStatus.RUNNING for s in picked)


def test_token_budget():
    sched = Scheduler(Config(max_num_seqs=8, max_num_batched_tokens=25))
    sched.add(make_seq(20, 2))
    sched.add(make_seq(20, 2))

    picked, _ = sched.schedule()
    assert len(picked) == 1
```

运行：

```bash
pytest -q tests/test_scheduler.py
```

---

# 21. Step 9：这时再对照 nano-vLLM 源码

到现在你再打开 nano-vLLM：

```text
nanovllm/engine/sequence.py
nanovllm/engine/scheduler.py
nanovllm/engine/llm_engine.py
```

你会发现你不再是在问：

```text
“这行代码是什么意思？”
```

而是在问：

```text
“它对我这个最小版本做了哪些系统级增强？”
```

重点比较：

```text
Sequence:
- token lifecycle
- cached/scheduled token accounting
- block_table

Scheduler:
- KV block availability
- allocation/deallocation
- prefix cache
- prefill/decode token budget

LLMEngine:
- tokenizer
- TP workers
- output reorder/decode
```

这才是“从零搭”的价值。

---

# 22. Day02 结束时的目录树

```text
tinyInfer/
├── pyproject.toml
├── README.md
├── examples/
│   ├── day01_toy_generate.py
│   └── day02_scheduler_trace.py
├── notes/
│   └── day02-scheduler.log
├── tests/
│   ├── test_sampling_params.py
│   ├── test_scheduler.py
│   └── test_sequence.py
└── tinyinfer/
    ├── __init__.py
    ├── config.py
    ├── llm.py
    ├── sampling_params.py
    ├── utils.py
    └── engine/
        ├── llm_engine.py
        ├── model_runner.py       # 目前仍是 ToyModelRunner
        ├── scheduler.py
        └── sequence.py
```

建议 commit：

```bash
git add .
git commit -m "day02: implement sequence lifecycle and continuous batching baseline"
```

---

# 23. 本篇必须真正掌握的 8 个结论

1. `Sequence` 不是 prompt 的包装器，而是跨 iteration 存活的 request runtime state。
2. `seq_id` 标识 Sequence 实例，不等价于输入 list 下标。
3. continuous batching 的“continuous”发生在 iteration 边界，而不是只在请求到达时。
4. prefill 一次可能消费很多 token budget；decode 通常每 seq 每轮只消费 1 token。
5. `max_num_seqs` 和 `max_num_batched_tokens` 约束不同维度。
6. Scheduler 是控制面，不做神经网络计算。
7. ModelRunner 是控制面到 GPU execution 的边界；今天用 ToyRunner 是为了先把接口固定下来。
8. 先把 baseline 做简单，后续的 chunked prefill/fairness 才有可比较对象。

下一篇开始，我们会把最关键的一块接进来：**Paged KV Cache + Prefix Cache + slot mapping**。从那一刻起，Scheduler 不再只是 queue，而会真正被 GPU memory 约束。
