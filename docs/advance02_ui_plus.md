# tinyInfer Advance 02：User 控制器 + 实时资源监控 + 性能统计 UI

> 前置：已经完成 `docs/advance01_ui.md`，能够用 FastAPI + WebSocket 跑起三终端动态推理。
>
> 本讲继续沿用原有 `Scheduler / BlockManager / ModelRunner`，不重写调度算法；主要新增三类能力：每个 User 独立采样参数与上一轮延迟统计、全局 GPU/KV block 资源可视化、所有已完成请求的平均性能统计。

---

# 0. 最终效果与目录变化

每个 User Panel 变成：

```text
User N
├── Output：流式输出
└── Bottom
    ├── Input + Start
    └── User tools
        ├── Controller
        │   ├── max_tokens: 32~1024，默认 128
        │   ├── temperature: 0.1~2.0，默认 0.8
        │   └── greedy: on/off，默认 on
        └── Last request statistics
            ├── TTFT
            ├── TPOT
            └── E2E Latency
```

右侧 Statistics Panel 变成：

```text
Statistics
├── Resource statistics
│   ├── 4-part GPU memory bar
│   │   ├── unused
│   │   ├── reserved
│   │   ├── allocated
│   │   └── available
│   └── KV block map
│       └── 每个小格严格对应一个 physical block
└── Performance statistics
    └── 所有已完成 seq 的 9 项指标平均值
```

这一讲只需要修改/新增：

```text
tinyinfer/
├── engine/
│   ├── model_runner.py                # 建立不可变 KVMemoryProfile 资源画像
│   └── llm_engine.py                  # 新增资源快照
├── runtime/
│   └── dynamic_engine.py              # 聚合统计 + 4Hz runtime event
└── ui/
    ├── app.py                         # 每请求独立 SamplingParams
    └── static/
        ├── index.html                 # Statistics / Controller 布局
        ├── style.css                  # 新 UI 样式
        └── app.js                     # slider、switch、实时统计刷新
examples/
└── advance02_ui_plus.py               # 新启动入口
```

Resource Statistics 统一采用同一套预算口径，四部分严格相加等于 GPU 总显存：

- `Unused`：`TOTAL_MEM * (1 - gpu_memory_utilization)`，策略上完全不使用的预留区域，灰色。
- `Reserved`：模型加载完成、KV pool 创建之前 `torch.cuda.memory_reserved()` 的值，包含 weights 在内的 PyTorch reserved memory，深橘色。
- `Allocated`：当前 `ref_count > 0` 的 KV blocks 总容量，浅橘色。
- `Available`：KV 预算中尚未被 active block 占用的容量，绿色。

因此 `Available = TOTAL_MEM - Unused - Reserved - Allocated`。当前 `PagedKVCache` 底层 tensor 仍会一次性预分配完整 KV pool，所以这里的 `Allocated/Available` 表示 **Paged KV allocator 的逻辑使用状态**，不是 CUDA driver 层面动态申请/释放的物理显存。

---

# 1. Step 1：统一 GPU 资源统计口径

## 1.1 原理

KV block 数本来就是在模型加载完成后通过下面这一步估算的：

```python
total = torch.cuda.get_device_properties(device).total_memory
budget = int(total * config.gpu_memory_utilization)
reserved = torch.cuda.memory_reserved(device)
available_for_kv = max(0, budget - reserved)
```

因此更标准的工程做法不是把运行时测量值塞回 `Config`，而是在 `ModelRunner` 初始化阶段创建一个只读的 `KVMemoryProfile`。它保存 KV pool 创建前这一次 profile 得到的 `total / budget / unused / reserved / block_bytes / num_blocks`，后续 UI 只读取这个 runtime profile。`Config` 仍只保存用户配置项。四段定义固定为：

```text
unused    = TOTAL_MEM * (1 - gpu_memory_utilization)
reserved  = KV profile 时的 torch.cuda.memory_reserved()
allocated = active_block_count * bytes_per_kv_block
available = TOTAL_MEM - unused - reserved - allocated
```

四段总和严格等于 `TOTAL_MEM`。KV block map 中 `ref_count > 0` 为 `allocated`，其余 block 为 `available`。

## 1.2 修改 `tinyinfer/engine/model_runner.py`：建立只读 `KVMemoryProfile`

`reserved` 是模型加载完成、KV pool 创建之前的一次运行时测量值，不属于静态配置，因此不要写入 `config.kv_profile_reserved_bytes`。在 `model_runner.py` 顶部加入 `dataclass`，并用一个独立 profile 对象保存这次测量结果。

先把文件开头 import 改为：

```python
from dataclasses import dataclass

import torch

from tinyinfer.utils.context import set_context, reset_context
from tinyinfer.engine.sequence import SchedulerOutput
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.layers.sampler import Sampler
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.loader import load_weights
```

然后把原来的 `estimate_num_kv_blocks()` 整段替换为：

```python
@dataclass(frozen=True, slots=True)
class KVMemoryProfile:
    total_bytes: int
    budget_bytes: int
    unused_bytes: int
    reserved_bytes: int
    block_bytes: int
    num_blocks: int


def build_kv_memory_profile(
    config,
    device: torch.device,
    dtype: torch.dtype,
) -> KVMemoryProfile:
    if device.type != "cuda":
        raise ValueError("Invalid device: cpu; Required: cuda")

    # 清掉 allocator 中可释放的历史缓存后再做 KV profile。
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    budget_bytes = int(total_bytes * config.gpu_memory_utilization)
    unused_bytes = total_bytes - budget_bytes
    reserved_bytes = int(torch.cuda.memory_reserved(device))
    block_bytes = bytes_per_kv_block(
        config.hf_config,
        config.kvcache_block_size,
        dtype,
    )

    if block_bytes <= 0:
        raise RuntimeError("invalid KV bytes per block")

    kv_budget_bytes = max(0, budget_bytes - reserved_bytes)
    num_blocks = kv_budget_bytes // block_bytes
    if num_blocks <= 0:
        raise RuntimeError(
            "no memory left for KV cache under gpu_memory_utilization"
        )

    return KVMemoryProfile(
        total_bytes=total_bytes,
        budget_bytes=budget_bytes,
        unused_bytes=unused_bytes,
        reserved_bytes=reserved_bytes,
        block_bytes=int(block_bytes),
        num_blocks=int(num_blocks),
    )
```

最后在 `ModelRunner.__init__()` 的 step3 中，把：

```python
num_blocks = estimate_num_kv_blocks(self.config, self.device, self.dtype)
self.config.num_kvcache_blocks = num_blocks
```

替换为：

```python
self.resource_profile = build_kv_memory_profile(
    self.config,
    self.device,
    self.dtype,
)
num_blocks = self.resource_profile.num_blocks
self.config.num_kvcache_blocks = num_blocks
```

这里 `self.resource_profile` 是一次初始化得到的 immutable runtime snapshot：

```text
Config
└── gpu_memory_utilization / block_size / model ...   # 用户配置

ModelRunner.resource_profile
└── total / reserved / block_bytes / num_blocks ...   # 运行时实测
```

这样配置和运行时测量不会混在一起；后面的 Resource Statistics 也不需要再次调用 `torch.cuda.memory_reserved()`。

## 1.3 完整替换 `tinyinfer/engine/llm_engine.py`

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


# 整体调用链条: LLM -> LLMEngine -> Scheduler -> ModelRunner
class LLMEngine:
    def __init__(self, config: Config | None = None, device: str = "cuda"):
        self.config = config or Config()
        self.config.validate_runtime_fields()
        self.config.load_hf_config()

        Sequence.block_size = self.config.kvcache_block_size

        # ModelRunner 先初始化，因为它会计算实际可用的 KV block 数。
        self.model_runner = ModelRunner(self.config, device=device)
        self.scheduler = Scheduler(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=True,
        )
        self.config.eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else -1
        )
        self.scheduler.eos_token_id = self.config.eos_token_id

    def add_request(self, token_ids: list[int], params: SamplingParams):
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq)
        return seq.seq_id

    def build_chat_prompt(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        if self.tokenizer.chat_template:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **kwargs,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        return text

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
        output = self.scheduler.schedule()
        if not output.items:
            return [], []

        sampled_tokens = self.model_runner.run(output)
        self.scheduler.postprocess(output, sampled_tokens)

        updates = []
        for item in output.items:
            token_id = sampled_tokens.get(item.seq.seq_id)
            if token_id is None:
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

        return output.items, updates

    def step(self):
        items, _ = self.step_with_updates()
        return items

    # 资源快照只读取 CPU metadata，不做 CUDA synchronize。
    def runtime_resource_snapshot(self) -> dict:
        device = self.model_runner.device
        if device.type != "cuda":
            return {
                "total_bytes": 0,
                "unused_bytes": 0,
                "reserved_bytes": 0,
                "allocated_kv_bytes": 0,
                "available_kv_bytes": 0,
                "block_bytes": 0,
                "num_blocks": 0,
                "active_block_ids": [],
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
        allocated_kv_bytes = len(active_block_ids) * block_bytes
        available_kv_bytes = max(
            0,
            total_bytes
            - unused_bytes
            - reserved_bytes
            - allocated_kv_bytes,
        )

        return {
            "total_bytes": total_bytes,
            "unused_bytes": unused_bytes,
            "reserved_bytes": reserved_bytes,
            "allocated_kv_bytes": int(allocated_kv_bytes),
            "available_kv_bytes": int(available_kv_bytes),
            "block_bytes": int(block_bytes),
            "num_blocks": len(blocks),
            "active_block_ids": active_block_ids,
        }

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
        for token_ids, params_item in zip(prompts, params):
            seq = Sequence(token_ids, params_item)
            seqs.append(seq)
            self.scheduler.add(seq)

        while not self.scheduler.is_finished():
            items = self.step()
            if not items and not self.scheduler.is_finished():
                raise RuntimeError(
                    "scheduler made no progress while requests remain"
                )

        seqs.sort(key=lambda x: x.seq_id)
        return [
            {
                "token_ids": seq.token_ids[seq.num_prompt_tokens:],
                "all_token_ids": list(seq.token_ids),
                "num_cached_tokens": seq.num_cached_tokens,
                "statistics": seq.statistics(),
            }
            for seq in seqs
        ]

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        prompt_token_ids = [
            self.tokenizer.encode(prompt, add_special_tokens=False)
            for prompt in prompts
        ]
        outputs = self.generate_token_ids(prompt_token_ids, sampling_params)

        for out in outputs:
            out["text"] = self.tokenizer.decode(
                out["token_ids"],
                skip_special_tokens=True,
            )
        return outputs
```

---

# 2. Step 2：DynamicEngineService 聚合全局统计并定期广播 runtime snapshot

## 2.1 原理

这里新增两件事：

1. 每个 seq 完成时，把 `Sequence.statistics()` 中 9 个指标累加，维护所有历史完成请求的算术平均值。
2. Engine worker 每约 `0.25s` 产生一次 `runtime` event；结束一条请求时立即强制刷新一次。

因此 token streaming 仍然逐 token 走原来的事件通道，而重量更大的资源统计只约 4Hz 更新，不会每个 token 都重建整个 KV block UI。

平均指标为：

```text
queue_time
prefill_time
decode_time
ttft
service_time
e2e_latency
tpot
decode_throughput
request_throughput
```

某个指标如果本条请求为 `None`，例如只生成一个 token 时没有 TPOT，则该请求不进入该指标的分母。

## 2.2 完整替换 `tinyinfer/runtime/dynamic_engine.py`

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
                seq_id = self.engine.add_text_request(
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
            self._metric_sums[key] += float(value)
            self._metric_counts[key] += 1

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

    def _worker_loop(self) -> None:
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

# 3. Step 3：WebSocket 接受每个 User 自己的 SamplingParams

## 3.1 原理

Advance 01 中 `max_tokens / temperature / greedy` 是 `create_app()` 的全局参数；现在它们必须随每个 request 一起从浏览器送到服务器。

单请求的数据变成：

```json
{
  "type": "submit",
  "user_id": "user1",
  "text": "hello",
  "max_tokens": 128,
  "temperature": 0.8,
  "greedy": true
}
```

`Start all ready` 仍然只发送一次 `submit_many`，但其中每条 request 保留各自独立的参数。

为了兼容 Advance 01，`create_app()` 仍保留三个默认值；只有旧客户端没发送参数时才使用它们。

## 3.2 完整替换 `tinyinfer/ui/app.py`

```python
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tinyinfer.runtime.dynamic_engine import (
    DynamicEngineService,
    SubmitRequest,
)
from tinyinfer.sampling_params import SamplingParams


STATIC_DIR = Path(__file__).with_name("static")


class SocketHub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.events: asyncio.Queue[dict] | None = None

    def bind_loop(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.events = asyncio.Queue()

    def publish_from_thread(self, event: dict) -> None:
        if self.loop is None or self.events is None:
            return
        self.loop.call_soon_threadsafe(self.events.put_nowait, event)

    async def broadcast_loop(self) -> None:
        assert self.events is not None
        while True:
            event = await self.events.get()
            dead = []
            for socket in self.clients:
                try:
                    await socket.send_json(event)
                except Exception:
                    dead.append(socket)
            for socket in dead:
                self.clients.discard(socket)


def _parse_sampling_params(
    item: dict,
    default_max_tokens: int,
    default_temperature: float,
    default_greedy: bool,
) -> SamplingParams:
    max_tokens = int(item.get("max_tokens", default_max_tokens))
    temperature = float(item.get("temperature", default_temperature))
    greedy = bool(item.get("greedy", default_greedy))

    if not 32 <= max_tokens <= 1024:
        raise ValueError("max_tokens must be in [32, 1024]")
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature must be in [0.1, 2.0]")

    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        greedy=greedy,
    )


def create_app(
    service: DynamicEngineService,
    max_tokens: int = 128,
    temperature: float = 0.8,
    greedy: bool = True,
) -> FastAPI:
    hub = SocketHub()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind_loop()
        service.start(hub.publish_from_thread)
        broadcaster = asyncio.create_task(hub.broadcast_loop())
        try:
            yield
        finally:
            service.stop()
            broadcaster.cancel()

    app = FastAPI(title="tinyInfer UI", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def websocket_endpoint(socket: WebSocket):
        await socket.accept()
        hub.clients.add(socket)

        snapshot = service.current_runtime_snapshot()
        if snapshot is not None:
            await socket.send_json({"type": "runtime", **snapshot})

        try:
            while True:
                payload = await socket.receive_json()
                request_type = payload.get("type")
                raw_requests = (
                    payload.get("requests", [])
                    if request_type == "submit_many"
                    else [payload]
                )

                requests = []
                requested_users = []
                for item in raw_requests:
                    user_id = str(item.get("user_id", ""))
                    text = str(item.get("text", ""))
                    if not user_id or not text.strip():
                        continue
                    requested_users.append(user_id)

                    try:
                        params = _parse_sampling_params(
                            item,
                            max_tokens,
                            temperature,
                            greedy,
                        )
                    except (TypeError, ValueError) as exc:
                        await socket.send_json(
                            {
                                "type": "error",
                                "user_id": user_id,
                                "message": str(exc),
                            }
                        )
                        continue

                    requests.append(
                        SubmitRequest(
                            user_id=user_id,
                            text=text,
                            params=params,
                        )
                    )

                accepted = service.submit_many(requests)
                await socket.send_json(
                    {
                        "type": "accepted",
                        "accepted_users": accepted,
                        "requested_users": requested_users,
                    }
                )
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(socket)

    return app
```

---

# 4. Step 4：替换前端，加入 User Controller、User Statistics、Resource/Performance Statistics

前端仍然只有原来的三个静态文件，不引入 React/Vue。这样 DOM 很轻，便于理解 WebSocket 数据流。

KV block grid 固定 **48 列**。Qwen3-0.6B + 24GB GPU + `block_size=16` 时 block 数通常会达到数千甚至上万；如果只用 15 列，页面会非常高。48 列仍保证“一个格子严格对应一个 block”，同时右侧面板可滚动。

性能方面，block DOM 只在 block 总数改变时创建一次；之后每个 runtime snapshot 只对“active 集合发生变化的 block”切换 class，而不是每 250ms 全量重绘数千个节点。

## 4.1 完整替换 `tinyinfer/ui/static/index.html`

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
            <div class="kv-key"><span class="swatch allocated"></span>allocated <span class="swatch available"></span>available</div>
          </div>
          <div class="block-grid" id="block-grid"></div>
          <div class="resource-note">Reserved is captured before KV pool creation. Allocated / available show the logical Paged KV block state.</div>
        </section>

        <section class="stats-section performance-section">
          <div class="section-title-row">
            <div class="section-title">Performance statistics</div>
            <div class="request-count" id="completed-requests">0 completed</div>
          </div>
          <div class="performance-grid" id="performance-grid"></div>
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

## 4.2 完整替换 `tinyinfer/ui/static/style.css`

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
  display: grid;
  grid-template-rows: repeat(3, minmax(0, 1fr));
  gap: 14px;
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
  padding: 14px;
  display: grid;
  grid-template-rows: auto minmax(70px, 1fr) auto;
  gap: 10px;
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
  min-height: 84px;
  max-height: 118px;
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
  margin-top: 8px;
  display: grid;
  gap: 7px;
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
.kv-key { display: flex; align-items: center; gap: 5px; }

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
  background: var(--available);
  opacity: .83;
}
.kv-block.active { background: var(--allocated); opacity: 1; }

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

## 4.3 完整替换 `tinyinfer/ui/static/app.js`

```javascript
const USERS = ["user1", "user2", "user3"];
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
            <input class="max-tokens" type="range" min="32" max="1024" step="32" value="128">
            <span class="control-value max-value">128</span>
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
            <div class="user-stat"><span>TPOT</span><strong class="stat-tpot">--</strong></div>
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

  const fragment = document.createDocumentFragment();
  for (let blockId = 0; blockId < numBlocks; blockId += 1) {
    const node = document.createElement("div");
    node.className = "kv-block";
    node.title = `block ${blockId}`;
    blockNodes[blockId] = node;
    fragment.appendChild(node);
  }
  blockGrid.appendChild(fragment);
}

function updateBlockGrid(numBlocks, activeBlockIds) {
  ensureBlockGrid(numBlocks);
  const next = new Set(activeBlockIds ?? []);

  for (const blockId of activeBlocks) {
    if (!next.has(blockId) && blockNodes[blockId]) {
      blockNodes[blockId].classList.remove("active");
    }
  }
  for (const blockId of next) {
    if (!activeBlocks.has(blockId) && blockNodes[blockId]) {
      blockNodes[blockId].classList.add("active");
    }
  }
  activeBlocks = next;
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
  document.querySelector("#kv-summary").textContent =
    `${activeIds.length} / ${numBlocks} allocated · ${formatBytes(resource.block_bytes ?? 0)} / block`;
  updateBlockGrid(numBlocks, activeIds);
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

# 5. Step 5：增加新的启动入口并运行

为了保留 Advance 01 的启动脚本不动，新建 `examples/advance02_ui_plus.py`。采样参数已经由每个 User Panel 自己控制，因此这个启动文件只需要模型、host、port。

## 5.1 新建 `examples/advance02_ui_plus.py`

```python
import argparse

import uvicorn

from tinyinfer import LLM
from tinyinfer.config import Config
from tinyinfer.runtime import DynamicEngineService
from tinyinfer.ui import create_app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/jason/huggingface/Qwen3-0.6B",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        model=args.model,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        max_model_len=2048,
        kvcache_block_size=16,
        gpu_memory_utilization=0.80,
    )
    engine = LLM(config)
    service = DynamicEngineService(engine, snapshot_interval_s=0.25)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

## 5.2 语法检查

在项目根目录执行：

```bash
python -m compileall -q tinyinfer examples
```

如果本机有 Node，也可以额外检查前端 JS：

```bash
node --check tinyinfer/ui/static/app.js
```

## 5.3 启动

```bash
python examples/advance02_ui_plus.py \
  --model /home/jason/huggingface/Qwen3-0.6B
```

然后打开：

```text
http://127.0.0.1:8000/
```

---

# 6. 最后检查功能是否符合预期

建议按下面顺序验证：

1. User1 输入 prompt。此时 Start 高亮，三个采样控件可编辑。
2. 把 `max_tokens` 拉到 256、temperature 改到 0.6、greedy 保持 ON，点击 Start。
3. 推理开始后，input 内容保留，Start、两个 slider、greedy switch 同时变灰。
4. 输出框继续逐 token 刷新；右侧 active KV blocks 数量和橘色 block 会动态变化。
5. User1 完成后，TTFT / TPOT / E2E 三个值立即刷新，input 清空，Controller 恢复可编辑。
6. User2、User3 分别设置不同参数，再点击 `Start all ready`，确认三条 request 一次进入 serving queue。
7. 每完成一条 seq，右侧 Performance statistics 的 `completed` 加一，并重新计算 9 个指标的全局平均值。
8. 请求结束后，Sequence 持有的 block `ref_count` 归零，对应方块恢复绿色；如果 block 内容仍作为 persistent prefix cache 留存，它依然是绿色，因为这里的颜色定义严格按照 `ref_count > 0`。

到这里，Advance 02 的数据流就是：

```text
User-specific controls
        │
        v
WebSocket submit / submit_many
        │
        v
DynamicEngineService
        │
        ├── Scheduler / ModelRunner / GPU
        │
        ├── token event ────────────────→ User output
        ├── finished + seq statistics ──→ User TTFT/TPOT/E2E
        └── runtime snapshot ~4Hz ──────→ Resource + global performance
```

最关键的是：**统计和 UI 都只是读取已经存在的 Scheduler/BlockManager metadata；GPU forward 仍然只由 `tinyinfer-engine` worker 驱动。** 因此这一步没有为了可视化再增加第二个模型执行线程，也没有在刷新路径里加入 CUDA synchronize。
