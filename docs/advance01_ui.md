# tinyInfer Advance 01：动态异步推理 + 多终端流式 UI

> 适用位置：完成 `02-A` 与真实 Qwen3 runtime 后。
>
> 这一讲不再把 `generate()` 看成“给定一组 prompt，阻塞直到全部结束”的离线函数，而是把 tinyInfer 升级成一个最小但完整的 **online serving runtime**：请求可以在任意时刻进入，Scheduler 在每个 GPU step 边界吸收新请求，三个独立终端实时接收 token，浏览器刷新与网络 I/O 不进入 GPU 推理主线程。

---

# 0. 目标与最终结构

最终页面保持你给出的布局思想，但换成更轻量的现代暗色界面：左侧三个独立 User Panel，右侧上方预留 Statistics，右侧下方是 Controller；`Start all ready` 只占 Controller 的 1/4 横向网格。

每个 User Panel 的状态机是：

```text
Idle + input empty
    ↓ 输入文字
Idle + ready
    ↓ Start
Busy: input 保留且只读，按钮变灰
    ↓ token / token / token ...
output 实时增长并自动滚到底部
    ↓ finished
input 自动清空，output 保留并追加空行，重新回到 Idle
```

服务端则变成：

```text
Browser / WebSocket
        │
        │ submit / submit_many
        v
FastAPI asyncio event loop
        │
        │ thread-safe command queue
        v
DynamicEngineService worker thread
        │
        ├── drain newly arrived requests
        ├── Scheduler.schedule()
        ├── ModelRunner.run()  ─────────────→ GPU
        ├── Scheduler.postprocess()
        └── stream token event
                  │
                  v
          WebSocket broadcast
```

这里“异步”的准确含义是：**UI/网络线程不会阻塞推理线程，请求在任何时刻都可以进入 CPU command queue；当前 GPU forward 一旦已经 launch，不会被中途插入新 sequence，新请求会在下一个 `schedule()` 边界进入 waiting queue。** 这正是 continuous/dynamic batching 应有的边界，而不是为三个用户各开一个 GPU 推理线程。

你当前 `Scheduler.schedule()` 已经支持 mixed batching：先推进 running decode，再用剩余 token/sequence budget 做 chunked prefill。因此这一讲不需要重写 Scheduler，重点是把原来 `generate()` 外层的静态 while-loop 拆成一个长期存在的 engine worker。

最终新增结构：

```text
tinyinfer/
├── engine/
│   └── llm_engine.py              # 增加 step_with_updates / text request
├── runtime/
│   ├── __init__.py
│   └── dynamic_engine.py          # online worker + command queue
└── ui/
    ├── __init__.py
    ├── app.py                     # FastAPI + WebSocket
    └── static/
        ├── index.html
        ├── style.css
        └── app.js
examples/
└── advance01_ui.py                # 一条命令启动
tests/
└── test_dynamic_engine.py
```

---

# 1. Step 1：让 LLMEngine 暴露“单步生成事件”，而不是只能一次性 generate

## 1.1 为什么原来的 `generate()` 不够

当前 `generate()` 的核心是：

```text
add all prompts
while scheduler not finished:
    step()
return all outputs
```

问题不是 Scheduler 不能动态接收请求，而是最外层 API 把整个过程封死成了一次阻塞调用。UI 需要的是：每做完一轮 `step`，立刻知道**本轮哪些 seq 新生成了 token、哪些 seq 已经 finished**。

因此保留原有 `step()` 和 `generate()`，新增 `step_with_updates()`。这样原来的静态 benchmark/test 不被破坏，online runtime 则使用新接口。

## 1.2 完整替换 `tinyinfer/engine/llm_engine.py`

```python
from dataclasses import dataclass

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

        # 先建立真实模型，再根据剩余显存建立 Scheduler/BlockManager。
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

    def build_chat_prompt(self, text: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        if self.tokenizer.chat_template:
            kwargs = dict(
                tokenize=False,
                add_generation_prompt=True,
            )
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

这里有三个关键点：

- `add_text_request()` 把 UI 字符串转换成真正 chat-template prompt，再进入原 Scheduler；Qwen3 支持时关闭 thinking，旧 tokenizer 不接受该参数时自动 fallback。
- `step_with_updates()` 仍然只做**一次** scheduler/model/postprocess，不自己 while 到结束，因此可以在 step 与 step 之间吸收新请求。
- `StreamUpdate` 保存累计 completion token ids。UI runtime 用累计 decode，而不是简单 `decode([single_token])`，避免 BPE/byte token 在单 token 解码时出现乱码或空格错位。

---

# 2. Step 2：建立真正长期运行的 DynamicEngineService

这一层是本讲最重要的新增代码。它不属于 UI，因此单独放在 `tinyinfer/runtime/`。

职责只有四个：

```text
1. UI 线程随时 submit，请求只进入 thread-safe Queue
2. 一个专用 worker thread 独占 Scheduler/ModelRunner
3. 每轮 GPU step 前 drain 新请求，所以请求动态加入 waiting queue
4. 每得到一个 sampled token 就发送 streaming event
```

这样避免 FastAPI/浏览器刷新直接碰 CUDA runtime，也避免多个线程同时调用同一个 ModelRunner。

## 2.1 新建 `tinyinfer/runtime/__init__.py`

```python
from tinyinfer.runtime.dynamic_engine import DynamicEngineService

__all__ = ["DynamicEngineService"]
```

## 2.2 新建 `tinyinfer/runtime/dynamic_engine.py`

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable

from tinyinfer.engine.llm_engine import LLMEngine
from tinyinfer.sampling_params import SamplingParams


@dataclass(slots=True)
class SubmitRequest:
    user_id: str
    text: str
    params: SamplingParams


@dataclass(slots=True)
class RuntimeEvent:
    type: str
    user_id: str
    seq_id: int | None = None
    text: str | None = None
    statistics: dict | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DynamicEngineService:
    def __init__(
        self,
        engine: LLMEngine,
        system_prompt: str = "You are a helpful, concise assistant.",
    ):
        self.engine = engine
        self.system_prompt = system_prompt
        self._commands: Queue[list[SubmitRequest] | None] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None
        self._event_sink: Callable[[dict], None] | None = None
        self._busy_users: set[str] = set()
        self._busy_lock = Lock()
        self._seq_to_user: dict[int, str] = {}

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

    def _handle_updates(self, updates) -> None:
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
        while not self._stop.is_set():
            if self.engine.scheduler.is_finished():
                try:
                    batch = self._commands.get(timeout=0.1)
                except Empty:
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
                self._handle_updates(updates)
            except Exception as exc:
                self._fail_active_requests(str(exc))
                return
```

### 这一层为什么算“动态调度”

假设 User1 正在 decode，而 User2 在这一轮 GPU forward 期间点击 Start：

```text
t0  GPU: User1 decode step k 正在运行
    CPU/UI: User2 submit → command queue

t1  User1 step k 返回
    worker: _drain_commands()
    User2 → scheduler.waiting

t2  Scheduler.schedule()
    User1: decode 1 token
    User2: chunked prefill N tokens
    二者进入同一个 mixed batch
```

因此新请求不需要等 User1 整条回答结束，只需要等**当前这一轮 forward**结束。这和 static `generate([prompt1, prompt2, prompt3])` 有本质区别。

### 为什么 Controller 的“全部开始”是真正同批到达

前端一次发送 `submit_many`，后端把它变成一个 `list[SubmitRequest]` command。worker 的 `_admit_batch()` 会把这个 batch 中所有请求连续 `scheduler.add()` 完之后，才进入下一次 `step_with_updates()`。所以从 Scheduler 视角，它们在同一个调度边界同时可见。

### busy 状态为什么服务端也要保存

不能只依赖按钮变灰。用户可能重复发 WebSocket 包，或者前端状态异常。`_busy_users` 是服务端最后一道保护：一个 User Panel 同时只允许一个 active request。

---

# 3. Step 3：增加 FastAPI + WebSocket UI 服务层

UI 代码全部放进 `tinyinfer/ui/`，不污染 engine/runtime。FastAPI 的 asyncio loop 只负责网络；GPU 推理由上一节的 worker thread 独占。

## 3.1 修改 `pyproject.toml`

完整内容如下：

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



[project.optional-dependencies]
dev = [
    "pytest>=8",
]
ui = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
]


[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

安装：

```bash
cd ~/tinyInfer
python -m pip install -e '.[ui,dev]'
```

## 3.2 新建 `tinyinfer/ui/__init__.py`

```python
from tinyinfer.ui.app import create_app

__all__ = ["create_app"]
```

## 3.3 新建 `tinyinfer/ui/app.py`

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


def create_app(
    service: DynamicEngineService,
    max_tokens: int = 128,
    temperature: float = 0.8,
    greedy: bool = False,
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
                for item in raw_requests:
                    user_id = str(item.get("user_id", ""))
                    text = str(item.get("text", ""))
                    if not user_id or not text.strip():
                        continue
                    requests.append(
                        SubmitRequest(
                            user_id=user_id,
                            text=text,
                            params=SamplingParams(
                                temperature=temperature,
                                max_tokens=max_tokens,
                                greedy=greedy,
                            ),
                        )
                    )

                accepted = service.submit_many(requests)
                await socket.send_json(
                    {
                        "type": "accepted",
                        "accepted_users": accepted,
                        "requested_users": [r.user_id for r in requests],
                    }
                )
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(socket)

    return app
```

这里没有 HTTP polling。浏览器和服务端保持一个 WebSocket：

```text
client → server: submit / submit_many
server → client: accepted / started / token / finished / error
```

`SocketHub.publish_from_thread()` 只做一次 `loop.call_soon_threadsafe(...)`，因此 inference worker 不会等待浏览器绘制，也不会等待 socket 发送完成。真正的 `send_json()` 在 asyncio event loop 里执行。

---

# 4. Step 4：实现三个独立 User Panel + Statistics 占位 + Controller

前端不用 React/Vue。这里只有三个固定 panel，用原生 HTML/CSS/JS 更小、更容易看清 serving 逻辑，也不会引入前端构建工具。

## 4.1 新建 `tinyinfer/ui/static/index.html`

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
      <section class="panel statistics">
        <div class="panel-title">Statistics</div>
        <div class="placeholder">Reserved for runtime metrics</div>
      </section>

      <section class="panel controller">
        <div class="panel-title">Controller</div>
        <div class="controller-grid">
          <button id="start-all" class="primary" disabled>Start all ready</button>
        </div>
      </section>
    </aside>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

## 4.2 新建 `tinyinfer/ui/static/style.css`

```css
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0b0d12;
  color: #edf1f7;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(73, 95, 160, .18), transparent 34rem),
    #0b0d12;
}

button, textarea { font: inherit; }

.shell {
  width: min(1500px, calc(100vw - 40px));
  height: calc(100vh - 40px);
  margin: 20px auto;
  display: grid;
  grid-template-columns: minmax(0, 2.1fr) minmax(300px, .9fr);
  gap: 16px;
}

.users {
  min-height: 0;
  display: grid;
  grid-template-rows: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.side {
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(0, 1fr) 170px;
  gap: 16px;
}

.panel {
  min-height: 0;
  border: 1px solid #272c37;
  border-radius: 18px;
  background: rgba(18, 21, 28, .88);
  box-shadow: 0 18px 50px rgba(0, 0, 0, .24);
  backdrop-filter: blur(18px);
}

.user-panel {
  padding: 16px;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 12px;
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.panel-title {
  color: #f7f9fc;
  font-size: 14px;
  font-weight: 650;
  letter-spacing: .02em;
}

.status {
  color: #8992a3;
  font-size: 12px;
}

.status.busy { color: #9ab5ff; }

.output {
  overflow-y: auto;
  padding: 14px 15px;
  border: 1px solid #232936;
  border-radius: 14px;
  background: #0f1218;
  color: #dce3ee;
  white-space: pre-wrap;
  line-height: 1.55;
}

.composer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  align-items: end;
}

textarea {
  width: 100%;
  min-height: 62px;
  max-height: 100px;
  resize: vertical;
  border: 1px solid #2b3240;
  border-radius: 14px;
  outline: none;
  padding: 12px 14px;
  background: #151922;
  color: #f5f7fb;
  transition: border-color .15s ease, box-shadow .15s ease;
}

textarea:focus {
  border-color: #657dc8;
  box-shadow: 0 0 0 3px rgba(101, 125, 200, .14);
}

textarea:read-only { color: #aeb6c5; }

button {
  border: 0;
  border-radius: 12px;
  padding: 11px 16px;
  font-weight: 650;
  cursor: pointer;
  transition: transform .12s ease, opacity .12s ease, background .12s ease;
}

button:not(:disabled):active { transform: translateY(1px); }

button:disabled {
  cursor: default;
  opacity: .38;
}

.primary {
  background: #dfe7ff;
  color: #111827;
}

.statistics, .controller { padding: 18px; }

.placeholder {
  height: calc(100% - 30px);
  display: grid;
  place-items: center;
  color: #626b79;
  font-size: 13px;
}

.controller-grid {
  margin-top: 16px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

#start-all { grid-column: span 1; }

@media (max-width: 900px) {
  .shell {
    height: auto;
    grid-template-columns: 1fr;
  }
  .users { grid-template-rows: repeat(3, 360px); }
  .side { grid-template-rows: 300px 170px; }
}
```

这个布局与参考图保持同一信息结构，但不是照抄配色：

```text
┌──────────────────────────────┬─────────────────┐
│ User 1                       │                 │
├──────────────────────────────┤   Statistics    │
│ User 2                       │   (reserved)    │
├──────────────────────────────┤                 │
│ User 3                       ├─────────────────┤
│                              │ Controller      │
└──────────────────────────────┴─────────────────┘
```

Controller 使用四列 grid，`Start all ready` 只占第一列，因此当前只占 1/4，另外 3/4 后续可以放 Pause、Abort all、Clear cache 等全局控制。

## 4.3 新建 `tinyinfer/ui/static/app.js`

```javascript
const USERS = ["user1", "user2", "user3"];
const states = new Map();
const usersRoot = document.querySelector("#users");
const startAll = document.querySelector("#start-all");
const ws = new WebSocket(`ws://${location.host}/ws`);

function makePanel(userId, index) {
  const panel = document.createElement("section");
  panel.className = "panel user-panel";
  panel.innerHTML = `
    <div class="panel-head">
      <div class="panel-title">User ${index + 1}</div>
      <div class="status">Idle</div>
    </div>
    <div class="output"></div>
    <div class="composer">
      <textarea placeholder="Type a prompt..."></textarea>
      <button class="primary" disabled>Start</button>
    </div>`;

  const state = {
    userId,
    panel,
    input: panel.querySelector("textarea"),
    button: panel.querySelector("button"),
    output: panel.querySelector(".output"),
    status: panel.querySelector(".status"),
    busy: false,
    outputBase: "",
  };
  states.set(userId, state);

  state.input.addEventListener("input", refreshControls);
  state.button.addEventListener("click", () => submitOne(state));
  usersRoot.appendChild(panel);
}

function ready(state) {
  return !state.busy && state.input.value.trim().length > 0;
}

function setBusy(state, busy) {
  state.busy = busy;
  state.input.readOnly = busy;
  state.status.textContent = busy ? "Running" : "Idle";
  state.status.classList.toggle("busy", busy);
  refreshControls();
}

function refreshControls() {
  for (const state of states.values()) {
    state.button.disabled = !ready(state);
  }
  startAll.disabled = ![...states.values()].some(ready);
}

function beginLocal(state) {
  state.outputBase = state.output.textContent;
  setBusy(state, true);
}

function submitOne(state) {
  if (!ready(state) || ws.readyState !== WebSocket.OPEN) return;
  const text = state.input.value;
  beginLocal(state);
  ws.send(JSON.stringify({type: "submit", user_id: state.userId, text}));
}

startAll.addEventListener("click", () => {
  if (ws.readyState !== WebSocket.OPEN) return;
  const requests = [];
  for (const state of states.values()) {
    if (!ready(state)) continue;
    requests.push({user_id: state.userId, text: state.input.value});
    beginLocal(state);
  }
  if (requests.length) {
    ws.send(JSON.stringify({type: "submit_many", requests}));
  }
});

ws.addEventListener("message", (message) => {
  const event = JSON.parse(message.data);

  if (event.type === "accepted") {
    const accepted = new Set(event.accepted_users);
    for (const userId of event.requested_users) {
      if (!accepted.has(userId)) setBusy(states.get(userId), false);
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
refreshControls();
```

前端状态机最值得注意的是 `outputBase`：每次开始新请求时记录旧输出；streaming event 传来的 `text` 是**本次请求的累计输出**，因此页面显示：

```text
old output + current request cumulative text
```

完成时：

```javascript
state.output.textContent = state.outputBase + event.text + "\n\n";
state.input.value = "";
```

所以完全符合要求：推理期间输入内容一直保留，结束后才清空；旧输出从不清空，并在每次回答后增加一个空行。

三个 panel 各有独立 `busy/input/outputBase`，一个 panel 正在推理不会阻止另外两个输入和点击。

---

# 5. Step 5：增加启动入口，一条命令运行

新建 `examples/advance01_ui.py`：

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
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
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
    service = DynamicEngineService(engine)
    app = create_app(
        service,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        greedy=args.greedy,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

启动：

```bash
cd ~/tinyInfer
python examples/advance01_ui.py \
  --model /home/jason/huggingface/Qwen3-0.6B \
  --max-tokens 128 \
  --greedy
```

看到：

```text
Uvicorn running on http://127.0.0.1:8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果想用 sampling：

```bash
python examples/advance01_ui.py \
  --model /home/jason/huggingface/Qwen3-0.6B \
  --max-tokens 128 \
  --temperature 0.8
```

`--greedy` 打开时仍会创建合法的正 temperature，但 sampler 按 `greedy_mask` 走 greedy 路径。

---

# 6. Step 6：加一个不依赖 GPU 行为的 runtime 回归测试

这个测试的目的不是验证 Qwen3 数值，而是验证：

```text
submit_many
→ 两个 user 同时被接收
→ 各自收到 started/token/token/finished
→ 输出互不串线
→ finished 后 busy 状态释放
```

新建 `tests/test_dynamic_engine.py`：

```python
import time
from dataclasses import dataclass

from tinyinfer.runtime.dynamic_engine import DynamicEngineService, SubmitRequest
from tinyinfer.sampling_params import SamplingParams


class FakeTokenizer:
    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeScheduler:
    def __init__(self, engine):
        self.engine = engine

    def is_finished(self):
        return not self.engine.active


@dataclass
class FakeUpdate:
    seq_id: int
    completion_token_ids: list[int]
    finished: bool
    statistics: dict | None = None


class FakeEngine:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.scheduler = FakeScheduler(self)
        self.active = {}
        self.next_seq_id = 0

    def add_text_request(self, text, params, system_prompt=None):
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.active[seq_id] = [ord("A"), ord("B")]
        return seq_id

    def step_with_updates(self):
        updates = []
        for seq_id in list(self.active):
            generated = self.active[seq_id]
            token = generated.pop(0)
            completion = [ord("A")] if token == ord("A") else [ord("A"), ord("B")]
            finished = not generated
            updates.append(
                FakeUpdate(
                    seq_id=seq_id,
                    completion_token_ids=completion,
                    finished=finished,
                    statistics={"ok": 1} if finished else None,
                )
            )
            if finished:
                del self.active[seq_id]
        return [object()], updates


def wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_submit_many_streams_independent_users():
    events = []
    service = DynamicEngineService(FakeEngine())
    service.start(events.append)

    params = SamplingParams(greedy=True, max_tokens=2)
    accepted = service.submit_many(
        [
            SubmitRequest("user1", "hello", params),
            SubmitRequest("user2", "world", params),
        ]
    )

    assert accepted == ["user1", "user2"]
    assert wait_until(
        lambda: sum(event["type"] == "finished" for event in events) == 2
    )

    for user_id in ("user1", "user2"):
        user_events = [e for e in events if e["user_id"] == user_id]
        assert [e["type"] for e in user_events] == [
            "started",
            "token",
            "token",
            "finished",
        ]
        assert user_events[-1]["text"] == "AB"
        assert not service.is_user_busy(user_id)

    service.stop()
```

运行：

```bash
python -m compileall -q tinyinfer examples/advance01_ui.py
python -m pytest -q tests/test_dynamic_engine.py
```

然后再跑原有测试，确认静态 API 没被破坏：

```bash
python -m pytest -q
```

最后用真实 GPU 做手工动态测试：

```text
1. User1 输入一个较长问题，点击 Start
2. 等 User1 已经开始流式输出后，再点击 User2 Start
3. 观察 User1 不停止，User2 在下一个调度 step 被插入
4. 再同时给 User2/User3 填内容，点击 Start all ready
5. 两条请求应在同一个 admission 边界进入 waiting queue
6. 任一请求结束时，仅对应 panel 恢复 Idle，其他 panel 不受影响
```

如果打开你已有的 Scheduler trace，还可以直接观察 waiting/running 的动态变化。

---

# 7. 这版设计中 CPU、GPU、UI 到底怎样并行

不要把“异步”理解成三个 Python thread 同时调用 CUDA model。当前结构刻意保持 **single owner**：

```text
FastAPI main thread
  ├── textarea/button/WebSocket
  ├── 接收任意时刻到达的请求
  └── 广播 token event

Engine worker thread
  ├── tokenizer admission
  ├── Scheduler
  ├── ModelRunner
  └── postprocess

GPU
  └── 当前 mixed batch forward
```

这个边界有三个好处：

1. Scheduler、BlockManager、Sequence 不需要为了 UI 到处加锁；只有 worker 能修改它们。
2. 浏览器再慢，也只是 WebSocket event queue 变慢，不会让 `send_json()` 卡在模型 forward 路径。
3. 请求可以在 GPU forward 期间到达 command queue，并在下一轮调度立刻并入 batch。

严格地说，Python 调用 `ModelRunner.run()` 本身仍然是 worker 的一个同步函数；PyTorch/CUDA 内部可能异步 launch kernel，也可能在某些算子处同步。因此这版的目标不是做复杂 CUDA stream overlap，而是先把 **online serving 的控制平面与 UI I/O 从 GPU engine 的所有权中分离出来**。这已经是从离线 inference loop 走向真实 serving runtime 最关键的一步。

---

# 8. 你运行后应该看到的请求时间线

例如：

```text
User1 Start
  ↓
seq0 enters waiting
  ↓
prefill(seq0) → token0 → UI1
  ↓
decode(seq0)  → token1 → UI1

             User2 Start
                  ↓
             command queue
                  ↓
next scheduler boundary
  ↓
[mixed batch]
  seq0: decode 1 token
  seq1: prefill chunk
  ↓
UI1 continues streaming
  ↓
seq1 final prefill → first token → UI2 starts streaming
```

这就是你现在应该建立的核心概念：**“三个终端独立”描述的是请求生命周期与 UI 状态独立；GPU 层不是三套模型，而是一个 Scheduler 把所有活动 sequence 动态组成 batch。**

---

# 9. 暂时不在 Advance 01 做的事情

为了让这一讲保持短而完整，下面内容先不加：

```text
Statistics Panel 的实时指标
Abort / cancel request
Pause / resume
多轮对话 history 重新拼 prompt
多个浏览器客户端的 session ownership
WebSocket backpressure / event coalescing
CUDA stream overlap
请求优先级 / deadline scheduling
```

但是当前目录和 Controller 的 3/4 预留空间已经为这些功能留好了位置。下一步最自然的是把你 `Sequence.statistics()` 里的 TTFT、TPOT、queue time、prefill/decode throughput 通过同一条 event path 推到 Statistics Panel，而不需要再改 GPU 主路径。

---

# 10. 最终检查清单

完成后只检查下面这些行为：

```text
[ ] 三个 input 初始为空，三个 Start 均 disabled
[ ] 输入非空后，对应 Start 立即可用
[ ] 点击后 input 内容保留但只读，Start disabled
[ ] token 到达时 output 连续增长并自动滚动
[ ] 推理结束后 input 清空，output 保留并增加空行
[ ] User1 busy 时 User2/User3 仍可独立提交
[ ] Start all ready 只提交“非 busy 且 input 非空”的 panel
[ ] submit_many 中所有请求在一次 worker admission 中加入 Scheduler
[ ] 新请求能在已有 decode 过程中加入，而不是等待旧请求整条结束
[ ] 原来的 LLM.generate() 仍然能工作
[ ] Statistics 区域保留但暂时不实现
```

做到这里，tinyInfer 就不再只是一个“离线 batch 推理 demo”，而已经具备一个最小 online serving 系统的核心形态：**continuous admission + mixed scheduling + streaming output + independent clients + isolated UI I/O**。
