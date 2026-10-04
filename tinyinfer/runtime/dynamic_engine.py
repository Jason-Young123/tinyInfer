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