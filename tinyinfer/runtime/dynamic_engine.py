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



