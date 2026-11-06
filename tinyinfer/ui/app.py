from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tinyinfer.runtime.dynamic_engine import (
    DynamicEngineService,
    SubmitRequest,
)
from tinyinfer.sampling_params import SamplingParams
from tinyinfer.utils.profiler import TimeProfiler


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
            for socket in list(self.clients):
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

    if not 32 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be in [32, 8192]")
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

    async def profiler_broadcast_loop() -> None:
        # Runs on the existing FastAPI event loop; it creates no thread/process.
        last_sent = {
            item["id"]: item["total_samples"]
            for item in TimeProfiler.snapshot_all()
        } if TimeProfiler.PROFILER else {}
        while True:
            if TimeProfiler.PROFILER:
                TimeProfiler.drain_all()
                updates = []
                for item in TimeProfiler.snapshot_all():
                    profiler_id = item["id"]
                    first = item["sample_start"]
                    total = item["total_samples"]
                    previous = last_sent.get(profiler_id)

                    if previous is None or previous < first:
                        # New profiler, or the backend's 1000-sample ring has
                        # advanced past samples not yet sent to the browser.
                        samples = item["samples"]
                        sample_start = first
                        reset = True
                    else:
                        offset = previous - first
                        samples = item["samples"][offset:]
                        sample_start = previous
                        reset = False

                    if previous is None or reset or samples:
                        updates.append({
                            "id": profiler_id,
                            "title": item["title"],
                            "sample_start": sample_start,
                            "total_samples": total,
                            "samples": samples,
                            "reset": reset,
                        })
                    last_sent[profiler_id] = total

                if updates and hub.events is not None:
                    hub.events.put_nowait({"type": "profiler", "profilers": updates})

            await asyncio.sleep(0.2)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind_loop()

        # WebSocket broadcast and profiler telemetry tasks share this event loop.
        broadcaster = asyncio.create_task(hub.broadcast_loop())
        profiler_task = (
            asyncio.create_task(profiler_broadcast_loop())
            if TimeProfiler.PROFILER else None
        )

        # Engine Worker remains managed by DynamicEngineService.
        service.start(hub.publish_from_thread)

        try:
            yield
        finally:
            # Stop Engine Worker.
            service.stop()

            # Stop the profiler telemetry and WebSocket tasks.
            if profiler_task is not None:
                profiler_task.cancel()
                with suppress(asyncio.CancelledError):
                    await profiler_task
            broadcaster.cancel()
            with suppress(asyncio.CancelledError):
                await broadcaster

    app = FastAPI(title="tinyInfer UI", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def websocket_endpoint(socket: WebSocket):
        await socket.accept()

        snapshot = service.current_runtime_snapshot()
        if snapshot is not None:
            await socket.send_json({"type": "runtime", **snapshot})
        if TimeProfiler.PROFILER:
            TimeProfiler.drain_all()
        await socket.send_json({
            "type": "profiler",
            "enabled": TimeProfiler.PROFILER,
            "profilers": TimeProfiler.snapshot_all() if TimeProfiler.PROFILER else [],
        })
        hub.clients.add(socket)

        try:
            while True:
                payload = await socket.receive_json()
                request_type = payload.get("type")

                if request_type == "configure_sd":
                    try:
                        raw_max = payload.get("max_draft_tokens")
                        if raw_max is None:
                            raise ValueError("max_draft_tokens is required")
                        snapshot = service.configure_speculative(int(raw_max))
                        await socket.send_json(
                            {"type": "sd_config", "speculative": snapshot}
                        )
                    except (TypeError, ValueError, RuntimeError) as exc:
                        await socket.send_json(
                            {"type": "sd_error", "message": str(exc)}
                        )
                    continue

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
