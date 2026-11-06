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


def run_profiler_gui(
    refresh_interval: float = 0.2,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """
    Run Matplotlib/GTK GUI on the calling thread.

    IMPORTANT:
    - Call this from the process main thread.
    - Uvicorn must run in a different thread.
    - This function owns the GUI event loop while profiler windows are open.
    """
    if not TimeProfiler.PROFILER:
        return

    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    non_interactive_backends = {
        "agg", "pdf", "ps", "svg", "cairo", "pgf", "template"
    }
    if backend in non_interactive_backends:
        print(
            f"[Profiler] Non-interactive Matplotlib backend: {backend}. "
            "Profiler windows cannot be displayed.",
            flush=True,
        )
        return

    print(f"[Profiler] GUI backend: {matplotlib.get_backend()}", flush=True)
    plt.ion()

    window_shown = False

    try:
        while should_stop is None or should_stop():
            # If the user closes all profiler windows, stop the GUI loop.
            # The Uvicorn server remains alive; the caller will wait for it.
            if window_shown and not plt.get_fignums():
                break

            with TimeProfiler._registry_lock:
                profilers = list(TimeProfiler.profilers)

            if not profilers:
                break

            # Drain completed CUDA Events and update each profiler figure.
            for profiler in profilers:
                profiler.drain()
                profiler.draw()

            # Create/show all figures once. draw() creates the figures.
            if not window_shown:
                plt.show(block=False)
                window_shown = True

            # Process GTK/Matplotlib events on the MAIN THREAD.
            # This is intentionally not inside FastAPI's asyncio loop.
            plt.pause(refresh_interval)

    finally:
        plt.ioff()
        plt.close("all")


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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind_loop()

        # WebSocket broadcast task.
        broadcaster = asyncio.create_task(hub.broadcast_loop())

        # Engine Worker remains managed by DynamicEngineService.
        service.start(hub.publish_from_thread)

        try:
            yield
        finally:
            # Stop Engine Worker.
            service.stop()

            # Stop WebSocket broadcaster.
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
        hub.clients.add(socket)

        snapshot = service.current_runtime_snapshot()
        if snapshot is not None:
            await socket.send_json({"type": "runtime", **snapshot})

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
