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




