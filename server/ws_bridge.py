from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from core.event_bus import EventBus
from core.health import HealthMonitor
from utils.logger import get_logger

logger = get_logger("dashboard")


class DashboardServer:
    def __init__(
        self,
        *,
        event_bus: EventBus,
        health_monitor: HealthMonitor,
        websocket_port: int,
        static_dir: str | Path,
        current_project_getter: Callable[[], Any] | None = None,
        current_provider_getter: Callable[[], str | None] | None = None,
        current_state_getter: Callable[[], str | None] | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self.event_bus = event_bus
        self.health_monitor = health_monitor
        self.websocket_port = int(websocket_port)
        self.host = host
        self.static_dir = Path(static_dir).expanduser().resolve()
        self.current_project_getter = current_project_getter
        self.current_provider_getter = current_provider_getter
        self.current_state_getter = current_state_getter
        self._clients: dict[int, WebSocket] = {}
        self._clients_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._subscriber_queue: asyncio.Queue | None = None
        self._event_task: asyncio.Task | None = None
        self._server: uvicorn.Server | None = None
        self._app = self._build_app()

    @property
    def url(self) -> str:
        return f"http://localhost:{self.websocket_port}"

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Dexter Dashboard", docs_url=None, redoc_url=None)

        if self.static_dir.exists():
            app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(self.static_dir / "index.html")

        @app.get("/static", include_in_schema=False)
        async def static_index() -> FileResponse:
            return FileResponse(self.static_dir / "index.html")

        @app.get("/health", include_in_schema=False)
        async def health() -> JSONResponse:
            return JSONResponse(self.health_monitor.get_health_summary())

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await self._accept_client(websocket)

        @app.on_event("startup")
        async def _startup() -> None:
            await self._start_event_fanout()

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            await self._stop_event_fanout()

        return app

    async def _start_event_fanout(self) -> None:
        if self._subscriber_queue is not None:
            return
        self._subscriber_queue = self.event_bus.subscribe(maxsize=0)
        self._event_task = asyncio.create_task(self._fan_out_events(), name="dexter-dashboard-events")
        logger.info("dashboard_event_bridge_started")

    async def _stop_event_fanout(self) -> None:
        self._stop_event.set()
        queue = self._subscriber_queue
        self._subscriber_queue = None
        if queue is not None:
            self.event_bus.unsubscribe(queue)
        task = self._event_task
        self._event_task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self._close_all_clients()

    async def _accept_client(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client_key = id(websocket)
        async with self._clients_lock:
            self._clients[client_key] = websocket
        try:
            await websocket.send_json({
                "type": "health_summary",
                "payload": self.health_monitor.get_health_summary(),
            })
            await websocket.send_json(self._build_dashboard_snapshot())
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("dashboard_client_error", error=str(exc))
        finally:
            async with self._clients_lock:
                self._clients.pop(client_key, None)

    async def _fan_out_events(self) -> None:
        queue = self._subscriber_queue
        if queue is None:
            return
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            await self._broadcast(event)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        async with self._clients_lock:
            clients = list(self._clients.items())
        if not clients:
            return

        dead_clients: list[int] = []
        for client_key, websocket in clients:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_clients.append(client_key)

        if dead_clients:
            async with self._clients_lock:
                for client_key in dead_clients:
                    self._clients.pop(client_key, None)

    async def _close_all_clients(self) -> None:
        async with self._clients_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for websocket in clients:
            with suppress(Exception):
                await websocket.close()

    def _build_dashboard_snapshot(self) -> dict[str, Any]:
        project = None
        if self.current_project_getter is not None:
            try:
                project = self.current_project_getter()
            except Exception as exc:
                logger.warning("dashboard_project_snapshot_failed", error=str(exc))

        provider = None
        if self.current_provider_getter is not None:
            try:
                provider = self.current_provider_getter()
            except Exception as exc:
                logger.warning("dashboard_provider_snapshot_failed", error=str(exc))

        state = None
        if self.current_state_getter is not None:
            try:
                state = self.current_state_getter()
            except Exception as exc:
                logger.warning("dashboard_state_snapshot_failed", error=str(exc))

        project_payload = None
        if project is not None:
            project_payload = {
                "name": getattr(project, "name", None),
                "source_path": getattr(project, "source_path", None),
                "confidence": getattr(project, "confidence", None),
                "last_confirmed_ts": getattr(project, "last_confirmed_ts", None),
            }

        return {
            "type": "dashboard_snapshot",
            "payload": {
                "health": self.health_monitor.get_health_summary(),
                "project": project_payload,
                "active_provider": provider,
                "assistant_state": state,
            },
        }

    async def serve(self) -> None:
        if not self.static_dir.exists():
            logger.warning("dashboard_static_dir_missing", path=str(self.static_dir))

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.websocket_port,
            loop="asyncio",
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        try:
            await self._server.serve()
        except OSError as exc:
            logger.warning("dashboard_server_failed", error=str(exc), port=self.websocket_port)
        finally:
            self._stop_event.set()
            await self._stop_event_fanout()

    def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.should_exit = True