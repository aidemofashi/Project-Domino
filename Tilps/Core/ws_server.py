"""
界面接口核心：本地 WebSocket 服务器。

负责与 Tilps/Ui/tauri_ui 前端双向通信：

    前端 -> 后端
        pet://user-message   { content }
        log://set-level      { level }
        log://clear          {}

    后端 -> 前端
        pet://user           { text, timestamp }
        pet://speak-start    {}
        pet://speak-chunk    { text }
        pet://speak-end      {}
        pet://status         { text }
        log://entry          { level, source, message, time }
        log://backlog        { entries: [...] }
        log://clear          {}

消息统一为 {"event": "...", "payload": {...}} 结构，
同时兼容旧版 {"type": "...", "content": "..."} 结构。
"""

import asyncio
import json
import threading
import time

import websockets
from websockets.exceptions import ConnectionClosed

from Tilps.Core import logger

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

EVENT_USER_MESSAGE = "pet://user-message"
EVENT_USER = "pet://user"
EVENT_SPEAK_START = "pet://speak-start"
EVENT_SPEAK_CHUNK = "pet://speak-chunk"
EVENT_SPEAK_END = "pet://speak-end"
EVENT_STATUS = "pet://status"

EVENT_LOG_ENTRY = "log://entry"
EVENT_LOG_ENTRIES = "log://entries"
EVENT_LOG_BACKLOG = "log://backlog"
EVENT_LOG_CLEAR = "log://clear"
EVENT_LOG_LEVEL = "log://set-level"

EVENT_API_CONFIG_REQUEST = "api://config-request"
EVENT_API_CONFIG = "api://config"
EVENT_API_CONFIG_SAVE = "api://config-save"
EVENT_API_CONFIG_SAVED = "api://config-saved"


class WebSocketServer:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.clients = set()
        self._loop = None
        self._thread = None
        self._server = None
        self._stopping = None
        self._ready = threading.Event()
        self._on_user_text = None
        self._on_command = None
        self._log = logger.get_logger("interface")

    # ── 生命周期 ─────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3)
        return True

    def set_on_user_text(self, callback):
        self._on_user_text = callback

    def set_on_command(self, callback):
        """注册通用事件回调 callback(event, payload)。"""
        self._on_command = callback

    def _run_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:  # pragma: no cover - 端口占用等
            self._log.error("WebSocket 服务器启动失败: %s", exc)
            self._ready.set()
        finally:
            self._loop.close()

    async def _serve(self):
        self._stopping = asyncio.Event()
        self._server = await websockets.serve(self._handler, self.host, self.port)
        self._ready.set()
        self._log.info("WebSocket 服务器已启动 ws://%s:%s", self.host, self.port)
        await self._stopping.wait()
        self._server.close()
        await self._server.wait_closed()

    async def _handler(self, websocket):
        self.clients.add(websocket)
        peer = getattr(websocket, "remote_address", None)
        self._log.info("界面已连接 %s", peer)
        try:
            await self._send_backlog(websocket)
            async for raw in websocket:
                self._handle_message(raw)
        except ConnectionClosed:
            pass
        except Exception as exc:  # pragma: no cover
            self._log.warning("界面消息处理失败: %s", exc)
        finally:
            self.clients.discard(websocket)
            self._log.info("界面已断开 %s", peer)

    async def _send_backlog(self, websocket):
        entries = logger.get_backlog()
        if not entries:
            return
        await websocket.send(json.dumps(
            {"event": EVENT_LOG_BACKLOG, "payload": {"entries": entries}},
            ensure_ascii=False,
        ))

    # ── 接收 ─────────────────────────────────────────────

    def _handle_message(self, raw):
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict):
            return

        event = data.get("event") or data.get("type")
        payload = data.get("payload")
        if payload is None:
            payload = data.get("data")
        if payload is None:
            payload = data

        if event in (EVENT_USER_MESSAGE, "user"):
            content = ""
            if isinstance(payload, dict):
                content = payload.get("content") or payload.get("text") or ""
            elif isinstance(payload, str):
                content = payload
            content = str(content).strip()
            if content and self._on_user_text:
                self._on_user_text(content)
        elif event == EVENT_LOG_LEVEL:
            level = payload.get("level") if isinstance(payload, dict) else payload
            if logger.set_level(level):
                logger.dispatch("INFO", "interface", f"日志级别已切换为 {logger.get_level()}")
        elif event == EVENT_LOG_CLEAR:
            logger.clear_backlog()
            self.broadcast(EVENT_LOG_CLEAR, {})
        elif self._on_command is not None:
            try:
                self._on_command(event, payload)
            except Exception as exc:  # pragma: no cover
                self._log.warning("命令处理失败 %s: %s", event, exc)

    # ── 发送 ─────────────────────────────────────────────

    def broadcast(self, event, payload=None):
        client_count = len(self.clients)
        if not client_count or self._loop is None or self._loop.is_closed():
            return
        message = {"event": event, "payload": payload if payload is not None else {}}
        data = json.dumps(message, ensure_ascii=False, default=str)
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    async def _broadcast(self, data):
        clients = list(self.clients)
        if not clients:
            return
        await asyncio.gather(
            *(client.send(data) for client in clients),
            return_exceptions=True,
        )

    def send_log(self, payload):
        self.broadcast(EVENT_LOG_ENTRY, payload)

    def send_logs(self, entries):
        if not entries:
            return
        self.broadcast(EVENT_LOG_ENTRIES, {"entries": list(entries)})

    def send_user(self, text):
        self.broadcast(EVENT_USER, {
            "text": text,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def send_speak_start(self):
        self.broadcast(EVENT_SPEAK_START, {})

    def send_speak_chunk(self, text):
        self.broadcast(EVENT_SPEAK_CHUNK, {"text": text})

    def send_speak_end(self):
        self.broadcast(EVENT_SPEAK_END, {})

    def send_status(self, text):
        self.broadcast(EVENT_STATUS, {"text": text})

    def send_api_config(self, data, path):
        self.broadcast(EVENT_API_CONFIG, {"data": data, "path": path})

    def send_api_saved(self, ok, error=None):
        self.broadcast(EVENT_API_CONFIG_SAVED, {"ok": bool(ok), "error": error or ""})

    def send_clear(self):
        self.broadcast(EVENT_LOG_CLEAR, {})

    # ── 关闭 ─────────────────────────────────────────────

    def shutdown(self):
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if self._stopping is not None:
            def _signal_stop():
                if self._stopping is not None:
                    self._stopping.set()
            try:
                loop.call_soon_threadsafe(_signal_stop)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)
