import asyncio
import json
import threading
import time

import websockets


class WebSocketServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self._loop = None
        self._thread = None
        self._on_user_text = None

    def start(self):
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

    async def _handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "user" and self._on_user_text:
                    self._on_user_text(data.get("content", ""))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)

    def set_on_user_text(self, callback):
        self._on_user_text = callback

    def broadcast(self, message: dict):
        if not self.clients or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast(message), self._loop
        )

    async def _broadcast(self, message: dict):
        if not self.clients:
            return
        data = json.dumps(message, ensure_ascii=False)
        await asyncio.gather(
            *(client.send(data) for client in self.clients.copy()),
            return_exceptions=True,
        )

    def send_user(self, text):
        self.broadcast({
            "type": "user",
            "content": text,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def send_domino(self, text, final=True):
        self.broadcast({
            "type": "domino",
            "content": text,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final": final,
        })

    def send_status(self, text):
        self.broadcast({
            "type": "status",
            "content": text,
        })

    def send_clear(self):
        self.broadcast({"type": "clear"})

    def shutdown(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
