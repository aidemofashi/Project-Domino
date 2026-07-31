import os
import sys

_ui_python = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tilps", "Ui", "tauri", "python"
)
if _ui_python not in sys.path:
    sys.path.insert(0, _ui_python)

from ws_server import WebSocketServer

__all__ = ["WebSocketServer"]
