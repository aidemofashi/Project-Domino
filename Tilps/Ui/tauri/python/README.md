# Domino UI - Python 模块

WebSocket 服务器模块，用于在 Python 后端和 Tauri 前端之间实时通信。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```python
from ws_server import WebSocketServer

ws = WebSocketServer()
ws.start()
ws.send_domino("你好，我是多咪喵~")
ws.send_user("用户说了什么")
ws.send_status("助手思考中...")
ws.shutdown()
```

## 协议

前端通过 WebSocket 连接到 `ws://127.0.0.1:8765`

### 消息格式

| type | content | 说明 |
|------|---------|------|
| user | str | 用户语音/文本输入 |
| domino | str | 多咪回复 |
| status | str | 状态更新（思考中/打断等） |
| clear | - | 清空聊天记录 |
