import { getCurrentWindow } from "@tauri-apps/api/window";

const WS_URL = "ws://127.0.0.1:8765";

const messagesEl = document.getElementById("messages");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const btnMinimize = document.getElementById("btn-minimize");
const btnClose = document.getElementById("btn-close");

let ws = null;
const appWindow = getCurrentWindow();

btnMinimize.addEventListener("click", (e) => {
  e.stopPropagation();
  appWindow.minimize();
});

btnClose.addEventListener("click", (e) => {
  e.stopPropagation();
  appWindow.close();
});

function connect() {
  setStatus("disconnected", "连接中...");

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus("listening", "聆听中");
  };

  ws.onclose = () => {
    setStatus("disconnected", "已断开");
    setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    setStatus("disconnected", "连接失败");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      console.error("消息解析失败:", e);
    }
  };
}

function setStatus(state, text) {
  const classes = { listening: "dot-listening", speaking: "dot-speaking", disconnected: "dot-disconnected" };
  statusDot.className = classes[state] || "dot-disconnected";
  statusText.textContent = text;
}

function handleMessage(data) {
  switch (data.type) {
    case "user":
      addMessage("user", data.content, data.timestamp);
      break;
    case "domino":
      addMessage("domino", data.content, data.timestamp, data.final);
      break;
    case "status":
      addStatus(data.content);
      if (data.content === "助手思考中") setStatus("speaking", "说话中");
      else if (data.content === "系统启动！" || data.content === "被语音打断") setStatus("listening", "聆听中");
      break;
    case "clear":
      messagesEl.innerHTML = "";
      break;
  }
}

function addMessage(role, content, timestamp, final = false) {
  const msg = document.createElement("div");
  msg.className = `msg msg-${role}`;

  const roleLabel = document.createElement("div");
  roleLabel.className = "msg-role";
  roleLabel.textContent = role === "user" ? "你" : "多咪";
  msg.appendChild(roleLabel);

  const contentEl = document.createElement("div");
  contentEl.className = "msg-content";
  contentEl.textContent = content;
  msg.appendChild(contentEl);

  if (timestamp) {
    const timeEl = document.createElement("div");
    timeEl.className = "msg-time";
    timeEl.textContent = timestamp;
    msg.appendChild(timeEl);
  }

  messagesEl.appendChild(msg);
  scrollToBottom();
}

function addStatus(text) {
  const msg = document.createElement("div");
  msg.className = "msg msg-status";
  msg.textContent = text;
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function scrollToBottom() {
  const container = document.getElementById("chat-container");
  container.scrollTop = container.scrollHeight;
}

function sendText(text) {
  if (!text.trim()) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "user", content: text }));
    addMessage("user", text, new Date().toLocaleString());
  } else {
    addStatus("未连接到服务");
  }
  textInput.value = "";
}

sendBtn.addEventListener("click", () => sendText(textInput.value));

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendText(textInput.value);
});

connect();
