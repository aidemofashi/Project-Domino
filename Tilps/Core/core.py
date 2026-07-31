import os
import queue
import subprocess
import threading
import time
import keyboard

from Tilps.Core.request import Request, RequestType, Priority
from Tilps.Core.state import AppState, StateManager
from Tilps.Core.pipeline import Pipeline
from Tilps.ASR.asr import ASR


class RequestCore:
    AUTO_TRIGGER_LIMIT = 2

    def __init__(self, enable_ws=True, enable_ui=True):
        """
        线程管理、状态机、WebSocket(可选)、UI(可选)
        """
        self.request_queue = queue.Queue()
        self.state = StateManager()
        self.modules = {}
        self.silence_timeout = 60
        self.make_memory = 16
        self._running = False
        self._llm_busy = False
        self._memory_lock = threading.Lock()
        self._compacting = False
        self.ws_server = None
        self.ui_process = None

        if enable_ws:
            from Tilps.Core.ws_server import WebSocketServer
            self.ws_server = WebSocketServer()
            self.ws_server.set_on_user_text(self._on_ws_text)
            self.ws_server.start()
            print("[WS] WebSocket 服务器已启动")
        else:
            print("[WS] 已禁用 WebSocket 服务器")

        if enable_ui:
            self._launch_ui()
        else:
            print("[UI] 已禁用界面")

    def _launch_ui(self):
        """启动 Tauri UI 进程（可选的 UI 模块）"""
        ui_exe = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Ui", "tauri", "src-tauri", "target", "release", "domino-ui.exe",
        )
        if not os.path.exists(ui_exe):
            print(f"[UI] 未找到 UI 程序: {ui_exe}")
            return
        try:
            self.ui_process = subprocess.Popen([ui_exe])
            print(f"[UI] 已启动 UI: {ui_exe}")
        except Exception as e:
            print(f"[UI] 启动失败: {e}")
            self.ui_process = None

    def _ws_send(self, method, *args, **kwargs):
        """仅在启用了 ws_server 时发送消息"""
        if self.ws_server is None:
            return
        getattr(self.ws_server, method)(*args, **kwargs)

    def register(self, name, module):
        """
        注册模块，将模块类放入字典
        """
        self.modules[name] = module

    def emit(self, request):
        if request.priority == Priority.HIGH:
            self.state.set_state(AppState.RECORDING)
            self.state.request_interrupt()
            if "tts" in self.modules:
                self.modules["tts"].stop()
        self.request_queue.put(request)

    def _on_vad_interrupt(self):
        self.state.set_state(AppState.RECORDING)

    def _on_ws_text(self, text):
        if text.strip():
            self.emit(
                Request(
                    type=RequestType.VOICE_INPUT,
                    payload={"text": text},
                    priority=Priority.HIGH,
                )
            )

    def _on_asr_result(self, text):
        filter = self.modules.get("filter")
        if filter and not filter.emo(text):
            print(">>> 消息被过滤")
            self.state.mark_activity()
            return

        self._ws_send("send_user", text)
        self.emit(
            Request(
                type=RequestType.VOICE_INPUT,
                payload={"text": text},
                priority=Priority.HIGH,
            )
        )

    def _process_voice_request(self, request):
        text = request.payload["text"]
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{date_time}] 用户: {text}")

        llm = self.modules["llm"]
        tts = self.modules["tts"]
        memory = self.modules["memory"]
        shot = self.modules["shot"]

        image_data = shot()
        image_data_url = f"data:image/jpeg;base64,{image_data}"

        messages = []
        with self._memory_lock:
            if initial_memory := getattr(self, "_initial_messages", None):
                messages.extend(initial_memory)

        user_msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": text},
            ],
        }
        messages.append(user_msg)

        print(">>> 助手思考中 (流式播报)...")
        self._ws_send("send_status", "助手思考中")
        full_response = ""

        self._llm_busy = True
        is_first_chunk = True
        for chunk_text in llm.send_llm_stream(messages):
            if self.state.consume_interrupt():
                tts.stop()
                print("\n>>> 被语音打断")
                self._ws_send("send_status", "被语音打断")
                self._llm_busy = False
                return

            if chunk_text.strip():
                tts.speak(chunk_text, interrupt=is_first_chunk, fast=is_first_chunk)
                full_response += chunk_text
                is_first_chunk = False

        self._llm_busy = False

        if full_response:
            self._ws_send("send_domino", full_response)
            self._append_chat({
                "role": "user",
                "content": text,
                "time": date_time,
            })
            self._append_chat({
                "role": "assistant",
                "content": full_response,
                "time": date_time,
            })
            memory.save_shot({"shot": image_data_url, "time": date_time})
            self.state.resume_auto_trigger()

        self.state.mark_activity()

    def _append_chat(self, entry):
        if not hasattr(self, "_chat_history"):
            self._chat_history = []
        self._chat_history.append(entry)
        if len(self._chat_history) >= self.make_memory and not self._compacting:
            self._compact_in_background()

    def _compact_in_background(self):
        self._compacting = True
        threading.Thread(target=self._do_compact, daemon=True).start()

    def _do_compact(self):
        memory = self.modules.get("memory")
        if not memory:
            self._compacting = False
            return
        chat = list(self._chat_history)
        try:
            new_messages = memory.chat_worker(chat)
            with self._memory_lock:
                self._initial_messages = new_messages
            self._chat_history.clear()
        except Exception as e:
            print(f"[记忆整理] 失败: {e}")
        finally:
            self._compacting = False

    def _process_timer_request(self):
        print("\n[主动触发] 静音已达阈值")
        self._ws_send("send_status", "多咪主动触发对话")
        llm = self.modules["llm"]
        tts = self.modules["tts"]
        shot = self.modules["shot"]

        image_data = shot()
        image_data_url = f"data:image/jpeg;base64,{image_data}"

        messages = []
        with self._memory_lock:
            if initial_memory := getattr(self, "_initial_messages", None):
                messages.extend(initial_memory)

        prompt_msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": "瞧"},
            ],
        }
        messages.append(prompt_msg)

        full_response = ""
        self._llm_busy = True
        try:
            is_first_chunk = True
            for chunk_text in llm.send_llm_stream(messages):
                if self.state.consume_interrupt():
                    tts.stop()
                    print("\n>>> 被语音打断")
                    self._llm_busy = False
                    return
                if chunk_text.strip():
                    tts.speak(chunk_text, interrupt=is_first_chunk, fast=is_first_chunk)
                    full_response += chunk_text
                    is_first_chunk = False
        except Exception as e:
            print(f"\n[LLM错误] 自主提问失败: {e}")
            self.state.pause_auto_trigger()
            self.state.mark_trigger()
            self._llm_busy = False
            return

        self._llm_busy = False

        if full_response:
            date_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self._ws_send("send_domino", full_response)
            self._append_chat({
                "role": "user",
                "content": "瞧",
                "time": date_time,
            })
            self._append_chat({
                "role": "assistant",
                "content": full_response,
                "time": date_time,
            })
            memory = self.modules["memory"]
            memory.save_shot({"shot": image_data_url, "time": date_time})
            self.state.resume_auto_trigger()
        else:
            print("\n[主动触发] LLM返回为空，暂停自主提问")
            self.state.pause_auto_trigger()

        self.state.mark_trigger()

    def run(self):
        self._running = True

        pipeline = Pipeline(self.state, self.modules, self.make_memory)
        initial_memory = self.modules.get("memory")
        if initial_memory:
            self._initial_messages = pipeline.load_history(initial_memory)

        self.modules["asr"].start_streaming(
            on_interrupt=self._on_vad_interrupt,
            on_result=self._on_asr_result,
        )

        print("\n" + "=" * 30)
        print("并行模式: ASR(流式VAD) + TTS(句级队列)")
        print("提示：按 [空格] 键开始，按 [Esc] 退出")
        print("=" * 30)

        keyboard.wait("space")
        print("\n>>> 系统启动！")

        while self._running:
            try:
                # 启用了 UI 时，UI 关闭则退出
                if self.ui_process is not None and self.ui_process.poll() is not None:
                    print("\n>>> UI 已关闭，退出")
                    break

                if self.state.get_state() == AppState.IDLE:
                    if self.state.should_trigger(self.silence_timeout, self.AUTO_TRIGGER_LIMIT):
                        self._process_timer_request()

                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                current_state = self.state.get_state()
                if current_state in (AppState.IDLE, AppState.RECORDING):
                    self.state.consume_interrupt()
                    self.state.set_state(AppState.PROCESSING)
                    try:
                        if request.type == RequestType.VOICE_INPUT:
                            self._process_voice_request(request)
                        elif request.type == RequestType.TIMER_TRIGGER:
                            self._process_timer_request()
                    finally:
                        if self.state.get_state() == AppState.PROCESSING:
                            self.state.set_state(AppState.IDLE)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[运行错误]: {e}")
                time.sleep(1)

    def shutdown(self):
        self._running = False
        if self.ws_server is not None:
            self.ws_server.shutdown()
        if self.ui_process is not None and self.ui_process.poll() is None:
            print("\n>>> 关闭 UI...")
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=5)
            except Exception:
                self.ui_process.kill()
        if "asr" in self.modules:
            self.modules["asr"].stop_streaming()
        if "tts" in self.modules:
            self.modules["tts"].shutdown()
