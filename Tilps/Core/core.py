import os
import queue
import threading
import time

from Tilps.Core import logger
from Tilps.Core.apim import ApiManager
from Tilps.Core.request import Request, RequestType, Priority
from Tilps.Core.state import AppState, StateManager
from Tilps.Core.pipeline import Pipeline
from Tilps.ASR.asr import ASR


class RequestCore:
    AUTO_TRIGGER_LIMIT = 1 
    """分钟"""

    def __init__(
        self,
        enable_ws=True,
        enable_ui=False,
        ws_host="127.0.0.1",
        ws_port=8765,
        log_level="INFO",
    ):
        """
        线程管理、状态机、WebSocket(可选)
        """
        logger.setup_logging(log_level)
        self._log = logger.get_logger("core")

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
        self.ws_host = ws_host
        self.ws_port = ws_port
        self._mic_enabled = True
        self._tts_enabled = True

        if enable_ws:
            from Tilps.Core.ws_server import WebSocketServer
            self.ws_server = WebSocketServer(host=ws_host, port=ws_port)
            self.ws_server.set_on_user_text(self._on_ws_text)
            self.ws_server.set_on_command(self._on_interface_command)
            logger.bind_sink(self.ws_server.send_logs)
            self.ws_server.start()
            self._log.info("WebSocket 接口已启用 ws://%s:%s", ws_host, ws_port)
        else:
            self._log.info("已禁用 WebSocket 接口")

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
            self._log.info("界面输入: %s", text)
            self.emit(
                Request(
                    type=RequestType.VOICE_INPUT,
                    payload={"text": text},
                    priority=Priority.HIGH,
                )
            )

    # ── 界面配置接口 ─────────────────────────────────────

    def _on_interface_command(self, event, payload):
        if event == "api://config-request":
            self._emit_api_config()
        elif event == "api://config-save":
            self._save_api_config(payload)
        elif event == "mic://set":
            enabled = True
            if isinstance(payload, dict) and "enabled" in payload:
                enabled = bool(payload.get("enabled"))
            self._set_microphone(enabled)
        elif event == "tts://set":
            enabled = True
            if isinstance(payload, dict) and "enabled" in payload:
                enabled = bool(payload.get("enabled"))
            self._set_tts(enabled)
        elif event == "tts://stop":
            self._stop_tts()
        else:
            self._log.debug("未处理的界面命令: %s", event)

    def _set_microphone(self, enabled):
        """开关麦克风输入（只影响 ASR，不停止 TTS）。"""
        enabled = bool(enabled)
        previous = self._mic_enabled
        self._mic_enabled = enabled

        asr = self.modules.get("asr")
        if asr is None:
            self._log.info("麦克风状态已记录: %s", "开启" if enabled else "关闭")
            return

        if enabled:
            if not previous:
                asr.start_streaming(
                    on_interrupt=self._on_vad_interrupt,
                    on_result=self._on_asr_result,
                )
                self._log.info("麦克风已开启")
                self._ws_send("send_status", "麦克风已开启")
        else:
            if previous:
                asr.stop_streaming()
                self._log.info("麦克风已关闭")
                self._ws_send("send_status", "麦克风已关闭")

    def _stop_tts(self):
        """立即停止当前语音播报。"""
        tts = self.modules.get("tts")
        if tts is None:
            return
        tts.stop()
        self._log.info("已停止语音播报")
        self._ws_send("send_status", "已停止语音播报")

    def _set_tts(self, enabled):
        """静音开关：关闭时停止当前播报且后续不再发声，开启时恢复。"""
        enabled = bool(enabled)
        previous = self._tts_enabled
        self._tts_enabled = enabled
        if enabled == previous:
            return
        if enabled:
            self._log.info("已恢复语音播报")
            self._ws_send("send_status", "已恢复语音播报")
        else:
            tts = self.modules.get("tts")
            if tts is not None:
                tts.stop()
            self._log.info("已静音语音播报")
            self._ws_send("send_status", "已静音语音播报")

    def _emit_api_config(self):
        try:
            api = ApiManager()
            self._ws_send("send_api_config", api.raw_config(), api.config_path())
        except Exception as e:
            self._log.error("读取 api.json 失败: %s", e)
            self._ws_send("send_api_saved", False, f"读取失败: {e}")

    def _save_api_config(self, payload):
        data = payload.get("data") if isinstance(payload, dict) else None
        try:
            api = ApiManager()
            path = api.save_config(data)
            self._log.info("api.json 已更新: %s", path)
            self._apply_api_config()
            self._ws_send("send_api_saved", True)
        except Exception as e:
            self._log.error("保存 api.json 失败: %s", e)
            self._ws_send("send_api_saved", False, str(e))

    def _apply_api_config(self):
        """把可热更新的配置应用到已注册模块，其余需重启生效。"""
        api = ApiManager()
        llm = self.modules.get("llm")
        if llm is None:
            return
        try:
            cfg = api.get_llm_config("main")
            llm.setting(cfg["api_base"], cfg["api_key"], cfg["model_name"])
            self._log.info("已热更新 LLM 配置")
        except Exception as e:
            self._log.warning("LLM 配置热更新失败: %s", e)

    def _on_asr_result(self, text):
        filter = self.modules.get("filter")
        if filter and not filter.emo(text):
            self._log.info("消息被过滤")
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

    def _build_messages(self, text, image_data_url):
        messages = []
        with self._memory_lock:
            if initial_memory := getattr(self, "_initial_messages", None):
                messages.extend(initial_memory)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": text},
            ],
        })
        return messages

    def _stream_reply(self, messages):
        """流式获取 LLM 回复并同步播报/推送到界面。

        :return: (full_response, interrupted)
        """
        llm = self.modules["llm"]
        tts = self.modules["tts"]

        full_response = ""
        interrupted = False
        self._llm_busy = True
        is_first_chunk = True
        req_start = time.time()
        self._ws_send("send_speak_start")

        try:
            for chunk_text in llm.send_llm_stream(messages):
                if self.state.consume_interrupt():
                    tts.stop()
                    self._log.info("回复被语音打断")
                    self._ws_send("send_status", "被语音打断")
                    interrupted = True
                    break

                if not chunk_text.strip():
                    continue

                if is_first_chunk:
                    self._log.info("LLM 首字 (%.2fs)", time.time() - req_start)
                self._ws_send("send_speak_chunk", chunk_text)
                if self._tts_enabled:
                    tts.speak(chunk_text, interrupt=is_first_chunk, fast=is_first_chunk)
                full_response += chunk_text
                is_first_chunk = False
        finally:
            self._llm_busy = False
            self._ws_send("send_speak_end")

        return full_response, interrupted

    def _process_voice_request(self, request):
        text = request.payload["text"]
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log.info("用户: %s", text)

        memory = self.modules["memory"]
        shot = self.modules["shot"]

        image_data = shot()
        image_data_url = f"data:image/jpeg;base64,{image_data}"

        messages = self._build_messages(text, image_data_url)

        self._log.info("助手思考中 (流式播报)...")
        self._ws_send("send_status", "助手思考中")

        full_response, interrupted = self._stream_reply(messages)
        if interrupted:
            return

        if full_response:
            self._append_chat({"role": "user", "content": text, "time": date_time})
            self._append_chat({"role": "assistant", "content": full_response, "time": date_time})
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
            self._log.error("记忆整理失败: %s", e)
        finally:
            self._compacting = False

    def _process_timer_request(self):
        self._log.info("[主动触发] 静音已达阈值")
        self._ws_send("send_status", "多咪主动触发对话")
        shot = self.modules["shot"]

        image_data = shot()
        image_data_url = f"data:image/jpeg;base64,{image_data}"

        messages = self._build_messages("瞧", image_data_url)

        try:
            full_response, interrupted = self._stream_reply(messages)
        except Exception as e:
            self._log.error("[LLM错误] 自主提问失败: %s", e)
            self.state.pause_auto_trigger()
            self.state.mark_trigger()
            self._llm_busy = False
            return

        if interrupted:
            return

        if full_response:
            date_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self._append_chat({"role": "user", "content": "瞧", "time": date_time})
            self._append_chat({"role": "assistant", "content": full_response, "time": date_time})
            memory = self.modules["memory"]
            memory.save_shot({"shot": image_data_url, "time": date_time})
            self.state.resume_auto_trigger()
        else:
            self._log.info("[主动触发] LLM返回为空，暂停自主提问")
            self.state.pause_auto_trigger()

        self.state.mark_trigger()

    def run(self):
        self._running = True

        pipeline = Pipeline(self.state, self.modules, self.make_memory)
        initial_memory = self.modules.get("memory")
        if initial_memory:
            self._initial_messages = pipeline.load_history(initial_memory)

        if self._mic_enabled:
            self.modules["asr"].start_streaming(
                on_interrupt=self._on_vad_interrupt,
                on_result=self._on_asr_result,
            )
        else:
            self._log.info("麦克风处于关闭状态，跳过启动")

        self._log.info("并行模式: ASR(流式VAD) + TTS(句级队列)")
        self._log.info("系统启动！")
        self._ws_send("send_status", "系统启动！")

        while self._running:
            try:
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
                self._log.error("运行错误: %s", e)
                time.sleep(1)

    def shutdown(self):
        self._running = False
        if self.ws_server is not None:
            self.ws_server.shutdown()
        if "asr" in self.modules:
            self.modules["asr"].stop_streaming()
        if "tts" in self.modules:
            self.modules["tts"].shutdown()
