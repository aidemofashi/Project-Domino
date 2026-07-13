import time
from Tilps.Core.request import RequestType


class Pipeline:
    def __init__(self, state, modules, make_memory=16):
        self.state = state
        self.modules = modules
        self.make_memory = make_memory
        self.chat = []
        self.messages = []

    def execute(self, request):
        filter = self.modules["filter"]
        llm = self.modules["llm"]
        tts = self.modules["tts"]
        memory = self.modules["memory"]
        shot = self.modules["shot"]

        rtype = request.type
        payload = request.payload

        if rtype == RequestType.TIMER_TRIGGER:
            return self._execute_timer(payload, filter, llm, tts, memory, shot)
        else:
            return self._execute_voice(payload, filter, llm, tts, memory, shot)

    def _execute_voice(self, payload, filter, llm, tts, memory, shot):
        raw_audio = payload["audio_data"]
        res = self.modules["asr"].audio_input(
            input_audio_data=raw_audio, lang="auto"
        )
        if not res:
            return True

        text = res[0]["text"]
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{date_time}] 用户: {text}")

        if not filter.emo(text):
            print(">>> 消息被过滤。")
            self.state.mark_activity()
            return True

        image_data = shot()
        image_data_url = f"data:image/jpeg;base64,{image_data}"

        self._strip_old_images()

        user_msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": text},
            ],
        }
        self.messages.append(user_msg)
        self.chat.append(
            {"role": "user", "content": [{"type": "text", "text": text}]}
        )

        print(">>> 助手思考中 (流式播报)...")
        self._stream_and_speak(llm, tts)

        if self._last_full_response:
            chat_entry = {
                "role": "assistant",
                "content": self._last_full_response,
                "time": date_time,
            }
            self.chat.append(chat_entry)
            memory.save_shot({"shot": image_data_url, "time": date_time})
            self.state.resume_auto_trigger()
        self.state.mark_activity()
        self._compact_memory(memory)
        return True

    def _execute_timer(self, payload, filter, llm, tts, memory, shot):
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[主动触发] 静音已达阈值")

        image_data = shot()
        image_data_url = f"data:image/jpeg;base64,{image_data}"

        self._strip_old_images()

        prompt_msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": "瞧"},
            ],
        }
        self.messages.append(prompt_msg)

        try:
            self._stream_and_speak(llm, tts)
        except Exception as e:
            print(f"\n[LLM错误] 自主提问失败: {e}")
            self.messages.pop()
            self.state.pause_auto_trigger()
            self.state.mark_trigger()
            return True

        if self._last_full_response:
            self.chat.append(
                {
                    "role": "assistant",
                    "content": self._last_full_response,
                    "time": date_time,
                }
            )
            memory.save_shot({"shot": image_data_url, "time": date_time})
            self.state.resume_auto_trigger()
        else:
            print("\n[主动触发] LLM返回为空，暂停自主提问")
            self.messages.pop()
            self.state.pause_auto_trigger()

        self.state.mark_trigger()
        self._compact_memory(memory)
        return True

    def _strip_old_images(self):
        for msg in self.messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                msg["content"] = [item for item in msg["content"] if item["type"] != "image_url"]

    def _stream_and_speak(self, llm, tts):
        self._last_full_response = ""
        is_first_chunk = True

        for chunk_text in llm.send_llm_stream(self.messages):
            if self.state.consume_interrupt():
                tts.stop()
                print("\n>>> 被语音打断")
                return
            tts.text_to_speech(chunk_text, interrupt=is_first_chunk)
            self._last_full_response += chunk_text
            is_first_chunk = False

    def _compact_memory(self, memory):
        if len(self.chat) >= self.make_memory:
            full_chat = memory.chat_worker(self.chat)
            self.messages = list(full_chat)
            self.chat = []

    def load_history(self, memory):
        full_chat = memory.chat_worker(chat=None)
        self.messages = list(full_chat)
