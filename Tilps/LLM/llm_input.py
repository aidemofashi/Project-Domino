import openai
import time

class LLMinput:
    def setting(self, api_base, api_key, model_name):
        openai.api_base = api_base
        openai.api_key = api_key
        self.model_name = model_name

    @staticmethod
    def _merge_system_messages(messages):
        """合并连续的多条 system 消息，避免严格模板（如 gemma3）报角色不交替错误"""
        merged = []
        for msg in messages:
            if msg.get("role") == "system" and merged and merged[-1].get("role") == "system":
                merged[-1]["content"] = str(merged[-1].get("content", "")) + "\n" + str(msg.get("content", ""))
            else:
                merged.append(dict(msg))
        return merged

    def _retry(self, func, max_retries=3, delay=2):
        """重试机制，遇到 503 自动重试"""
        for attempt in range(max_retries):
            try:
                return func()
            except openai.error.ServiceUnavailableError:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise

    def send_llm(self,messages):
        messages = self._merge_system_messages(messages)
        response = self._retry(lambda: openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.6,
            max_tokens=1500,
            stream=False,
        ))
        if response and response.choices:
            return response.choices[0].message.content
        return ""

    def send_llm_stream(self, messages):
        messages = self._merge_system_messages(messages)
        response = self._retry(lambda: openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.6,
            max_tokens=500,
            stream=True,
            ))
        
        buffer = ""
        preview_yielded = False
        # 预览块之后：遇到逗号/句号（中英文）即切段输出
        split_marks = ["。", "，", ",", "."]

        for chunk_data in response:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    buffer += content

                    if not preview_yielded and len(buffer) >= 6:
                        # 预览块：最早凑满 6 个字符立即输出
                        yield buffer.strip()
                        buffer = ""
                        preview_yielded = True
                    elif preview_yielded and any(m in buffer for m in split_marks):
                        # 预览块之后：遇到逗号/句号即切段
                        yield buffer.strip()
                        buffer = ""

        if buffer.strip():
            yield buffer.strip()
        print()