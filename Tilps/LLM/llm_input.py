import openai
import time

class LLMinput:
    def setting(self, api_base, api_key, model_name):
        openai.api_base = api_base
        openai.api_key = api_key
        self.model_name = model_name

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
        response = self._retry(lambda: openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.6,
            max_tokens=500,
            stream=True,
            ))
        
        buffer = ""
        preview_yielded = False
        sentence_end = ["。", "！", "？", "\n"]

        for chunk_data in response:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    buffer += content

                    if not preview_yielded and len(buffer) >= 6:
                        yield buffer.strip()
                        buffer = ""
                        preview_yielded = True

                    hit_end = any(d in content for d in sentence_end)
                    if hit_end and len(buffer) > 30:
                        yield buffer.strip()
                        buffer = ""
                    elif len(buffer) > 50:
                        yield buffer.strip()
                        buffer = ""

        if buffer.strip():
            yield buffer.strip()
        print()