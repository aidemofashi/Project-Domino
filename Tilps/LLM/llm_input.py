import openai

class LLMinput:
    def setting(self, api_base, api_key, model_name):
        openai.api_base = api_base
        openai.api_key = api_key
        self.model_name = model_name

    def send_llm_stream(self, messages):
        """流式发送请求，按标点符号切割并立即返回片段"""
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.6,
            max_tokens=500,
            stream=True, # 必须开启流式
        )

        buffer = ""
        # 遇到这些符号就切分，确保 TTS 尽快开始
        delimiters = ["，", "。", "！", "？", "；", "\n"]

        for chunk in response:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True) # 终端实时显示内容
                    buffer += content
                    
                    # 检查是否包含切分符号
                    if any(d in content for d in delimiters):
                        if buffer.strip():
                            yield buffer.strip()
                            buffer = "" 

        # 吐出最后剩下的内容
        if buffer.strip():
            yield buffer.strip()
        print()