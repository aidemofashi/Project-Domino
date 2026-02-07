import openai
# 本地 API 地址
openai.api_base = "http://192.168.111.1:8888/v1" # LM Studio 默认端口
openai.api_key = " " # 可随意填写，本地不会校验

class llm_input:
    def send_llm(input):
            # input 应该是列表，且内部是 dict
        if not isinstance(input, list):
            raise TypeError("input(messages) 应为 list")
        if not all(isinstance(msg, dict) for msg in input):
            raise TypeError("messages 每项都需为 dict, 请检查 chat.json 数据格式！")
        response = openai.ChatCompletion.create(
        model="qwen/qwen3-4b-2507b", # 必须与 LM Studio 中加载的模型名称一致
        messages=input,
        temperature=0.6,
        max_tokens=500,
        stream=True,
        )
        full_reply = ""
        for chunk in response:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_reply += content

        print()
        return full_reply
        # 如果后续要用完整回复，可用 full_reply