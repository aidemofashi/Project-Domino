import openai
# 配置本地 API 地址
openai.api_base = "http://192.168.48.1:8888/v1" # LM Studio 默认端口
openai.api_key = "" # 可随意填写，本地不会校验
# 调用本地模型
response = openai.ChatCompletion.create(
   model="qwen/qwen3-4b-2507b", # 必须与 LM Studio 中加载的模型名称一致
   messages=[{"role": "system", "content": "真的吗"}],
   temperature=0.6,
   max_tokens=1000,
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

print()  # 换行
# 如果后续要用完整回复，可用 full_reply