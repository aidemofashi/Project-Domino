import sounddevice as sd
import numpy as np
import os
import json
from funasr import AutoModel
import time
#路径处理

dir = "./"

# 声音处理
fs = 16000  #声音采样
duration = 5  # 录制 5 秒
print("正在录音 (请说话)...")

# 录音逻辑
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("录音结束，正在识别...")

# 关键修正点：将二维数组转为一维
# sounddevice 录制的是 (n, 1)，模型需要的是 (n,)
audio_data = audio.flatten()

# 初始化模型 (建议加上 disable_update=True 跳过每次启动的检查)
model = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)

# 进行识别
res = model.generate(input=audio_data, cache={}, language="zh")

if res:
    print("\n识别结果：")
    print(res[0]['text'])

if res:
    # 1. 记录当前时间
    # 秒级浮点数 (例如: 1706096633.123)
    unix_time = time.time() 
    # 格式化时间字符串 (例如: "2026-01-24 20:45:00")
    readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unix_time))

    # 2. 将新字段注入到结果字典中
    res[0]['datetime'] = readable_time
    res[0].pop("key", None)
    # 3. 保存文件
    with open(os.path.join(dir, "output_full.json"), mode='w', encoding='utf-8') as file:
        # res[0] 现在已经包含了 text, timestamp 和 datetime
        json.dump(res[0], file, ensure_ascii=False, indent=4)