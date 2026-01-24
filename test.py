import sounddevice as sd
import numpy as np
from funasr import AutoModel

fs = 16000
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