import sounddevice as sd
import numpy as np
import os
import json
from funasr import AutoModel
import time
#路径处理,当前先不用
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
model = AutoModel(model="iic/SenseVoiceSmall",vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", device="cpu", disable_update=True)

# 启动模型
res = model.generate(input=audio_data, cache={}, language="zh")

if res:
    print("\n识别结果：")
    print(res[0]['text'])

if res:
    # 1. 准备数据
    res[0]['datetime'] = time.strftime("%Y-%m-%d %H:%M:%S")
    res[0].pop("key", None)

    # 2. 读取并累加 (最少修改点)
    if os.path.exists("output_full.json") and os.path.getsize("output_full.json") > 0:
        with open("output_full.json", 'r', encoding='utf-8') as file:
            contents = json.load(file)
    else:
        contents = [] # 如果文件不存在或为空，初始化列表

    contents.append(res[0]) # 使用 append 实现累加

    # 3. 保存
    with open("output_full.json", 'w', encoding='utf-8') as file:
        json.dump(contents, file, ensure_ascii=False, indent=4)