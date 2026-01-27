import os
import json
from funasr import AutoModel
import time
from audio_input import AudioInput
import keyboard

#路径处理,当前先不用
dir = "./"
s = 0
# 初始化模型 (建议加上 disable_update=True 跳过每次启动的检查)
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    device="cpu",
    disable_pbar=True,
    disable_update=True)

print("\n提示：按[空格]键开始")
keyboard.wait('space')

while True:
    # 这里会进入监听状态，直到检测到声音并录制完毕
    audio_data = AudioInput.record()
    
    if audio_data.size == 0:
        print("未检测到有效声音")
        continue

    print("正在识别...")
    # SenseVoiceSmall 模型处理
    res = model.generate(input=audio_data, cache={}, language="auto")
    
    if res and res[0]['text'].strip():
        text = res[0]['text']
        print(f"识别结果：{text}")

        # 1. 准备数据
        res[0]['datetime'] = time.strftime("%Y-%m-%d %H:%M:%S")
        res[0].pop("key", None)

        # 2. 读取并累加
        filename = "output_full.json"
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r', encoding='utf-8') as file:
                contents = json.load(file)
        else:
            contents = []

        contents.append(res[0])

        # 3. 保存
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(contents, file, ensure_ascii=False, indent=4)
