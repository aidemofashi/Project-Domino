import os
import json
import time
import keyboard
#模型
from funasr import AutoModel
#文件
from audio_input import AudioInput  #语音输入  
from filter import Filter
from llm_input import llm_input

#路径处理,当前先不用
dir = "./"
s = 0
model_dir = "./models/SenseVoiceSmall"  
vad_model_dir = "./models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
# 初始化模型 (建议加上 disable_update=True 跳过每次启动的检查)
model = AutoModel(
    model= model_dir,
    vad_model=vad_model_dir,
    device="cpu",
    disable_pbar=True,
    disable_update=True,
    local_files_only=True
    )

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
    res = model.generate(input=audio_data, cache={}, language="zh")
    
    if res and res[0]['text'].strip():
        text = res[0]['text']
        print(f"识别结果：{text}")
        print()
        # 1. 准备数据
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        res[0]['datetime'] = time.strftime("%Y-%m-%d %H:%M:%S")
        res[0].pop("key", None)

        # 2. 读取并累加
        audio_input_file = "output_full.json"
        chat_file = "chat.json"

        if os.path.exists(audio_input_file) and os.path.getsize(audio_input_file) and os.path.exists(chat_file) and os.path.getsize(chat_file) > 0:
            with open(audio_input_file, 'r', encoding='utf-8') as file:
                contents = json.load(file)
            with open(chat_file, 'r', encoding='utf-8') as file:
                chat = json.load(file)
        else:
            contents = []
            chat = []
        if Filter.emo(res[0]['text']) != False:
            contents.append(res[0])
            # chat.append('"role": "Aidemofashi", "content": '+ res[0]['text'] + "/no_think")  # 错误
            chat.append({"role": "user", "content": res[0]['text'] + "/no_think","time" : date_time})      # 正确
            # 讲识别的发送到llm
            response = llm_input.send_llm(chat)
            chat.append({"role": "assistant","content": response,"time" : date_time})
        # 3. 保存
        with open(audio_input_file, 'w', encoding='utf-8') as file:
            json.dump(contents, file, ensure_ascii=False, indent=4)
        with open(chat_file, 'w', encoding='utf-8') as file:
            json.dump(chat, file, ensure_ascii=False, indent=4)