import os
import json
import time
import keyboard

from funasr import AutoModel

from Tilps.Audio.audio_input import AudioInput 
from Tilps.LLM.filter import Filter
from Tilps.LLM.llm_input import LLMinput
from Tilps.Audio.audio_output import AudioOutput
from Tilps.ASR.asr import ASR

dir = "./"
model_dir = "./models/SenseVoiceSmall"  
vad_model_dir = "./models/speech_fsmn_vad_zh-cn-16k-common-pytorch"

LLM_CONFIG = {
    "api_base": "https://api.deepseek.com/v1",
    "api_key": os.getenv("deepseek_api"),
    "model_name": "deepseek-chat"
}

llm = LLMinput()
llm.setting(LLM_CONFIG["api_base"], LLM_CONFIG["api_key"], LLM_CONFIG["model_name"])
TTS_api = os.getenv('ALI_API')
asr_setting = {"model":model_dir,"vad_model":vad_model_dir,"device":"cuda","disable_pbar":True,"disable_update":True,"local_files_only":True}
# 初始化模型:
AudioOutput.input_api(TTS_api)
audio_output=AudioOutput()
ASR.set(asr_setting)

print("\n提示：按[空格]键开始")
keyboard.wait('space')

while True:
    try:
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        audio_data = AudioInput.record()
        print("正在识别...")
        res = ASR.audio_input(input_audio_data=audio_data,lang="zh")
        if not res or not res[0].get('text'):
            print("未检测到有效声音")
            continue
        if keyboard.is_pressed("esc"):
            exit
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在退出...")
        break  # 添加这行来退出循环
    except:
        print("声音输入出错")
        
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
        chat.append({"role": "user", "content": res[0]['text'] + "/no_think"+"{data_time}","time" : date_time})
        # 讲识别的发送到llm
        try:
            response = llm.send_llm(chat)
            if response:
                try:
                    print("\n说话中...")
                    audio_output.text_to_speech(response)
                    chat.append({"role": "assistant","content": response,"time" : date_time})
                except:
                    print("说话出问题")
        except:
            print("模型出问题")
    else:
        print(res[0]['text'])
    # 3. 保存
        with open(audio_input_file, 'w', encoding='utf-8') as file:
            json.dump(contents, file, ensure_ascii=False, indent=4)
        with open(chat_file, 'w', encoding='utf-8') as file:
            json.dump(chat, file, ensure_ascii=False, indent=4)