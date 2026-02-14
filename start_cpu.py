import os
import json
import time
import keyboard
import sys
import threading
import queue
import numpy as np

# 骨牌组件导入
from Tilps.LLM.filter import Filter
from Tilps.LLM.llm_input import LLMinput
from Tilps.Audio.audio_output import AudioOutput
from Tilps.ASR.asr import ASR
from Tilps.VAD.vad_vosk import AudioInput  # 确保你已将上一步给你的 Vosk 代码保存为此路径

# --- 配置区域 ---
MODEL_DIR = "./models/SenseVoiceSmall"
# 注意：因为使用了 Vosk 做 VAD，ASR 内部的 vad_model 可以设为 None 以提升速度
ASR_SETTING = {
    "model": MODEL_DIR,
    "vad_model": None, 
    "device": "cuda", # 如果没有 GPU 请改为 "cpu"
    "disable_pbar": True,
    "disable_update": True,
    "local_files_only": True
}

LLM_CONFIG = {
    "api_base": "https://api.deepseek.com/v1",
    "api_key": os.getenv("deepseek_api"),
    "model_name": "deepseek-chat"
}

AUDIO_INPUT_FILE = "output_full.json"
CHAT_FILE = "chat.json"

# --- 初始化 ---
def initialize():
    # 环境变量获取 TTS API
    tts_api = os.getenv("ALI_API")
    
    # 组件实例化
    llm = LLMinput()
    llm.setting(LLM_CONFIG["api_base"], LLM_CONFIG["api_key"], LLM_CONFIG["model_name"])
    
    AudioOutput.input_api(tts_api)
    audio_out = AudioOutput()
    ASR.set(ASR_SETTING)
    
    return llm, audio_out

def load_history():
    if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
        with open(CHAT_FILE, 'r', encoding='utf-8') as f:
            chat = json.load(f)
    else:
        chat = []
    return chat

def save_history(chat):
    with open(CHAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chat, f, ensure_ascii=False, indent=4)

def vosk_worker(res_queue):
    """专门负责运行 Vosk 并在拿到结果后塞入队列"""
    while True:
        # 假设你的 AudioInput.record() 已经改成了返回识别后的 String 文本
        audio_data = AudioInput.record() 
        if np.any(audio_data > 0):
            res_queue.put(audio_data)

# --- 主程序逻辑 ---
def main():
    llm_input, audio_output = initialize()
    # 初始化队列
    res_queue = queue.Queue()
    
    print("\n" + "="*30)
    print("Vosk 线程模式已就绪")
    print("提示：按 [空格] 键开始运行，按 [Esc] 退出")
    print("="*30)
    
    keyboard.wait('space')
    
    # 启动后台 Vosk 线程
    t = threading.Thread(target=vosk_worker, args=(res_queue,), daemon=True)
    t.start()
    
    print("\n>>> 系统启动！请直接说话...")

    while True:
        try:
            if keyboard.is_pressed("esc"):
                break
            # 关键修改：从队列获取结果（阻塞直到 Vosk 线程塞入新文本）
            # timeout=0.1 是为了让循环能回到顶部检查 esc 键
            try:
                user_text = ASR.audio_input(input_audio_data=res_queue.get(timeout=0.1), lang="auto")
            except queue.Empty:
                continue

            date_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{date_time}] 用户: {user_text}")

            # 4. 过滤与对话处理
            chat= load_history()

            if Filter.emo(user_text[0]['text']) != False:
                if not res_queue.empty(): # 发现新输入
                    audio_output.stop()  
                chat.append({
                    "role": "user", 
                    "content": f"{user_text}/no_think", 
                    "time": date_time
                })
                print("思考中...")
                response = llm_input.send_llm(chat)
                
                if response:
                    s = threading.Thread(
                            target=audio_output.text_to_speech, # 传入函数名，不带括号
                            args=(response,),                   # 参数放在 args 元组里
                            daemon=True)
                    s.start()
                    print("说话中...")
                    
                    chat.append({
                        "role": "assistant", 
                        "content": response, 
                        "time": date_time
                    })
                    save_history(chat)
            else:
                print(">>> 消息被 Filter 过滤。")

        except Exception as e:
            print(f"\n[运行错误]: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()