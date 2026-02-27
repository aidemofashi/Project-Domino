import os
import json
import time
import keyboard
import sys
import threading
import queue


# 骨牌组件导入
from Tilps.LLM.filter import Filter
from Tilps.LLM.llm_input import LLMinput
from Tilps.TTS.genie import AudioOutput
from Tilps.ASR.asr import ASR
from Tilps.VAD.vad_vosk import AudioInput  # 确保你已将上一步给你的 Vosk 代码保存为此路径
from Tilps.mcp.shot import shot_screen
from Tilps.LLM.view import viewllm_input
from Tilps.LLM.trigger import TimerTrigger

# --- 配置区域 ---
MODEL_DIR = "./models/SenseVoiceSmall"
# 注意：因为使用了 Vosk 做 VAD，ASR 内部的 vad_model 可以设为 None 以提升速度
ASR_SETTING = {
    "model": MODEL_DIR,
    "vad_model": "./models/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
    "device": "cuda", # 如果没有 GPU 请改为 "cpu"
    "disable_pbar": True,
    "disable_update": True,
    "local_files_only": True
}

LLM_CONFIG = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("ALI_API"),
    "model_name": "qwen3-max"
}

VIEW_LLM_CONFIG = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("ALI_API"),
    "model_name": "qwen-vl-max-latest"
}

AUDIO_INPUT_FILE = "output_full.json"
CHAT_FILE = "chat.json" 
SILENCE_TIMEOUT = 15  # 10秒没声音就触发

# --- 初始化 ---
def initialize():
    # 环境变量获取 TTS API
    tts_api = os.getenv("ALI_API")
    # 组件实例化
    llm = LLMinput()
    llm.setting(LLM_CONFIG["api_base"], LLM_CONFIG["api_key"], LLM_CONFIG["model_name"])
    
    AudioOutput.input_api(tts_api)
    audio_out = AudioOutput()
    
    view=viewllm_input()
    view.setting(VIEW_LLM_CONFIG["api_base"], VIEW_LLM_CONFIG["api_key"], VIEW_LLM_CONFIG["model_name"])
    return llm, audio_out ,view

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
        text = AudioInput.record() 
        if text and text.strip():
            res_queue.put(text)

# --- 主程序逻辑 ---
def main():
    llm_input, audio_output, view = initialize()
    timer = TimerTrigger()
    
    # 初始化队列
    res_queue = queue.Queue()
    
    print("\n" + "="*30)
    print("Vosk 线程模式已就绪")
    print(f"静音 {SILENCE_TIMEOUT} 秒后助手会主动说话")
    print("提示：按 [空格] 键开始运行，按 [Esc] 退出")
    print("="*30)
    
    keyboard.wait('space')
    
    # 启动后台 Vosk 线程
    t = threading.Thread(target=vosk_worker, args=(res_queue,), daemon=True)
    t.start()
    
    print("\n>>> 系统启动！请直接说话...")
    
    while True:
        try:
            # if keyboard.is_pressed("esc"):
            #     break

            # 从队列获取结果
            try:
                user_text = ASR.audio_input(input_audio_data=res_queue.get(timeout=0.1), lang="auto")
                
                date_time = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{date_time}] 用户: {user_text}")

                # 过滤与对话处理
                chat = load_history()
                
                if Filter.emo(user_text) != False:
                    image = shot_screen()
                    see = view.llm_view(image_path="shot.png")
                    chat.append({
                        "role": "user", 
                        "content": f"{user_text} \n 屏幕: {see} \n{date_time}", 
                        "time": date_time
                    })

                    print("思考中...")
                    response = llm_input.send_llm(chat)
                    
                    if response:
                        print("说话中...")
                        audio_output.text_to_speech(response)

                        chat.append({
                            "role": "assistant", 
                            "content": response, 
                            "time": date_time
                        })
                        save_history(chat)
                        # 有用户输入，标记活动
                        timer.mark_activity()
                else:
                    print(">>> 消息被 Filter 过滤。")
                    
            except queue.Empty:
                # 没有语音输入时，检查是否需要主动说话
                if timer.should_trigger(SILENCE_TIMEOUT):
                    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[主动触发] 已静音 {SILENCE_TIMEOUT} 秒")
                    
                    # 截屏
                    shot_screen()
                    see = view.llm_view(image_path="shot.png")
                    
                    # 加载历史并添加
                    chat = load_history()
                    chat.append({
                        "role": "assistant", 
                        "content": f"主人没有说话，我看到屏幕上显示的是：{see}", 
                        "time": date_time
                    })
                    print("思考中...")
                    response = llm_input.send_llm(chat)

                    if response:
                        print(f"[助手主动]: {response}")
                        print("说话中...")
                        audio_output.text_to_speech(response)

                        chat.append({
                            "role": "assistant", 
                            "content": response, 
                            "time": date_time
                        })
                        save_history(chat)
                    
                    # 标记已触发
                    timer.mark_trigger()
                    
        except Exception as e:
            print(f"\n[运行错误]: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()