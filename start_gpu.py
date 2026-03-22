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
from Tilps.TTS.edgetts import AudioOutput  
from Tilps.ASR.asr import ASR
from Tilps.VAD.vad_vosk import AudioInput 
from Tilps.mcp.shot import shot_screen
from Tilps.LLM.trigger import TimerTrigger
from Tilps.TTS.edge_test import tts_test
from Tilps.LLM.memorymanager import MemoryManager

# --- 配置区域 ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/SenseVoiceSmall")

ASR_SETTING = {
    "model": MODEL_DIR,
    "vad_model": None, 
    "device": "cuda",
    "disable_pbar": True,
    "disable_update": True,
    "local_files_only": True,
    "batch_size": 1,
    "max_single_segment_length": 20000,
}

LLM_CONFIG = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("ALI_API"),
    "model_name": "qwen3.5-flash"
}

CHAT_FILE = "chat.json" 
SHOT_FILE = "shot.json"
SILENCE_TIMEOUT = 60  # 静音触发阈值
MAKE_MEMORY =  16

# --- 初始化 ---
def initialize():
    tts_api = os.getenv("ALI_API")
    llm = LLMinput()
    llm.setting(LLM_CONFIG["api_base"], LLM_CONFIG["api_key"], LLM_CONFIG["model_name"])
    
    # 初始化最新的流式 AudioOutput
    audio_out = AudioOutput()
    llm_memory=MemoryManager()
    return llm, audio_out,llm_memory

def vosk_worker(res_queue, audio_output):
    """后台 ASR 录制线程"""
    while True:
        # VAD 内部会根据 audio_output 的状态判断是否需要打断
        audio_data = AudioInput.record(audio_output=audio_output) 
        if audio_data is not None and len(audio_data) > 0:
            res_queue.put(audio_data)

# --- 主程序逻辑 ---
def main():
    ASR.set(ASR_SETTING)
    filter = Filter()
    llm_input, audio_output,llm_memory = initialize()
    timer = TimerTrigger()
    
    res_queue = queue.Queue()
    
    tts_test()

    print("\n" + "="*30)
    print("双向流式模式已就绪 (LLM Stream + TTS Stream)")
    print("提示：按 [空格] 键开始，按 [Esc] 退出")
    print("="*30)
    
    keyboard.wait('space')
    
    t = threading.Thread(target=vosk_worker, args=(res_queue, audio_output), daemon=True)
    t.start()
    print("\n>>> 系统启动！")

    chat = []
    full_chat = llm_memory.chat_worker(chat=None)
    messege = list(full_chat)  # 消息列表用于发送给 LLM
    while True:
        try:
            if len(chat) >= MAKE_MEMORY:
                full_chat = llm_memory.chat_worker(chat)
                messege = list(full_chat)  # 消息列表用于发送给 LLM
                chat = []

            # 1. 检查 ASR 识别结果
            try:
                raw_audio = res_queue.get(timeout=0.05)
                user_text_res = ASR.audio_input(input_audio_data=raw_audio, lang="auto")

                if user_text_res:
                    filtertext = user_text_res[0]['text']
                    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{date_time}] 用户: {filtertext}")

                    if filter.emo(filtertext):
                        # 识别到有效指令，立即停止之前的 TTS 播报
                        audio_output.stop()
                        
                        image_data = shot_screen()
                        image_data_url = f"data:image/jpeg;base64,{image_data}"

                        # 构建上下文
                        user_msg = {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                                {"type": "text", "text": filtertext}
                            ]
                        }
                        messege.append(user_msg)
                        chat.append({"role": "user", "content": [{"type": "text", "text": filtertext}]})

                        print(">>> 助手思考中 (流式播报)...")
                        full_response = ""
                        is_first_chunk = True
                        
                        # --- 核心改进：流式迭代 LLM 输出 ---
                        for chunk_text in llm_input.send_llm_stream(messege):
                            # 将每一段文字（遇到逗号/句号等）立即送去 TTS
                            audio_output.text_to_speech(chunk_text, interrupt=is_first_chunk)
                            full_response += chunk_text
                            is_first_chunk = False # 仅第一段需要 interrupt 以清空旧缓存
                        # 记录回复
                        if full_response:
                            chat.append({"role": "assistant", "content": full_response, "time": date_time})
                            llm_memory.save_shot({"shot": image_data_url, "time": date_time})
                            timer.mark_activity()
                    else:
                        print(">>> 消息被过滤。")
                    
            except queue.Empty:
                # 2. 检查定时主动触发逻辑
                if timer.should_trigger(SILENCE_TIMEOUT):
                    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[主动触发] 静音已达 {SILENCE_TIMEOUT} 秒")
                    
                    image_data = shot_screen()
                    image_data_url = f"data:image/jpeg;base64,{image_data}"
                    
                    prompt_msg = {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": "主人没有说话"}
                        ]
                    }
                    messege.append(prompt_msg)

                    full_response = ""
                    is_first_chunk = True
                    # 同样使用流式处理
                    for chunk_text in llm_input.send_llm_stream(messege):
                        audio_output.text_to_speech(chunk_text, interrupt=is_first_chunk)
                        full_response += chunk_text
                        is_first_chunk = False

                    if full_response:
                        chat.append({"role": "assistant", "content": full_response, "time": date_time})
                        llm_memory.save_shot({"shot": image_data_url, "time": date_time})
                    
                    timer.mark_trigger()
                    
        except Exception as e:
            print(f"\n[运行错误]: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()