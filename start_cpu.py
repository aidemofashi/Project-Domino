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
from Tilps.VAD.vad_vosk import AudioInput  # 确保你已将上一步给你的 Vosk 代码保存为此路径
from Tilps.mcp.shot import shot_screen
from Tilps.LLM.view import viewllm_input
from Tilps.LLM.trigger import TimerTrigger

# --- 配置区域 ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/SenseVoiceSmall")

ASR_SETTING = {
    "model": MODEL_DIR,
    "vad_model": None, 
    "device": "cpu", # 如果没有 GPU 请改为 "cpu"
    "disable_pbar": True,
    "disable_update": True,
    "local_files_only": True,
    "batch_size": 1,
    "max_single_segment_length": 20000,
}

LLM_CONFIG = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("ALI_API"),
    "model_name": "qwen-vl-max-latest"
}

VIEW_LLM_CONFIG = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("ALI_API"),
    "model_name": "qwen-vl-max-latest"
}

AUDIO_INPUT_FILE = "output_full.json"
CHAT_FILE = "chat.json" 
SHOT_FILE = "shot.json"
SILENCE_TIMEOUT = 60  # 10秒没声音就触发

# --- 初始化 ---
def initialize():
    # 环境变量获取 TTS API
    tts_api = os.getenv("ALI_API")
    # 组件实例化
    llm = LLMinput()
    llm.setting(LLM_CONFIG["api_base"], LLM_CONFIG["api_key"], LLM_CONFIG["model_name"])
    
    AudioOutput.input_api(tts_api)
    audio_out = AudioOutput()
    
    #view=viewllm_input()
    #view.setting(VIEW_LLM_CONFIG["api_base"], VIEW_LLM_CONFIG["api_key"], VIEW_LLM_CONFIG["model_name"])
    return llm, audio_out

def load_history():
    chat = []
    shot = []
    if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
        with open(CHAT_FILE, 'r', encoding='utf-8') as f:
            chat = json.load(f)
    if os.path.exists(SHOT_FILE) and os.path.getsize(SHOT_FILE) > 0:
        with open(SHOT_FILE, 'r', encoding='utf-8') as f:
            shot = json.load(f)
    return chat,shot

def save_history(chat):
    with open(CHAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chat, f, ensure_ascii=False, indent=4)

def save_shot(image_data):
    with open(SHOT_FILE, 'w', encoding='utf-8') as f:
        json.dump(image_data, f, ensure_ascii=False, indent=4)


def vosk_worker(res_queue, audio_output):
    while True:
        # 这里把 audio_output 传进去，VAD 内部就能直接控制打断
        audio_data = AudioInput.record(audio_output=audio_output) 
        if audio_data is not None and len(audio_data) > 0:
            res_queue.put(audio_data)

# --- 主程序逻辑 ---
def main():
    ASR.set(ASR_SETTING)
    filter = Filter()
    llm_input, audio_output = initialize()
    timer = TimerTrigger()
    
    # 初始化队列
    res_queue = queue.Queue()
    
    print("\n" + "="*30)
    print("Vosk 线程模式已就绪")
    print(f"静音 {SILENCE_TIMEOUT} 秒后助手会主动说话")
    print("提示：按 [空格] 键开始运行，按 [Esc] 退出")
    print("="*30)
    
    keyboard.wait('space')
    chat,shot = load_history()
    messege,_ = load_history()
    # 启动后台 Vosk 线程
    t = threading.Thread(target=vosk_worker, args=(res_queue, audio_output), daemon=True)
    t.start()
    print("\n>>> 系统启动！请直接说话...")

    while True:
        try:
            # if keyboard.is_pressed("esc"):
            #     break

            # 从队列获取结果
            try:
                user_text = ASR.audio_input(input_audio_data=res_queue.get(timeout=0.05), lang="auto")

                date_time = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{date_time}] 用户: {user_text}")

                # 过滤与对话处理
                print(user_text)
                if user_text:
                    filtertext = user_text[0]['text'] 
                    
                if filter.emo(filtertext):
                    image_data = shot_screen()
                    image_data_url = f"data:image/jpeg;base64,{image_data}"
                    chat.append({
                        "role": "user",
                        "content": [{
                        "type": "text",
                        "text": filtertext}]})
                    messege.append({
                        "role": "user",
                        "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    },
                    {
                        "type": "text",
                        "text": filtertext}]})

                    print("思考中...")
                    response = llm_input.send_llm(messege)
                    
                    if response:
                        print("说话中...")
                        audio_output.text_to_speech(response)
                        chat.append({
                            "role": "assistant", 
                            "content": response, 
                            "time": date_time
                        })
                        shot.append({
                            "shot": image_data_url, 
                            "time": date_time
                        })
                        save_history(chat)
                        save_shot(shot)
                        timer.mark_activity()
                else:
                    print(">>> 消息被 Filter 过滤。")
                    
            except queue.Empty:
                # 没有语音输入时，检查是否需要主动说话
                if timer.should_trigger(SILENCE_TIMEOUT):
                    # 在这里加载聊天历史
                    chat,shot = load_history()
                    messege, _ = load_history()
                    
                    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[主动触发] 已静音 {SILENCE_TIMEOUT} 秒")
                    
                    # 截屏
                    image_data = shot_screen()
                    image_data_url = f"data:image/jpeg;base64,{image_data}"
                    chat.append({
                        "role": "user",
                        "content": [{
                        "type": "text",
                        "text": "主人没有说话"}]})
                    messege.append({
                        "role": "user",
                        "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    },
                    {
                        "type": "text",
                        "text": "主人没有说话"}]})

                    print("思考中...")
                    response = llm_input.send_llm(messege)

                    if response:
                        print(f"[助手主动]: {response}")
                        print("说话中...")
                        audio_output.text_to_speech(response)
                        chat.append({
                            "role": "assistant", 
                            "content": response, 
                            "time": date_time
                        })
                        shot.append({
                            "shot": image_data_url, 
                            "time": date_time
                        })
                        save_history(chat)
                        save_shot(shot)
                    
                    # 标记已触发
                    timer.mark_trigger()
                    
        except Exception as e:
            print(f"\n[运行错误]: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()