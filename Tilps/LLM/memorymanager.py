import json
import os
import time
from Tilps.LLM.llm_input import LLMinput
CHAT_FILE = "./chat.json" 
SHOT_FILE = "./shot.json"
MEMORY_FILE = "./memorise.json"
CHARACTER_SETTING = "./character.json"
PROMOTE = ""
LLM_CONFIG = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("ALI_API"),
    "model_name": "qwen3-max"
}
llm_send = LLMinput()
llm_send.setting(LLM_CONFIG["api_base"], LLM_CONFIG["api_key"], LLM_CONFIG["model_name"])

class MemoryManager:
    @classmethod
    def load_history(cls):  #加载对话记录
        chat = []
        shot = []
        if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
            with open(CHAT_FILE, 'r', encoding='utf-8') as f:
                chat = json.load(f)
        if os.path.exists(SHOT_FILE) and os.path.getsize(SHOT_FILE) > 0:
            with open(SHOT_FILE, 'r', encoding='utf-8') as f:
                shot = json.load(f)
        return chat,shot
    
    @classmethod
    def load_character_setting(cls):
        character = []
        if os.path.exists(CHARACTER_SETTING) and os.path.getsize(CHARACTER_SETTING) > 0:
            with open(CHARACTER_SETTING,'r',encoding='utf-8') as f:
                character = json.load(f)
        return character

    @classmethod
    def load_memorise(cls):
        memorise = []
        if os.path.exists(MEMORY_FILE) and os.path.getsize(MEMORY_FILE) > 0:  
            with open(MEMORY_FILE,'r',encoding='utf-8') as f:
                memorise = json.load(f)
        return memorise
    
    @classmethod
    def save_chat(cls, chat_record):
        tmp, _ = cls.load_history()
        tmp.append(chat_record)   # chat_record 是单个字典
        with open(CHAT_FILE, 'w', encoding='utf-8') as f:
            json.dump(tmp, f, ensure_ascii=False, indent=4)

    @classmethod
    def save_shot(cls, shot_record):
        _, tmp = cls.load_history()
        tmp.append(shot_record)   # shot_record 是单个字典
        with open(SHOT_FILE, 'w', encoding='utf-8') as f:
            json.dump(tmp, f, ensure_ascii=False, indent=4)

    @classmethod
    def save_memory(cls, memory_record):
        tmp = cls.load_memorise()
        tmp.append(memory_record)  # memory_record 可以是字符串或字典
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(tmp, f, ensure_ascii=False, indent=4)


    def chat_worker(self,chat):
        memorise=""
        print("正在整理记忆...")
        def add(chat):
            character = self.load_character_setting()
            memory = self.load_memorise()
            messages = []
            if character:
                messages.append(character)
            if memory: 
                messages.append({
                    "role": "system",
                    "content": f"{json.dumps(memory, ensure_ascii=False)}"
                })
            if chat:
                messages.extend(chat)
            print(messages)
            return messages

        send_messages = [{"role": "system","content": "你是记忆整理的专家，会根据输入的信息提出人物和事件，每次精简不超过50字。\n"},{"role":"user","content": json.dumps(chat, ensure_ascii=False)}]
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")

        if chat:
            try:
                memorise= llm_send.send_llm(send_messages)
                print("记忆整理发生成功")
                print(memorise)
            except Exception as e:
                print(f"\n记忆整理llm出错: {e}")
                time.sleep(0.5)
        if memorise:
            try:
                self.save_memory({"回忆": memorise, "time": date_time})
            except Exception as e:
                print(f"\n记忆存储出错:{e}")
        chat_full=add(chat)
        print("记忆整理完成！")
        return chat_full