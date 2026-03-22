import json
import os
import time
from Tilps.LLM.llm_input import LLMinput
CHAT_FILE = "./Data/chat.json" 
SHOT_FILE = "./Data/shot.json"
MEMORY_FILE = "./Data/memorise.json"
CHARACTER_SETTING = "./Data/character.json"
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

        send_messages = [{"role": "system","content": "你是记忆提取专家。从对话中提取关于用户的事实信息，按以下格式输出：\n\n【固定事实】\n- 身份信息：地域、职业/身份、语言能力等\n- 近期事件：正在做的事、重要日程、压力来源\n- 行为习惯：作息、偏好、常用工具/平台\n- 情绪状态：当前情绪、情绪变化趋势\n- 人际关系：提及的人/角色/群体\n\n【提取规则】\n1. 只提取“事实”，不添加推断和评价\n2. 如果同一信息在不同时间出现，只保留最新或最具体的版本\n3. 每条回忆控制在20字以内，用短横线列出\n4. 如果用户明确表达情绪（如“好累”“谢谢你”），一并记录\n5. 输出格式必须为严格的JSON，结构如下：\n{\n  \"summary\": [\"事实1\", \"事实2\", \"事实3\"],\n  \"tags\": [\"标签1\", \"标签2\", \"标签3\"]\n}\n\n注意：只输出JSON对象，不要输出任何其他文字、注释或Markdown标记。"},{"role":"user","content":json.dumps(chat, ensure_ascii=False)}]
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