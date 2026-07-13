import json
import os
import time
from Tilps.Core.apim import ApiManager

CHAT_FILE = "./Data/chat.json" 
SHOT_FILE = "./Data/shot.json"
MEMORY_FILE = "./Data/memorise.json"
CHARACTER_SETTING = "./Data/character.json"

api = ApiManager()
llm_send = api.create_llm("memory")


class MemoryManager:
    @classmethod
    def load_history(cls):
        chat = []
        shot = []
        if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
            with open(CHAT_FILE, 'r', encoding='utf-8') as f:
                chat = json.load(f)
        if os.path.exists(SHOT_FILE) and os.path.getsize(SHOT_FILE) > 0:
            with open(SHOT_FILE, 'r', encoding='utf-8') as f:
                shot = json.load(f)
        return chat, shot

    @classmethod
    def load_character_setting(cls):
        character = []
        if os.path.exists(CHARACTER_SETTING) and os.path.getsize(CHARACTER_SETTING) > 0:
            with open(CHARACTER_SETTING, 'r', encoding='utf-8') as f:
                character = json.load(f)
        return character

    @classmethod
    def load_memorise(cls):
        memorise = []
        if os.path.exists(MEMORY_FILE) and os.path.getsize(MEMORY_FILE) > 0:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                memorise = json.load(f)
        return memorise

    @classmethod
    def save_chat(cls, chat_record):
        tmp, _ = cls.load_history()
        tmp.append(chat_record)
        with open(CHAT_FILE, 'w', encoding='utf-8') as f:
            json.dump(tmp, f, ensure_ascii=False, indent=4)

    @classmethod
    def save_shot(cls, shot_record):
        _, tmp = cls.load_history()
        tmp.append(shot_record)
        with open(SHOT_FILE, 'w', encoding='utf-8') as f:
            json.dump(tmp, f, ensure_ascii=False, indent=4)

    @classmethod
    def save_memory(cls, memory_record):
        tmp = cls.load_memorise()
        tmp.append(memory_record)
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(tmp, f, ensure_ascii=False, indent=4)

    def chat_worker(self, chat):
        memorise = ""
        print("正在整理记忆...")

        def add(chat):
            character = self.load_character_setting()
            memory = self.load_memorise()
            messages = []
            for entry in character:
                if entry.get("use") == "character":
                    messages.append({"role": entry.get("role", "system"), "content": entry.get("content", "")})
                    break
            if memory:
                messages.append({
                    "role": "system",
                    "content": f"{json.dumps(memory, ensure_ascii=False)}"
                })
            if chat:
                messages.extend(chat)
            print(messages)
            return messages

        memory_prompt = {"role": "system", "content": "你是记忆提取专家。从对话中提取关于用户的信息，并且将每一条对话总结成对应每一条不超过10个字的回忆。注意：只输出纯文字，不要输出任何其他非普通书写文本格式注释的标记。"}
        character = self.load_character_setting()
        for entry in character:
            if entry.get("use") == "memory":
                memory_prompt = {"role": entry.get("role", "system"), "content": entry.get("content", memory_prompt["content"])}
                break
        send_messages = [
            memory_prompt,
            {"role": "user", "content": json.dumps(chat, ensure_ascii=False)}
        ]
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")

        if chat:
            try:
                memorise = llm_send.send_llm(send_messages)
                print("记忆整理成功")
                print(memorise)
            except Exception as e:
                print(f"\n记忆整理llm出错: {e}")
                time.sleep(0.5)

        if memorise:
            try:
                self.save_memory({"回忆": memorise, "time": date_time})
            except Exception as e:
                print(f"\n记忆存储出错:{e}")

        chat_full = add(chat)
        print("记忆整理完成！")
        return chat_full
