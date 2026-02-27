import genie_tts as genie
import sounddevice as sd
import numpy as np
import re
from pathlib import Path

class AudioOutput:
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.model_dir = self.script_dir / "models" / "feibi" 
        self.ref_wav = self.script_dir / "reference" / "feibi.wav"
        self.character = "feibi"

        try:
            if self.model_dir.exists():
                print(f">>> [TTS] 加载路径模型: {self.model_dir}")
                genie.load_character(
                    character_name=self.character,
                    onnx_model_dir=str(self.model_dir),
                    language='Chinese'
                )
                
                if self.ref_wav.exists():
                    genie.set_reference_audio(
                        character_name=self.character,
                        audio_path=str(self.ref_wav),
                        audio_text="在此之前，请您务必享受旅居拉古娜的时光"
                    )
                
                # --- 彻底移除预热 tts() 调用，这是省内存的关键 ---
                print(f">>> [TTS] 角色 {self.character} 加载成功")

        except Exception as e:
            print(f">>> [TTS] 初始化失败: {e}")

    def clean_text(self, text):
        text = re.sub(r'[\(\uff08].*?[\)\uff09]', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text)
        return text
    
    def input_api(api):
        pass

    def stop(self):
        # 即使是 genie 内部播放，sd.stop() 通常也能关掉大多数 Python 驱动的音频
        sd.stop()
        print(">>> [TTS] 播放停止")

    def text_to_speech(self, text):
        if not text or text.strip() == "":
            return
        cleaned_text = self.clean_text(text)
        self._generate_and_play(cleaned_text)

    def _generate_and_play(self, text):
        try:
            # 改成在里面播放，问题是依旧能打断？
            genie.tts(
                character_name=self.character,
                text=text,
                play=True
            )
        except Exception as e:
            print(f">>> [TTS] 合成失败: {e}")