import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
import sounddevice as sd
import numpy as np
import re
import time

class AudioOutput:
    def __init__(self):
        print(">>> 初始化语音合成...")
        self.model = "cosyvoice-v1"
        self.voice = "longmiao"
        self.sample_rate = 24000
        print(f">>> 当前使用音色 ID: {self.voice}")

    def input_api(api):
        dashscope.api_key = api

    def text_to_speech(self, text):
        if not text:
            return
        # 仅保留文本清洗：去除特殊符号，保留基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,.? !，。！？]', '', text)
        AudioOutput.input_api
        try:
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.PCM_24000HZ_MONO_16BIT
            )
            audio_bytes = synthesizer.call(text) 
        except Exception as e:
            print(f">>> 语音合成失败: {e}")
            return

        # 播放音频
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio_data, samplerate=self.sample_rate)
        sd.wait()
        time.sleep(0.5)