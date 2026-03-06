import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
import sounddevice as sd
import numpy as np
import re
import threading
import time

class AudioOutput:
    """阿里云语音合成输出模块（稳定版，每次新建连接，同步合成）"""

    _api_key = None  # 类变量，用于存储 API Key

    @classmethod
    def input_api(cls, api_key: str):
        """类方法设置阿里云 API Key"""
        cls._api_key = api_key
        dashscope.api_key = api_key
        print(">>> API Key 已设置")

    def __init__(self):
        print(">>> 初始化阿里云语音合成（稳定版）...")
        self.model = "cosyvoice-v1"
        self.voice = "longmiao"
        self.sample_rate = 24000
        self._is_stopping = False
        print(f">>> 当前使用音色 ID: {self.voice}，采样率: {self.sample_rate} Hz")

    def _synthesize_and_play(self, text: str):
        """
        同步合成完整文本并播放（在子线程中运行）
        """
        self._is_stopping = False

        # 文本清洗：去除括号内的注释及特殊符号
        text = re.sub(r'[\(\uff08].*?[\)\uff09]', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？,.!?]', '', text)

        if not text:
            return

        # 按句子分割，但这里我们一次合成整个文本（因为同步call不支持流式）
        # 如果想分句播放，可以循环创建多个合成器，但会增加延迟
        try:
            # 每次新建合成器
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.PCM_24000HZ_MONO_16BIT
            )
            print(f">>> 合成文本: {text[:30]}...")
            audio_bytes = synthesizer.call(text)  # 同步合成

            if self._is_stopping:
                return

            # 转换为 float32 并播放
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()  # 等待播放完成
            time.sleep(0.2)  # 避免过于紧凑

        except Exception as e:
            print(f">>> 合成失败: {e}")

    def text_to_speech(self, text: str):
        """
        对外接口：将文本合成为语音并播放（在独立线程中执行）
        """
        if not text:
            return

        if not self.__class__._api_key:
            print(">>> 错误：未设置 API Key，请先调用 AudioOutput.input_api()")
            return

        # 在新线程中执行合成与播放，避免阻塞主线程
        play_thread = threading.Thread(target=self._synthesize_and_play, args=(text,), daemon=True)
        play_thread.start()

    def stop(self):
        """停止当前播放"""
        self._is_stopping = True
        sd.stop()
        print(">>> 已停止合成与播放")