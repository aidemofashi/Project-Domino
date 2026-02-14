import edge_tts
import asyncio
import sounddevice as sd
import numpy as np
import re
import io
import pydub

class AudioOutput:
    def __init__(self):
        print(">>> 初始化 Edge-TTS 输出...")
        self.voice = "zh-CN-XiaoxiaoNeural"
        self.rate = "+10%"
        # 标记是否需要停止播放
        self._is_stopping = False 
        print(f">>> 当前使用 Edge-TTS 音色: {self.voice}")

    @staticmethod
    def input_api(api):
        pass

    async def _generate_and_play(self, text):
        self._is_stopping = False  # 开始前重置停止状态
        # 清洗文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,.? !，。！？]', '', text)
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        audio_data = b""
        async for chunk in communicate.stream():
            # 如果在获取流的过程中点击了停止，直接退出
            if self._is_stopping:
                return
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        if not audio_data or self._is_stopping:
            return

        # 转换音频
        audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        samples /= (2**15) 
        # 播放
        sd.play(samples, samplerate=audio_segment.frame_rate)
        # 使用循环检查来模拟非阻塞等待，这样可以在播放途中响应 stop()
        while sd.get_stream().active:
            if self._is_stopping:
                sd.stop()
                break
            await asyncio.sleep(0.1)

    def text_to_speech(self, text):
        if not text:
            return
        try:
            # 注意：如果在多线程环境（如 GUI）下，这里可能需要不同的异步处理方式
            asyncio.run(self._generate_and_play(text))
        except Exception as e:
            print(f">>> Edge-TTS 播放失败: {e}")

    def stop(self):
        self._is_stopping = True
        sd.stop()  # 立即停止声卡输出