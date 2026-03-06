import edge_tts
import asyncio
import sounddevice as sd
import numpy as np
import re
import io
import pydub
import threading

class AudioOutput:
    def __init__(self):
        print(">>> 初始化 Edge-TTS 输出...")
        self.voice = "zh-CN-XiaoxiaoNeural"
        self.rate = "+10%"
        self._is_stopping = False 
        print(f">>> 当前使用 Edge-TTS 音色：{self.voice}")

    @staticmethod
    def input_api(api):
        pass

    async def _generate_and_play(self, text):
        self._is_stopping = False
        text = re.sub(r'[\(\uff08].*?[\)\uff09]', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text)
        section = text.split(',')
        
        audio_datas = []
        lock = threading.Lock()
        
        async def get_data(index, content):
            try:
                communicate = edge_tts.Communicate(content, self.voice, rate=self.rate)
                audio_data = b""

                # 使用 async for 来迭代异步生成器
                async for chunk in communicate.stream():
                    if self._is_stopping:
                        return
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]

                if not audio_data or self._is_stopping:
                    return
                
                with lock:
                    audio_datas.append((index, audio_data))
            except Exception as e:
                print(f"获取音频数据失败：{e}")

        # 创建异步任务
        tasks = []
        for idx, content in enumerate(section):
            task = asyncio.create_task(get_data(idx, content))
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        if self._is_stopping:
            return
        
        audio_datas.sort(key=lambda x: x[0])
        
        for _, audio_data in audio_datas:
            if self._is_stopping:
                break
            audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            samples /= (2**15) 
            sd.play(samples, samplerate=audio_segment.frame_rate)
            while sd.get_stream().active:
                if self._is_stopping:
                    sd.stop()
                    break
                await asyncio.sleep(0.1)

    def text_to_speech(self, text):
        if not text:
            return
        # 使用 asyncio.run_coroutine_threadsafe 来在事件循环中运行协程
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 如果没有运行中的事件循环，创建一个新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 在新线程中运行事件循环
        def run_async():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._generate_and_play(text))
        
        play_thread = threading.Thread(target=run_async, daemon=True)
        play_thread.start()

    def stop(self):
        self._is_stopping = True
        sd.stop()