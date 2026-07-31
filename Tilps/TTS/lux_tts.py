import re
import threading
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


class AudioOutput:
    def __init__(self, max_workers=1, threads=8, num_steps=3, speed=0.8):
        from .tts_engine.luxtts import LuxTTS

        self.script_dir = Path(__file__).parent.absolute()
        self.ref_wav = str(self.script_dir / "reference" / "feibi.wav")
        self.threads = threads
        self.num_steps = num_steps
        self.speed = speed

        self._is_running = True
        self.stop_event = threading.Event()
        self._playing = False

        t0 = time.time()
        self._lux = self._init_lux()
        print(f">>> [LuxTTS] 模型加载完成 ({time.time() - t0:.1f}s)")

        self._encoded_prompt = None
        self._encoded_prompt_lock = threading.Lock()
        self._encode_prompt()

        # 生产队列：text → 生成线程
        self.task_queue = queue.Queue()
        # 播放队列：音频数据 → 播放线程
        self._play_queue = queue.Queue()

        for _ in range(max_workers):
            threading.Thread(target=self._gen_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()

    def _init_lux(self):
        from .zipvoice.luxvoice import LuxTTS
        model_dir = str(self.script_dir / "models" / "lux")
        return LuxTTS(model_dir, device='cpu', threads=self.threads)

    def _encode_prompt(self):
        try:
            if Path(self.ref_wav).exists():
                self._encoded_prompt = self._lux.encode_prompt(
                    self.ref_wav, rms=0.01
                )
                print(f">>> [LuxTTS] 参考音频编码完成")
        except Exception as e:
            print(f">>> [LuxTTS] 参考音频编码失败: {e}")

    def _gen_worker(self):
        while self._is_running:
            try:
                item = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            text, steps = item
            if self.stop_event.is_set():
                continue
            try:
                with self._encoded_prompt_lock:
                    encoded = self._encoded_prompt

                final_wav = self._lux.generate_speech(
                    text,
                    encoded,
                    num_steps=steps,
                    speed=self.speed,
                )
                audio = final_wav.numpy().squeeze()
                audio = np.pad(audio, (0, int(48000 * 0.4)))

                self._play_queue.put(audio)
            except Exception as e:
                print(f">>> [LuxTTS] 合成失败: {e}")

    def _play_worker(self):
        while self._is_running:
            try:
                audio = self._play_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio is None:
                break
            try:
                self._playing = True
                sd.play(audio, samplerate=48000)
                sd.wait()
                self._playing = False
            except Exception as e:
                print(f">>> [LuxTTS] 播放失败: {e}")

    def speak(self, text, interrupt=False, fast=False):
        cleaned = text.strip()
        if not cleaned:
            return

        # 去掉 tokenizer 无法处理的字符（emoji、特殊符号等）
        cleaned = re.sub(
            r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？,.!?\n ]',
            '', cleaned
        ).strip()
        if not cleaned:
            return

        # vocoder kernel=7 需要至少 ~4 个字产生的帧数，短于 4 个字直接跳过
        if len(cleaned) < 4:
            print(f">>> [LuxTTS] 文本过短跳过 ({len(cleaned)}c): {cleaned}")
            return

        if interrupt:
            self.stop()

        # fast=预览块(2步) > interrupt=首块(2步) > 普通块(self.num_steps)
        steps = 2 if fast else (2 if interrupt else self.num_steps)
        self.task_queue.put((cleaned, steps))

    text_to_speech = speak

    def stop(self):
        if self._playing:
            sd.stop()
            self._playing = False
        self.stop_event.set()
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
            except queue.Empty:
                break
        self.stop_event.clear()

    def shutdown(self):
        self._is_running = False
        self.stop()
        self.task_queue.put(None)
        self._play_queue.put(None)
