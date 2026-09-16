import re
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


def _use_project_zipvoice():
    """把 zipvoice 指向项目内副本（tts_engine/luxtts），避免依赖 site-packages"""
    import importlib
    if sys.modules.get("zipvoice") is not None:
        return
    luxtts = importlib.import_module("Tilps.TTS.tts_engine.luxtts")
    sys.modules["zipvoice"] = luxtts


class AudioOutput:
    def __init__(self, max_workers=1, threads=8, num_steps=3, speed=0.8, flush_delay=1.0):
        _use_project_zipvoice()
        from .tts_engine.luxtts.luxvoice import LuxTTS

        self.script_dir = Path(__file__).parent.absolute()
        self.ref_wav = str(self.script_dir / "reference" / "feibi.wav")
        self.threads = threads
        self.num_steps = num_steps
        self.speed = speed
        self.flush_delay = flush_delay

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
        # DirectML/ONNX 会话非线程安全，多 worker 时必须串行化
        self._synth_lock = threading.Lock()

        # 不分段缓冲：所有后续文本累积到这里，直到流暂停再一次性合成
        self._pending_text = ""
        self._pending_lock = threading.Lock()
        self._flush_timer = None

        # 播放顺序控制：保证按入队顺序播放，杜绝乱序
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._expected_seq = 0
        self._pending_play = {}

        for _ in range(max_workers):
            threading.Thread(target=self._gen_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()

    def _init_lux(self):
        _use_project_zipvoice()
        from .tts_engine.luxtts.luxvoice import LuxTTS
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

    def _enqueue(self, text, steps):
        if len(text) < 4:
            print(f">>> [LuxTTS] 文本过短跳过 ({len(text)}c): {text}")
            return
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        self.task_queue.put((text, steps, time.time(), seq))

    def _append_pending(self, text):
        if not text:
            return
        with self._pending_lock:
            self._pending_text += text

    def _schedule_flush(self):
        with self._pending_lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
            self._flush_timer = threading.Timer(self.flush_delay, self._do_flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _do_flush(self):
        """流暂停后，把整段剩余文本作为一段合成（不分段）"""
        with self._pending_lock:
            self._flush_timer = None
            text = self._pending_text
            self._pending_text = ""
        text = text.strip()
        if not text or len(text) < 4:
            return
        self._enqueue(text, self.num_steps)

    def _gen_worker(self):
        while self._is_running:
            try:
                item = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            text, steps, enqueued_at, seq = item
            if self.stop_event.is_set():
                continue
            try:
                with self._synth_lock:
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

                self._play_queue.put((seq, audio, enqueued_at))
            except Exception as e:
                print(f">>> [LuxTTS] 合成失败: {e}")

    def _play_one(self, audio, enqueued_at):
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t}] TTS 播放 (生成延迟 {time.time()-enqueued_at:.2f}s)")
        self._playing = True
        sd.play(audio, samplerate=48000)
        sd.wait()
        self._playing = False

    def _play_worker(self):
        while self._is_running:
            try:
                item = self._play_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            seq, audio, enqueued_at = item
            if seq != self._expected_seq:
                self._pending_play[seq] = (audio, enqueued_at)
                continue
            try:
                self._play_one(audio, enqueued_at)
            except Exception as e:
                print(f">>> [LuxTTS] 播放失败: {e}")
            self._expected_seq += 1
            while self._expected_seq in self._pending_play:
                a, e = self._pending_play.pop(self._expected_seq)
                try:
                    self._play_one(a, e)
                except Exception as ex:
                    print(f">>> [LuxTTS] 播放失败: {ex}")
                self._expected_seq += 1

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

        if interrupt:
            self.stop()

        if fast:
            # 预览：只合成前 4 个字，保证尽快出声
            preview = cleaned[:4]
            rest = cleaned[4:]
            if len(preview) >= 4:
                self._enqueue(preview, 2)
            self._append_pending(rest)
        else:
            self._append_pending(cleaned)

        self._schedule_flush()

    text_to_speech = speak

    def stop(self):
        if self._playing:
            sd.stop()
            self._playing = False
        self.stop_event.set()
        # 取消未触发的合成，清空缓冲，重置顺序编号
        with self._pending_lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
            self._pending_text = ""
        with self._seq_lock:
            self._seq = 0
        self._expected_seq = 0
        self._pending_play.clear()
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