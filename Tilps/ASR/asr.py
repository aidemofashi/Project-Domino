import sounddevice as sd
import numpy as np
import collections
import time
import threading
from funasr import AutoModel


class ASR:
    def __init__(self):
        self.model = None
        self.vad_model = None
        self._vad_cache = {}
        self._on_interrupt = None
        self._on_result = None
        self._listening = False
        self._thread = None
        self._input_device = None
        self.fs = 16000
        self.BLOCK_SIZE = 2000

    def set(self, setting):
        self._input_device = setting.pop("input_device", None)
        if self._input_device is None:
            devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
            if devs:
                self._input_device = devs[0]["index"]
                print(f'[ASR] Input device: {self._input_device} ({devs[0]["name"]})')
        else:
            print(f"[ASR] Input device: {self._input_device} (config)")
        vad_cfg = setting.pop("vad_model", "fsmn-vad")
        self.model = AutoModel(**setting)
        if vad_cfg:
            try:
                self.vad_model = AutoModel(model=vad_cfg, disable_update=True, local_files_only=False, disable_pbar=True)
                print(f"[ASR] VAD model loaded: {vad_cfg}")
            except Exception as e:
                print(f"[ASR] VAD model unavailable ({e}), using energy-based fallback")
                self.vad_model = None

    def start_streaming(self, on_interrupt=None, on_result=None):
        """流程启动前准备配置"""
        self._on_interrupt = on_interrupt
        self._on_result = on_result
        self._listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[ASR] 工作流已启动")

    def stop_streaming(self):
        """ 停止监听循环 """
        self._listening = False

    def _listen_loop(self):
        """主监听循环"""
        PRE_ROLL_SEC = 0.5
        MAX_SILENCE_BLOCKS = 8
        pre_roll_chunks = int(PRE_ROLL_SEC * self.fs / self.BLOCK_SIZE)
        ring_buffer = collections.deque(maxlen=pre_roll_chunks)

        while self._listening:
            self._vad_cache = {}  # 最好不要动，使用蓝牙耳机集成麦克风有时候会卡死，这个对问题有帮助  
            recording = []
            is_triggered = False
            silence_counter = 0
            done = False
            speech_start_fired = False

            def callback(indata, frames, time_info, status):
                nonlocal is_triggered, silence_counter, done, recording, speech_start_fired

                if status:
                    print(f"[ASR] Status: {status}")

                is_speech, is_end = self._detect_speech(indata)

                if is_speech:
                    if not speech_start_fired:
                        speech_start_fired = True
                        if self._on_interrupt:
                            self._on_interrupt()
                    if not is_triggered:
                        is_triggered = True
                        silence_counter = 0
                        for buf in ring_buffer:
                            recording.append(buf.copy())
                    recording.append(indata.copy())
                    if is_end:
                        done = True
                        raise sd.CallbackStop
                else:
                    if is_triggered:
                        if is_end:
                            done = True
                            raise sd.CallbackStop
                        silence_counter += 1
                        recording.append(indata.copy())
                        if silence_counter >= MAX_SILENCE_BLOCKS:
                            done = True
                            raise sd.CallbackStop
                    else:
                        ring_buffer.append(indata.copy())

            try:
                with sd.InputStream(
                    samplerate=self.fs, channels=1, callback=callback,
                    blocksize=self.BLOCK_SIZE, dtype="float32",
                    device=self._input_device,
                ):
                    start_time = time.time()
                    max_duration = 30
                    while not done and self._listening:
                        sd.sleep(100)
                        if time.time() - start_time >= max_duration:
                            break
            except sd.CallbackStop:
                pass
            except Exception as e:
                print(f"[ASR] Error: {e}")
                time.sleep(0.1)
                continue

            if not recording or not is_triggered:
                continue

            post_roll = np.zeros((int(self.fs * 0.1), 1), dtype="float32")
            recording.append(post_roll)
            audio_data = np.concatenate(recording).flatten()

            try:
                res = self.model.generate(input=audio_data, cache={}, language="auto")
                if res and res[0]["text"].strip():
                    text = res[0]["text"]
                    print(f"\n[ASR] {text}")
                    if self._on_result:
                        self._on_result(text)
            except Exception as e:
                print(f"[ASR] Recognition error: {e}")

    def _detect_speech(self, indata):
        """备选语音检测、在没有vad模型时使用"""
        if self.vad_model is not None:
            try:
                result = self.vad_model.generate(
                    input=indata, is_final=False, cache=self._vad_cache,
                    chunk_size=int(len(indata) / self.fs * 1000)
                )
                if result and len(result) > 0:
                    segments = result[0].get("value", [])
                    for seg in segments:
                        if isinstance(seg, (list, tuple)) and len(seg) == 2:
                            has_speech = seg[0] >= 0
                            has_end = seg[1] >= 0
                            if has_speech or has_end:
                                return has_speech, has_end
            except Exception as e:
                print("VAD error: " + str(e))
                pass
        energy = np.linalg.norm(indata) / np.sqrt(len(indata))
        return energy > 0.015, False

    def audio_input(self, input_audio_data, lang):
        if self.model is None:
            raise RuntimeError("ASR model not initialized")
        res = self.model.generate(input=input_audio_data, cache={}, language=lang)
        if res and res[0]["text"].strip():
            text = res[0]["text"]
            print(f"[ASR] {text}")
            res[0]["datetime"] = time.strftime("%Y-%m-%d %H:%M:%S")
            res[0].pop("key", None)
        return res
