import sounddevice as sd
import numpy as np
import collections
import time
import threading
from funasr import AutoModel


class ASR:
    model = None
    vad_model = None
    _vad_cache = {}
    _on_interrupt = None
    _on_result = None
    _listening = False
    _thread = None

    fs = 16000
    BLOCK_SIZE = 2000

    @classmethod
    def set(cls, setting):
        vad_cfg = setting.pop("vad_model", "fsmn-vad")
        cls.model = AutoModel(**setting)
        if vad_cfg:
            try:
                cls.vad_model = AutoModel(model=vad_cfg, disable_update=True, local_files_only=False, disable_pbar=True)
                print(f"[ASR] VAD model loaded: {vad_cfg}")
            except Exception as e:
                print(f"[ASR] VAD model unavailable ({e}), using energy-based fallback")
                cls.vad_model = None

    @classmethod
    def start_streaming(cls, on_interrupt=None, on_result=None):
        cls._on_interrupt = on_interrupt
        cls._on_result = on_result
        cls._listening = True
        cls._thread = threading.Thread(target=cls._listen_loop, daemon=True)
        cls._thread.start()
        print("[ASR] Streaming started (FunASR VAD + SenseVoice)")

    @classmethod
    def stop_streaming(cls):
        cls._listening = False

    @classmethod
    def _listen_loop(cls):
        PRE_ROLL_SEC = 1.0
        MAX_SILENCE_BLOCKS = 16
        pre_roll_chunks = int(PRE_ROLL_SEC * cls.fs / cls.BLOCK_SIZE)
        ring_buffer = collections.deque(maxlen=pre_roll_chunks)

        while cls._listening:
            cls._vad_cache = {}
            recording = []
            is_triggered = False
            silence_counter = 0
            done = False
            speech_start_fired = False

            def callback(indata, frames, time_info, status):
                nonlocal is_triggered, silence_counter, done, recording, speech_start_fired

                if status:
                    print(f"[ASR] Status: {status}")

                is_speech, is_end = cls._detect_speech(indata)

                if is_speech:
                    if not speech_start_fired:
                        speech_start_fired = True
                        if cls._on_interrupt:
                            cls._on_interrupt()
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
                    samplerate=cls.fs, channels=1, callback=callback,
                    blocksize=cls.BLOCK_SIZE, dtype="float32"
                ):
                    start_time = time.time()
                    max_duration = 30
                    while not done and cls._listening:
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

            post_roll = np.zeros((int(cls.fs * 0.3), 1), dtype="float32")
            recording.append(post_roll)
            audio_data = np.concatenate(recording).flatten()

            try:
                res = cls.model.generate(input=audio_data, cache={}, language="auto")
                if res and res[0]["text"].strip():
                    text = res[0]["text"]
                    print(f"\n[ASR] {text}")
                    if cls._on_result:
                        cls._on_result(text)
            except Exception as e:
                print(f"[ASR] Recognition error: {e}")

    @classmethod
    def _detect_speech(cls, indata):
        """Returns (is_speech: bool, is_speech_end: bool)"""
        if cls.vad_model is not None:
            try:
                result = cls.vad_model.generate(
                    input=indata, is_final=False, cache=cls._vad_cache,
                    chunk_size=int(len(indata) / cls.fs * 1000)
                )
                if result and len(result) > 0:
                    segments = result[0].get("value", [])
                    for seg in segments:
                        if isinstance(seg, (list, tuple)) and len(seg) == 2:
                            has_speech = seg[0] >= 0
                            has_end = seg[1] >= 0
                            if has_speech or has_end:
                                return has_speech, has_end
            except Exception:
                pass
        energy = np.linalg.norm(indata) / np.sqrt(len(indata))
        return energy > 0.015, False

    @classmethod
    def audio_input(cls, input_audio_data, lang):
        if cls.model is None:
            raise RuntimeError("ASR model not initialized")
        res = cls.model.generate(input=input_audio_data, cache={}, language=lang)
        if res and res[0]["text"].strip():
            text = res[0]["text"]
            print(f"[ASR] {text}")
            res[0]["datetime"] = time.strftime("%Y-%m-%d %H:%M:%S")
            res[0].pop("key", None)
        return res
