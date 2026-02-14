import sounddevice as sd
import numpy as np
import json
import collections
import time
from vosk import Model, KaldiRecognizer

class AudioInput:
    fs = 16000
    MODEL_PATH = "./models/vosk-model-small-cn-0.22"
    _model = None
    _rec = None

    @classmethod
    def _init_vosk(cls):
        if cls._model is None:
            cls._model = Model(cls.MODEL_PATH)
            cls._rec = KaldiRecognizer(cls._model, cls.fs)

    @classmethod
    def record(cls):
        cls._init_vosk()
        cls._rec.Reset()

        # 环形缓冲区，用于保存说话前 0.5 秒的音频
        pre_roll_len = int(1.5 * cls.fs / 2000)  # 修正为0.5秒
        ring_buffer = collections.deque(maxlen=pre_roll_len)
        
        recording = []
        is_triggered = False
        done = False  # 直接使用变量而不是字典

        print(">>> [系统监听中] ...")

        def callback(indata, frames, time_info, status):
            nonlocal is_triggered, done
            
            # 转换给 Vosk 进行 VAD 判断
            audio_bytes = (indata * 32768).astype(np.int16).tobytes()
            
            # Vosk 流式判断
            if cls._rec.AcceptWaveform(audio_bytes):
                if is_triggered:
                    done = True
                    raise sd.CallbackStop
            else:
                partial = json.loads(cls._rec.PartialResult())
                if partial.get("partial", "").strip() != "":
                    if not is_triggered:
                        is_triggered = True
                        # 触发时，把缓冲区里的“预录”数据塞进去
                        recording.extend(list(ring_buffer))
                        print("[VAD] 检测到语音，开始录音...")

            if is_triggered:
                recording.append(indata.copy())
            else:
                # 还没说话时，音频进环形缓冲区
                ring_buffer.append(indata.copy())

        try:
            # 移除外层 while 循环，直接使用 InputStream
            with sd.InputStream(samplerate=cls.fs, channels=1, callback=callback, 
                              blocksize=2000, dtype='float32'):
                
                # 等待直到录制完成或超时
                start_time = time.time()
                while not done:
                    sd.sleep(100)
                    
                    # 超时处理（可选）
                    if time.time() - start_time >= 10:  # 10秒超时
                        if not is_triggered:
                            print("[VAD] 监听超时，未检测到语音")
                            break
                        else:
                            # 如果已经开始录制但长时间没结束，强制结束
                            print("[VAD] 录音超时，强制结束")
                            break
                            
        except sd.CallbackStop:
            print("[VAD] 录音完成")
        except Exception as e:
            print(f"[VAD] 错误: {e}")
            return np.array([], dtype='float32')

        if not recording:
            return np.array([], dtype='float32')
            
        # 录音结束后，手动追加一小段静音/尾音缓冲
        post_roll = np.zeros((int(cls.fs * 0.3), 1), dtype='float32')
        recording.append(post_roll)
        
        return np.concatenate(recording).flatten()