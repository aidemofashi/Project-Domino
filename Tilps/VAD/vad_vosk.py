import sounddevice as sd
import numpy as np
import json
import collections
import time
from vosk import Model, KaldiRecognizer

class AudioInput:
    fs = 12000
    MODEL_PATH = "./models/vosk-model-small-cn-0.22"
    _model = None
    _rec = None

    @classmethod
    def _init_vosk(cls):
        if cls._model is None:
            cls._model = Model(cls.MODEL_PATH)
            cls._rec = KaldiRecognizer(cls._model, cls.fs)

    @classmethod
    def record(cls, audio_output=None): # 仅在此处增加参数接收，用于打断
        cls._init_vosk()
        cls._rec.Reset()
        MAX_SILENCE_BLOCKS = 6
        silence_counter = 0

        # 环形缓冲区，用于保存说话前 1.5 秒的音频
        pre_roll_len = int(1.5 * cls.fs / 2000)
        ring_buffer = collections.deque(maxlen=pre_roll_len)
        
        recording = []
        is_triggered = False
        done = False

        print(">>> [系统监听中] ...")

        def callback(indata, frames, time_info, status):
            nonlocal is_triggered, done, silence_counter
            
            if status:
                print(f"Status: {status}")
            
            # 转换为 int16 给 Vosk
            audio_int16 = (indata * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Vosk 处理
            if cls._rec.AcceptWaveform(audio_bytes):
                # 有完整的一句话
                result = json.loads(cls._rec.Result())
                if result.get("text", ""):
                    print(f"[识别] {result['text']}")
            else:
                # 获取部分结果进行 VAD 判断
                partial = json.loads(cls._rec.PartialResult())
                partial_text = partial.get("partial", "").strip()
                
                if partial_text:
                    # 【核心修改点】：只要识别到初步文字，立即打断 TTS 播放
                    if audio_output:
                        audio_output.stop() 

                    if not is_triggered:
                        # 检测到语音，开始录音
                        is_triggered = True
                        silence_counter = 0
                        # 把缓冲区里的预录音频加入
                        for buf in ring_buffer:
                            recording.append(buf.copy())
                        print("[VAD] 检测到语音，开始录音...")
                        print(f"[实时] {partial_text}")
                    else:
                        # 已经在录音中，有声音输出
                        silence_counter = 0
                        print(f"[实时] {partial_text}")
                else:
                    # 没有检测到语音
                    if is_triggered:
                        silence_counter += 1
                        print(f"[静音] 静音计数: {silence_counter}/{MAX_SILENCE_BLOCKS}")
                        
                        # 静音达到阈值，结束录音
                        if silence_counter >= MAX_SILENCE_BLOCKS:
                            done = True
                            print("[VAD] 静音超时，结束录音")
                            raise sd.CallbackStop

            if is_triggered:
                # 录音中，保存音频
                recording.append(indata.copy())
            else:
                # 未触发，保存到环形缓冲区
                ring_buffer.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=cls.fs, 
                channels=1, 
                callback=callback, 
                blocksize=2000, 
                dtype='float32'
            ):
                # 等待直到录制完成
                start_time = time.time()
                max_duration = 30  # 最长录音时间 30秒
                
                while not done:
                    sd.sleep(100)
                    
                    # 超时处理
                    if time.time() - start_time >= max_duration:
                        if not is_triggered:
                            print("[VAD] 监听超时，未检测到语音")
                        else:
                            print("[VAD] 录音超时，强制结束")
                        break
                            
        except sd.CallbackStop:
            print("[VAD] 录音正常结束")
        except Exception as e:
            print(f"[VAD] 错误: {e}")
            return np.array([], dtype='float32')

        if not recording:
            print("[VAD] 没有录制到音频")
            return np.array([], dtype='float32')
            
        # 录音结束后，追加一小段静音
        post_roll = np.zeros((int(cls.fs * 0.3), 1), dtype='float32')
        recording.append(post_roll)
        
        audio_data = np.concatenate(recording).flatten()
        print(f"[VAD] 录音完成，长度: {len(audio_data)/cls.fs:.2f}秒")
        
        return audio_data