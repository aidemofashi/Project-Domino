import sounddevice as sd
import numpy as np
import collections

class AudioInput:
    fs = 16000
    # --- 阈值设置 ---
    THRESHOLD = 0.015       # 音量阈值（根据环境调整，0.01-0.05 常用）
    SILENCE_LIMIT = 1.5     # 静音持续多久（秒）后停止录音
    PRE_ROLL = 1          # 预录时长（秒），把说话前的一小段也存下来，防止掐头

    @classmethod
    def record(cls):
        """声控录音：检测到声音开始，安静后自动结束"""
        print(">>> 正在监听环境音量，请直接说话...")
        
        chunk_size = 1024
        recording = []     # 存放实际录音块
        
        # 环形缓冲区，用于保存触发前的声音（预录）
        pre_roll_chunks = int(cls.PRE_ROLL * cls.fs / chunk_size)
        ring_buffer = collections.deque(maxlen=pre_roll_chunks)
        
        is_triggered = False  # 是否已触发录制
        silent_chunks = 0     # 静音计数
        limit_chunks = int(cls.SILENCE_LIMIT * cls.fs / chunk_size)

        def callback(indata, frames, time, status):
            nonlocal is_triggered, silent_chunks, recording 
            # 计算当前块的音量 (RMS)
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            
            if not is_triggered:
                if volume_norm > cls.THRESHOLD:
                    print("\n[检测到声音] 录制中...")
                    is_triggered = True
                    # 将预录的缓冲区数据先塞进录音列表
                    recording.extend(list(ring_buffer))
                else:
                    # 还没触发时，先把数据存入循环缓冲区
                    ring_buffer.append(indata.copy())
            
            if is_triggered:
                recording.append(indata.copy())
                if volume_norm < cls.THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                
                # 如果静音时间超过阈值，停止流
                if silent_chunks > limit_chunks:
                    raise sd.CallbackStop

        # 使用 InputStream 实时处理数据
        try:
            with sd.InputStream(samplerate=cls.fs, channels=1, callback=callback, blocksize=chunk_size):
                while True:
                    sd.sleep(100)
                    if is_triggered and silent_chunks > limit_chunks:
                        break
        except sd.CallbackStop:
            pass

        print("录音结束")
        
        if not recording:
            return np.array([], dtype='float32')
            
        # 拼接并展平为一维数组
        return np.concatenate(recording).flatten()