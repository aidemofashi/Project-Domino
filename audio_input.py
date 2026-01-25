import sounddevice as sd

class audio_input:
    fs = 16000  #声音采样
    duration = 5  # 录制 5 秒
    print("正在录音 (请说话)...")

    # 录音逻辑
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("录音结束")
    #将二维数组转为一维
    audio_data = audio.flatten()