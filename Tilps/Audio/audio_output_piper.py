import subprocess
import os
import sounddevice as sd
import soundfile as sf

class AudioOutput:
    def __init__(self):
        self.piper_path = r"./piper/piper.exe"
        self.model_path = r"./zh_CN-huayan-medium.onnx"
        self.temp_wav = "output.wav"

    @staticmethod
    def input_api(api):
        pass

    def text_to_speech(self, text):
        if not text: return
        
        # 直接让 Piper 把完整的音频写入文件
        cmd = [self.piper_path, "--model", self.model_path, "--output_file", self.temp_wav]
        
        try:
            # 运行生成
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.communicate(input=text.encode('utf-8'))

            # 播放文件（soundfile会自动处理采样率和完整度）
            if os.path.exists(self.temp_wav):
                data, fs = sf.read(self.temp_wav)
                sd.play(data, fs)
                sd.wait() 
        except Exception as e:
            print(f"播放出错: {e}")