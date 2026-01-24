import os
from faster_whisper import WhisperModel

dir = './'
# 自动检测 DirectML 设备（AMD GPU）
model = WhisperModel("small", device="auto", compute_type="int8")
segments, info = model.transcribe("test.wav", language="zh")
for segment in segments:
    print(segment.text)

with open(os.path.join(dir,"output.txt"),mode='w',encoding='utf-8') as file:
    file.write(segment.text)