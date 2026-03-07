import time
from edgetts import AudioOutput

def main():
    print("开始测试 TTS 流式播放...")
    tts = AudioOutput()
    # 播放一段较长的文本，以便观察流式效果
    test_text = "这是一个测试语音，用于验证流式播放功能是否正常。如果一切顺利，你应该很快听到声音，而不需要等待所有音频生成完毕。"
    print(f"正在合成并播放: {test_text}")
    tts.text_to_speech(test_text)
    # 等待足够长的时间让语音播完（根据文本长度估计）
    time.sleep(15)
    print("测试结束。如果听到声音，说明 AudioOutput 工作正常。")
    # 可选：调用 stop 测试打断
    # tts.stop()

if __name__ == "__main__":
    main()