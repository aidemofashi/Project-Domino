import genie_tts as genie
import sounddevice as sd
import numpy as np
import re
import threading
import queue
from pathlib import Path

class AudioOutput:
    def __init__(self, max_workers=2):
        self.script_dir = Path(__file__).parent.absolute()
        self.model_dir = self.script_dir / "models" / "feibi" 
        self.ref_wav = self.script_dir / "reference" / "feibi.wav"
        self.character = "feibi"

        # --- 核心并发控制 ---
        self.task_queue = queue.Queue()       # 存放 (seq, text)
        self.result_buffer = {}               # 存放已合成音频 {seq: audio_data}
        self.buffer_lock = threading.Lock()
        self.next_seq = 0                     # 下一个该播放的序号
        self.seq_counter = 0                  # 输入计数器
        self.counter_lock = threading.Lock()
        
        self.stop_event = threading.Event()   # 停止/打断信号
        self._is_running = True

        self._init_genie()

        # --- 启动工作线程池 (合成) ---
        for _ in range(max_workers):
            threading.Thread(target=self._synthesis_worker, daemon=True).start()

        # --- 启动播放线程 ---
        self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self.play_thread.start()

    def _init_genie(self):
        try:
            if self.model_dir.exists():
                genie.load_character(self.character, str(self.model_dir), 'Chinese')
                if self.ref_wav.exists():
                    genie.set_reference_audio(self.character, str(self.ref_wav), "在此之前，请您务必享受旅居拉古娜的时光")
                print(f">>> [TTS] {self.character} 加载成功")
        except Exception as e:
            print(f">>> [TTS] 初始化失败: {e}")

    def _synthesis_worker(self):
        """并发合成线程：只负责把文本转成音频数据，不直接播放"""
        while self._is_running:
            try:
                seq, text = self.task_queue.get(timeout=0.1)
                if text is None: break
                
                if self.stop_event.is_set():
                    continue

                # 注意：这里需要确保 genie.tts 能返回音频数组而非直接播放
                # 如果 genie.tts 不支持返回数组，需查看其文档是否有 get_audio 接口
                # 假设 genie.tts(play=False) 返回音频 numpy 数组
                try:
                    audio = genie.tts(
                        character_name=self.character,
                        text=text,
                        play=False  # 设置为 False，我们手动控制播放顺序
                    )
                    
                    with self.buffer_lock:
                        self.result_buffer[seq] = audio
                except Exception as e:
                    print(f">>> [TTS] 合成序号 {seq} 失败: {e}")
                
            except (queue.Empty, TypeError):
                continue

    def _play_worker(self):
        """独占播放线程：严格按照 seq 顺序从缓冲区提取并播放"""
        while self._is_running:
            with self.buffer_lock:
                if self.next_seq in self.result_buffer:
                    audio = self.result_buffer.pop(self.next_seq)
                    self.next_seq += 1
                else:
                    audio = None

            if audio is not None:
                # 检查是否在播放前被打断
                if not self.stop_event.is_set():
                    sd.play(audio, samplerate=24000) # 请根据模型实际采样率调整
                    sd.wait() # 等待当前段落播完
            else:
                threading.Event().wait(0.05) # 稍微歇会儿，等待合成

    def text_to_speech(self, text, interrupt=False):
        if interrupt:
            self.stop()

        cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text).strip()
        if not cleaned_text:
            return

        with self.counter_lock:
            current_seq = self.seq_counter
            self.seq_counter += 1
        
        self.task_queue.put((current_seq, cleaned_text))

    def stop(self):
        """打断逻辑"""
        self.stop_event.set()
        sd.stop()
        
        # 清空所有状态
        while not self.task_queue.empty():
            try: self.task_queue.get_nowait()
            except queue.Empty: break
            
        with self.buffer_lock:
            self.result_buffer.clear()
            self.next_seq = 0
            
        with self.counter_lock:
            self.seq_counter = 0
            
        self.stop_event.clear()
        print(">>> [TTS] 播放已重置")

    def shutdown(self):
        self._is_running = False