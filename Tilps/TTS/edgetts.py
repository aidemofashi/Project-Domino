import edge_tts
import asyncio
import sounddevice as sd
import numpy as np
import re
import threading
import queue
import subprocess
import time

class AudioOutput:
    def __init__(self, max_workers=3):
        """
        max_workers: 并发合成的工作线程数（Edge TTS 并发数不宜过多，建议 2-3）
        """
        print(f">>> 初始化 Edge-TTS 输出（并发合成，顺序播放），工作线程数={max_workers}...")
        self.voice = "zh-CN-XiaoxiaoNeural"
        self.rate = "+10%"
        self.max_workers = max_workers

        # 播放队列存放按序号排序后的 PCM 数据（已保证顺序）
        self.play_queue = queue.Queue()
        # 原始任务队列（文本 + 序号）
        self.task_queue = queue.Queue()
        # 结果缓冲区：暂存已合成但未到播放顺序的音频数据 {序号: audio_data}
        self.result_buffer = {}
        self.buffer_lock = threading.Lock()
        # 下一个期望播放的序号
        self.next_seq = 0
        self.seq_lock = threading.Lock()

        # 用于停止当前正在执行的任务（新对话打断）
        self._stop_current = threading.Event()
        self._synthesis_stop = threading.Event()  # 全局停止信号

        # 启动播放线程
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()

        # 启动合成工作线程池
        self._workers = []
        for i in range(self.max_workers):
            t = threading.Thread(target=self._synthesis_worker, daemon=True)
            t.start()
            self._workers.append(t)

    def _play_worker(self):
        """从播放队列取数据并推送到扬声器"""
        samplerate = 24000
        stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype='float32')
        stream.start()

        while not self._synthesis_stop.is_set():
            try:
                item = self.play_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:  # 退出信号
                break

            if isinstance(item, str) and item == "STOP_RESET":
                stream.stop()
                stream.start()
                continue

            # 写入音频数据 (n_samples, 1)
            stream.write(item.reshape(-1, 1))

    def _synthesis_worker(self):
        """工作线程：从任务队列取任务，合成音频，将结果放入缓冲区"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not self._synthesis_stop.is_set():
            try:
                seq, text = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # 退出信号
                break

            # 检查是否被打断（新对话清空任务时可能设置此事件）
            if self._stop_current.is_set():
                continue

            # 合成音频，得到 PCM 数据（float32 数组）
            audio_data = self._synthesize(text, loop)
            if audio_data is not None:
                # 将合成结果放入缓冲区
                with self.buffer_lock:
                    self.result_buffer[seq] = audio_data
                # 触发播放检查
                self._try_play()

    def _synthesize(self, text, loop):
        """实际合成逻辑，返回 numpy 数组 (float32)"""
        # 预处理文本
        text = re.sub(r'[\(\uff08].*?[\)\uff09]', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？]', '', text)
        if not text.strip():
            return None

        # FFmpeg 命令
        ffmpeg_cmd = [
            'ffmpeg', '-i', 'pipe:0', '-f', 'f32le', '-ar', '24000', '-ac', '1', '-v', 'quiet', 'pipe:1'
        ]
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        # 收集所有 PCM 数据
        pcm_chunks = []

        def pull_pcm():
            while True:
                raw_data = process.stdout.read(4096)
                if not raw_data:
                    break
                samples = np.frombuffer(raw_data, dtype=np.float32)
                pcm_chunks.append(samples)

        pull_thread = threading.Thread(target=pull_pcm, daemon=True)
        pull_thread.start()

        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            async def run():
                async for chunk in communicate.stream():
                    if self._stop_current.is_set():
                        break
                    if chunk["type"] == "audio":
                        process.stdin.write(chunk["data"])
                        process.stdin.flush()
            loop.run_until_complete(run())
        finally:
            if process.stdin:
                process.stdin.close()
            pull_thread.join()
            process.wait()

        if pcm_chunks:
            return np.concatenate(pcm_chunks)
        return None

    def _try_play(self):
        """检查缓冲区，将可连续播放的音频按顺序放入播放队列"""
        with self.buffer_lock:
            while True:
                if self.next_seq in self.result_buffer:
                    audio = self.result_buffer.pop(self.next_seq)
                    self.play_queue.put(audio)
                    self.next_seq += 1
                else:
                    break

    def text_to_speech(self, text, interrupt=False):
        """
        text: 文字片段
        interrupt: 是否打断当前正在播放的旧对话（新对话的第一个片段设为 True）
        """
        if interrupt:
            # 停止当前所有合成任务
            self._stop_current.set()
            # 清空播放队列（停止当前播放）
            self.stop()
            # 清空任务队列中尚未处理的任务
            with self.buffer_lock:
                self.result_buffer.clear()
            with self.seq_lock:
                self.next_seq = 0
            # 清空任务队列
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            # 重置停止标志
            self._stop_current.clear()

        # 分配序号（必须在锁内保证顺序）
        with self.seq_lock:
            seq = self.next_seq + len(self.result_buffer) + self.task_queue.qsize()
            # 注意：这里分配的序号可能不是严格递增的，因为多个线程可能同时调用 text_to_speech？
        # 但 LLM 流式输出是在主线程中顺序调用的，所以 seq 可以在主线程中生成，无需锁。
        # 我们可以直接在 interrupt 处理时重置 next_seq，并在主线程中累加一个计数器。
        # 为了简化，我们可以在类内部维护一个递增的计数器，仅在 interrupt 时重置。
        # 但这里为了兼容并发调用（实际上主线程是串行调用），我们使用一个线程安全的计数器。
        # 简化：使用 threading.Lock 保护 seq_counter
        if not hasattr(self, '_seq_counter'):
            self._seq_counter = 0
            self._seq_counter_lock = threading.Lock()

        with self._seq_counter_lock:
            if interrupt:
                self._seq_counter = 0
            seq = self._seq_counter
            self._seq_counter += 1

        self.task_queue.put((seq, text))

        # 如果是第一个片段（seq == 0），可以立即触发播放检查（实际上已经有结果？但结果还没出来）
        # 无需特殊处理，等合成完成会调用 _try_play

    def stop(self):
        """立即停止播放，并清空播放队列"""
        sd.stop()
        while not self.play_queue.empty():
            try:
                self.play_queue.get_nowait()
            except queue.Empty:
                break
        self.play_queue.put("STOP_RESET")

    def shutdown(self):
        """关闭所有线程"""
        self._synthesis_stop.set()
        for _ in self._workers:
            self.task_queue.put((None, None))