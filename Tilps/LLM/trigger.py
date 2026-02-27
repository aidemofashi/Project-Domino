import time

class TimerTrigger:
    def __init__(self):
        self.last_trigger_time = time.time()
        self.last_activity_time = time.time()
    
    def should_trigger(self, silence_timeout, current_time=None):
        """
        判断是否应该触发主动说话
        
        Args:
            silence_timeout: 静音超时时间（秒）
            current_time: 当前时间，如果不传则自动获取
        
        Returns:
            bool: True表示应该触发，False表示不应该触发
        """
        if current_time is None:
            current_time = time.time()
        
        # 如果超过静音阈值，且距离上次触发也超过阈值
        if (current_time - self.last_activity_time > silence_timeout and 
            current_time - self.last_trigger_time > silence_timeout):
            return True
        return False
    
    def mark_activity(self):
        """标记用户活动（有语音输入时调用）"""
        self.last_activity_time = time.time()
        self.last_trigger_time = time.time()  # 也重置触发时间
    
    def mark_trigger(self):
        """标记已经触发（主动说话后调用）"""
        self.last_trigger_time = time.time()
    
    def get_status(self):
        """获取当前状态（用于调试）"""
        current = time.time()
        return {
            "沉默时长": round(current - self.last_activity_time, 1),
            "距离上次触发": round(current - self.last_trigger_time, 1),
            "静音阈值": SILENCE_TIMEOUT if 'SILENCE_TIMEOUT' in globals() else '未设置'
        }