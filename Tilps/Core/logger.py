"""
统一日志核心。

所有模块的日志（包括遗留的 print）统一汇聚到这里，
再通过绑定的 sink（通常是 WebSocket 接口）推送到前端。

设计要点：
    * logging 负责结构化日志（各级别、来源明确）
    * 标准输出/错误被重定向，使历史遗留的 print 也能进入同一条链路
    * 保留最近 N 条日志作为 backlog，前端连接到接口后可回放
"""

import logging
import sys
import threading
import time

_LEVEL_VALUES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_LEVEL_ALIASES = {
    "WARN": "WARNING",
    "ERR": "ERROR",
    "FATAL": "CRITICAL",
}

_SOURCE_TAGS = {
    "ASR": "asr",
    "VAD": "vad",
    "TTS": "tts",
    "LLM": "llm",
    "WS": "interface",
    "UI": "ui",
    "MCP": "mcp",
    "记忆整理": "memory",
    "运行错误": "core",
    "主动触发": "core",
    "启动": "core",
    "LuxTTS": "tts",
}

_ERROR_HINTS = (
    "error",
    "错误",
    "失败",
    "exception",
    "traceback",
    "failed",
    "cannot",
    "unable",
    "refused",
    "timeout",
)

# 这些库在 DEBUG 级别会记录每一次网络收发，若转发会造成日志风暴
_NOISY_LOGGERS = (
    "websockets",
    "asyncio",
    "urllib3",
    "httpcore",
    "httpx",
    "PIL",
)

_lock = threading.RLock()
_sink = None
_backlog = []
_backlog_limit = 800
_pending = []
_pending_limit = 200
_flush_interval = 0.12
_flusher = None
_last_signature = None
_last_signature_time = 0.0
_duplicate_window = 1.0
_max_message_length = 1200
_configured = False
_local = threading.local()
_console_handler = None


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_level(level):
    name = str(level or "INFO").upper()
    return _LEVEL_ALIASES.get(name, name)


def _detect_source(line):
    for tag, source in _SOURCE_TAGS.items():
        if f"[{tag}]" in line[:16] or line.lstrip().startswith(f">>> [{tag}]"):
            return source
    return None


def _infer(line, is_stderr):
    """根据输出内容推断日志来源与级别。"""
    text = line.strip()
    lowered = text.lower()
    is_error = any(hint in lowered for hint in _ERROR_HINTS)
    source = _detect_source(text)

    if is_stderr:
        return "core", ("ERROR" if is_error else "WARNING")
    if source:
        return source, ("ERROR" if is_error else "INFO")
    if text.startswith(">>>"):
        return "domino", ("ERROR" if is_error else "INFO")
    if is_error:
        return "domino", "WARNING"
    return "domino", "INFO"


def _record(level, source, message):
    text = str(message)
    if len(text) > _max_message_length:
        text = text[:_max_message_length] + " …(已截断)"
    return {
        "level": _normalize_level(level),
        "source": source or "domino",
        "message": text,
        "time": _timestamp(),
    }


def dispatch(level, source, message):
    """把一条日志写入 backlog，并加入待推送批次。

    同一来源、同一级别、内容完全相同且间隔很短的连续日志会被合并，
    避免音频回调等高频输出淹没界面。
    """
    global _last_signature, _last_signature_time

    text = str(message)
    signature = (level, source, text)
    now = time.time()

    with _lock:
        if signature == _last_signature and (now - _last_signature_time) < _duplicate_window:
            _last_signature_time = now
            return None
        _last_signature = signature
        _last_signature_time = now

        payload = _record(level, source, text)
        _backlog.append(payload)
        if len(_backlog) > _backlog_limit:
            del _backlog[: len(_backlog) - _backlog_limit]
        _pending.append(payload)
        if len(_pending) > _pending_limit:
            del _pending[: len(_pending) - _pending_limit]
        _ensure_flusher()
    return payload


def _ensure_flusher():
    global _flusher
    if _flusher is not None and _flusher.is_alive():
        return
    _flusher = threading.Thread(target=_flush_loop, name="log-flusher", daemon=True)
    _flusher.start()


def _flush_loop():
    while True:
        time.sleep(_flush_interval)
        flush()


def flush():
    """把待推送日志成批交给 sink。"""
    with _lock:
        if not _pending:
            return 0
        batch = list(_pending)
        _pending.clear()
        sink = _sink
    if sink is not None:
        try:
            sink(batch)
        except Exception:
            pass
    return len(batch)


def get_backlog():
    with _lock:
        return list(_backlog)


def clear_backlog():
    with _lock:
        _backlog.clear()
        _pending.clear()


def bind_sink(sink):
    """绑定日志出口，sink 接收一批日志 payload 列表。"""
    global _sink
    with _lock:
        _sink = sink


def set_level(level):
    """设置控制台输出的最低级别，接口始终接收全部级别。"""
    global _console_handler
    name = _normalize_level(level)
    if name not in _LEVEL_VALUES:
        return False
    if _console_handler is not None:
        _console_handler.setLevel(_LEVEL_VALUES[name])
    return True


def get_level():
    if _console_handler is None:
        return "INFO"
    return logging.getLevelName(_console_handler.level)


def get_logger(name="domino"):
    return logging.getLogger(name)


class InterfaceLogHandler(logging.Handler):
    """把 logging 记录转发到接口。"""

    def emit(self, record):
        try:
            name = record.name or ""
            if name.startswith(_NOISY_LOGGERS):
                return
            dispatch(record.levelname, record.name, record.getMessage())
        except Exception:
            pass


class _StreamRouter:
    """把标准输出/错误按行转换为 logging 记录。"""

    def __init__(self, real_stream, is_stderr):
        self._real = real_stream
        self._is_stderr = is_stderr
        self._buffer = ""
        self._local_lock = threading.RLock()

    def write(self, data):
        if data is None:
            return 0
        text = str(data)
        if not text:
            return 0
        with self._local_lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit_line(line)
        return len(text)

    def flush(self):
        try:
            if self._real is not None:
                self._real.flush()
        except Exception:
            pass

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._real, "errors", "replace")

    @property
    def buffer(self):
        return getattr(self._real, "buffer", None)

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def isatty(self):
        return False

    def fileno(self):
        if self._real is None:
            raise OSError("no underlying stream")
        return self._real.fileno()

    def _emit_line(self, line):
        text = line.rstrip("\r")
        if not text.strip():
            return
        source, level = _infer(text, self._is_stderr)
        self._log(source, level, text)
        # 保证控制台仍能看到原始输出（由 console handler 统一写出）
        if _console_handler is None:
            self._write_real(text + "\n")

    def _log(self, source, level, text):
        if getattr(_local, "forwarding", False):
            self._write_real(text + "\n")
            return
        _local.forwarding = True
        try:
            logging.getLogger(source).log(_LEVEL_VALUES[level], text)
        except Exception:
            self._write_real(text + "\n")
        finally:
            _local.forwarding = False

    def _write_real(self, text):
        try:
            if self._real is not None:
                self._real.write(text)
        except Exception:
            pass


def setup_logging(level="INFO", capture_streams=True):
    """初始化统一日志。可重复调用，仅第一次生效。"""
    global _configured, _console_handler
    with _lock:
        if _configured:
            set_level(level)
            return
        _configured = True

        real_stdout = sys.stdout if sys.stdout is not None else sys.__stdout__
        real_stderr = sys.stderr if sys.stderr is not None else sys.__stderr__

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        for handler in list(root.handlers):
            root.removeHandler(handler)

        console = logging.StreamHandler(real_stdout)
        console.setLevel(_LEVEL_VALUES.get(_normalize_level(level), logging.INFO))
        console.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"
        ))
        root.addHandler(console)
        _console_handler = console

        interface = InterfaceLogHandler()
        interface.setLevel(logging.DEBUG)
        interface.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(interface)

        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        if capture_streams:
            if real_stdout is not None:
                sys.stdout = _StreamRouter(real_stdout, False)
            if real_stderr is not None:
                sys.stderr = _StreamRouter(real_stderr, True)
