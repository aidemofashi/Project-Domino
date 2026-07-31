import os
import json


class ApiManager:
    """
    API管理  
    加载文件 Data/api.json
    """
    _instance = None
    _config = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        if self._initialized:
            return
        self._initialized = True
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "Data", "api.json"
            )
        self._config_path = config_path
        self._config = self._load(config_path)

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload(self):
        """重新加载配置文件"""
        self._config = self._load(self._config_path)

    def _resolve(self, value):
        """将 env:XXX 格式的值解析为环境变量"""
        if isinstance(value, str) and value.startswith("env:"):
            return os.getenv(value[4:], "")
        return value

    # ── LLM ──────────────────────────────────────────────

    def get_llm_config(self, profile="main"): 
        """获取 LLM 配置字典 {api_base, api_key, model_name}"""
        cfg = self._config["llm"][profile]
        return {
            "api_base": cfg["api_base"],
            "api_key": self._resolve(cfg["api_key"]),
            "model_name": cfg["model_name"],
        }

    def create_llm(self, profile="main"):
        """创建并配置好的 LLMinput 实例"""
        from Tilps.LLM.llm_input import LLMinput
        cfg = self.get_llm_config(profile)
        llm = LLMinput()
        llm.setting(cfg["api_base"], cfg["api_key"], cfg["model_name"])
        return llm

    # ── ASR ──────────────────────────────────────────────

    def get_asr_config(self, device="cpu"):
        """获取 ASR 配置字典（可指定推理设备）"""
        cfg = dict(self._config["asr"])
        for k, v in cfg.items():
            cfg[k] = self._resolve(v)
        cfg["device"] = device
        return cfg

    # ── TTS ──────────────────────────────────────────────

    def get_tts_config(self):
        """获取 TTS 配置字典"""
        return dict(self._config["tts"])

    def create_tts(self):
        """根据配置创建对应 TTS 引擎实例"""
        cfg = self.get_tts_config()
        engine = cfg.get("engine", "edge")
        if engine == "edge":
            from Tilps.TTS.edgetts import AudioOutput
            tts = AudioOutput()
            edge_cfg = cfg.get("edge", {})
            tts.voice = edge_cfg.get("voice", tts.voice)
            tts.rate = edge_cfg.get("rate", tts.rate)
            return tts
        elif engine == "ali":
            from Tilps.TTS.ali_tts import AudioOutput
            AudioOutput.input_api(self._resolve(cfg["ali"]["api_key"]))
            tts = AudioOutput()
            ali_cfg = cfg.get("ali", {})
            tts.voice = ali_cfg.get("voice", tts.voice)
            tts.model = ali_cfg.get("model", tts.model)
            return tts
        elif engine == "genie":
            from Tilps.TTS.genie import AudioOutput
            genie_cfg = cfg.get("genie", {})
            max_workers = genie_cfg.get("max_workers", 2)
            tts = AudioOutput(max_workers=max_workers)
            return tts
        elif engine == "lux":
            from Tilps.TTS.lux_tts import AudioOutput
            lux_cfg = cfg.get("lux", {})
            max_workers = lux_cfg.get("max_workers", 1)
            threads = lux_cfg.get("threads", 8)
            num_steps = lux_cfg.get("num_steps", 3)
            speed = lux_cfg.get("speed", 0.8)
            tts = AudioOutput(max_workers=max_workers, threads=threads, num_steps=num_steps, speed=speed)
            return tts
        else:
            raise ValueError(f"不支持的 TTS 引擎: {engine}")

    # ── VAD ──────────────────────────────────────────────

    def get_vad_config(self):
        """获取 VAD 配置字典"""
        cfg = dict(self._config["vad"])
        for k, v in cfg.items():
            cfg[k] = self._resolve(v)
        return cfg

    # ── System ──────────────────────────────────────────────

    def get_system_config(self):
        """获取系统运行参数"""
        return dict(self._config["system"])
