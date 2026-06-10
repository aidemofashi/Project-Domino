import os

from Tilps.ASR.asr import ASR
from Tilps.LLM.llm_input import LLMinput
from Tilps.LLM.filter import Filter
from Tilps.LLM.memorymanager import MemoryManager
from Tilps.TTS.edgetts import AudioOutput
from Tilps.TTS.edge_test import tts_test
from Tilps.mcp.shot import shot_screen
from Tilps.Core.core import RequestCore


DEVICE = os.getenv("DEVICE", "cpu")

if DEVICE == "cuda":
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/SenseVoiceSmall")
    ASR_SETTING = {
        "model": MODEL_DIR,
        "vad_model": None,
        "device": "cuda",
        "disable_pbar": True,
        "disable_update": True,
        "local_files_only": True,
        "batch_size": 1,
        "max_single_segment_length": 20000,
    }
    LLM_CONFIG = {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("ALI_API"),
        "model_name": "qwen3.5-flash",
    }
else:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/SenseVoiceSmall")
    ASR_SETTING = {
        "model": MODEL_DIR,
        "vad_model": None,
        "device": "cpu",
        "disable_pbar": True,
        "disable_update": True,
        "local_files_only": True,
        "batch_size": 1,
        "max_single_segment_length": 20000,
    }
    LLM_CONFIG = {
        "api_base": "https://api.vectorengine.ai/v1",
        "api_key": os.getenv("V_API"),
        "model_name": "grok-4.1-fast",
    }

SILENCE_TIMEOUT = 60
MAKE_MEMORY = 16


def main():
    ASR.set(ASR_SETTING)

    llm = LLMinput()
    llm.setting(LLM_CONFIG["api_base"], LLM_CONFIG["api_key"], LLM_CONFIG["model_name"])

    tts = AudioOutput()
    memory = MemoryManager()
    filter = Filter()

    #tts_test()

    core = RequestCore()
    core.register("asr", ASR)
    core.register("llm", llm)
    core.register("tts", tts)
    core.register("filter", filter)
    core.register("memory", memory)
    core.register("shot", shot_screen)
    core.silence_timeout = SILENCE_TIMEOUT
    core.make_memory = MAKE_MEMORY

    try:
        core.run()
    except KeyboardInterrupt:
        pass
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
