import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from Tilps.Core.apim import ApiManager
from Tilps.ASR.asr import ASR
from Tilps.LLM.filter import Filter
from Tilps.LLM.memorymanager import MemoryManager
from Tilps.mcp.shot import shot_screen
from Tilps.Core.core import RequestCore


DEVICE = os.getenv("DEVICE", "cpu")


def main():
    api = ApiManager()
    core = RequestCore()

    asr_config = api.get_asr_config(DEVICE)
    asr = ASR()
    asr.set(asr_config)

    profile = "main_cuda" if DEVICE == "cuda" else "main"
    llm = api.create_llm(profile)

    tts = api.create_tts()
    memory = MemoryManager()
    filter = Filter()

    core.register("asr", asr)
    core.register("llm", llm)
    core.register("tts", tts)
    core.register("filter", filter)
    core.register("memory", memory)
    core.register("shot", shot_screen)

    sys_cfg = api.get_system_config()
    core.silence_timeout = sys_cfg["silence_timeout"]
    core.make_memory = sys_cfg["make_memory"]

    try:
        core.run()
    except KeyboardInterrupt:
        pass
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
