from funasr import AutoModel
import time

class ASR:
    model = None
    @classmethod
    def set(cls,setting):
        cls.model = AutoModel(**setting)
        return cls.model
    @classmethod
    def audio_input(cls,input_audio_data,lang):
        if cls.model is None:
            raise RuntimeError("ASR model not initialized. Call ASR.set() first.")
        
        res = cls.model.generate(input=input_audio_data, cache={}, language=lang)
        if res and res[0]['text'].strip():
            text = res[0]['text']
            print(f"识别结果：{text}")
            print()
            res[0]['datetime'] = time.strftime("%Y-%m-%d %H:%M:%S")
            res[0].pop("key", None)
        return res