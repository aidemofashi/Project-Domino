# THIRD PARTY NOTICES

PROJECT:Domino 基于以下第三方代码、模型与服务构建。分发的任何版本必须保留本文件及各组件要求的许可声明。

License 对照：
- **Apache-2.0**：需附带许可证全文、保留版权与 NOTICE、修改过的文件加修改说明。
- **MIT / BSD**：需附带版权与许可声明。
- **LGPL-3.0**：需附带许可证全文，允许用户替换该库（动态链接视为合规）。
- **FunASR Model License v1.1**：模型权重专用条款，商用允许但要求署名。

---

## 一、仓库内置代码 / 模型（随项目分发）

### 1. LuxTTS 引擎（vendored）
- 位置：`Tilps/TTS/tts_engine/luxtts/`
- 来源：https://github.com/ysharma3501/LuxTTS
- 代码许可：Apache-2.0
- 模型 `YatharthS/LuxTTS`（HuggingFace）：Apache-2.0（模型卡明示 "Model and code released under Apache-2.0"）
- 说明：LuxTTS 基于 k2-fsa/ZipVoice（Apache-2.0）蒸馏。已按 Apache-2.0 要求在本目录保留来源说明。

### 2. Genie 菲比（feibi）音色模型（未接入主流程）
- 位置：`Tilps/TTS/models/feibi/*.onnx`、`Tilps/TTS/reference/feibi.wav`
- 来源：https://huggingface.co/High-Logic/Genie
- 模型卡许可：MIT
- ⚠️ 风险提示：该参考音色为《鸣潮》角色「菲比」的游戏配音。即使仓库标注 MIT，**角色声音本身的权利属于游戏著作权人（库洛游戏）**，对外分发 / 商用前需自行确认权利方许可，本文件不构成对该音色可用性的保证。

---

## 二、Python 运行时依赖（requirements.txt）

| 依赖 | 版本 | 许可 |
|------|------|------|
| sounddevice | >=0.4.0 | MIT |
| numpy | >=1.21.0 | BSD-3-Clause |
| soundfile | any | BSD-3-Clause |
| pydub | any | MIT |
| miniaudio | any | MIT |
| torch | any | BSD-3-Clause |
| torchaudio | any | BSD-3-Clause |
| funasr | ==1.4.0 | MIT（框架） |
| vosk | any | Apache-2.0 |
| edge-tts | any | **LGPL-3.0**（`src/edge_tts/srt_composer.py` 为 MIT） |
| dashscope | >=1.15.0 | Apache-2.0 |
| openai | >=0.28.0,<1.0.0 | MIT |
| keyboard | >=0.13.5 | MIT |
| Pillow | any | HPND（PIL 历史许可）/ MIT-CMU |

---

## 三、模型权重与在线服务

### 1. SenseVoiceSmall（语音识别，经 funasr 加载）
- 来源：https://modelscope.cn/models/iic/SenseVoiceSmall
- 权重许可：**FunASR Model License v1.1**（"FunASR Model Open Source License Agreement"）
- 要求：商用允许；保留版权声明；对外提供模型时需注明出处（署名 FunASR / Alibaba）。

### 2. FunASR（框架）
- 来源：https://github.com/modelscope/FunASR
- 代码许可：MIT
- ⚠️ 注意：框架 MIT ≠ 其托管模型的权重许可（模型单独适用上一条 / 各模型自身条款）。

### 3. Vosk 模型（若使用 vosk 模式）
- 来源：https://alphacephei.com/vosk/models
- 许可：Apache-2.0（与 vosk-api 一致）

### 4. Edge TTS（在线合成）
- 来源：https://github.com/rany2/edge-tts
- 许可：LGPL-3.0（见上表）
- 说明：调用微软 Edge 在线服务，需遵守 Microsoft 服务条款；合成内容不可离线再分发。使用中若修改了 edge_tts 本身，需按 LGPL-3.0 开放对应修改。

### 5. 阿里云 DashScope / 通义（在线合成与 LLM API）
- 来源：https://help.aliyun.com/zh/model-studio/
- 许可：SDK 为 Apache-2.0；**API 服务适用阿里云服务条款**（商用付费）。

---

## 四、合规待办（强烈建议）

1. **根目录补充 Apache-2.0 / MIT / BSD / LGPL-3.0 许可证全文**（或指向本文件并附下载地址）。
2. **确认 feibi 音色的权利**（见上文 ⚠️）。若用于对外发布，建议替换为自有录音或明确授权的音色。
3. **LGPL-3.0（edge-tts）**：确保最终用户可替换该依赖；如需避免传染，可改为通过独立进程调用。
4. Apache-2.0 组件若修改过源码（如 luxtts），请在对应目录保留修改记录。

—— 生成日期：2026-07-31。此清单基于当前仓库内容与依赖版本，随依赖升级需同步更新。
