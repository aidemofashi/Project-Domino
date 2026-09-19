<div align="center">
  <img src="./Domino.png" width="140" alt="PROJECT:Domino">

  <h1>Project:Domino</h1>

  <p>
    <a href="./readme.md  "><strong>简体中文</strong></a>
  </p>

  <p>
    <a href="https://github.com/aidemofashi/Project-Domino"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=whit" alt="Codex Custom Pet"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="MIT License"></a>
    <h1></h1>
  </p>
</div>
<center><b>完全本地化的陪伴形态Agent</b></center>

演示视频：  

[![AI桌宠，没有api没有N卡？没有问题（演示1）](https://i0.hdslb.com/bfs/archive/7296d0b5bb74b6028500ac2fbf46bfc92c366eed.jpg)](https://www.bilibili.com/video/BV1Hyeh6VENJ)

## 为什么会有Project_Domino？  

Project_Domion 是致力于让任何人和任何机器都能体验到全模态的本地ai陪伴的项目。  

## Domino有什么作用？  

- Domino致力于全模态的陪伴和快捷的互动。  

- 现在的智能设备上设备助手已经习以为常，但Domino主要致力于能为大家在夜深人静的时候还有一个人能和你一起工作和娱乐。  

- Domino的各个功能开发相对简单，借助ai工具每个人都能为Domino最基础的组件、或是硬件控制主键进行修改和创作。  

## 相比于寻常的陪伴型Agent,Domino有什么特点？  

- Domino致力于在消费级设备上本地运行，即便没有可以用来加速推理的GPU或是云端的大模型提供商，Domino依旧可以以最佳状态运行。本地优化过的语音转文字、语音合成、大模型推理等体验的优化聚合是本项目主要的努力方向。  

- 在下面给出的开发环境中，开启桌面视觉、本地麦克风语言转文字、本地大模型推理、本地文字合成语言（含音色克隆）的Dominon的已经能做到稳定4s的回复延迟。  

## Domino不能使用网络的模型提供商吗？  

- Domino后续将会提供完整的各项功能自定义网络提供商接口，用户可以自行配置      

## 开发环境：  

System: windows10   
CPU: intel i5-12490f   
GPU: AMD RX6650XT   
RAM: 16GB  

## 将要实现

1. 实时语音识别  ✅  
2. 实时屏幕识别处理  ✅  
3. 语音输出  ✅   
4. ui模块  ✅  
5. MCP模块  

## 已经实现

1. 从计算机输入设备直接捕捉并处理音频信息_模块  
2. 将输入的信息转为文字发送给llm并保存对话_模块  
3. 将llm输出使用api合成为朗读声音_模块  

## 进行中
  
1. 优化互动流程  
2. 使用Tauri 创建ui  

## 记录  

实现非实时语音识别  --2026/1/24    

更换阿里的SenseVoice，提升处理速度，使用sounddevice，实现从计算机输入设备直接捕捉并处理音频信息  --2026/1/24 

添加VAD模型，累加历史对话  --2026/1/25    

放弃流式语音处理，通过监控音量大小的方式来模仿实时语音识别  --2026/1/28  

新增filter  --2026/1/29  

实现对话保存和输入传输到大模型的api  --2026/2/6  

暂时放弃编辑本地语音合成模块，新增audio_output使用线上api实现语音合成，（测试环境为阿里云api）将llm输出的文字转为朗读语音  --2026/2/7  

重构主流程和添加vosk模式  --2026/2/15  

完成记忆功能，完成除了MCP部分外的全部基础功能  --2026/3/14    

替换成sensevoice自带vad，以及大幅优化完整流程  --2026/7/16   

使用tauri创建ui模块、默认tts引擎转为Lux-tts  --2026/7/16   

已经实现tauri桌宠，占时并未commit  --2026/9/16   
  
