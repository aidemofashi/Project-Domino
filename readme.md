# PROJECT:Domino

## 项目描述  
一个语音陪伴agents

在目前运行环：
cpu:12490f
gpu:6650xt
ram:16gb

## 语音部分  
先要有单独的语音识别模型  

由于没办法使用cuda，尝试使用ONNX Runtime提供的directML来跑Whisper 模型  


## 将要实现：  

1. 实时语音识别  
2. 实时屏幕识别处理    
3. 语音输出  
4. 反控处理   

## 记录  

实现非实时语音识别  --2026/1/24  