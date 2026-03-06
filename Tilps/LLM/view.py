import base64
import os
from PIL import Image
import io
from Tilps.LLM.llm_input import LLMinput  # 直接导入LLMinput

class viewllm_input:
    def setting(self, api_base, api_key, model_name):
        """设置视觉模型的API参数"""
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        # 初始化自己的LLM客户端
        self.llm = LLMinput()
        self.llm.setting(api_base, api_key, model_name)
        print(f"视觉模型已设置: {model_name}")
    
    def compress_image(self, image_path, max_size=1024, quality=85):
        """简单压缩图片"""
        # 打开图片
        img = Image.open(image_path)
        
        # 调整尺寸（保持宽高比）
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 保存到内存，用原格式或转JPEG 
        output = io.BytesIO()
        if img.mode == 'RGBA':
            # PNG带透明通道转RGB
            img = img.convert('RGB')
            save_format = 'JPEG'
        else:
            save_format = 'JPEG'
        
        img.save(output, format=save_format, quality=quality, optimize=True)
        return output.getvalue()

    def encode_image(self, image_path):
        """压缩并编码图片"""
        try:
            # 先压缩再编码
            compressed = self.compress_image(image_path)
            return base64.b64encode(compressed).decode("utf-8")
        except FileNotFoundError:
            print(f"图片文件未找到: {image_path}")
            return None
        except Exception as e:
            print(f"处理图片失败: {e}")
            return None

    def llm_view(self, image_path=None):
        """直接处理图片，不需要传入llm_client"""
        # 如果没有传入image_path，使用默认
        if image_path is None:
            error_msg = "未添加图片路径"
            return error_msg
            
        print(f"开始处理图片: {image_path}")
        
        # 压缩并编码
        base64_image = self.encode_image(image_path)
        if not base64_image:
            return "图片处理失败"
            
        image_data_url = f"data:image/jpeg;base64,{base64_image}"
        
        # 显示压缩效果
        original_size = os.path.getsize(image_path)
        compressed_size = len(base64.b64decode(base64_image))
        print(f"原始: {original_size/1024:.1f}KB → 压缩后: {compressed_size/1024:.1f}KB")
        
        # 构造符合LLMinput格式的消息列表
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    },
                    {
                        "type": "text",
                        "text": "用一句话详细描述当前画面，并且推测相关信息，若是文字信息过多就聚焦描述突出内容，回复不得超过100字"
                    }
                ]
            }
        ]
        
        try:
            # 用自己的llm发送请求
            print("\n正在调用视觉模型...")
            response = self.llm.send_llm(messages)
            
            print("\n模型回答：")
            return response,image_data_url
            
        except Exception as e:
            error_msg = f"API调用失败: {e}"
            print(error_msg)
            return error_msg