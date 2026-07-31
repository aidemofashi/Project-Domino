import base64
import os
import time
import subprocess
from PIL import Image
import io

def shot_screen():
    """
    截图并使用WebP格式压缩
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nircmd_path = os.path.join(current_dir, 'nircmd.exe')
    shot_path = os.path.join(current_dir, "shot.png")
    
    # 截图
    if not os.path.exists(nircmd_path):
        raise FileNotFoundError(f"nircmd.exe 不存在：{nircmd_path}")
    
    subprocess.run([nircmd_path, "savescreenshotwin", shot_path])
    time.sleep(0.3)
    
    # 压缩并编码图片
    img = Image.open(shot_path)
    
    # 调整尺寸到1024以内
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    
    # 保存为JPEG格式
    output = io.BytesIO()
    img = img.convert("RGB")
    img.save(output, format='JPEG', quality=50)
    
    base64_data = base64.b64encode(output.getvalue()).decode("utf-8")
    
    # 清理临时文件
    if os.path.exists(shot_path):
        os.remove(shot_path)
    
    return base64_data