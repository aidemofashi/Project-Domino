import base64
import os
import time
import subprocess

def shot_screen():
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # nircmd.exe 的完整绝对路径
    nircmd_path = os.path.join(current_dir, 'nircmd.exe')
    
    # 截图保存路径（保存到当前文件同级目录）
    shot_path = os.path.join("shot.png")
    
    # 检查 nircmd.exe 是否存在
    if not os.path.exists(nircmd_path):
        raise FileNotFoundError(f"nircmd.exe 不存在：{nircmd_path}")
    
    # 使用 subprocess 执行截图命令（比 os.system 更可靠）
    try:
        result = subprocess.run(
            [nircmd_path, "savescreenshotwin", shot_path],
            capture_output=True,
            text=True
        )
        
        # 等待文件写入完成
        time.sleep(0.3)
        
        # 检查截图文件是否成功创建
        if not os.path.exists(shot_path):
            raise FileNotFoundError(f"截图失败，文件不存在：{shot_path}")
            
    except Exception as e:
        raise Exception(f"截图执行失败：{str(e)}")