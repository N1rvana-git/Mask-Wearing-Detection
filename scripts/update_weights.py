
import shutil
import os
import datetime
from pathlib import Path

# 定义源路径和目标路径
src_dir = Path(r"e:\graduation_project\Mask-Wearing-Detection\runs\detect\runs\yolov11_mask_detection\custom_v2_accum\weights")
dst_dir = Path(r"e:\graduation_project\Mask-Wearing-Detection\weights")

# 确保目标目录存在
dst_dir.mkdir(parents=True, exist_ok=True)

# 定义要复制的文件
files_to_copy = ["best.pt", "best.onnx"]

# 创建这个时间的备份目录
backup_dir = dst_dir / f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
backup_dir.mkdir(exist_ok=True)

print(f"开始更新模型权重...")
print(f"源目录: {src_dir}")
print(f"目标目录: {dst_dir}")
print(f"备份目录: {backup_dir}")

for filename in files_to_copy:
    src_file = src_dir / filename
    dst_file = dst_dir / filename
    
    if src_file.exists():
        # 如果目标文件存在，先备份
        if dst_file.exists():
            print(f"备份现有文件: {filename}")
            shutil.move(str(dst_file), str(backup_dir / filename))
            
        # 复制新文件
        print(f"复制新文件: {filename}")
        shutil.copy2(str(src_file), str(dst_file))
        print(f"成功更新 {filename}")
    else:
        print(f"警告: 源文件不存在 {filename}")

print("模型权重更新完成！请重启后端以加载新模型。")
