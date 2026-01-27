# YOLOv11n结构魔改集成说明
# 用于train_yolov11_mask_detection.py加载自定义YAML、Attention、WIoU等

import sys
import torch
from ultralytics.nn.tasks import DetectionModel
from models.modules.attention import GAMAttention
from models.losses.wiou import WIoULoss

class CustomYOLOv11n(DetectionModel):
    """
    自定义YOLOv11n结构，支持P2检测头、GAM注意力、WIoU损失
    """
    def __init__(self, cfg='models/configs/yolo11n_mask_custom.yaml', ch=3, nc=3, **kwargs):
        super().__init__(cfg=cfg, ch=ch, nc=nc, **kwargs)
        # 注意: GAMAttention 已在 YAML 配置文件中定义 (models/configs/yolo11n_mask_custom.yaml)
        # 因此此处无需手动 append 到 backbone，Ultralytics 会根据 YAML 自动解析并构建层级结构。
        
        # 替换损失函数标记 (实际生效需配合 Trainer 使用)
        self.criterion = WIoULoss()
        # 可扩展：根据cfg动态插入P2检测头等

# 用于Ultralytics自定义模型注册
CUSTOM_MODEL_REGISTRY = {
    'yolo11n_mask_custom': CustomYOLOv11n,
}
