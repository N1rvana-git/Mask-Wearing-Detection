
# 基于YOLOv11n的口罩佩戴检测系统

## 项目简介

本项目为毕业设计，基于YOLOv11n深度学习模型，支持高效、轻量的口罩佩戴检测。系统包含模型训练、ONNX导出、FastAPI/ONNXRuntime后端推理、前端实时检测等完整流程，适配4GB显存环境，支持微信小程序等Web端部署。

## 主要特性

- **AI模型**：YOLOv11n，支持自定义结构与消融实验
- **训练脚本**：支持自动batch size优化、混合精度、丰富数据增强
- **后端服务**：FastAPI/ONNXRuntime，RESTful接口，支持Base64和文件上传
- **前端**：现代化网页端，支持实时视频流检测
- **部署**：支持Docker、云服务器、小程序端推理
- **学术复现**：训练曲线自动保存，实验结果可复查

## 项目结构

```
├── models/                  # 训练脚本与模型
│   ├── train_yolov11_mask_detection.py  # YOLOv11n训练主脚本
│   └── weights/            # 训练权重与ONNX模型
├── backend/                 # FastAPI后端
│   ├── app.py               # 主服务（YOLOv11n）
│   ├── app_onnx_fastapi.py  # ONNX推理服务
│   └── api/                 # API接口
├── data/                    # 数据集与配置
│   └── mask_detection.yaml
├── runs/                    # 训练输出
│   └── yolov11_mask_detection/custom_v2_accum/
│       ├── training_curves.png
│       ├── results.csv
│       └── weights/
│           ├── best.pt
│           ├── last.pt
│           └── best.onnx
├── tests/                   # API与集成测试
├── frontend/                # 前端页面
├── requirements.txt         # 依赖
└── README.md
```

## 训练与实验结果

- 训练命令示例：
   ```bash
   python models/train_yolov11_mask_detection.py --device 0 --batch-size 16 --epochs 200
   ```
- 训练耗时：约4.3小时（RTX 3050 4GB）
- 最佳模型：`runs/yolov11_mask_detection/custom_v2_accum/weights/best.pt`（5.5MB）
- ONNX模型：`best.onnx`（10.1MB，满足≤80MB约束）
- 训练曲线：`training_curves.png`，自动生成
- 验证集表现：
   - mAP@0.5：0.84
   - mAP@0.5:0.95：0.54
   - Precision：0.83
   - Recall：0.79
   - R_mask类别：P=0.97，R=0.96，mAP@0.5=0.99
   - W_mask类别：P=0.68，R=0.62，mAP@0.5=0.69

## 快速开始

### 1. 环境准备

```bash
git clone <repository-url>
cd graduation_project
pip install -r requirements.txt
```

### 2. 数据准备

```bash
python data/prepare_dataset.py
# 或根据 data/mask_detection.yaml 配置数据集路径
```

### 3. 模型训练

```bash
python models/train_yolov11_mask_detection.py --device 0 --batch-size 16 --epochs 200
```

### 4. 启动后端服务

```bash
# FastAPI开发模式
uvicorn backend.app:app --host 0.0.0.0 --port 5000 --reload
# ONNX推理服务
python backend/app_onnx_fastapi.py
```

### 5. API调用示例

```python
import requests, base64
with open('test.jpg', 'rb') as f:
      img = base64.b64encode(f.read()).decode()
r = requests.post('http://localhost:5000/api/detect_base64', json={'image': img})
print(r.json())
```

## 部署说明

- Docker部署：
   ```bash
   docker build -t mask-detection -f deployment/docker/Dockerfile .
   docker run -p 5000:5000 --gpus all mask-detection
   ```
- 微信小程序/前端：通过WebSocket或Base64 API实时推理

## 主要变更与亮点

- 新增YOLOv11n训练主脚本，支持自动batch size与混合精度
- 训练结果自动保存曲线图与CSV，便于论文复现
- ONNX模型导出与推理服务，适配小程序端
- 完善API测试用例，保证接口稳定
- 训练与推理均适配4GB显存，模型体积远小于80MB

## 运行结果样例

- 训练曲线与详细指标见 `runs/yolov11_mask_detection/custom_v2_accum/`
- ONNX模型可直接用于微信小程序或Web端部署

## 许可证

MIT License

---

如需更详细的实验分析或曲线图展示，可进一步联系作者。
