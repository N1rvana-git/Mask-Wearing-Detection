<div align=""center"">

# 😷 基于 YOLOv11n 的口罩佩戴实时检测项目

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-High_Performance-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED.svg?style=for-the-badge&logo=onnx&logoColor=white)]()
[![WeChat](https://img.shields.io/badge/WeChat-MicroProgram-07C160.svg?style=for-the-badge&logo=wechat&logoColor=white)]()
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)]()

*轻量级、高精度、端到端的深度学习目标检测解决方案*

[立即开始](#-快速开始) • [详细架构](#-架构与详细目录结构) • [性能评估](#-模型训练与实验结果) • [部署指南](#-部署说明)

</div>

---

## 📖 项目简介

本项目是一套完整的**端到端人脸口罩佩戴检测系统**。系统核心采用轻量级目标检测模型 **YOLOv11n**，专为资源受限环境（如 4GB 显存设备与移动终端）设计。
不仅提供了模型训练、ONNX 模型转换全路径，还包含基于 **FastAPI/ONNXRuntime** 的高性能后端验证平台与**微信小程序**原生调用方案。可作为计算机视觉应用落地、毕业设计或深度学习实战绝佳案例。

## ✨ 核心特性

- 🚀 **前沿且轻量**：采用全新 YOLOv11n 架构，支持自定义网络消融实验，导出 ONNX 模型仅 10.1MB。
- ⚡ **高性能业务**：基于 FastAPI 的异步 RESTful 接口与 WebSocket 服务，完美适配 CPU/GPU 混合推理。
- 📱 **多终端支持**：内置现代化 Web 端与微信小程序前端工程，支持摄像头实时视频流检测。
- 🔬 **严谨的学术论证**：训练脚本自带动态 Batch-Size 优化与混合精度训练（AMP），一键生成训练曲线、PR 曲线分析图表以便于论文引用。
- 🛠️ **开发者友好**：提供自动获取本机 IP 并注入前端配置的一键拉起本地联调脚本。

---

## 🏗️ 架构与详细目录结构

本系统采用经典的前后端分离与模型解耦架构，主要分为四大模块：**前端展示层**、**后端推理层**、**模型控制层** 及 **数据预处理层**。

<details>
<summary><b>📂 点击展开查看详细项目目录树与模块作用</b></summary>

`	ext
Mask-Wearing-Detection/
├── Web/                     # 📱 微信小程序前端工程 (核心展示层)
│   ├── pages/               # 视图层 (包含 realtime 实时摄像头检测模块)
│   ├── app.js               # 小程序入口与网络接口全局变量配置
│   └── project.config.json  # 微信开发者平台工作区配置
├── backend/                 # ⚙️ 高并发后端推理服务 (核心业务层)
│   ├── api/                 # API路由控制器 (暴露Base64图像识别与WS视频流接口)
│   ├── utils/               # 模型加载器设计 (ONNX热插拔) 与图像转换基建
│   ├── app.py               # FastAPI 主服务应用 (基于PyTorch)
│   └── app_onnx_fastapi.py  # 纯 ONNXRuntime 轻量版服务 (摆脱笨重Torch包)
├── models/                  # 🧠 深度学习研究与控制层 
│   ├── train_yolov11_mask_detection.py # YOLOv11核心训练脚本 (提供超参重写接口)
│   └── weights/             # 模型库：存放 .pt 原生权重与 .onnx 生产部署权重
├── data/                    # 📊 数据集管理层
│   ├── mask_detection.yaml  # 训练所需的数据集路径及类别规范配置
│   └── prepare_dataset.py   # 数据洗牌、增强、清洗及标注检查脚本
├── runs/                    # 📈 实验记录与可视化层
│   └── yolov11_mask_detection/ # 每次运行自动留痕：包含训练Loss图表、F1-Score、混淆矩阵
├── run.ps1                  # 🚀 全栈快捷脚本：自适应获取本地IP并一键启动前后端桥接
├── requirements.txt         # API项目及训练环境核心依赖抽象集
└── README.md                # 当前说明手册
`
</details>

---

## 🚀 快速开始

### 1. 环境准备

建议使用 Virtualenv 或 Conda 管理环境。下载源码后装载基础环境包：

`ash
git clone https://github.com/N1rvana-git/Mask-Wearing-Detection.git
cd Mask-Wearing-Detection
pip install -r requirements.txt
`

### 2. 本地全栈一键拉起（强烈推荐 ⭐）

无需手动对照改 IP 代码，项目根目录下直接使用内置的高级 PowerShell 脚本，它能一键完成：**获取本机活动IP → 注入修改小程序环境 → 释放被占用的5000端口 → 拉起 FastAPI 运行后端**。

`powershell
# 在 Windows PowerShell 下执行：
.\run.ps1
`
*启动完毕后，双击微信开发者工具直接打开 Web/ 小程序目录即可开展真机无缝调测！*

<details>
<summary><b>🛠️ 备选：手动操作指南</b></summary>
    
1. 数据准备 (若首次需跑模型)
   `ash
   python data/prepare_dataset.py
   `
2. 手动启动 FastAPI
   `ash
   uvicorn backend.app:app --host 0.0.0.0 --port 5000 --reload
   `
</details>

---

## 📊 模型训练与实验结果

支持低成本算力落地，在常见的 4GB 环境下 (如 NVIDIA RTX 3050) 也能自适应跑通全量目标数据。

- **启动定制训练** (自带早停及参数调优):
   `ash
   python models/train_yolov11_mask_detection.py --device 0 --batch-size 16 --epochs 200
   `
- **关键测评指标**：
   - 综合 mAP@0.5: **0.84** | mAP@0.5:0.95: **0.54**
   - 目标规范戴口罩准确率 (R_mask): *Precision = 0.97*, *Recall = 0.96*
- **落地性能**：模型转换为 .onnx 后体积极致压缩仅需 **10.1MB**，完美符合微信小程序引擎通常要求 ≤80MB 的边端部署硬性约束。

> _详尽的损失下降曲线、评估 CSV 文件以及混淆矩阵生成图均以标准科研体例保存于 uns/yolov11/...，方便作者在论文编撰和开题报告中随取随用。_

---

## 🔌 API 接口参考

本项目自带现代化交互式 API 调试文档（基于 Swagger UI）。服务启动后，使用浏览器访问 http://localhost:5000/docs 即可视化测试各项接口。

<details>
<summary><b>💻 点击获取 Base64 图片推理请求代码示例 (Python)</b></summary>

**调用地址**: POST /api/detect_base64
    
**请求样例**:
`python
import requests
import base64

# 读取图像并将其直接推入 Base64 编码池
with open('test_image.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# 向后台发出检测校验
res = requests.post(
    'http://localhost:5000/api/detect_base64', 
    json={'image': img_b64}
)

# 打印检测框坐标与置信度结果
print(res.json())
`
</details>

---

## 📦 部署说明

- **微服务化容器部署**：建议通过 Docker 进行云端快速交付。支持显卡透传加速推理。
  `ash
  docker build -t mask-detection -f Dockerfile .
  docker run -d -p 5000:5000 --gpus all mask-detection
  `
- **边缘端/客户端接入**：得益于统一的 REST API 与 WebSocket 基建，模型已准备好直接对接各类平台（H5、微信小程序、原生 Android）。

---

## 📄 许可证
本项目采用 **MIT License** 开源开源协议。赋予高度的代码修改商用及二次创新权利，但引用请遵循相关要求并保留核心作者的 Attribution 溯源许可。

---
<div align="center">
  <i>基于 YOLO 构建，赋能 AI 智能物联落地验证 🚀</i><br>
  如果你觉得有帮助，欢迎随手点一个 <b>Star</b> ⭐ 鼓励！
</div>
