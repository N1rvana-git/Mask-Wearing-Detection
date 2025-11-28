
"""
YOLOv11口罩检测训练脚本
基于Ultralytics官方库实现的YOLOv11n模型训练
符合开题报告要求的技术路线
"""

# 保证models等自定义包可被import
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import os
import sys
import yaml
import torch
import argparse
import importlib.util
from pathlib import Path

try:
    from ultralytics import YOLO
    from ultralytics import settings
except ImportError:
    print("错误：未安装ultralytics库")
    print("请运行：pip install ultralytics")
    sys.exit(1)


class YOLOv11MaskDetectionTrainer:
    """YOLOv11口罩检测训练器"""
    
    def __init__(self,
                 data_path="data/mask_detection.yaml",
                 model_size="yolo11n",
                 img_size=640,
                 batch_size=16,
                 epochs=100,
                 device="0",
                 project="runs/yolov11_mask_detection",
                 workers: int | None = None,
                 export_onnx: bool = True):
        """
        初始化训练器
        
        Args:
            data_path: 数据集配置文件路径
            model_size: 模型大小 (yolo11n, yolo11s, yolo11m等)
            img_size: 输入图像尺寸
            batch_size: 批次大小（-1表示自动优化）
            epochs: 训练轮数
            device: 设备 (cpu, 0, 1等)
            project: 训练输出目录
        """
        self.data_path = data_path
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.project = project
        self.workers = workers if workers is not None else max(0, min((os.cpu_count() or 1) - 1, 8))
        self.export_onnx_flag = export_onnx
        
        # 设置路径
        self.project_root = Path(__file__).parent.parent
        self.weights_dir = self.project_root / "weights"
        self.runs_dir = self.project_root / "runs"
        
        # 创建必要目录
        self.weights_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        
        # GPU优化
        if self.batch_size == -1:
            self.batch_size = self._auto_optimize_batch_size()
    
    def _auto_optimize_batch_size(self):
        """根据GPU显存自动优化批次大小"""
        if not torch.cuda.is_available():
            print("⚠️ 未检测到GPU，使用CPU训练（batch_size=8）")
            return 8
        
        # 获取GPU信息
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔍 检测到GPU显存: {gpu_memory:.1f}GB")
        
        # RTX 3050 Laptop GPU = 4GB显存
        # 根据开题报告的硬件约束优化
        if gpu_memory <= 4:
            # YOLOv11n在4GB显存下的推荐配置
            batch_size = 32
            print(f"✅ 自动配置批次大小: {batch_size} (适配4GB显存)")
        elif gpu_memory <= 8:
            batch_size = 64
            print(f"✅ 自动配置批次大小: {batch_size} (适配8GB显存)")
        else:
            batch_size = 128
            print(f"✅ 自动配置批次大小: {batch_size} (充足显存)")
        
        return batch_size
    
    def verify_data_config(self):
        """验证数据集配置"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据集配置文件不存在: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查必要字段
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"数据集配置缺少必要字段: {field}")
        
        # 验证类别数与名称一致性
        names = config.get('names', [])
        if not isinstance(names, list) or not names:
            raise ValueError("数据集配置中的 names 必须是非空列表")
        if config['nc'] != len(names):
            raise ValueError(f"类别数与名称数量不一致：nc={config['nc']}，names数={len(names)}")
        
        dataset_root = Path(config.get('path', '.')).expanduser()
        if not dataset_root.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {dataset_root}")

        for split in ('train', 'val', 'test'):
            split_dir = config.get(split)
            if not split_dir:
                raise ValueError(f"数据集配置缺少 {split} 路径")

            split_path = Path(split_dir)
            if not split_path.is_absolute():
                split_path = dataset_root / split_dir

            if not split_path.exists():
                raise FileNotFoundError(f"{split} 路径不存在: {split_path}")

        print(f"✅ 数据集类别配置: {config['nc']} 类 -> {names}")
        
        print(f"✅ 数据集配置验证通过")
        print(f"   - 训练集: {config['train']}")
        print(f"   - 验证集: {config['val']}")
        print(f"   - 类别数: {config['nc']}")
        print(f"   - 类别名: {config['names']}")
        
        return config
    
    def train(self):
        """开始训练（支持自定义结构魔改）"""
        print("="*60)
        print("🚀 YOLOv11n 口罩检测训练开始")
        print("="*60)
        print(f"模型: {self.model_size}")
        print(f"图像尺寸: {self.img_size}")
        print(f"批次大小: {self.batch_size}")
        print(f"训练轮数: {self.epochs}")
        print(f"设备: {self.device}")
        print("="*60)

        # 验证数据集
        self.verify_data_config()

        # 判断是否使用自定义结构
        use_custom = self.model_size in ["yolo11n_mask_custom", "custom"]
        if use_custom:
            print("\n📥 加载自定义YOLOv11n结构（含GAM/WIoU/P2）...")
            custom_cfg = str(self.project_root / "models" / "configs" / "yolo11n_mask_custom.yaml")
            model = YOLO(custom_cfg)
        else:
            print(f"\n📥 加载YOLOv11n模型...")
            model = YOLO(f'{self.model_size}.pt')

        # 训练配置
        train_args = {
            'data': self.data_path,
            'epochs': 200,  # Dr. Chen: 延长训练轮数到 200
            'imgsz': self.img_size,
            'batch': 16,    # Dr. Chen: 显存受限，物理 Batch 设为 16 (或 32，视显存而定)
            
            # --- 陈金鹏老师的建议 (梯度累积) ---
            # nbs (Nominal Batch Size) 设为 64。
            # 逻辑：如果物理 batch=16，框架会自动累计 64/16 = 4 次梯度再更新。
            # 效果：等效于 64 batch size 的稳定性，解决 Precision 震荡。
            'nbs': 64,      
            # -------------------------------

            'device': self.device,
            'project': self.project,
            'name': 'custom_v2_accum', # 改个名字方便区分
            'exist_ok': True,
            
            # --- Hawkeye 的增强策略 (提升 Recall) ---
            'optimizer': 'auto',
            'workers': self.workers,
            'amp': True,    # 必须开启混合精度以节省显存
            
            # 针对遮挡和密集人群的增强
            'mosaic': 1.0,
            'mixup': 0.2,       # 从 0.1 提升到 0.2
            'copy_paste': 0.3,  # 从 0.1 提升到 0.3 (关键！)
            
            # 几何增强
            'degrees': 15.0,    # 增加旋转角度
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0005,
            'flipud': 0.0,
            'fliplr': 0.5,
            
            # 光照增强
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            
            # 训练策略
            'close_mosaic': 15, # 最后 15 轮关闭 Mosaic，精细微调
            'save': True,
            'save_period': 10,
            'plots': True,
            'verbose': True,
        }

        # 开始训练
        try:
            print(f"\n🎯 开始训练（{self.epochs}轮）...")
            results = model.train(**train_args) if not use_custom else model.train(**train_args)

            print("\n" + "="*60)
            print("✅ 训练完成！")
            print("="*60)

            # 显示训练结果
            print(f"\n📊 训练结果摘要:")
            if hasattr(results, 'results_dict'):
                print(f"   - 最佳mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
                print(f"   - 最佳mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
                print(f"   - 模型保存路径: {results.save_dir}")
            self.visualize_training_curves(getattr(results, 'save_dir', self.project))

            # 导出ONNX（用于微信小程序部署）
            if self.export_onnx_flag:
                self.export_onnx(model, getattr(results, 'save_dir', self.project))

            return results

        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            raise
    
    def export_onnx(self, model, save_dir):
        """导出ONNX格式（开题报告要求：≤80MB）"""
        print(f"\n📦 导出ONNX格式...")
        try:
            self._ensure_onnx_dependencies()

            onnx_path = model.export(
                format='onnx',
                imgsz=self.img_size,
                simplify=True,
                dynamic=False,
            )
            
            # 检查模型大小
            onnx_size = os.path.getsize(onnx_path) / 1024 / 1024  # MB
            print(f"✅ ONNX模型已导出: {onnx_path}")
            print(f"   - 模型大小: {onnx_size:.2f}MB")
            
            if onnx_size > 80:
                print(f"⚠️ 警告：模型大小({onnx_size:.2f}MB)超过开题报告要求(80MB)")
                print(f"   建议：使用量化或剪枝进一步压缩")
            else:
                print(f"✅ 模型大小符合开题报告要求(<80MB)")
            
        except Exception as e:
            print(f"⚠️ ONNX导出失败: {e}")
    
    def _ensure_onnx_dependencies(self):
        required = {
            'onnx': 'onnx==1.19.1',
            'onnxslim': 'onnxslim>=0.1.71',
            'onnxruntime': 'onnxruntime-gpu',
        }

        missing = []
        for module_name, pip_name in required.items():
            if importlib.util.find_spec(module_name) is None:
                missing.append(pip_name)

        if missing:
            install_cmd = 'pip install ' + ' '.join(missing)
            raise RuntimeError(
                "缺少ONNX导出依赖: "
                + ', '.join(missing)
                + f"。请先运行: {install_cmd}"
            )

    def visualize_training_curves(self, save_dir):
        """根据results.csv绘制训练曲线"""
        save_dir = Path(save_dir)
        csv_path = save_dir / 'results.csv'

        if not csv_path.exists():
            print(f"⚠️ 未找到results.csv，跳过训练曲线绘制: {csv_path}")
            return

        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"⚠️ 无法绘制训练曲线，缺少依赖: {exc}")
            return

        df = pd.read_csv(csv_path)
        if 'epoch' not in df.columns:
            print("⚠️ results.csv 缺少epoch列，跳过训练曲线绘制")
            return

        metrics = [
            ("Box Loss", 'train/box_loss'),
            ("Cls Loss", 'train/cls_loss'),
            ("DFL Loss", 'train/dfl_loss'),
            ("mAP@0.5", 'metrics/mAP50(B)'),
            ("mAP@0.5:0.95", 'metrics/mAP50-95(B)'),
            ("Precision", 'metrics/precision(B)'),
            ("Recall", 'metrics/recall(B)'),
        ]

        plotted = [(title, col) for title, col in metrics if col in df.columns]
        if not plotted:
            print("⚠️ results.csv 中没有可绘制的标准列")
            return

        import math

        rows = math.ceil(len(plotted) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for idx, (title, column) in enumerate(plotted):
            axes[idx].plot(df['epoch'], df[column], label=title, color='#1f77b4')
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(column)
            axes[idx].grid(True, linestyle='--', alpha=0.4)

        for idx in range(len(plotted), len(axes)):
            axes[idx].axis('off')

        fig.tight_layout()
        output_path = save_dir / 'training_curves.png'
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"✅ 训练曲线已保存: {output_path}")

    def validate(self, weights_path=None):
        """验证模型"""
        print(f"\n🔍 开始验证...")
        
        if weights_path:
            model = YOLO(weights_path)
        else:
            model = YOLO(f'{self.model_size}.pt')
        
        results = model.val(
            data=self.data_path,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
        )
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv11口罩检测训练')
    parser.add_argument('--data', type=str, default='data/mask_detection.yaml', 
                       help='数据集配置文件')
    parser.add_argument('--model', type=str, default='yolo11n', 
                       help='模型大小 (yolo11n, yolo11s, yolo11m等)')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=-1, 
                       help='批次大小（-1表示自动优化）')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数')
    parser.add_argument('--device', type=str, default='0', 
                       help='设备 (cpu, 0, 1等)')
    parser.add_argument('--project', type=str, default='runs/yolov11_mask_detection',
                       help='训练输出目录')
    parser.add_argument('--workers', type=int, default=None,
                       help='DataLoader线程数量（默认自动）')
    parser.add_argument('--no-export-onnx', action='store_true',
                       help='训练结束后不导出ONNX文件')
    parser.add_argument('--validate', action='store_true',
                       help='仅验证模型')
    parser.add_argument('--weights', type=str, default=None,
                       help='验证时使用的权重文件路径')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YOLOv11MaskDetectionTrainer(
        data_path=args.data,
        model_size=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    project=args.project,
    workers=args.workers,
    export_onnx=not args.no_export_onnx
    )
    
    # 执行训练或验证
    if args.validate:
        trainer.validate(args.weights)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
