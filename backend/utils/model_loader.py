"""
Universal model loader
Priority: YOLOv11 (Ultralytics) while keeping YOLOv7 as baseline comparison.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = ("yolov11", "yolov7")
DEFAULT_PREDICT_CONF = 0.5
DEFAULT_PREDICT_IOU = 0.45
DEFAULT_PREDICT_MAX_DET = 300
DEFAULT_IMAGE_SIZE = 640


@dataclass
class PredictConfig:
    """Configuration used when running inference."""

    conf: float = DEFAULT_PREDICT_CONF
    iou: float = DEFAULT_PREDICT_IOU
    max_det: int = DEFAULT_PREDICT_MAX_DET
    imgsz: int = DEFAULT_IMAGE_SIZE

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


class ModelLoader:
    """Generic loader that supports YOLOv11 and archived YOLOv7 models."""
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_type: str = "auto",
    ) -> None:
        self.project_root = Path(__file__).resolve().parent
        # 修正为项目根目录下的weights目录
        self.weights_dir = Path(__file__).resolve().parents[2] / "weights"
        self.baseline_dir = self.project_root / "baselines" / "yolov7"

        self.device = self._resolve_device(device)
        self.predict_config = PredictConfig()

        self.model_path = model_path or self._get_default_model_path()
        self.model_type = (
            model_type if model_type in SUPPORTED_MODEL_TYPES else self._detect_model_type(self.model_path)
        )

        self.model: Any = None
        self.model_info: Dict[str, Any] = {}
        self._loaded = False

        logger.info("ModelLoader initialised")
        logger.info("  model_path: %s", self.model_path)
        logger.info("  model_type: %s", self.model_type.upper())
        logger.info("  device: %s", self.device)

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device is None or device == "auto":
            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU")
                return torch.device("cuda:0")
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")

        try:
            return torch.device(device)
        except (TypeError, RuntimeError) as exc:  # noqa: BLE001
            logger.warning("Invalid device %s, falling back to CPU (%s)", device, exc)
            return torch.device("cpu")

    def _get_default_model_path(self) -> str:
        # 调试输出权重目录和目标文件
        print(f"[DEBUG] weights_dir: {self.weights_dir.resolve()}")
        custom_onnx = self.weights_dir / "best.onnx"
        print(f"[DEBUG] custom_onnx: {custom_onnx.resolve()} exists: {custom_onnx.exists()}")
        if custom_onnx.exists():
            logger.info("Using custom ONNX weights: %s", custom_onnx)
            return str(custom_onnx)
        if self.weights_dir.exists():
            preferred = [
                "yolov11_mask_best.pt",
                "mask_detection_best.pt",
                "yolov11n_mask_best.pt",
                "yolov11n_mask_last.pt",
                "yolov11n.pt",
            ]
            for name in preferred:
                candidate = self.weights_dir / name
                print(f"[DEBUG] candidate: {candidate.resolve()} exists: {candidate.exists()}")
                if candidate.exists():
                    logger.info("Using local YOLOv11 weights: %s", candidate)
                    return str(candidate)
            for ext in ("*.pt", "*.onnx"):
                matches = sorted(self.weights_dir.glob(ext))
                print(f"[DEBUG] glob {ext}: {[m.resolve() for m in matches]}")
                if matches:
                    logger.info("Using detected local weights: %s", matches[0])
                    return str(matches[0])
        logger.warning("No local weights found, defaulting to Ultralytics yolo11n.pt")
        return "yolo11n.pt"

    def _detect_model_type(self, model_path: str) -> str:
        lower = str(model_path).lower()
        if "yolov7" in lower or "baselines" in lower:
            return "yolov7"
        if Path(model_path).suffix in {".onnx", ".engine"}:
            return "yolov11"
        return "yolov11" if "yolo11" in lower else "yolov11"

    def load_model(self, model_path: Optional[str] = None) -> bool:
        if model_path:
            self.model_path = model_path
            self.model_type = self._detect_model_type(self.model_path)
            logger.info("Updated model path: %s (%s)", self.model_path, self.model_type)

        logger.info("Loading model: %s", self.model_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            if self.model_type == "yolov11":
                return self._load_yolov11_model()
            if self.model_type == "yolov7":
                return self._load_yolov7_model()
            logger.error("Unsupported model type: %s", self.model_type)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load model: %s", exc)
            self._loaded = False
            return False

    def _load_yolov11_model(self) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics is not installed. Please run: pip install ultralytics")
            return False

        self.model = YOLO(self.model_path)
        try:
            self.model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not move model to target device, keeping default (%s)", exc)

        names = getattr(self.model, "names", {0: "no_mask", 1: "mask"})
        parameters = sum(p.numel() for p in self.model.model.parameters()) if hasattr(self.model, "model") else None
        input_size = getattr(self.model.model, "args", {}).get("imgsz", DEFAULT_IMAGE_SIZE)

        self.model_info = {
            "model_path": self.model_path,
            "model_type": "YOLOv11",
            "device": str(self.device),
            "class_names": names,
            "num_classes": len(names),
            "parameters": parameters,
            "input_size": input_size,
        }

        self._loaded = True
        logger.info("YOLOv11 model loaded successfully")
        return True

    def _load_yolov7_model(self) -> bool:
        if not self.baseline_dir.exists():
            logger.error("YOLOv7 baseline directory not found: %s", self.baseline_dir)
            return False

        baseline_path = str(self.baseline_dir)
        if baseline_path not in sys.path:
            sys.path.insert(0, baseline_path)

        try:
            experimental = importlib.import_module("models.experimental")
            torch_utils = importlib.import_module("utils.torch_utils")
        except ModuleNotFoundError as exc:
            logger.error("Failed to import YOLOv7 modules: %s", exc)
            return False

        attempt_load = getattr(experimental, "attempt_load")
        select_device = getattr(torch_utils, "select_device")
        device = select_device(str(self.device))

        weight_path = Path(self.model_path)
        if not weight_path.exists():
            alternative = self.baseline_dir / "weights" / weight_path.name
            if alternative.exists():
                weight_path = alternative
            else:
                logger.error("YOLOv7 weights not found: %s", self.model_path)
                return False

        self.model = attempt_load(str(weight_path), map_location=device)
        self.model.eval()

        names = getattr(self.model, "names", {0: "no_mask", 1: "mask"})
        parameters = sum(p.numel() for p in self.model.parameters())

        self.model_info = {
            "model_path": str(weight_path),
            "model_type": "YOLOv7",
            "device": str(device),
            "class_names": names,
            "num_classes": len(names),
            "parameters": parameters,
            "input_size": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
        }

        self._loaded = True
        logger.info("YOLOv7 model loaded successfully (baseline)")
        return True

    def is_loaded(self) -> bool:
        if self.model_type == "onnx":
            return self._loaded and self.ort_session is not None
        return self._loaded and self.model is not None

    def get_model(self) -> Any:
        if not self.is_loaded():
            logger.warning("Model is not loaded")
            return None
        if self.model_type == "onnx":
            return self.ort_session
        return self.model

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info.copy()

    def get_device_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "device": str(self.device),
            "device_type": self.device.type,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available() and self.device.type == "cuda":
            index = self.device.index or 0
            info.update(
                {
                    "gpu_name": torch.cuda.get_device_name(index),
                    "gpu_memory_total": torch.cuda.get_device_properties(index).total_memory,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(index),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(index),
                }
            )
        return info

    def set_confidence_threshold(self, conf: float) -> None:
        self.predict_config.conf = conf
        if self.model_type == "yolov7" and hasattr(self.model, "conf"):
            self.model.conf = conf

    def set_iou_threshold(self, iou: float) -> None:
        self.predict_config.iou = iou
        if self.model_type == "yolov7" and hasattr(self.model, "iou"):
            self.model.iou = iou

    def set_max_detections(self, max_det: int) -> None:
        self.predict_config.max_det = max_det
        if self.model_type == "yolov7" and hasattr(self.model, "max_det"):
            self.model.max_det = max_det

    def set_image_size(self, imgsz: int) -> None:
        self.predict_config.imgsz = imgsz

    def load_model(self) -> bool:
        if self._loaded:
            logger.info("Model already loaded, skipping reload.")
            return True
        try:
            model_path = self._get_default_model_path()
            print(f"[DEBUG] model_path: {model_path}")
            logger.info(f"  model_path: {model_path}")
            if model_path.endswith(".onnx"):
                try:
                    import onnxruntime as ort
                    print("[DEBUG] Loading ONNX model via onnxruntime...")
                    self.ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                    self.model_type = "onnx"
                    logger.info(f"ONNX model loaded: {model_path}")
                except Exception as e:
                    print(f"[ERROR] ONNX load failed: {e}")
                    logger.error(f"ONNX load failed: {e}")
                    raise
            else:
                raise RuntimeError("仅支持ONNX推理，请先导出best.onnx到weights目录！")
            self._loaded = True
            logger.info(f"ModelLoader initialised")
            logger.info(f"  model_path: {model_path}")
            logger.info(f"  model_type: {self.model_type}")
            logger.info(f"  device: {self.device}")
            return True
        except Exception as exc:
            print(f"[FATAL] Model loading failed: {exc}")
            import traceback
            traceback.print_exc()
            logger.error(f"Model loading failed: {exc}")
            return False

        import time
        import numpy as np

        logger.info("Starting benchmark for %d runs", 10)
        imgsz = self.predict_config.imgsz
        if self.model_type == "onnx":
            logger.info("ONNX model benchmark skipped.")
            return {}
        dummy = torch.randn(1, 3, imgsz, imgsz, device=self.device)

        if self.model_type == "yolov11":
            for _ in range(3):
                _ = self.model.model(dummy)  # type: ignore[arg-type]
        else:
            for _ in range(3):
                _ = self.model(dummy)

        times = []
        for _ in range(10):
            start = time.time()
            if self.model_type == "yolov11":
                _ = self.model.model(dummy)  # type: ignore[arg-type]
            else:
                _ = self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

        arr = np.array(times)
        stats = {
            "mean_time": float(arr.mean()),
            "std_time": float(arr.std()),
            "min_time": float(arr.min()),
            "max_time": float(arr.max()),
            "fps": float(1.0 / arr.mean()),
            "runs": 10,
        }
        logger.info("Benchmark finished, mean latency %.2f ms", stats["mean_time"] * 1e3)
        return stats
