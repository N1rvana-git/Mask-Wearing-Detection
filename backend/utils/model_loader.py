"""
Universal model loader
Priority: YOLOv11 (Ultralytics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = ("yolov11",)
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
    """Generic loader that supports YOLOv11 models."""
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_type: str = "auto",
    ) -> None:
        self.project_root = Path(__file__).resolve().parent
        # 修正为项目根目录下的weights目录
        self.weights_dir = Path(__file__).resolve().parents[2] / "weights"

        self.device = self._resolve_device(device)
        self.predict_config = PredictConfig()

        self.model_path = model_path or self._get_default_model_path()
        self.model_type = "yolov11" # Default to yolov11/onnx

        self.model: Any = None
        self.ort_session: Any = None
        self.model_info: Dict[str, Any] = {}
        self._loaded = False

        logger.info("ModelLoader initialised")
        logger.info("  model_path: %s", self.model_path)
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
            
            # Fallback to any pt/onnx
            for ext in ("*.pt", "*.onnx"):
                matches = sorted(self.weights_dir.glob(ext))
                if matches:
                    logger.info("Using detected local weights: %s", matches[0])
                    return str(matches[0])
                    
        logger.warning("No local weights found, defaulting to Ultralytics yolo11n.pt")
        return "yolo11n.pt"

    def load_model(self, model_path: Optional[str] = None) -> bool:
        if self._loaded and not model_path:
            logger.info("Model already loaded, skipping reload.")
            return True
            
        if model_path:
            self.model_path = model_path
            
        logger.info("Loading model: %s", self.model_path)
        
        try:
            if self.model_path.endswith(".onnx"):
                return self._load_onnx_model()
            else:
                return self._load_pt_model()
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            self._loaded = False
            return False

    def _load_onnx_model(self) -> bool:
        try:
            import onnxruntime as ort
            logger.info("Loading ONNX model via onnxruntime...")
            
            providers = ["CPUExecutionProvider"]
            if torch.cuda.is_available():
                providers.insert(0, "CUDAExecutionProvider")
                
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
            self.model_type = "onnx"
            
            # Basic info
            inputs = self.ort_session.get_inputs()
            input_shape = inputs[0].shape
            self.model_info = {
                "model_path": self.model_path,
                "model_type": "ONNX",
                "device": str(self.device),
                "input_shape": input_shape
            }
            
            self._loaded = True
            logger.info(f"ONNX model loaded: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX load failed: {e}")
            raise

    def _load_pt_model(self) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics is not installed. Please run: pip install ultralytics")
            return False

        self.model = YOLO(self.model_path)
        try:
            self.model.to(self.device)
        except Exception as exc:
            logger.warning("Could not move model to target device, keeping default (%s)", exc)

        names = getattr(self.model, "names", {0: "no_mask", 1: "mask"})
        
        self.model_info = {
            "model_path": self.model_path,
            "model_type": "YOLOv11",
            "device": str(self.device),
            "class_names": names,
            "num_classes": len(names),
        }

        self._loaded = True
        logger.info("YOLOv11 model loaded successfully")
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

    def set_iou_threshold(self, iou: float) -> None:
        self.predict_config.iou = iou

    def set_max_detections(self, max_det: int) -> None:
        self.predict_config.max_det = max_det

    def set_image_size(self, imgsz: int) -> None:
        self.predict_config.imgsz = imgsz
