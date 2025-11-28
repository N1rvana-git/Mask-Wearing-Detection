"""
Mask detection API built around YOLOv11 as the primary model.
Falls back to the archived YOLOv7 implementation for baseline comparisons.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from backend.utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_MAX_DETECTIONS = 300


class MaskDetectionAPI:
    """High level detection API that wraps the model loader."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_loader = ModelLoader(model_path=model_path, device=device)

        self.conf_threshold = DEFAULT_CONFIDENCE
        self.iou_threshold = DEFAULT_IOU
        self.max_detections = DEFAULT_MAX_DETECTIONS
        self.img_size = DEFAULT_IMAGE_SIZE

        self.class_names: Dict[int, str] = {0: "mask", 1: "no_mask"}
        
        self.colors = {
            "no_mask": (0, 0, 255), # 没戴口罩显示红色框
            "mask": (0, 255, 0),    # 戴口罩显示绿色框
        }

        self.detection_stats = {
            "total_detections": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0,
        }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        try:
            if not self.model_loader.load_model():
                logger.error("Failed to load model")
                return False

            # Sync detection parameters to loader
            self.model_loader.set_confidence_threshold(self.conf_threshold)
            self.model_loader.set_iou_threshold(self.iou_threshold)
            self.model_loader.set_max_detections(self.max_detections)
            self.model_loader.set_image_size(self.img_size)

            info = self.model_loader.get_model_info()
            names = info.get("class_names")
            if isinstance(names, dict):
                self.class_names = {int(k): v for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                self.class_names = {idx: name for idx, name in enumerate(names)}

            logger.info("Detection API initialised using %s", info.get("model_type", "unknown model"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialise detection API: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Detection entry points
    # ------------------------------------------------------------------
    def detect_image(self, image: np.ndarray, return_annotated: bool = True) -> Dict[str, Any]:
        if not self.model_loader.is_loaded():
            logger.error("Model is not loaded")
            return self._create_error_result("Model not loaded")

        start_time = time.time()
        try:
            if self.model_loader.model_type in ("yolov11", "onnx"):
                result = self._detect_with_yolov11(image, return_annotated)
            else:
                result = self._detect_with_yolov7(image, return_annotated)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Detection failed: %s", exc)
            return self._create_error_result(f"Detection failed: {exc}")

        inference_time = time.time() - start_time
        result["inference_time"] = inference_time
        self._update_stats(inference_time)

        logger.info(
            "Detection finished: %d objects, %.2f ms",
            result.get("detection_count", 0),
            inference_time * 1000,
        )
        print("[DEBUG] 推理结果:", result)
        return result

    def detect_batch(self, images: List[np.ndarray], return_annotated: bool = True) -> List[Dict[str, Any]]:
        results = []
        for index, image in enumerate(images):
            logger.info("Processing image %d/%d", index + 1, len(images))
            results.append(self.detect_image(image, return_annotated=return_annotated))
        return results

    # ------------------------------------------------------------------
    # YOLOv11 path
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 替换 backend/api/detection.py 中的 _detect_with_yolov11 方法
    # ------------------------------------------------------------------
    def _detect_with_yolov11(self, image: np.ndarray, return_annotated: bool) -> Dict[str, Any]:
        import cv2
        import numpy as np

        # 1. ONNX 推理分支
        if self.model_loader.model_type == "onnx":
            ort_session = self.model_loader.ort_session
            input_name = ort_session.get_inputs()[0].name
            
            # 图像预处理
            img_h, img_w = image.shape[:2]
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1)) 
            img = np.expand_dims(img, axis=0)
            
            # 执行推理
            outputs = ort_session.run(None, {input_name: img})
            
            # 2. 后处理 (Decoding & NMS)
            # 输出形状通常是 (1, 4+nc, 8400) -> 需要转置为 (1, 8400, 4+nc)
            pred = np.transpose(outputs[0], (0, 2, 1))
            data = pred[0] # 取第一个 batch, 形状 (8400, 7) [cx, cy, w, h, score_cls1, score_cls2, score_cls3]
            
            boxes = []
            confidences = []
            class_ids = []
            
            # 提取数据
            # data[:, 0:4] 是坐标
            # data[:, 4:] 是类别概率
            scores = data[:, 4:]
            # 获取每个框的最大类别分数和索引
            max_scores = np.max(scores, axis=1)
            argmax_indices = np.argmax(scores, axis=1)
            
            # 过滤低置信度
            mask = max_scores > self.conf_threshold
            filtered_boxes = data[mask, 0:4]
            filtered_scores = max_scores[mask]
            filtered_classes = argmax_indices[mask]
            
            # 转换坐标 & 收集用于 NMS 的数据
            # YOLO 输出是 [cx, cy, w, h] (归一化或像素值，这里通常是基于 640x640 的像素值)
            # 我们需要还原到原图尺寸
            scale_x = img_w / self.img_size
            scale_y = img_h / self.img_size
            
            for i in range(len(filtered_boxes)):
                cx, cy, w, h = filtered_boxes[i]
                
                # 还原到原图坐标
                # 转换中心点到左上角
                x = int((cx - w / 2) * scale_x)
                y = int((cy - h / 2) * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                
                boxes.append([x, y, w, h])
                confidences.append(float(filtered_scores[i]))
                class_ids.append(int(filtered_classes[i]))
            
            # 执行 NMS (非极大值抑制)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
            
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    conf = confidences[i]
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detections.append({
                        "bbox": [x, y, x + w, y + h], # 转为 [x1, y1, x2, y2]
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": class_name,
                        "center": [x + w // 2, y + h // 2],
                        "area": w * h,
                    })

            if return_annotated:
                annotated_image = self._annotate_image(image.copy(), detections)
                
            return {
                "success": True,
                "detections": detections,
                "image_shape": image.shape,
                "detection_count": len(detections),
                "annotated_image": annotated_image if return_annotated else None,
            }

        # PyTorch/Ultralytics 分支 (保持不变)
        model = self.model_loader.get_model()
        device_arg = str(self.model_loader.device)
        half_precision = self.model_loader.device.type == "cuda"

        results = model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            imgsz=self.img_size,
            device=device_arg,
            half=half_precision,
            verbose=False,
        )
        
        # ... (后续 PyTorch 处理逻辑保持原样，或者直接复制原来的代码)
        # 为简洁起见，如果你的系统只用 ONNX，上面的 ONNX 分支已经足够。
        # 如果需要保留 PyTorch 分支，请保留原文件里 `if results:` 之后的部分
        
        detections: List[Dict[str, Any]] = []
        annotated_image: Optional[np.ndarray] = None

        if results:
            result = results[0]
            names = result.names if hasattr(result, "names") else self.class_names
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for idx, coords in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = coords.tolist()
                    conf = float(confidences[idx])
                    class_id = int(classes[idx])
                    class_name = names[class_id] if isinstance(names, dict) else names[class_id]
                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class_id": class_id,
                            "class_name": class_name,
                            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            "area": int((x2 - x1) * (y2 - y1)),
                        }
                    )

            if return_annotated:
                annotated_image = result.plot()

        return {
            "success": True,
            "detections": detections,
            "image_shape": image.shape,
            "detection_count": len(detections),
            "annotated_image": annotated_image if return_annotated else None,
        }

    # ------------------------------------------------------------------
    # YOLOv7 path (baseline only)
    # ------------------------------------------------------------------
    def _detect_with_yolov7(self, image: np.ndarray, return_annotated: bool) -> Dict[str, Any]:
        try:
            letterbox_module = importlib.import_module("utils.datasets")
            general_module = importlib.import_module("utils.general")
        except ModuleNotFoundError as exc:
            logger.error("YOLOv7 utilities not available: %s", exc)
            return self._create_error_result("YOLOv7 utilities not available")

        letterbox = getattr(letterbox_module, "letterbox")
        non_max_suppression = getattr(general_module, "non_max_suppression")
        scale_coords = getattr(general_module, "scale_coords")

        processed = letterbox(image, self.img_size, stride=32)[0]
        processed = processed[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB
        processed = np.ascontiguousarray(processed)
        tensor = torch.from_numpy(processed).to(self.model_loader.device)
        tensor = tensor.float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = self.model_loader.get_model()(tensor)

        predictions = non_max_suppression(
            predictions,
            self.conf_threshold,
            self.iou_threshold,
            max_det=self.max_detections,
        )

        detections: List[Dict[str, Any]] = []
        for pred in predictions:
            if pred is None or not len(pred):
                continue
            pred[:, :4] = scale_coords((self.img_size, self.img_size), pred[:, :4], image.shape[:2]).round()
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class_id": class_id,
                        "class_name": class_name,
                        "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                        "area": (x2 - x1) * (y2 - y1),
                    }
                )

        annotated_image = self._annotate_image(image.copy(), detections) if return_annotated else None
        return {
            "success": True,
            "detections": detections,
            "image_shape": image.shape,
            "detection_count": len(detections),
            "annotated_image": annotated_image,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _annotate_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            color = self.colors.get(class_name, (255, 255, 255))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return image

    def _create_error_result(self, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": message,
            "detections": [],
            "detection_count": 0,
        }

    def _update_stats(self, inference_time: float) -> None:
        self.detection_stats["total_detections"] += 1
        self.detection_stats["total_inference_time"] += inference_time
        self.detection_stats["average_inference_time"] = (
            self.detection_stats["total_inference_time"] / max(1, self.detection_stats["total_detections"])
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = self.detection_stats.copy()
        stats.update(
            {
                "model_info": self.model_loader.get_model_info(),
                "device_info": self.model_loader.get_device_info(),
                "detection_params": {
                    "conf_threshold": self.conf_threshold,
                    "iou_threshold": self.iou_threshold,
                    "max_detections": self.max_detections,
                    "img_size": self.img_size,
                },
            }
        )
        return stats

    def set_detection_params(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        img_size: Optional[int] = None,
    ) -> None:
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            self.model_loader.set_confidence_threshold(conf_threshold)
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            self.model_loader.set_iou_threshold(iou_threshold)
        if max_detections is not None:
            self.max_detections = max_detections
            self.model_loader.set_max_detections(max_detections)
        if img_size is not None:
            self.img_size = img_size
            self.model_loader.set_image_size(img_size)

        logger.info(
            "Detection params updated: conf=%.2f, iou=%.2f, max_det=%d, img_size=%d",
            self.conf_threshold,
            self.iou_threshold,
            self.max_detections,
            self.img_size,
        )

    def reset_stats(self) -> None:
        self.detection_stats = {
            "total_detections": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0,
        }

    def cleanup(self) -> None:
        self.model_loader.unload_model()


_detection_api: Optional[MaskDetectionAPI] = None


def get_detection_api() -> Optional[MaskDetectionAPI]:
    global _detection_api
    if _detection_api is None:
        api = MaskDetectionAPI()
        if not api.initialize():
            return None
        _detection_api = api
    return _detection_api


def initialize_detection_api(model_path: Optional[str] = None, device: Optional[str] = None) -> bool:
    global _detection_api
    api = MaskDetectionAPI(model_path=model_path, device=device)
    if not api.initialize():
        return False
    _detection_api = api
    return True
