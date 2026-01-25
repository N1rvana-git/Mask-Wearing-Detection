
import onnx
import sys

path = "E:/graduation_project/Mask-Wearing-Detection/runs/yolov11_mask_detection/custom_v2_accum/weights/best.onnx"
try:
    print(f"Checking {path}...")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    print("✅ Model is valid ONNX.")
    sys.exit(0)
except Exception as e:
    print(f"❌ Model is invalid: {e}")
    sys.exit(1)
