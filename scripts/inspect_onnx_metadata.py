
import onnxruntime as ort
import json

file_path = r"e:\graduation_project\Mask-Wearing-Detection\weights\best.onnx"

try:
    sess = ort.InferenceSession(file_path, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta()
    custom_metadata_map = meta.custom_metadata_map
    print("Custom Metadata Map Keys:", custom_metadata_map.keys())
    
    if "names" in custom_metadata_map:
        print("Names:", custom_metadata_map["names"])
    
    # 也可以尝试从 inputs/outputs 推断形状，虽然看不出类别名
    for input_meta in sess.get_inputs():
        print(f"Input: {input_meta.name}, Shape: {input_meta.shape}")
    for output_meta in sess.get_outputs():
        print(f"Output: {output_meta.name}, Shape: {output_meta.shape}")

except Exception as e:
    print(f"Error loading model: {e}")
