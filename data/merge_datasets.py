import os
import shutil
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import yaml
import random

def setup_dirs(base_path):
    dirs = [
        base_path / 'images' / 'train',
        base_path / 'images' / 'val',
        base_path / 'labels' / 'train',
        base_path / 'labels' / 'val'
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_maskdata4(xml_dir, img_dir, out_img_dirs, out_label_dirs, class_mapping):
    print("Converting maskdata4 (XML) to YOLO...")
    xml_files = list(Path(xml_dir).glob('*.xml'))
    
    for xml_file in tqdm(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        # Handle potential filename mismatch or extension issues
        src_img_path = Path(img_dir) / filename
        if not src_img_path.exists():
            # Try png/jpg variations
            if (Path(img_dir) / f"{xml_file.stem}.png").exists():
                src_img_path = Path(img_dir) / f"{xml_file.stem}.png"
            elif (Path(img_dir) / f"{xml_file.stem}.jpg").exists():
                src_img_path = Path(img_dir) / f"{xml_file.stem}.jpg"
            else:
                # print(f"Skipping {filename}, image not found.")
                continue

        # Get image size
        size = root.find('size')
        if size is None:
            continue
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w == 0 or h == 0:
            continue
            
        yolo_lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_mapping:
                continue
            cls_id = class_mapping[cls_name]
            
            xml_box = obj.find('bndbox')
            b = (float(xml_box.find('xmin').text), float(xml_box.find('xmax').text),
                 float(xml_box.find('ymin').text), float(xml_box.find('ymax').text))
            bb = convert_box((w, h), b)
            yolo_lines.append(f"{cls_id} {' '.join([str(a) for a in bb])}")
            
        if yolo_lines:
            # 90/10 split
            is_train = random.random() < 0.9
            target_img_dir = out_img_dirs[0] if is_train else out_img_dirs[1]
            target_lbl_dir = out_label_dirs[0] if is_train else out_label_dirs[1]
            
            dst_img_path = target_img_dir / f"m4_{src_img_path.name}"
            dst_lbl_path = target_lbl_dir / f"m4_{src_img_path.stem}.txt"
            
            shutil.copy(src_img_path, dst_img_path)
            with open(dst_lbl_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

def copy_yolo_dataset(src_img_dir, src_lbl_dir, out_img_dirs, out_lbl_dirs, prefix):
    print(f"Copying YOLO dataset from {src_img_dir}...")
    
    img_files = list(Path(src_img_dir).glob('*.*')) 
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = [x for x in img_files if x.suffix.lower() in valid_extensions]
    
    for img_path in tqdm(img_files):
        lbl_name = img_path.stem + ".txt"
        src_lbl_path = Path(src_lbl_dir) / lbl_name
        
        if not src_lbl_path.exists():
            continue
            
        is_train = random.random() < 0.9
        target_img_dir = out_img_dirs[0] if is_train else out_img_dirs[1]
        target_lbl_dir = out_lbl_dirs[0] if is_train else out_lbl_dirs[1]
        
        dst_img_path = target_img_dir / f"{prefix}_{img_path.name}"
        dst_lbl_path = target_lbl_dir / f"{prefix}_{lbl_name}"
        
        shutil.copy(img_path, dst_img_path)
        shutil.copy(src_lbl_path, dst_lbl_path)

def main():
    base_dir = Path(r"e:\graduation_project\Mask-Wearing-Detection")
    combined_dir = base_dir / "data" / "combined_dataset"
    
    # 1. Setup Directories
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    pass
    # Re-create
    img_dirs = [combined_dir / 'images' / 'train', combined_dir / 'images' / 'val']
    lbl_dirs = [combined_dir / 'labels' / 'train', combined_dir / 'labels' / 'val']
    
    for d in img_dirs + lbl_dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Define Class Mapping (for Maskdata4)
    mapping = {
        'with_mask': 0,
        'mask_weared_incorrect': 1,
        'without_mask': 2
    }
    
    # 3. Process Maskdata1
    # Path: maskdataset\maskdata1\mark-datas\images\train
    m1_img = base_dir / "maskdataset/maskdata1/mark-datas/images/train"
    m1_lbl = base_dir / "maskdataset/maskdata1/mark-datas/lables/train" # Note typo lables
    if m1_img.exists() and m1_lbl.exists():
        copy_yolo_dataset(m1_img, m1_lbl, img_dirs, lbl_dirs, "m1")
    else:
        print(f"Maskdata1 paths not found: {m1_img}")

    # 4. Process Maskdata3
    # Path: maskdataset\maskdata3\images\train
    m3_img = base_dir / "maskdataset/maskdata3/images/train"
    m3_lbl = base_dir / "maskdataset/maskdata3/labels/train"
    if m3_img.exists() and m3_lbl.exists():
        copy_yolo_dataset(m3_img, m3_lbl, img_dirs, lbl_dirs, "m3")
    else:
        print(f"Maskdata3 paths not found: {m3_img}")

    # 5. Process Maskdata4
    m4_xml = base_dir / "maskdataset/maskdata4/annotations"
    m4_img = base_dir / "maskdataset/maskdata4/images"
    if m4_xml.exists() and m4_img.exists():
        convert_maskdata4(m4_xml, m4_img, img_dirs, lbl_dirs, mapping)
    else:
        print("Maskdata4 paths not found")

    # 6. Create YAML
    yaml_content = {
        'path': f"{combined_dir.absolute().as_posix()}",
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val', 
        'nc': 3,
        'names': ['R_mask', 'W_mask', 'N_mask']
    }
    
    with open(base_dir / "data/mask_detection_combined.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print("Dataset merge completed. YAML saved to data/mask_detection_combined.yaml")

if __name__ == "__main__":
    main()
