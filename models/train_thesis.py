from ultralytics import YOLO
import os

def main():
    # Load the model
    # User mentioned "my yolo model" and yolo11n.pt is in the root. 
    # Using 'yolo11n.pt' for transfer learning.
    pt_path = r"E:\graduation_project\Mask-Wearing-Detection\yolo11n.pt"
    model = YOLO(pt_path) 

    # Train the model
    results = model.train(
        data=r'E:\graduation_project\Mask-Wearing-Detection\data\mask_detection_thesis.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/mask_detection_thesis',
        name='yolov11_200epochs',
        exist_ok=True,
        plots=True,       # Save plots
        save=True,        # Save checkpoints
        save_period=10,   # Save checkpoint every 10 epochs (more records)
        verbose=True
    )
    
    print("Training finished.")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()
