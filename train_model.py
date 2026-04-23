import torch
from ultralytics import YOLO
import os

def train():
    # 1. Hardware Check
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not detected. Training will be slow on CPU.")

    # 2. Load the model
    # We use the nano version for speed and efficiency
    model = YOLO('yolov8n.pt') 

    # 3. Start Training
    # The 'data' path should be the absolute path to your .yaml file
    results = model.train(
        data='aruco.yaml', 
        epochs=30,         
        imgsz=640,         
        batch=16,          # Reduced slightly to ensure no memory errors on 3060
        device=0,          
        workers=4,         # Number of CPU cores for data loading
        project='aruco_project',
        name='hybrid_model'
    )

if __name__ == '__main__':
    # This block is MANDATORY on Windows to prevent the RuntimeError
    train()