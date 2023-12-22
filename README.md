## YOLOv8 Root Detection

This repository contains code for training a YOLOv8 model on custom datasets for root detection.

### Steps to Train the Model:
1. **Model Initialization:** 
    Initialize YOLOv8 using the provided pretrained weights (`yolov8s.pt`).
    ```
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")
    ```

2. **Custom Dataset Setup:**
    - Set the path to your custom dataset in the training configuration file (`data.yaml`).
  
3. **Model Training:**
    Train the model with your dataset for a specified number of epochs (e.g., 30).
    ```
    model.train(data="C:/Users/Admin/Desktop/yolov8Root-Detection/datasets/data.yaml", epochs=30)
    ```
4. **Import Modules:**
    pip install -r requirement.txt

5. **For Running the root_detector.py**
    python3 root_detector.py 
    open any browser and enter 
    localhost:8080