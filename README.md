# Custom YOLOv7 Model for Human Face Detection

Detecting human faces with precision using a custom YOLOv7 model. Train your own YOLOv7 model specialized in detecting human faces with high accuracy. This repository provides a comprehensive guide and codebase for training, testing, and deploying a custom YOLOv7 model tailored specifically for face detection tasks. Join the realm of computer vision and empower your applications with robust face detection capabilities.

## Instructions

Follow these steps to get started with the project:

1. Clone this repository:
```bash
   git clone https://github.com/ManikSinghSarmaal/YOLO_custom
```
# Create a new conda environment and activate it
conda create --name yolov7_env python=3.11
conda activate yolov7_env

# Navigate to the cloned repository
cd YOLOv7_custom

# Install the required dependencies
pip install -r requirements.txt

## YOLOV7 MODEL

# Run the check_preds.py script to test detections using webcam feed for YOLOV7
```bash
python check_preds.py
```

## YOLOV7_light_model

# Run the check_preds.py script to test detections using webcam feed for YOLOV7_light
```bash
python test_light.py
```
# Optionally, you can modify the check_preds.py script to change the input source to an image file or video by adjusting the source_path variable.

```bash
#Note : Model weights are in MODEL_DATA dir 
├── yolov7
│   └── weights
│       └── best.pt #yolov7
└── yolov7_light
    └── weights
        └── best.pt #yolov7_light_model
```