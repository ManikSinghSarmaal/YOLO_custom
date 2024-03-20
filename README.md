# YOLO_custom
 Detecting human faces with precision using custom YOLOv7 model. Train your own YOLOv7 model specialized in detecting human faces with high accuracy. This repository provides a comprehensive guide and codebase for training, testing, and deploying a custom YOLOv7 model tailored specifically for face detection tasks.

# Custom YOLOv7 Model for Human Face Detection

Detecting human faces with precision using a custom YOLOv7 model. Train your own YOLOv7 model specialized in detecting human faces with high accuracy. This repository provides a comprehensive guide and codebase for training, testing, and deploying a custom YOLOv7 model tailored specifically for face detection tasks. Join the realm of computer vision and empower your applications with robust face detection capabilities.

## Instructions

Follow these steps to get started with the project:

1. Clone this repository:
```bash
   git clone https://github.com/ManikSinghSarmaal/YOLO_custom

# Create a new conda environment and activate it
conda create --name yolov7_env python=3.8
conda activate yolov7_env

# Navigate to the cloned repository
cd YOLOv7_custom

# Install the required dependencies
pip install -r requirements.txt

# Run the check_preds.py script to test detections using webcam feed
python check_preds.py