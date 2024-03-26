# Custom YOLOv7 Model for Human Face Detection

Detecting human faces with precision using a custom YOLOv7 model. Train your own YOLOv7 model specialized in detecting human faces with high accuracy. This repository provides a comprehensive guide and codebase for training, testing, and deploying a custom YOLOv7 model tailored specifically for face detection tasks. Join the realm of computer vision and empower your applications with robust face detection capabilities.

## Instructions

Follow these steps to get started with the project:

1. Clone this repository:
```bash
   git clone https://github.com/ManikSinghSarmaal/YOLO_custom
```
2. Create a new conda environment and activate it
conda create --name yolov7_env python=3.11
conda activate yolov7_env

3. Navigate to the cloned repository
cd YOLOv7_custom

4. Install the required dependencies
pip install -r requirements.txt

# Run YOLOV7 MODEL

5. Run the check_preds.py script to test detections using webcam feed for YOLOV7
```bash
python check_preds.py
```

# Run YOLOV7_light_model

5. Run the check_preds.py script to test detections using webcam feed for YOLOV7_light
```bash
python test_light.py
```
Optional: Testing with Images or Videos:

Modify the source_path variable within the check_preds.py script to specify an image or video file path (refer to the script for details).

* Model Weights Location: 

yolov7/weights/best.pt: Weights for the yolov7 model.
yolov7_light/weights/best.pt: Weights for the yolov7_light model.
** Training Your Own Custom Model**

```bash
#Note : Model weights are in MODEL_DATA dir 
├── yolov7
│   └── weights
│       └── best.pt #yolov7
└── yolov7_light
    └── weights
        └── best.pt #yolov7_light_model
```