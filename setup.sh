#!/bin/bash

# Define your project banner
echo "
========================================
   Welcome to the Trash Detection Project
   🚀 AI-Powered Object Detection System
========================================
📂 Current Directory: $(pwd)
📅 Date: $(date)
🔧 Running on: $(uname -a)
"
echo "Creating necessary files.."

git clone https://github.com/ultralytics/ultralytics.git
git clone https://github.com/pedropro/TACO.git
$(python -m venv yolo-flask-env)
source yolo-flask-env/bin/activate
$(pip install -r requirements.txt)
# Stole data from here cudos to person who started this project, we need more higher quality data
# $(python ./TACO/download.py)
# $(python ./convert_to_yolo.py)
$(python ./generate_yaml.py)
$(yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640)
$(yolo val model=runs/detect/train/weights/best.pt data=data.yaml)
$(yolo predict model=runs/detect/train/weights/best.pt source=./test_image.jpg)
$(python ./test_detection.py)

tree -d -I "__pycache__" > directory_structure.md  