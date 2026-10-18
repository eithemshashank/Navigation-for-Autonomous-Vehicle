"""Step 1: train the object-detection model (YOLOv8 nano) on your dataset."""
from ultralytics import YOLO

# Start from the pre-trained YOLOv8 nano weights and fine-tune them.
model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="object_detection_model",
)
