"""Step 3: train the road-segmentation model (YOLOv8 nano, segmentation)."""
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="seg_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="road_segmentation_model",
)
