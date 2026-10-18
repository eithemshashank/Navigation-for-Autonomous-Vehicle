"""Step 2: run the trained detection model on test images."""
from ultralytics import YOLO

model = YOLO("runs/detect/object_detection_model/weights/best.pt")

model.predict(source="test_images/", save=True, conf=0.5)
