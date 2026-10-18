"""Step 4: run the trained segmentation model on test images."""
from ultralytics import YOLO

model = YOLO("runs/segment/road_segmentation_model/weights/best.pt")

model.predict(source="test_images/", save=True, conf=0.5)
