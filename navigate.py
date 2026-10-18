"""Step 5: run both models on a video and show the navigation command.

The detection model finds objects and signs in each frame. If it detects a
class named stop / left / right / forward with confidence above 0.5, that
becomes the command shown on screen (the default is FORWARD). The
segmentation model draws the road area as a blue overlay.

Press q to quit.
"""
import cv2
import numpy as np
from ultralytics import YOLO

COMMANDS = {"stop": "STOP", "left": "LEFT", "right": "RIGHT", "forward": "FORWARD"}

det_model = YOLO("runs/detect/object_detection_model/weights/best.pt")
seg_model = YOLO("runs/segment/road_segmentation_model/weights/best.pt")

cap = cv2.VideoCapture("test_video.mp4")  # use 0 instead for a webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Object detection: draw boxes and pick the command from the labels.
    direction = "FORWARD"
    for result in det_model(frame):
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf <= 0.5:
                continue
            label = det_model.names[int(box.cls[0])]
            direction = COMMANDS.get(label.lower(), direction)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 2. Road segmentation: overlay the predicted road mask in blue.
    combined = frame.copy()
    for result in seg_model(frame):
        if result.masks is None:
            continue
        for mask in result.masks.data:
            mask = mask.cpu().numpy()
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = (mask * 255).astype(np.uint8)
            blue = np.zeros_like(frame)
            blue[:, :, 0] = mask
            combined = cv2.addWeighted(combined, 1, blue, 0.3, 0)

    # 3. Show the command.
    cv2.putText(combined, f"Direction: {direction}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Autonomous Vehicle Navigation", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
