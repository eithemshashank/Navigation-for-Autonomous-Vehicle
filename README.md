# Navigation for Autonomous Vehicle

A computer-vision project that runs two YOLOv8 models on a video and shows a navigation command (STOP, LEFT, RIGHT or FORWARD) on screen, with the road area highlighted.

**Tech:** Python, YOLOv8 (Ultralytics), OpenCV, NumPy

## How it works

1. **Detection model** (`yolov8n.pt`, fine-tuned for 50 epochs) detects objects in each frame and draws a labelled box for every detection with confidence above 0.5.
2. **Command:** if a detection is labelled `stop`, `left`, `right` or `forward`, that label becomes the command shown on screen. With no such detection the command is FORWARD.
3. **Segmentation model** (`yolov8n-seg.pt`, fine-tuned for 50 epochs) predicts the road area, which is drawn as a blue overlay.

## Files

| File | Purpose |
|---|---|
| `train_detection.py` | train the detection model |
| `test_detection.py` | run it on test images |
| `train_segmentation.py` | train the road-segmentation model |
| `test_segmentation.py` | run it on test images |
| `navigate.py` | run both models on a video and show the command |

## Run it

```
pip install -r requirements.txt
python train_detection.py
python train_segmentation.py
python navigate.py
```

You need your own datasets in YOLO format (`dataset/data.yaml` for detection, `seg_dataset/data.yaml` for segmentation) and a `test_video.mp4`. Datasets, trained weights and videos are not stored in this repository.

## Limitations

- The command comes only from the detection labels. The segmentation result is shown as an overlay and does not change the command.
- It does not measure distance or judge where an obstacle is relative to the vehicle's path.
- It is a demonstration on recorded video, not a tested driving system.
- No accuracy figures are included in this repository yet.

## Author

Eithem Shashank
