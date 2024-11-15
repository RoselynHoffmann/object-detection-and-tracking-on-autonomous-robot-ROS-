from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")