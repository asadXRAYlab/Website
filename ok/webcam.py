from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# object classes

def webcam():

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
