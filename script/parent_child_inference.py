from ultralytics import YOLO
import cv2
from typing import TypedDict

class DetectionResult(TypedDict):
    bbox: list
    label: list

class YOLODetector:
    def __init__(self, model_path: str, class_ids: list[int] = None):
        self.model = YOLO(model_path)
        self.class_ids = class_ids

    def detect(self, frame):
        result = self.model(frame, classes=self.class_ids if self.class_ids else None)
        bbox = result[0].boxes.xyxy
        
        return bbox

class Inference:
    def __init__(self, model_list):
        self.model_list = model_list

    def run(self, camera_path):
        cap = cv2.VideoCapture(camera_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

    def process(self, frame):
        images = frame
        det_result = []
        for model in self.model_list:
            result = model.detect(images)
            
            images = [images[int(bbox[1]):int(bbox[3]),int(bbox[1]):int(bbox[2])] for bbox in result]

if __name__ == "__main__":
    # model = YOLODetector(r"C:\Users\AQI\JOB\ppe_detection\yolo26n.pt")

    # print(model.detect(r"C:\Users\AQI\Pictures\vlcsnap-2026-08-10-17h21m31s716.png"))

    model = YOLO(r"C:\Users\AQI\JOB\ppe_detection\yolo26n.pt")

    result = model(r"C:\Users\AQI\Pictures\vlcsnap-2026-08-10-17h21m31s716.png")

    print(result[0].boxes.xyxyn)

