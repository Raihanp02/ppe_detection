from ultralytics import YOLO

class YOLODetection:
    def __init__(self, yolo):
        self.yolo = YOLO(yolo)

    def predict(self, image):
        result = self.yolo(source=image,classes=[0])
        return result

    def postprocess(self, result):
        bbox = []

        for box in result[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())

            bbox.append([x1,y1,x2,y2])

        return bbox