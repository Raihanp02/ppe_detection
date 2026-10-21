from pathlib import Path
import cv2
from uuid import uuid4

from people_detection import YOLODetection

BASE_DIR = Path(__file__).resolve().parents[1]

class ImageCropper:
    def __init__(self, model, saved_path):
        self.model = model
        self.saved_path = saved_path

    def process(self, directory_path):
        for path in (BASE_DIR / Path(directory_path)).glob("*"):
            image = cv2.imread(path)

            results = self.model.predict(image)
            results = self.model.postprocess(results)

            for bbox in results:
                x1,y1,x2,y2 = self._expand_bbox(bbox, image.shape)

                filename = path.name
                new_name = f"{uuid4().hex}_{filename}"

                saved_path = BASE_DIR / Path(self.saved_path) / new_name
                cv2.imwrite(saved_path, image[y1:y2, x1:x2])

                print(f"file saved into: {saved_path}")

    def _expand_bbox(self, bbox, img_shape, scale=0.25):
        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1

        cx = x1 + w/2
        cy = y1 + h/2

        new_w = w * (1 + scale)
        new_h = h * (1 + scale)

        x1 = int(max(0, cx - new_w/2))
        y1 = int(max(0, cy - new_h/2))
        x2 = int(min(img_shape[1], cx + new_w/2))
        y2 = int(min(img_shape[0], cy + new_h/2))
        
        return x1,y1,x2,y2

if __name__ == "__main__":
    people_model = YOLODetection("yolov8s.pt")
    cropper = ImageCropper(model = people_model, saved_path="raw_data")

    cropper.process("roboflow_dataset/valid/images")