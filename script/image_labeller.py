from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

model = YOLO(
    r"C:\Users\AQI\JOB\ppe_detection\roboflow_dataset\PPE_detection_5400\output\train\weights\best.pt"
)

image_folder = BASE_DIR / "raw_data/video_dataset3/images"

model.predict(
    source=str(image_folder),
    save_txt=True,
    save=False,
    project=BASE_DIR / "raw_data",
    name="video_dataset3",
    exist_ok=True,
)

print("Finished labeling")