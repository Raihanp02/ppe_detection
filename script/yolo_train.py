from ultralytics import YOLO
from pathlib import Path
import albumentations as A

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "yolo26s.pt"
MODEL_PATH = BASE_DIR / MODEL_NAME

model = YOLO("yolo26s.pt")

custom_augmentations = [
    # Apply one blur type with 20% probability
    # A.OneOf(
    #     [
    #         A.MotionBlur(blur_limit=(3, 7), p=1.0),
    #         A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    #     ],
    #     p=0.2,
    # ),

    # Add mild pixel noise
    A.GaussNoise(
        std_range=(0.02, 0.06),
        p=0.2,
    ),

    # Brightness and contrast jitter
    A.RandomBrightnessContrast(
        brightness_limit=0.0,
        contrast_limit=0.2,
        p=0.3,
    ),
]

def main():
    model.train(
        data=BASE_DIR / "yoloformat_dataset/PPE_person_crops_v40_filtered/data.yaml",
        epochs=100,
        imgsz=160,
        batch=16,
        #workers=8,
        project=BASE_DIR / "yoloformat_dataset/PPE_person_crops_v40_filtered/output",

        device=0,

        optimizer="auto",
        lr0=0.01,

        cos_lr=True,

        patience=30,

        cache=True,

        amp=True,

        augmentations=custom_augmentations,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,

        mosaic=0,
        #mixup=0.1,

        #close_mosaic=10
    )

if __name__ == "__main__":
    main()