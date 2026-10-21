from pathlib import Path
import shutil

def move_files(images_dir, labels_dir, new_images_dir = None, new_labels_dir = None, extensions=[".jpg", ".jpeg", ".png"], method = "copy"):
    if new_images_dir is not None and new_labels_dir is not None:
        raise ValueError("new_images_dir and new_labels_dir cannot be used together")

    if new_images_dir is None and new_labels_dir is None:
        raise ValueError("Either new_images_dir or new_labels_dir must be provided")
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    new_images_dir = Path(new_images_dir) if new_images_dir else None
    new_labels_dir = Path(new_labels_dir) if new_labels_dir else None

    if new_labels_dir:
        new_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_file in images_dir.glob("*.*"):
            stem = image_file.stem

            for ext in extensions:
                txt_file = labels_dir / f"{stem}{ext}"
                if txt_file.exists():
                    print(f"Found label file {txt_file} for image {image_file}")
                    print(f"Processing {txt_file}")
                    if method == "copy":
                        shutil.copy2(txt_file, new_labels_dir / txt_file.name)
                    elif method == "move":
                        shutil.move(txt_file, new_labels_dir / txt_file.name)
                    break

    if new_images_dir:
        new_images_dir.mkdir(parents=True, exist_ok=True)
        for label_file in labels_dir.glob("*.txt"):
            stem = label_file.stem

            for ext in extensions:
                image_file = images_dir / f"{stem}{ext}"
                if image_file.exists():
                    print(f"Processing {image_file}")
                    if method == "copy":
                        shutil.copy2(image_file, new_images_dir / image_file.name)
                    elif method == "move":
                        shutil.move(image_file, new_images_dir / image_file.name)
                    break

                print(f"Image file for {stem} not found in {images_dir}")

if __name__ == "__main__":
    images_dir = r"C:\Users\AQI\JOB\ppe_detection\yoloformat_dataset\project-4-at-2026-06-14-15-15-a88338c2\valid\images"
    labels_dir = r"C:\Users\AQI\JOB\ppe_detection\yoloformat_dataset\project-4-at-2026-06-14-15-15-a88338c2\train\labels"
    new_images_dir = None
    new_labels_dir = r"C:\Users\AQI\JOB\ppe_detection\yoloformat_dataset\project-4-at-2026-06-14-15-15-a88338c2\valid\labels"

    move_files(images_dir, labels_dir, new_labels_dir=new_labels_dir, method="move", extensions=[".txt"])