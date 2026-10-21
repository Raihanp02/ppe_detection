import json
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def yolo_to_labelstudio(
    images_dir: str,
    labels_dir: str,
    class_names: dict[int, str],
    output_file: str = "labelstudio_import.json",
    name_folder_labelstudio = "images"
):
    """
    Convert YOLO labels to Label Studio import JSON.

    Parameters
    ----------
    images_dir : str
        Directory containing images.

    labels_dir : str
        Directory containing YOLO txt labels.

    class_names : dict[int, str]
        Mapping from class id to class name.

    output_file : str
        Output Label Studio JSON file.
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    tasks = []

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"

        results = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, xc, yc, w, h = map(float, line.strip().split())

                    results.append(
                        {
                            "id": str(uuid.uuid4()),
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": (xc - w / 2) * 100,
                                "y": (yc - h / 2) * 100,
                                "width": w * 100,
                                "height": h * 100,
                                "rectanglelabels": [
                                    class_names[int(cls)]
                                ],
                            },
                        }
                    )

        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=pakuwon_data/{image_path.name}"
                },
                "predictions": [
                    {
                        "model_version": "yolo_import",
                        "result": results,
                    }
                ],
            }
        )

    with open(output_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Saved {len(tasks)} tasks to {output_file}")

if __name__ == "__main__":
    images_dir = BASE_DIR / "raw_data/video_dataset3/images"
    labels_dir = BASE_DIR / "raw_data/video_dataset3/labels"
    class_names = {0: 'boots', 1: 'gloves', 2: 'hardhat', 3: 'no_boots', 4: 'no_gloves', 5: 'no_hardhat', 6: 'no_vest', 7: 'person', 8: 'vest'}
    output_file = BASE_DIR / "raw_data/video_dataset3/labelstudio_import.json"
    name_folder_labelstudio = "pakuwon_car"

    yolo_to_labelstudio(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=class_names,
        output_file=output_file,
        name_folder_labelstudio=name_folder_labelstudio
    )