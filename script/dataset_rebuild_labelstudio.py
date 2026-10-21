from pathlib import Path
import shutil

images_dir = Path("mydata/images")
labels_dir = Path("export/labels")

out_images = Path("filtered/images")
out_labels = Path("filtered/labels")

out_images.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

for label_file in labels_dir.glob("*.txt"):
    stem = label_file.stem

    for ext in [".jpg", ".jpeg", ".png"]:
        img_file = images_dir / f"{stem}{ext}"
        if img_file.exists():
            shutil.copy2(img_file, out_images / img_file.name)
            shutil.copy2(label_file, out_labels / label_file.name)
            break