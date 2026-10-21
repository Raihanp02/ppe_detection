import fiftyone as fo

ds = fo.Dataset.from_dir(
    dataset_dir=r"C:\Users\AQI\JOB\ppe_detection\yoloformat_dataset\PPE_person_crops_v40_filtered",
    dataset_type=fo.types.YOLOv5Dataset,
    split="test",
    yaml_path=r"C:\Users\AQI\JOB\ppe_detection\yoloformat_dataset\PPE_person_crops_v40_filtered\data.yaml"
)
session = fo.launch_app(ds)
session.wait() 