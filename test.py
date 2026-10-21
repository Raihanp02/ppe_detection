from ultralytics import YOLO

model = YOLO(
    r"C:\Users\AQI\Downloads\drive-download-20260807T040227Z-1-001\ppe-1938.pt"
)

print(model.names)

if (0 % 3 == 0):
    print("True")