from ultralytics import YOLO
import cv2
import time
from pathlib import Path

def runner(model_path, camera_path, target_fps=None):
    detector = YOLO(model_path)

    cap = cv2.VideoCapture(camera_path)

    if Path(camera_path).suffix in ['.mp4', '.avi', '.mkv']:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original FPS: {fps}")
        if target_fps is None:
            target_fps = fps

    frame_time = 1.0 / target_fps

    while True:
        start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(
            source=frame,
            conf=0.5,
            verbose=False,
            imgsz=640, 
            rect=True
        )

        annotated_frame = results[0].plot(font_size=0.1, line_width=1)

        cv2.imshow("YOLO Realtime", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed = time.perf_counter() - start
        sleep_time = frame_time - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r"C:\Users\AQI\JOB\ppe_detection\best (2).pt"
    camera_path = r"C:\Users\AQI\JOB\ppe_detection\videos\ppe-5.mp4"
    
    #print(YOLO(model_path).names)
    runner(model_path, camera_path)