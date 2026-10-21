import cv2
from uuid import uuid4
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def capture_frame(video_path, interval_in_seconds, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_in_seconds)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{uuid4().hex}.jpg"
            saved_path = BASE_DIR / Path(output_path) / filename
            cv2.imwrite(saved_path, frame)
            print(f"Frame saved: {saved_path}")
            saved_count += 1

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"Total frames processed: {frame_count}, Total frames saved: {saved_count}")

def read_video_from_folder(folder_path, interval_in_seconds, output_path):
    for video_file in (BASE_DIR / Path(folder_path)).glob("*"):
        if video_file.suffix in ['.mp4', '.avi', '.mkv']:
            print(f"Processing video: {video_file}")
            capture_frame(str(video_file), interval_in_seconds, output_path)

if __name__ == "__main__":
    video_path = "videos/TIDAK MENGGUNAKAN BODY HARNES.mp4"
    output_folder = "raw_data/wastec_apd"
    interval_seconds = 3

    capture_frame(video_path, interval_seconds, output_folder)