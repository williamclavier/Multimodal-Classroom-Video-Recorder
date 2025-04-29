import cv2
import os
import datetime
from models.ocr.core.ocr_engine import extract_text
from models.ocr.core.change_detector import is_different

VIDEO_PATH = "models/ocr/test_videos/your_video.mp4"
OUTPUT_PATH = "models/ocr/results/ocr_log.txt"
FRAME_INTERVAL = 10
SSIM_THRESHOLD = 0.85

def frame_to_timestamp(cap, frame_idx):
    """Convert frame number to timestamp based on video fps."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = frame_idx / fps
    return str(datetime.timedelta(seconds=seconds))

def run_video_ocr():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")
    print(f" Opened: {VIDEO_PATH}")

    prev_frame = None
    frame_idx = 0
    logs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            if prev_frame is None or is_different(prev_frame, frame, SSIM_THRESHOLD):
                text = extract_text(frame)
                if text.strip():
                    timestamp = frame_to_timestamp(cap, frame_idx)
                    logs.append(f"[Frame {frame_idx}] [Timestamp {timestamp}]\n{text.strip()}\n")
                prev_frame = frame
        frame_idx += 1

    cap.release()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        with open(OUTPUT_PATH, "w") as f:
            f.write("\n".join(logs))
        print(f"OCR saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Failed to save OCR results: {e}")

if __name__ == "__main__":
    run_video_ocr()
