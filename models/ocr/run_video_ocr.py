import cv2
import os
import datetime
import argparse
from models.ocr.core.ocr_engine import extract_text
from models.ocr.core.change_detector import is_different

FRAME_INTERVAL = 10
SSIM_THRESHOLD = 0.85

def frame_to_timestamp(cap, frame_idx):
    """Convert frame number to timestamp based on the video fps."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = int(frame_idx / fps)
    return str(datetime.timedelta(seconds=seconds))

def run_video_ocr(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    print(f"📹 Opened: {video_path}")

    prev_frame = None
    frame_idx = 0
    logs = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    try:
        with open(output_path, "w") as f:
            f.write("\n".join(logs))
        print(f"OCR saved to {output_path}")
    except Exception as e:
        print(f"Failed to save OCR results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on a classroom video.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="models/ocr/results/ocr_log.txt", help="Path to output text file")
    args = parser.parse_args()

    run_video_ocr(args.video, args.output)
