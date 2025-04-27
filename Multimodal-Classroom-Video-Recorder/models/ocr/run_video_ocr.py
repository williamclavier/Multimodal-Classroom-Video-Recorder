import cv2
import os
from models.ocr.core.ocr_engine import extract_text
from models.ocr.core.change_detector import is_different

# === Settings ===
VIDEO_PATH = "models/ocr/test_videos/your_video.mov"  # Update path if needed
OUTPUT_PATH = "models/ocr/results/ocr_log.txt"
FRAME_INTERVAL = 10  # Only process every 10th frame
SSIM_THRESHOLD = 0.85  # Threshold for slide change detection

# === Main Function ===
def run_video_ocr():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f" Cannot open video: {VIDEO_PATH}")
    print(f"📹 Opened video: {VIDEO_PATH}")

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
                    log_entry = f"[Frame {frame_idx}]\n{text.strip()}\n"
                    logs.append(log_entry)
                    print(f"📝 OCR at frame {frame_idx}: {text.strip()[:50]}...")

                prev_frame = frame

        frame_idx += 1

    cap.release()

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(logs))

    print(f"OCR results saved to {OUTPUT_PATH}")

# === Entry Point ===
if __name__ == "__main__":
    run_video_ocr()
