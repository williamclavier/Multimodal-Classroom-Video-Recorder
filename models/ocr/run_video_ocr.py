import cv2
import os
import json
from datetime import datetime
from .core.ocr_engine import extract_text, OCRResult
from .core.change_detector import is_different

def run_video_ocr(video_path: str, output_path: str, frame_interval: int = 10, ssim_threshold: float = 0.85):
    """
    Run OCR on a video file and save results to JSON.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the OCR results JSON file
        frame_interval: Number of frames to skip between OCR processing
        ssim_threshold: Threshold for structural similarity to detect changes
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open video: {video_path}")
    print(f"📹 Opened: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_frame = None
    frame_idx = 0
    results = []

    # Create progress bar
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate timestamp
        timestamp = frame_idx / fps if fps > 0 else 0
        
        if frame_idx % frame_interval == 0:
            if prev_frame is None or is_different(prev_frame, frame, ssim_threshold):
                # Extract text with timestamp
                ocr_result = extract_text(frame, timestamp, frame_idx)
                
                if ocr_result.text:
                    # Convert OCRResult to dict for JSON serialization
                    result_dict = {
                        "timestamp": ocr_result.timestamp,
                        "frame_index": ocr_result.frame_index,
                        "text": ocr_result.text,
                        "confidence": ocr_result.confidence,
                        "bounding_boxes": ocr_result.bounding_boxes
                    }
                    results.append(result_dict)
                    
                prev_frame = frame
                
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results with metadata
    output_data = {
        "video_path": video_path,
        "processing_date": datetime.now().isoformat(),
        "total_frames": total_frames,
        "fps": fps,
        "frame_interval": frame_interval,
        "ssim_threshold": ssim_threshold,
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"✅ OCR results saved to {output_path}")
    print(f"📊 Processed {len(results)} frames with text")

if __name__ == "__main__":
    # Default paths for testing
    VIDEO_PATH = "ocr/test_videos/your_video.mov"
    OUTPUT_PATH = "ocr/results/ocr_results.json"
    run_video_ocr(VIDEO_PATH, OUTPUT_PATH)
