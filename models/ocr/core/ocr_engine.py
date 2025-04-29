import cv2
import pytesseract
from dataclasses import dataclass
from typing import Optional

@dataclass
class OCRResult:
    text: str
    timestamp: float
    frame_index: int
    confidence: Optional[float] = None
    bounding_boxes: Optional[list] = None

def extract_text(image, timestamp: float, frame_index: int) -> OCRResult:
    """
    Extract text from an image with timestamp information.
    
    Args:
        image: The input image
        timestamp: Current timestamp in seconds
        frame_index: Current frame index
        
    Returns:
        OCRResult object containing the extracted text and metadata
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    
    # Get detailed OCR data including confidence and bounding boxes
    ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    # Calculate average confidence
    confidences = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    
    # Extract bounding boxes
    boxes = []
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():
            boxes.append({
                'text': ocr_data['text'][i],
                'confidence': float(ocr_data['conf'][i]),
                'bbox': {
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i]
                }
            })
    
    return OCRResult(
        text=text.strip(),
        timestamp=timestamp,
        frame_index=frame_index,
        confidence=avg_confidence,
        bounding_boxes=boxes
    )
