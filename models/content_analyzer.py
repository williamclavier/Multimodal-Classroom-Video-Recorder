from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from pathlib import Path

@dataclass
class ContentAnalysisResult:
    timestamp: float
    slide_text: str
    transcription_text: str
    similarity_score: float
    is_slide_related: bool
    confidence: float
    safe_corners: Dict[str, float]  # Confidence scores for each corner
    slide_edges: Dict[str, List[int]]  # Edge positions
    slide_region: Optional[Dict] = None

class ContentAnalyzer:
    def __init__(self, min_similarity_threshold: float = 0.3, edge_threshold: float = 0.7):
        self.min_similarity_threshold = min_similarity_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.edge_threshold = edge_threshold
        
    def load_ocr_results(self, ocr_path: str) -> List[Dict]:
        """Load OCR results from JSON file."""
        with open(ocr_path, 'r') as f:
            data = json.load(f)
        return data['results']
    
    def load_transcription_results(self, transcription_path: str) -> List[Dict]:
        """Load transcription results from JSON file."""
        with open(transcription_path, 'r') as f:
            data = json.load(f)
        return data['segments']
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using TF-IDF and cosine similarity."""
        if not text1 or not text2:
            return 0.0
            
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _find_closest_ocr_result(self, timestamp: float, ocr_results: List[Dict]) -> Optional[Dict]:
        """Find the OCR result closest to the given timestamp."""
        if not ocr_results:
            return None
            
        closest_result = min(ocr_results, 
                           key=lambda x: abs(x['timestamp'] - timestamp))
        return closest_result
    
    def _find_safe_corners(self, frame: np.ndarray) -> Tuple[Dict[str, float], Dict[str, List[int]]]:
        """Find safe corners for overlay placement using edge detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to better detect text
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Get image dimensions
        h, w = edges.shape
        
        # Define corner regions (1/4 of the image)
        corner_size = min(h, w) // 4
        corners = {
            'top_left': (0, 0, corner_size, corner_size),
            'top_right': (w - corner_size, 0, w, corner_size),
            'bottom_left': (0, h - corner_size, corner_size, h),
            'bottom_right': (w - corner_size, h - corner_size, w, h)
        }
        
        # Calculate edge and text density in each corner
        corner_scores = {}
        edge_positions = {}
        
        for corner_name, (x1, y1, x2, y2) in corners.items():
            # Extract corner regions
            corner_edges = edges[y1:y2, x1:x2]
            corner_text = thresh[y1:y2, x1:x2]
            
            # Calculate edge density (percentage of edge pixels)
            edge_density = np.sum(corner_edges > 0) / (corner_size * corner_size)
            
            # Calculate text density (percentage of text pixels)
            text_density = np.sum(corner_text > 0) / (corner_size * corner_size)
            
            # Combine edge and text density
            content_density = max(edge_density, text_density)
            
            # Lower density means safer corner (less content)
            safety_score = 1.0 - content_density
            
            # Apply corner preference
            if corner_name == 'bottom_right':
                safety_score *= 1.2  # 20% boost for bottom-right corner
            elif corner_name == 'bottom_left':
                safety_score *= 1.1  # 10% boost for bottom-left corner
            elif corner_name == 'top_right':
                safety_score *= 0.9  # 10% penalty for top-right corner
            else:  # top_left
                safety_score *= 0.8  # 20% penalty for top-left corner
            
            # Clip safety score to valid range
            safety_score = max(0.0, min(1.0, safety_score))
            
            # Find edge positions
            edge_pos = []
            if edge_density > 0:
                # Find non-zero positions
                y_indices, x_indices = np.where(corner_edges > 0)
                if len(y_indices) > 0:
                    # Get average position
                    avg_y = int(np.mean(y_indices)) + y1
                    avg_x = int(np.mean(x_indices)) + x1
                    edge_pos = [avg_x, avg_y]
            
            corner_scores[corner_name] = safety_score
            edge_positions[corner_name] = edge_pos
        
        return corner_scores, edge_positions
    
    def analyze_content(self, 
                       ocr_path: str, 
                       transcription_path: str,
                       window_size: float = 5.0) -> List[ContentAnalysisResult]:
        """
        Analyze the relationship between OCR and transcription content.
        
        Args:
            ocr_path: Path to OCR results JSON file
            transcription_path: Path to transcription results JSON file
            window_size: Time window in seconds to consider for analysis
            
        Returns:
            List of ContentAnalysisResult objects
        """
        ocr_results = self.load_ocr_results(ocr_path)
        transcription_results = self.load_transcription_results(transcription_path)
        
        analysis_results = []
        
        for trans_segment in transcription_results:
            # Get the closest OCR result
            closest_ocr = self._find_closest_ocr_result(
                trans_segment['start'], 
                ocr_results
            )
            
            if not closest_ocr:
                continue
                
            # Calculate similarity between OCR and transcription text
            similarity = self._calculate_text_similarity(
                closest_ocr['text'],
                trans_segment['text']
            )
            
            # Determine if the content is slide-related
            is_slide_related = similarity >= self.min_similarity_threshold
            
            # Initialize safe corners and slide edges
            safe_corners = {}
            slide_edges = {}
            
            # Load frame image and analyze if available
            frame_path = closest_ocr.get('frame_path')
            if frame_path:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Find safe corners
                    safe_corners, slide_edges = self._find_safe_corners(frame)
            
            # Create analysis result
            result = ContentAnalysisResult(
                timestamp=trans_segment['start'],
                slide_text=closest_ocr['text'],
                transcription_text=trans_segment['text'],
                similarity_score=similarity,
                is_slide_related=is_slide_related,
                confidence=min(similarity, closest_ocr.get('confidence', 1.0)),
                safe_corners=safe_corners,
                slide_edges=slide_edges,
                slide_region=closest_ocr.get('bounding_boxes')
            )
            
            analysis_results.append(result)
            
        return analysis_results
    
    def save_analysis_results(self, 
                            results: List[ContentAnalysisResult],
                            output_path: str):
        """Save analysis results to a JSON file."""
        output_data = {
            "analysis_date": datetime.now().isoformat(),
            "min_similarity_threshold": self.min_similarity_threshold,
            "results": [
                {
                    "timestamp": r.timestamp,
                    "slide_text": r.slide_text,
                    "transcription_text": r.transcription_text,
                    "similarity_score": r.similarity_score,
                    "is_slide_related": r.is_slide_related,
                    "confidence": r.confidence,
                    "slide_region": r.slide_region,
                    "safe_corners": r.safe_corners,
                    "slide_edges": r.slide_edges
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ Content analysis results saved to {output_path}") 