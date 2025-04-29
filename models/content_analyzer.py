from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ContentAnalysisResult:
    timestamp: float
    slide_text: str
    transcription_text: str
    similarity_score: float
    is_slide_related: bool
    confidence: float
    slide_region: Optional[Dict] = None

class ContentAnalyzer:
    def __init__(self, min_similarity_threshold: float = 0.3):
        self.min_similarity_threshold = min_similarity_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
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
            
            # Create analysis result
            result = ContentAnalysisResult(
                timestamp=trans_segment['start'],
                slide_text=closest_ocr['text'],
                transcription_text=trans_segment['text'],
                similarity_score=similarity,
                is_slide_related=is_slide_related,
                confidence=min(similarity, closest_ocr.get('confidence', 1.0)),
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
                    "slide_region": r.slide_region
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ Content analysis results saved to {output_path}") 