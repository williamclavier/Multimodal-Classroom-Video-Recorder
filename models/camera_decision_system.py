from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
from models.content_analyzer import ContentAnalysisResult

@dataclass
class CameraDecision:
    timestamp: float
    primary_feed: str  # 'slide' or 'professor'
    overlay_enabled: bool
    slide_position: Optional[str] = None  # 'full', 'corner', or None
    confidence: float = 1.0
    reasoning: Optional[str] = None

class CameraDecisionSystem:
    def __init__(self, 
                 min_slide_confidence: float = 0.5,
                 min_pose_confidence: float = 0.7):
        self.min_slide_confidence = min_slide_confidence
        self.min_pose_confidence = min_pose_confidence
        
    def load_content_analysis(self, analysis_path: str) -> List[ContentAnalysisResult]:
        """Load content analysis results from JSON file."""
        with open(analysis_path, 'r') as f:
            data = json.load(f)
        return [ContentAnalysisResult(**r) for r in data['results']]
    
    def load_pose_data(self, pose_path: str) -> List[Dict]:
        """Load pose estimation results from JSON file."""
        with open(pose_path, 'r') as f:
            data = json.load(f)
        return data['results']
    
    def _find_closest_pose_data(self, 
                              timestamp: float, 
                              pose_data: List[Dict]) -> Optional[Dict]:
        """Find the pose data closest to the given timestamp."""
        if not pose_data:
            return None
            
        closest_pose = min(pose_data, 
                          key=lambda x: abs(x['timestamp'] - timestamp))
        return closest_pose
    
    def _analyze_pose_for_gestures(self, pose_data: Dict) -> Dict:
        """
        Analyze pose data for relevant gestures.
        Returns a dictionary with gesture information and confidence scores.
        """
        if not pose_data:
            return {
                'is_pointing': False,
                'is_writing': False,
                'is_gesturing': False,
                'confidence': 0.0
            }
            
        # Extract relevant pose information
        is_pointing = pose_data.get('is_pointing', False)
        confidence = pose_data.get('confidence', 0.0)
        
        return {
            'is_pointing': is_pointing,
            'is_writing': False,  # Not currently implemented
            'is_gesturing': is_pointing,  # Use pointing as a proxy for gesturing
            'confidence': confidence
        }
    
    def make_decision(self, 
                     content_analysis: ContentAnalysisResult,
                     pose_data: Optional[Dict] = None) -> CameraDecision:
        """
        Make a camera feed decision based on content analysis and pose data.
        """
        # Analyze pose if available
        pose_analysis = self._analyze_pose_for_gestures(pose_data) if pose_data else None
        
        # Determine if we should show slides based on content analysis
        show_slides = (content_analysis.is_slide_related and 
                      content_analysis.confidence >= self.min_slide_confidence)
        
        # Determine if we should overlay professor
        overlay_professor = False
        if show_slides:
            # Check if professor is gesturing or pointing
            if pose_analysis and pose_analysis['confidence'] >= self.min_pose_confidence:
                overlay_professor = (pose_analysis['is_pointing'] or 
                                   pose_analysis['is_gesturing'])
        
        # Determine slide position
        slide_position = None
        if show_slides:
            if overlay_professor:
                slide_position = 'full'
            else:
                slide_position = 'corner'
        
        # Create decision
        decision = CameraDecision(
            timestamp=content_analysis.timestamp,
            primary_feed='slide' if show_slides else 'professor',
            overlay_enabled=overlay_professor,
            slide_position=slide_position,
            confidence=min(content_analysis.confidence, 
                         pose_analysis['confidence'] if pose_analysis else 1.0),
            reasoning=f"Showing {'slides' if show_slides else 'professor'} with {'overlay' if overlay_professor else 'no overlay'}"
        )
        
        return decision
    
    def process_analysis(self,
                        content_analysis_path: str,
                        pose_data_path: str,
                        output_path: str):
        """Process content analysis and pose data to make camera decisions."""
        content_results = self.load_content_analysis(content_analysis_path)
        pose_data = self.load_pose_data(pose_data_path)
        
        decisions = []
        
        for content_result in content_results:
            # Find closest pose data
            closest_pose = self._find_closest_pose_data(
                content_result.timestamp,
                pose_data
            )
            
            # Make decision
            decision = self.make_decision(content_result, closest_pose)
            decisions.append(decision)
        
        # Save decisions
        output_data = {
            "processing_date": datetime.now().isoformat(),
            "min_slide_confidence": self.min_slide_confidence,
            "min_pose_confidence": self.min_pose_confidence,
            "decisions": [
                {
                    "timestamp": d.timestamp,
                    "primary_feed": d.primary_feed,
                    "overlay_enabled": d.overlay_enabled,
                    "slide_position": d.slide_position,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning
                }
                for d in decisions
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ Camera decisions saved to {output_path}") 