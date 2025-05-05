from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
from models.content_analyzer import ContentAnalysisResult
from models.config import (
    MIN_SLIDE_CONFIDENCE,
    MIN_POSE_CONFIDENCE,
    OVERLAY_THRESHOLD,
    MIN_CORNER_SAFETY,
    MIN_GESTURE_CONFIDENCE,
    CONTENT_WEIGHT,
    POSE_WEIGHT,
    CORNER_WEIGHT,
    GESTURE_WEIGHT,
    PRIMARY_SLIDE_PENALTY,
    MODEL_AGREEMENT_BOOST,
    MODEL_DISAGREEMENT_PENALTY,
    NO_POSE_PENALTY
)

@dataclass
class CameraDecision:
    timestamp: float
    primary_feed: str  # 'slide' or 'professor'
    overlay_enabled: bool
    slide_position: Optional[str] = None  # 'full', 'corner', or None
    confidence: float = 1.0
    reasoning: Optional[str] = None
    professor_bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2) normalized coordinates

class CameraDecisionSystem:
    def __init__(self, 
                 min_slide_confidence: float = MIN_SLIDE_CONFIDENCE,
                 min_pose_confidence: float = MIN_POSE_CONFIDENCE):
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
                'confidence': 0.0
            }
            
        # Extract relevant pose information directly from the pose data
        is_pointing = pose_data.get('is_pointing', False)
        is_writing = pose_data.get('is_writing', False)
        confidence = pose_data.get('confidence', 0.0)
        
        return {
            'is_pointing': is_pointing,
            'is_writing': is_writing,
            'confidence': confidence
        }
    
    def _get_professor_bbox(self, pose_data: Dict) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculate a bounding box around the professor using pose landmarks.
        Returns normalized coordinates (x1, y1, x2, y2).
        """
        if not pose_data or 'landmarks' not in pose_data:
            return None
            
        landmarks = pose_data['landmarks']
        if not landmarks:
            return None
            
        # Get key points for bounding box
        key_points = []
        for landmark in landmarks:
            if landmark.visibility > 0.5:  # Only use visible landmarks
                key_points.append((landmark.x, landmark.y))
                
        if not key_points:
            return None
            
        # Calculate bounding box
        x_coords = [x for x, _ in key_points]
        y_coords = [y for _, y in key_points]
        
        # Add padding (20% of width/height)
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        padding_x = width * 0.2
        padding_y = height * 0.2
        
        x1 = max(0.0, min(x_coords) - padding_x)
        y1 = max(0.0, min(y_coords) - padding_y)
        x2 = min(1.0, max(x_coords) + padding_x)
        y2 = min(1.0, max(y_coords) + padding_y)
        
        return (x1, y1, x2, y2)
    
    def make_decision(self, 
                     content_analysis: ContentAnalysisResult,
                     pose_data: Optional[Dict] = None) -> CameraDecision:
        """
        Make a camera feed decision based on content analysis and pose data.
        """
        # Get professor bounding box if pose data is available
        professor_bbox = self._get_professor_bbox(pose_data) if pose_data else None
        
        # Analyze pose if available
        pose_analysis = self._analyze_pose_for_gestures(pose_data) if pose_data else None
        
        # Determine if we should show slides based on content analysis
        show_slides = (content_analysis.is_slide_related and 
                      content_analysis.confidence >= self.min_slide_confidence)
        
        # Calculate overlay confidence based on multiple factors
        overlay_confidence = 0.0
        
        # 1. Content analysis contribution
        if content_analysis.safe_corners:
            # Use the best corner's safety score
            best_corner_score = max(content_analysis.safe_corners.values())
            overlay_confidence += best_corner_score * CORNER_WEIGHT
            
            # Only consider corners with good safety scores
            if best_corner_score < MIN_CORNER_SAFETY:
                overlay_confidence = 0.0
        
        # 2. Pose analysis contribution
        if pose_analysis:
            # Writing takes highest priority - if writing is detected, show professor
            if pose_analysis['is_writing'] and pose_analysis['confidence'] > MIN_GESTURE_CONFIDENCE:
                show_slides = False
                overlay_confidence = 0.0  # No overlay when writing
            # Pointing is secondary - consider context
            elif pose_analysis['is_pointing'] and pose_analysis['confidence'] > MIN_GESTURE_CONFIDENCE:
                # If pointing and content is relevant, prefer slides with professor overlay
                if show_slides:
                    overlay_confidence += pose_analysis['confidence'] * GESTURE_WEIGHT
                else:
                    # If pointing but no relevant content, show professor
                    show_slides = False
                    overlay_confidence = 0.0
        
        # 3. Primary feed consideration
        # If showing slides as primary, be more conservative with overlay
        if show_slides:
            overlay_confidence *= PRIMARY_SLIDE_PENALTY
        
        # Determine if overlay should be enabled
        overlay_enabled = overlay_confidence >= OVERLAY_THRESHOLD
        
        # Determine best corner for overlay based on safe corners
        best_corner = 'bottom_right'  # Default
        best_score = 0.0
        
        if content_analysis.safe_corners:
            for corner, score in content_analysis.safe_corners.items():
                if score > best_score:
                    best_score = score
                    best_corner = corner
        
        # Build detailed reasoning
        reasoning_parts = []
        
        # Content analysis reasoning
        if content_analysis.is_slide_related:
            reasoning_parts.append(f"Slide content detected (confidence: {content_analysis.confidence:.2f})")
        else:
            reasoning_parts.append("No slide content detected")
            
        # Safe corner reasoning
        if content_analysis.safe_corners:
            reasoning_parts.append(f"Best corner: {best_corner} (safety: {best_score:.2f})")
            
        # Pose analysis reasoning
        if pose_analysis:
            if pose_analysis['is_writing']:
                reasoning_parts.append("Writing detected - showing professor")
            elif pose_analysis['is_pointing']:
                if show_slides:
                    reasoning_parts.append("Pointing with relevant content - showing slides with overlay")
                else:
                    reasoning_parts.append("Pointing without relevant content - showing professor")
            reasoning_parts.append(f"Pose confidence: {pose_analysis['confidence']:.2f}")
        else:
            reasoning_parts.append("No pose data available")
            
        # Overlay reasoning
        reasoning_parts.append(f"Overlay {'enabled' if overlay_enabled else 'disabled'} (confidence: {overlay_confidence:.2f})")
        
        # Calculate overall confidence
        if pose_analysis:
            # Normalize pose confidence based on threshold
            normalized_pose_confidence = min(1.0, pose_analysis['confidence'] / self.min_pose_confidence)
            
            # Weighted average of both confidences
            overall_confidence = (
                content_analysis.confidence * CONTENT_WEIGHT +
                normalized_pose_confidence * POSE_WEIGHT
            )
            
            # Boost confidence if both models agree
            if (show_slides and pose_analysis['is_pointing']) or \
               (not show_slides and pose_analysis['is_writing']):
                # Larger boost when models agree
                overall_confidence = min(1.0, overall_confidence * MODEL_AGREEMENT_BOOST)
                
            # Penalize if models disagree
            elif (show_slides and pose_analysis['is_writing']) or \
                 (not show_slides and pose_analysis['is_pointing']):
                overall_confidence *= MODEL_DISAGREEMENT_PENALTY
        else:
            # If no pose data, use content confidence with a larger penalty
            overall_confidence = content_analysis.confidence * NO_POSE_PENALTY
        
        # Create decision with professor bounding box
        decision = CameraDecision(
            timestamp=content_analysis.timestamp,
            primary_feed='slide' if show_slides else 'professor',
            overlay_enabled=overlay_enabled,
            slide_position=best_corner if overlay_enabled else None,
            confidence=overall_confidence,
            reasoning=" | ".join(reasoning_parts),
            professor_bbox=professor_bbox
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
            
            # Add pose data to the decision for visualization
            if closest_pose and 'debug_info' in closest_pose:
                decision_dict = {
                    "timestamp": decision.timestamp,
                    "primary_feed": decision.primary_feed,
                    "overlay_enabled": decision.overlay_enabled,
                    "slide_position": decision.slide_position,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "pose_data": closest_pose['debug_info']  # Include pose data for visualization
                }
            else:
                decision_dict = {
                    "timestamp": decision.timestamp,
                    "primary_feed": decision.primary_feed,
                    "overlay_enabled": decision.overlay_enabled,
                    "slide_position": decision.slide_position,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning
                }
            
            decisions.append(decision_dict)
        
        # Save decisions
        output_data = {
            "processing_date": datetime.now().isoformat(),
            "min_slide_confidence": self.min_slide_confidence,
            "min_pose_confidence": self.min_pose_confidence,
            "decisions": decisions
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ Camera decisions saved to {output_path}") 