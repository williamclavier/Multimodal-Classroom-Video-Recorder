import whisper
from typing import Dict, List, Tuple, Optional
import re
import os
from dataclasses import dataclass
from pathlib import Path
import logging
from transformers import pipeline
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualReference:
    start_time: float
    end_time: float
    phrase: str
    text_context: str
    confidence: float
    reference_type: str  # "slide", "board", "general"
    specific_content: Optional[str] = None  # e.g., "equation", "diagram", "bullet point"

class WhisperTranscriber:
    def __init__(self, model="base") -> None:
        self.model = whisper.load_model(model)
        # Load zero-shot classification for better reference detection
        self.classifier = pipeline("zero-shot-classification")
        
        # Get the absolute path to the phrases directory
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.slide_phrases = self._load_phrases(current_dir / "phrases" / "slide_phrases.txt")
        self.board_phrases = self._load_phrases(current_dir / "phrases" / "board_phrases.txt")
        
        # Define reference type categories
        self.reference_categories = {
            "slide": ["slide", "presentation", "powerpoint", "next page", "previous page"],
            "board": ["board", "whiteboard", "blackboard", "write", "drawing", "draw"],
            "equation": ["equation", "formula", "calculation"],
            "diagram": ["diagram", "figure", "graph", "chart"],
            "list": ["bullet point", "list", "items"]
        }

    def _load_phrases(self, file_path: Path) -> List[str]:
        """Load phrases from a text file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Phrase file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            phrases = [line.strip() for line in f if line.strip()]
        return phrases

    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio and return full result with timestamps."""
        result = self.model.transcribe(audio_path)
        logger.debug(f"Transcription text: {result.get('text', '')}")
        return result

    def _find_segment_for_match(self, match_start: int, match_end: int, segments: List[Dict], full_text: str) -> Dict:
        """Find the segment that contains the given match position."""
        for segment in segments:
            segment_start = full_text.find(segment["text"].lower())
            if segment_start == -1:
                continue
            segment_end = segment_start + len(segment["text"])
            if segment_start <= match_start < segment_end:
                return segment
        return None

    def _analyze_reference_type(self, text: str) -> Tuple[str, float, Optional[str]]:
        """
        Analyze the type of reference using zero-shot classification.
        Returns (reference_type, confidence, specific_content)
        """
        # First determine if it's a slide or board reference
        candidate_labels = ["slide reference", "board reference", "general reference"]
        result = self.classifier(text, candidate_labels)
        
        reference_type = result["labels"][0].split()[0]  # "slide" or "board"
        confidence = result["scores"][0]

        # Then determine the specific content type
        content_labels = ["equation", "diagram", "list", "text"]
        content_result = self.classifier(text, content_labels)
        specific_content = content_result["labels"][0] if content_result["scores"][0] > 0.5 else None

        return reference_type, confidence, specific_content

    def _extract_deictic_references(self, text: str) -> List[str]:
        """Extract phrases that indicate pointing or referring to something."""
        deictic_patterns = [
            r"(?:look|see|notice|observe) (?:at|here|this)",
            r"(?:here|this) (?:we|you) (?:can|have)",
            r"(?:as|like) (?:shown|displayed|illustrated)",
            r"(?:this|that|these|those) (?:is|are|shows?|represents?)",
            r"(?:on|in) (?:the|this) (?:slide|board|figure|diagram)"
        ]
        
        references = []
        for pattern in deictic_patterns:
            matches = re.finditer(pattern, text.lower())
            references.extend([m.group() for m in matches])
        return references

    def analyze_visual_references(self, transcription_result: Dict) -> Dict[str, List[VisualReference]]:
        """
        Enhanced analysis of transcribed text for references to visual elements.
        """
        references = {
            "slides": [],
            "board": []
        }
        
        segments = transcription_result.get("segments", [])
        full_text = transcription_result.get("text", "").lower()
        
        # 1. Check for explicit phrases
        self._process_explicit_phrases(full_text, segments, references)
        
        # 2. Extract deictic references
        for segment in segments:
            segment_text = segment["text"].lower()
            deictic_refs = self._extract_deictic_references(segment_text)
            
            for ref in deictic_refs:
                # Analyze the broader context (3 seconds before and after)
                context_text = self._get_temporal_context(segments, segment, window=3)
                
                # Determine reference type and confidence
                ref_type, confidence, specific_content = self._analyze_reference_type(context_text)
                
                if confidence > 0.4:  # Threshold can be adjusted
                    reference = VisualReference(
                        start_time=segment["start"],
                        end_time=segment["end"],
                        phrase=ref,
                        text_context=context_text,
                        confidence=confidence,
                        reference_type=ref_type,
                        specific_content=specific_content
                    )
                    references[f"{ref_type}s"].append(reference)

        # Sort references by timestamp
        for ref_type in references:
            references[ref_type].sort(key=lambda x: x.start_time)
            
        return references

    def _get_temporal_context(self, segments: List[Dict], current_segment: Dict, window: float) -> str:
        """Get text context within a time window around the current segment."""
        context_segments = []
        current_time = current_segment["start"]
        
        for segment in segments:
            if (current_time - window) <= segment["start"] <= (current_time + window):
                context_segments.append(segment["text"])
                
        return " ".join(context_segments)

    def _process_explicit_phrases(self, full_text: str, segments: List[Dict], references: Dict):
        """Process explicit phrase matches"""
        for phrase in self.slide_phrases:
            phrase_lower = phrase.lower()
            for match in re.finditer(re.escape(phrase_lower), full_text):
                segment = self._find_segment_for_match(match.start(), match.end(), segments, full_text)
                if segment:
                    ref_type, confidence, specific_content = self._analyze_reference_type(segment["text"])
                    references[f"{ref_type}s"].append(VisualReference(
                        start_time=segment["start"],
                        end_time=segment["end"],
                        phrase=phrase,
                        text_context=segment["text"],
                        confidence=confidence,
                        reference_type=ref_type,
                        specific_content=specific_content
                    ))

        # Similar process for board phrases
        for phrase in self.board_phrases:
            phrase_lower = phrase.lower()
            for match in re.finditer(re.escape(phrase_lower), full_text):
                segment = self._find_segment_for_match(match.start(), match.end(), segments, full_text)
                if segment:
                    ref_type, confidence, specific_content = self._analyze_reference_type(segment["text"])
                    references[f"{ref_type}s"].append(VisualReference(
                        start_time=segment["start"],
                        end_time=segment["end"],
                        phrase=phrase,
                        text_context=segment["text"],
                        confidence=confidence,
                        reference_type=ref_type,
                        specific_content=specific_content
                    ))

    def get_visual_context(self, audio_path: str) -> Tuple[Dict, Dict[str, List[VisualReference]]]:
        """
        Transcribes audio and analyzes it for visual references.
        """
        transcription = self.transcribe(audio_path)
        references = self.analyze_visual_references(transcription)
        return transcription, references