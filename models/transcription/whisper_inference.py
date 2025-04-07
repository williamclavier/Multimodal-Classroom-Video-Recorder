import whisper
from typing import Dict, List, Tuple
import re
import os
from dataclasses import dataclass
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualReference:
    start_time: float
    end_time: float
    phrase: str
    text_context: str

class WhisperTranscriber:
    def __init__(self, model="base") -> None:
        self.model = whisper.load_model(model)
        # Get the absolute path to the phrases directory
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.slide_phrases = self._load_phrases(current_dir / "phrases" / "slide_phrases.txt")
        self.board_phrases = self._load_phrases(current_dir / "phrases" / "board_phrases.txt")
        logger.debug(f"Loaded {len(self.slide_phrases)} slide phrases: {self.slide_phrases}")
        logger.debug(f"Loaded {len(self.board_phrases)} board phrases: {self.board_phrases}")

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

    def analyze_visual_references(self, transcription_result: Dict) -> Dict[str, List[VisualReference]]:
        """
        Analyzes the transcribed text for references to visual elements.
        Returns a dictionary with timestamps of detected references.
        """
        references = {
            "slides": [],
            "board": []
        }
        
        # Get segments with timestamps
        segments = transcription_result.get("segments", [])
        full_text = transcription_result.get("text", "").lower()
        logger.debug(f"Analyzing text: {full_text}")
        
        # Search for slide references
        for phrase in self.slide_phrases:
            phrase_lower = phrase.lower()
            logger.debug(f"Searching for slide phrase: {phrase_lower}")
            for match in re.finditer(re.escape(phrase_lower), full_text):
                logger.debug(f"Found slide phrase match: {phrase_lower} at position {match.start()}")
                segment = self._find_segment_for_match(match.start(), match.end(), segments, full_text)
                if segment:
                    references["slides"].append(VisualReference(
                        start_time=segment["start"],
                        end_time=segment["end"],
                        phrase=phrase,
                        text_context=segment["text"]
                    ))
                    logger.debug(f"Added slide reference: {phrase} at {segment['start']}s")
        
        # Search for board references
        for phrase in self.board_phrases:
            phrase_lower = phrase.lower()
            logger.debug(f"Searching for board phrase: {phrase_lower}")
            for match in re.finditer(re.escape(phrase_lower), full_text):
                logger.debug(f"Found board phrase match: {phrase_lower} at position {match.start()}")
                segment = self._find_segment_for_match(match.start(), match.end(), segments, full_text)
                if segment:
                    references["board"].append(VisualReference(
                        start_time=segment["start"],
                        end_time=segment["end"],
                        phrase=phrase,
                        text_context=segment["text"]
                    ))
                    logger.debug(f"Added board reference: {phrase} at {segment['start']}s")
        
        logger.debug(f"Found {len(references['slides'])} slide references and {len(references['board'])} board references")
        return references

    def get_visual_context(self, audio_path: str) -> Tuple[Dict, Dict[str, List[VisualReference]]]:
        """
        Transcribes audio and analyzes it for visual references.
        """
        transcription = self.transcribe(audio_path)
        references = self.analyze_visual_references(transcription)
        return transcription, references