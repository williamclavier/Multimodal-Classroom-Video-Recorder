import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import shutil

class VideoProcessor:
    def __init__(self, 
                 slide_video_path: str, 
                 professor_video_path: str, 
                 output_dir: str,
                 ocr_results_path: Optional[str] = None,
                 pose_results_path: Optional[str] = None,
                 transcription_path: Optional[str] = None,
                 analysis_path: Optional[str] = None,
                 load_only: bool = False):
        self.slide_video_path = slide_video_path
        self.professor_video_path = professor_video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure main output directory exists
        self.load_only = load_only
        
        # Store paths to saved results
        self.saved_ocr_path = ocr_results_path
        self.saved_pose_path = pose_results_path
        self.saved_transcription_path = transcription_path
        self.saved_analysis_path = analysis_path
        
        # Only import and initialize components if not in load-only mode
        if not load_only:
            from models.transcription.whisper_inference import WhisperTranscriber
            from models.content_analyzer import ContentAnalyzer
            from models.camera_decision_system import CameraDecisionSystem
            
            self.transcriber = WhisperTranscriber()
            self.content_analyzer = ContentAnalyzer()
            self.camera_decision_system = CameraDecisionSystem(
                min_slide_confidence=0.3,
                min_pose_confidence=0.5
            )
        
        # Create subdirectories for outputs
        self.ocr_dir = self.output_dir / "ocr"
        self.pose_dir = self.output_dir / "pose"
        self.transcription_dir = self.output_dir / "transcription"
        self.analysis_dir = self.output_dir / "analysis"
        self.decisions_dir = self.output_dir / "decisions"
        self.visualization_dir = self.output_dir / "visualization"
        
        # Create all subdirectories
        for dir_path in [self.ocr_dir, self.pose_dir, self.transcription_dir, 
                        self.analysis_dir, self.decisions_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _normalize_path(self, path: str) -> str:
        """Normalize a path by converting it to absolute path and using forward slashes."""
        return str(Path(path).resolve()).replace('\\', '/')
    
    def _copy_if_different(self, src: str, dst: str) -> None:
        """Copy a file only if source and destination are different files."""
        src_norm = self._normalize_path(src)
        dst_norm = self._normalize_path(dst)
        
        if src_norm != dst_norm:
            shutil.copy(src, dst)
        else:
            print(f"Source and destination paths are the same ({src}), skipping copy")
    
    def process_video(self):
        """Process both videos through all components."""
        from models.pose_estimation.pose_detector import process_video
        print("Starting video processing...")
        
        # Ensure all output directories exist
        for dir_path in [self.ocr_dir, self.pose_dir, self.transcription_dir, 
                        self.analysis_dir, self.decisions_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Run OCR on slide video (or use saved results)
        if not self.load_only:
            print("\n1. Running OCR on slide video...")
            ocr_output = self.ocr_dir / "ocr_results.json"
            if self.saved_ocr_path:
                print("Using saved OCR results from:", self.saved_ocr_path)
                self._copy_if_different(self.saved_ocr_path, str(ocr_output))
            else:
                from models.ocr.run_video_ocr import run_video_ocr
                run_video_ocr(self.slide_video_path, str(ocr_output))
        
        # 2. Run pose detection on professor video (or use saved results)
        if not self.load_only:
            print("\n2. Running pose detection on professor video...")
            pose_output = self.pose_dir / "pose_results.json"
            if self.saved_pose_path:
                print("Using saved pose results from:", self.saved_pose_path)
                self._copy_if_different(self.saved_pose_path, str(pose_output))
            else:
                process_video(self.professor_video_path, str(pose_output))
        
        # 3. Run transcription on professor video (or use saved results)
        if not self.load_only:
            print("\n3. Running transcription on professor video...")
            transcription_output = self.transcription_dir / "transcription.json"
            if self.saved_transcription_path:
                print("Using saved transcription from:", self.saved_transcription_path)
                self._copy_if_different(self.saved_transcription_path, str(transcription_output))
            else:
                transcription_result = self.transcriber.get_visual_context(self.professor_video_path)
                with open(transcription_output, 'w') as f:
                    json.dump(transcription_result, f, indent=2)
        
        # 4. Run content analysis (or use saved results)
        if not self.load_only:
            print("\n4. Running content analysis...")
            analysis_output = self.analysis_dir / "analysis_results.json"
            if self.saved_analysis_path:
                print("Using saved analysis from:", self.saved_analysis_path)
                self._copy_if_different(self.saved_analysis_path, str(analysis_output))
            else:
                analysis_results = self.content_analyzer.analyze_content(
                    str(ocr_output),
                    str(transcription_output)
                )
                self.content_analyzer.save_analysis_results(analysis_results, str(analysis_output))
        
        # 5. Make camera decisions
        if not self.load_only:
            print("\n5. Making camera decisions...")
            decisions_output = self.decisions_dir / "decisions.json"
            self.camera_decision_system.process_analysis(
                str(analysis_output),
                str(pose_output),
                str(decisions_output)
            )
        else:
            # In load-only mode, use the provided decisions file
            decisions_output = self.saved_analysis_path or (self.decisions_dir / "decisions.json")
        
        return decisions_output

class DecisionVisualizer:
    def __init__(self, slide_video_path: str, professor_video_path: str, 
                 decisions_path: str, output_dir: str, debug_mode: bool = False,
                 quality: str = 'high'):
        self.slide_video_path = slide_video_path
        self.professor_video_path = professor_video_path
        self.decisions_path = decisions_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug_mode = debug_mode
        # If in debug mode, force quality to debug
        self.quality = 'debug' if debug_mode else quality
        
        # Load decisions
        with open(decisions_path, 'r') as f:
            self.decisions_data = json.load(f)
        self.decisions = self.decisions_data['decisions']
        
        # Open videos
        self.slide_cap = cv2.VideoCapture(slide_video_path)
        self.professor_cap = cv2.VideoCapture(professor_video_path)
        
        if not self.slide_cap.isOpened() or not self.professor_cap.isOpened():
            raise IOError("Cannot open one or both video files")
            
        # Get video properties and determine which is higher quality
        slide_fps = self.slide_cap.get(cv2.CAP_PROP_FPS)
        prof_fps = self.professor_cap.get(cv2.CAP_PROP_FPS)
        slide_width = int(self.slide_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        slide_height = int(self.slide_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        prof_width = int(self.professor_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        prof_height = int(self.professor_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # In debug mode, use lower resolution
        if debug_mode:
            slide_width = slide_width // 2
            slide_height = slide_height // 2
            prof_width = prof_width // 2
            prof_height = prof_height // 2
        
        # Use the higher quality video's properties
        self.fps = max(slide_fps, prof_fps)
        self.width = max(slide_width, prof_width)
        self.height = max(slide_height, prof_height)
        
        # Store which video is higher quality
        self.slide_is_higher_quality = (slide_width * slide_height) > (prof_width * prof_height)
        
        # Calculate total frames (use the shorter video's length)
        slide_frames = int(self.slide_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prof_frames = int(self.professor_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = min(slide_frames, prof_frames)
        
        # Set up paths
        if debug_mode:
            self.output_path = self.output_dir / "debug_output.mp4"
        else:
            self.output_path = self.output_dir / "output.mp4"
        self.temp_output_path = self.output_dir / "temp_output.mp4"
        
        # Initialize FFmpeg process
        self.ffmpeg_process = None
        self.frame_count = 0
    
    def _enhance_frame_quality(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Enhance frame quality to match target resolution."""
        # Use bicubic interpolation for better quality
        enhanced = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply slight sharpening to compensate for upscaling
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _get_zoomed_region(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Get a zoomed region based on pose data."""
        if not pose_data or 'keypoints' not in pose_data:
            return frame
        
        # Get hand keypoints
        keypoints = pose_data['keypoints']
        right_hand = next((kp for kp in keypoints if kp['part'] == 'rightWrist'), None)
        left_hand = next((kp for kp in keypoints if kp['part'] == 'leftWrist'), None)
        
        # If pointing, use the pointing hand
        if right_hand and right_hand['score'] > 0.5:
            center_x, center_y = right_hand['position']
        elif left_hand and left_hand['score'] > 0.5:
            center_x, center_y = left_hand['position']
        else:
            return frame
        
        # Calculate zoom region (1/4 of the frame size)
        h, w = frame.shape[:2]
        zoom_size = min(h, w) // 4
        x1 = max(0, int(center_x - zoom_size//2))
        y1 = max(0, int(center_y - zoom_size//2))
        x2 = min(w, x1 + zoom_size)
        y2 = min(h, y1 + zoom_size)
        
        # Extract and resize the region
        zoomed = frame[y1:y2, x1:x2]
        if zoomed.size > 0:
            return cv2.resize(zoomed, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
        return frame
    
    def _draw_pose_boxes(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw pose detection boxes and keypoints on the frame."""
        if not pose_data or 'keypoints' not in pose_data:
            return frame
        
        # Get current frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate scale factors if in debug mode
        if self.debug_mode:
            # Get original video dimensions
            orig_h = int(self.professor_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_w = int(self.professor_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            # Calculate scale factors
            scale_x = w / orig_w
            scale_y = h / orig_h
        else:
            scale_x = scale_y = 1.0
        
        # Draw bounding box around person
        if 'bbox' in pose_data:
            x, y, w_box, h_box = pose_data['bbox']
            # Scale coordinates
            x = int(x * scale_x)
            y = int(y * scale_y)
            w_box = int(w_box * scale_x)
            h_box = int(h_box * scale_y)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        
        # Draw keypoints and connections
        keypoints = pose_data['keypoints']
        
        # Define connections between keypoints (MediaPipe Pose connections)
        connections = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP')
        ]
        
        # Create a dictionary of keypoints for easy lookup
        kp_dict = {kp['part']: kp for kp in keypoints if kp['score'] > 0.5}
        
        # Draw connections
        for start_part, end_part in connections:
            if start_part in kp_dict and end_part in kp_dict:
                start_kp = kp_dict[start_part]
                end_kp = kp_dict[end_part]
                start_pos = start_kp['position']
                end_pos = end_kp['position']
                
                # Scale coordinates
                start_x = int(start_pos[0] * scale_x)
                start_y = int(start_pos[1] * scale_y)
                end_x = int(end_pos[0] * scale_x)
                end_y = int(end_pos[1] * scale_y)
                
                # Draw line between keypoints
                cv2.line(frame, 
                        (start_x, start_y),
                        (end_x, end_y),
                        (0, 255, 0), 2)
        
        # Draw keypoints
        for kp in keypoints:
            if kp['score'] > 0.5:  # Only draw high confidence keypoints
                x, y = kp['position']
                # Scale coordinates
                x = int(x * scale_x)
                y = int(y * scale_y)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, kp['part'], (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def _draw_decision_info(self, frame: np.ndarray, decision: Dict) -> np.ndarray:
        """Draw decision information on the frame."""
        if not self.debug_mode:
            return frame
            
        # Create semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add decision information
        cv2.putText(frame, f"Primary Feed: {decision['primary_feed']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Overlay: {'Enabled' if decision['overlay_enabled'] else 'Disabled'}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Slide Position: {decision['slide_position'] or 'None'}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add confidence bar
        confidence = decision['confidence']
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = 160
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_width * confidence), bar_y + bar_height), 
                     (0, 255, 0), -1)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (bar_x + bar_width + 10, bar_y + bar_height - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add reasoning
        if 'reasoning' in decision:
            cv2.putText(frame, f"Reason: {decision['reasoning']}", 
                       (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _start_ffmpeg_writer(self):
        """Start FFmpeg process for writing frames."""
        import subprocess
        
        # Quality settings based on command line option
        quality_settings = {
            'ultra': {'preset': 'veryslow', 'crf': '15'},
            'high': {'preset': 'slow', 'crf': '18'},
            'medium': {'preset': 'medium', 'crf': '23'},
            'low': {'preset': 'fast', 'crf': '28'},
            'debug': {'preset': 'ultrafast', 'crf': '35'}  # Fastest encoding for debugging
        }
        
        settings = quality_settings[self.quality]
        
        # Command to write frames to video using FFmpeg with quality settings
        command = [
            'ffmpeg', '-y',
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', settings['preset'],
            '-crf', settings['crf'],
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(self.temp_output_path)
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=10**8,
                universal_newlines=False
            )
        except Exception as e:
            print(f"Error starting FFmpeg process: {e}")
            raise
    
    def _write_frame_to_ffmpeg(self, frame):
        """Write a frame to the FFmpeg process."""
        if self.ffmpeg_process is None:
            self._start_ffmpeg_writer()
        
        try:
            # Convert frame to PNG format
            _, buffer = cv2.imencode('.png', frame)
            
            # Write the binary data
            self.ffmpeg_process.stdin.write(buffer.tobytes())
            self.ffmpeg_process.stdin.flush()
            self.frame_count += 1
            
            # Read progress from stderr in a non-blocking way
            try:
                # Try to read a line from stderr without blocking
                import msvcrt
                if msvcrt.kbhit():
                    line = self.ffmpeg_process.stderr.readline()
                    if line:
                        try:
                            line_str = line.decode('utf-8', errors='ignore')
                            if 'frame=' in line_str:
                                print(f"FFmpeg progress: {line_str.strip()}")
                        except UnicodeDecodeError:
                            pass
            except ImportError:
                # If msvcrt is not available, just skip progress reading
                pass
            
            return True
        except Exception as e:
            print(f"Error writing frame {self.frame_count}: {e}")
            return False
    
    def _find_decision_for_frame(self, frame_idx: int) -> Optional[Dict]:
        """Find the decision closest to the current frame."""
        timestamp = frame_idx / self.fps
        closest_decision = min(self.decisions, 
                             key=lambda x: abs(x['timestamp'] - timestamp))
        return closest_decision
    
    def _combine_frames(self, slide_frame: np.ndarray, professor_frame: np.ndarray, 
                       decision: Dict) -> np.ndarray:
        """Combine slide and professor frames according to the decision."""
        # Enhance quality of both frames to match the higher resolution
        if self.slide_is_higher_quality:
            professor_frame = self._enhance_frame_quality(professor_frame, self.width, self.height)
        else:
            slide_frame = self._enhance_frame_quality(slide_frame, self.width, self.height)
        
        if decision['primary_feed'] == 'slide':
            if decision['overlay_enabled']:
                # Get zoomed region of professor frame if pose data is available
                if 'pose_data' in decision and decision['pose_data']:
                    prof_small = self._get_zoomed_region(professor_frame, decision['pose_data'])
                else:
                    # If no pose data, just resize the professor frame
                    h, w = slide_frame.shape[:2]
                    prof_small = cv2.resize(professor_frame, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
                
                # Create a copy of the slide frame to avoid modifying the original
                result = slide_frame.copy()
                
                # Determine corner position
                h, w = slide_frame.shape[:2]
                corner_size = min(h, w) // 4
                
                if decision['slide_position'] == 'bottom_right':
                    y_start = h - prof_small.shape[0]
                    x_start = w - prof_small.shape[1]
                elif decision['slide_position'] == 'bottom_left':
                    y_start = h - prof_small.shape[0]
                    x_start = 0
                elif decision['slide_position'] == 'top_right':
                    y_start = 0
                    x_start = w - prof_small.shape[1]
                else:  # top_left
                    y_start = 0
                    x_start = 0
                
                # Ensure the overlay is within bounds
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                
                # Place the professor frame in the corner
                result[y_start:y_start+prof_small.shape[0], 
                      x_start:x_start+prof_small.shape[1]] = prof_small
                
                if self.debug_mode and 'pose_data' in decision:
                    # Draw pose boxes on the professor frame before placing it
                    prof_small = self._draw_pose_boxes(prof_small, decision['pose_data'])
                    result[y_start:y_start+prof_small.shape[0], 
                          x_start:x_start+prof_small.shape[1]] = prof_small
                
                return result
            if self.debug_mode and 'pose_data' in decision:
                professor_frame = self._draw_pose_boxes(professor_frame, decision['pose_data'])
            return slide_frame
        else:  # professor is primary
            if decision['overlay_enabled']:
                h, w = professor_frame.shape[:2]
                slide_small = cv2.resize(slide_frame, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
                
                # Create a copy of the professor frame to avoid modifying the original
                result = professor_frame.copy()
                
                # Determine corner position
                if decision['slide_position'] == 'bottom_right':
                    y_start = h - slide_small.shape[0]
                    x_start = w - slide_small.shape[1]
                elif decision['slide_position'] == 'bottom_left':
                    y_start = h - slide_small.shape[0]
                    x_start = 0
                elif decision['slide_position'] == 'top_right':
                    y_start = 0
                    x_start = w - slide_small.shape[1]
                else:  # top_left
                    y_start = 0
                    x_start = 0
                
                # Ensure the overlay is within bounds
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                
                # Place the slide frame in the corner
                result[y_start:y_start+slide_small.shape[0], 
                      x_start:x_start+slide_small.shape[1]] = slide_small
                
                if self.debug_mode and 'pose_data' in decision:
                    result = self._draw_pose_boxes(result, decision['pose_data'])
                
                return result
            if self.debug_mode and 'pose_data' in decision:
                professor_frame = self._draw_pose_boxes(professor_frame, decision['pose_data'])
            return professor_frame
    
    def create_visualization(self):
        """Create a video showing the combined feeds with decisions."""
        print(f"Creating video (quality: {self.quality})")

        # Create progress bar
        pbar = tqdm(total=self.total_frames, desc="Processing frames")
        
        frame_idx = 0
        frames_written = 0
        
        try:
            while True:
                # Read frames from both videos
                slide_ret, slide_frame = self.slide_cap.read()
                prof_ret, prof_frame = self.professor_cap.read()
                
                if not slide_ret or not prof_ret:
                    break
                
                # Resize frames in debug mode
                if self.debug_mode:
                    slide_frame = cv2.resize(slide_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    prof_frame = cv2.resize(prof_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    
                # Find decision for current frame
                decision = self._find_decision_for_frame(frame_idx)
                
                # Combine frames according to decision
                combined_frame = self._combine_frames(slide_frame, prof_frame, decision)
                
                # Draw decision information
                frame = self._draw_decision_info(combined_frame, decision)
                
                # Ensure frame is in BGR format and has correct dimensions
                if frame.ndim == 2:  # If grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 4:  # If RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # Write frame using FFmpeg
                if self._write_frame_to_ffmpeg(frame):
                    frames_written += 1
                else:
                    print(f"Warning: Failed to write frame {frame_idx}")
                
                frame_idx += 1
                pbar.update(1)
            
            pbar.close()
            
            # Release resources
            self.slide_cap.release()
            self.professor_cap.release()
            
            # Close FFmpeg process
            if self.ffmpeg_process is not None:
                print("Closing FFmpeg process...")
                try:
                    # Close stdin first
                    if self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.close()
                    
                    # Wait for FFmpeg to finish with timeout
                    import time
                    timeout = 30  # 30 seconds timeout
                    start_time = time.time()
                    
                    while self.ffmpeg_process.poll() is None:
                        if time.time() - start_time > timeout:
                            print("FFmpeg process timeout - terminating")
                            self.ffmpeg_process.terminate()
                            break
                        time.sleep(0.1)
                    
                    # Read any remaining output
                    if self.ffmpeg_process.stderr:
                        stderr = self.ffmpeg_process.stderr.read()
                        if stderr:
                            try:
                                print("FFmpeg stderr:", stderr.decode('utf-8', errors='ignore'))
                            except UnicodeDecodeError:
                                print("FFmpeg stderr: [binary data]")
                    
                    print(f"FFmpeg process completed with return code: {self.ffmpeg_process.returncode}")
                except Exception as e:
                    print(f"Error closing FFmpeg process: {e}")
            
            # Verify the temporary video file exists and has content
            if not self.temp_output_path.exists():
                raise IOError(f"Temporary video file not created at {self.temp_output_path}")
            
            file_size = self.temp_output_path.stat().st_size
            if file_size == 0:
                raise IOError(f"Temporary video file is empty (0 bytes)")
            
            print(f"Successfully wrote {frames_written} frames to temporary video file")
            print(f"Temporary video file size: {file_size / (1024*1024):.2f} MB")
            
            # Add audio from professor video
            print("Adding audio...")
            import subprocess
            try:
                # Extract audio from professor video
                audio_path = self.output_dir / "temp_audio.aac"
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(self.professor_video_path),
                    '-vn', '-acodec', 'copy',
                    str(audio_path)
                ], check=True)
                
                # Verify audio file was created
                if not audio_path.exists():
                    raise IOError(f"Audio file not created at {audio_path}")
                
                audio_size = audio_path.stat().st_size
                if audio_size == 0:
                    raise IOError("Audio file is empty (0 bytes)")
                
                print(f"Successfully extracted audio file size: {audio_size / 1024:.2f} KB")
                
                # Combine video and audio
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(self.temp_output_path),
                    '-i', str(audio_path),
                    '-c:v', 'copy',  # Copy video stream without re-encoding
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-movflags', '+faststart',
                    str(self.output_path)
                ], check=True)
                
                # Clean up temporary files
                self.temp_output_path.unlink()
                audio_path.unlink()
                
                print(f"Visualization saved to {self.output_path}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error adding audio: {e}")
                # If audio addition fails, just rename the temp file
                self.temp_output_path.rename(self.output_path)
                print(f"Visualization saved without audio to {self.output_path}")
            except FileNotFoundError:
                print("ffmpeg not found. Visualization saved without audio.")
                self.temp_output_path.rename(self.output_path)
                print(f"Visualization saved without audio to {self.output_path}")
            except Exception as e:
                print(f"Error during audio processing: {e}")
                if self.temp_output_path.exists():
                    self.temp_output_path.rename(self.output_path)
                    print(f"Visualization saved without audio to {self.output_path}")
                else:
                    print("Failed to create visualization file")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
            # Clean up resources
            self.slide_cap.release()
            self.professor_cap.release()
            if self.ffmpeg_process is not None:
                try:
                    if self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.terminate()
                except:
                    pass
            raise

def run_pose_estimation(video_path: str, output_path: str):
    """Run pose estimation on a video and save results to JSON file."""
    from models.pose_estimation.pose_detector import process_video
    
    print(f"Running pose estimation on {video_path}...")
    process_video(video_path, output_path)
    print(f"Pose estimation results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process lecture videos and generate combined output')
    parser.add_argument('slide_video', help='Path to slide video')
    parser.add_argument('professor_video', help='Path to professor video')
    parser.add_argument('--output-dir', default='output', help='Directory to save outputs (default: output)')
    parser.add_argument('--ocr-results', help='Path to pre-computed OCR results')
    parser.add_argument('--pose-results', help='Path to pre-computed pose results')
    parser.add_argument('--transcription', help='Path to pre-computed transcription')
    parser.add_argument('--analysis', help='Path to pre-computed content analysis')
    parser.add_argument('--load-only', action='store_true', help='Only load pre-computed results')
    parser.add_argument('--skip-video', action='store_true', help='Skip video creation after processing')
    parser.add_argument('--pose-only', action='store_true', help='Only run pose estimation and exit')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'], default='high',
                      help='Output video quality')
    parser.add_argument('--report', action='store_true', help='Generate confidence analysis report')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir.absolute()}")

    # If pose-only mode, just run pose estimation and exit
    if args.pose_only:
        print("Running pose estimation only...")
        pose_output = output_dir / "pose" / "pose_results.json"
        pose_output.parent.mkdir(parents=True, exist_ok=True)
        run_pose_estimation(args.professor_video, str(pose_output))
        return

    # Process videos
    processor = VideoProcessor(
        slide_video_path=args.slide_video,
        professor_video_path=args.professor_video,
        output_dir=args.output_dir,
        ocr_results_path=args.ocr_results,
        pose_results_path=args.pose_results,
        transcription_path=args.transcription,
        analysis_path=args.analysis,
        load_only=args.load_only
    )

    decisions_path = processor.process_video()

    # Generate confidence analysis report if requested
    if args.report:
        print("\nGenerating confidence analysis report...")
        from evaluation.confidence_analysis import plot_multi_model_analysis
        plot_multi_model_analysis(output_dir)
        print("✅ Confidence analysis report generated")

    # Create visualization unless explicitly skipped
    if not args.skip_video:
        visualizer = DecisionVisualizer(
            slide_video_path=args.slide_video,
            professor_video_path=args.professor_video,
            decisions_path=str(decisions_path),
            output_dir=args.output_dir,
            debug_mode=args.debug,
            quality=args.quality
        )
        visualizer.create_visualization()
    else:
        print("Skipping video creation as requested")

if __name__ == "__main__":
    main()