import cv2
import numpy as np
import time
from pathlib import Path
import math
import os
import json

class ProfessorPoseDetector:
    def __init__(self):
        """
        Initialize the pose detector with OpenCV DNN
        """
        # Load the pre-trained model
        self.model_dir = Path("../pose_estimation/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model file paths
        proto_file = "pose_deploy_linevec.prototxt"
        model_file = "pose_iter_440000.caffemodel"
        
        # Download models if not present
        self._ensure_model_files_exist(proto_file, model_file)
        
        # Load model
        self.net = cv2.dnn.readNetFromCaffe(
            str(self.model_dir / proto_file), 
            str(self.model_dir / model_file)
        )
        
        # Model parameters
        self.in_width = 368
        self.in_height = 368
        self.threshold = 0.1
        
        # Define the body parts
        self.BODY_PARTS = {
            0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
            5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
            10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
            15: "LEye", 16: "REar", 17: "LEar"
        }
        
        # Define the pairs of body parts for drawing skeleton lines
        self.POSE_PAIRS = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
            [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
            [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
        ]
        
        # For smoothing
        self.recent_poses = []
        self.max_recent_poses = 5
    
    def _ensure_model_files_exist(self, proto_file, model_file):
        """Download model files if they don't exist"""
        proto_path = self.model_dir / proto_file
        model_path = self.model_dir / model_file
        
        # Download proto file if needed
        if not proto_path.exists():
            print(f"Downloading {proto_file}...")
            proto_url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
            os.system(f"curl -o {proto_path} {proto_url}")
        
        # Download model file if needed - using a direct model URL instead of Google Drive
        if not model_path.exists():
            print(f"Downloading {model_file}...")
            print("This may take a while as it's a large file (200MB)...")
            print("Attempting multiple download sources...")
            
            # Try multiple sources for the model file
            download_success = False
            
            # Alternative model sources
            model_urls = [
                "https://huggingface.co/camenduru/openpose/resolve/f5bb0c0a16060ac8b373472a5456c76bd68eb202/pose_iter_440000.caffemodel"
            ]
            
            for url in model_urls:
                try:
                    print(f"Trying to download from: {url}")
                    # Use curl with retry and progress display
                    download_cmd = f"curl -L -o {model_path} {url} --retry 3 --retry-delay 2 --progress-bar"
                    result = os.system(download_cmd)
                    
                    if result == 0 and model_path.exists() and model_path.stat().st_size > 1000000:  # Ensure file is at least 1MB
                        download_success = True
                        print(f"Successfully downloaded model from {url}")
                        break
                    else:
                        print(f"Download from {url} failed or file is incomplete")
                except Exception as e:
                    print(f"Error downloading from {url}: {str(e)}")
            
            if not download_success:
                print("\nDirect download failed. Please download the model manually from one of these sources:")
                for url in model_urls:
                    print(f"- {url}")
                print(f"And save it to: {model_path}")
                
                # Try to run without the model to see if it's already downloaded to a standard location
                standard_locations = [
                    "/usr/local/share/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
                    "/usr/share/OpenPose/models/pose/coco/pose_iter_440000.caffemodel", 
                    str(Path.home() / "Downloads" / "pose_iter_440000.caffemodel")
                ]
                
                for loc in standard_locations:
                    if Path(loc).exists():
                        print(f"Found model at standard location: {loc}")
                        print(f"Copying to: {model_path}")
                        import shutil
                        shutil.copy(loc, model_path)
                        download_success = True
                        break
                
                if not download_success:
                    raise FileNotFoundError(f"Could not download or find model file. Please download manually.")
        
        # Verify files exist
        if not proto_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Model files could not be downloaded. Please download manually:\n"
                                   f"- Proto file: {proto_path}\n"
                                   f"- Model file: {model_path}")
        else:
            print("Model files found and ready.")
    
    def detect_pose(self, frame):
        """
        Detect poses in the given frame
        Args:
            frame: Input video frame
        Returns:
            frame: Frame with detected pose
            pose_info: Dictionary containing pose information
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Prepare input blob
        input_blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (self.in_width, self.in_height),
            (0, 0, 0), swapRB=False, crop=False
        )
        
        # Set input and make forward pass
        self.net.setInput(input_blob)
        output = self.net.forward()
        
        # Get dimensions
        H = output.shape[2]
        W = output.shape[3]
        
        # List to store detected points
        points = []
        
        # For calculating confidence
        point_confidence = []
        
        # Process each body part
        for i in range(len(self.BODY_PARTS)):
            # Probability map for current part
            prob_map = output[0, i, :, :]
            
            # Find max probability point
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            # Scale to frame size
            x = int((frame_width * point[0]) / W)
            y = int((frame_height * point[1]) / H)
            
            if prob > self.threshold:
                points.append((x, y))
                point_confidence.append(prob)
            else:
                points.append(None)
                point_confidence.append(0)
        
        # Create a copy for drawing
        annotated_frame = frame.copy()
        
        # Draw the skeleton
        for pair in self.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            
            if points[partA] and points[partB]:
                cv2.line(annotated_frame, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(annotated_frame, points[partA], 5, (0, 0, 255), thickness=-1)
                cv2.circle(annotated_frame, points[partB], 5, (0, 0, 255), thickness=-1)
        
        # Analyze pose
        pose_info = self._analyze_pose(points, point_confidence, annotated_frame)
        
        # Add text with pose information
        self._add_pose_info_text(annotated_frame, pose_info)
        
        return annotated_frame, pose_info
    
    def _analyze_pose(self, points, point_confidence, frame):
        """
        Analyze the detected pose and determine if the professor is pointing
        Args:
            points: Detected keypoints
            point_confidence: Confidence for each keypoint
            frame: Frame for visualization
        Returns:
            dict: Pose information
        """
        # Default pose info
        pose_info = {
            "pose_type": "Unknown",
            "confidence": 0.0,
            "is_pointing": False,
            "pointing_direction": None
        }
        
        # Basic check if we have enough points
        if not points[1] or not points[2] or not points[5]:  # Neck, shoulders
            return pose_info
        
        # Calculate overall confidence (average of detected points)
        detected_points = sum(1 for p in points if p is not None)
        point_confidence_avg = np.mean([c for c in point_confidence if c > 0]) * 100
        confidence = (detected_points / len(points)) * 100
        
        # Extract key points
        neck = points[1]
        right_shoulder = points[2]
        left_shoulder = points[5]
        
        # Check for pointing with right arm
        right_pointing = False
        right_angle = 0
        if points[3] and points[4]:  # Right elbow and wrist
            right_elbow = points[3]
            right_wrist = points[4]
            
            # Calculate angle
            right_angle = self._calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )
            
            # Draw angle on frame
            if right_elbow[0] < frame.shape[1] - 80:  # Ensure text fits on screen
                cv2.putText(frame, f"R: {right_angle:.1f}°", 
                           (right_elbow[0] + 10, right_elbow[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Check if pointing (arm nearly straight and horizontal)
            is_arm_extended = right_angle > 150
            is_horizontal = abs(right_wrist[0] - right_elbow[0]) > abs(right_wrist[1] - right_elbow[1])
            wrist_right_of_elbow = right_wrist[0] > right_elbow[0]
            
            right_pointing = is_arm_extended and is_horizontal and wrist_right_of_elbow
        
        # Check for pointing with left arm
        left_pointing = False
        left_angle = 0
        if points[6] and points[7]:  # Left elbow and wrist
            left_elbow = points[6]
            left_wrist = points[7]
            
            # Calculate angle
            left_angle = self._calculate_angle(
                left_shoulder, left_elbow, left_wrist
            )
            
            # Draw angle on frame
            if left_elbow[0] > 80:  # Ensure text fits on screen
                cv2.putText(frame, f"L: {left_angle:.1f}°", 
                           (left_elbow[0] - 80, left_elbow[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Check if pointing (arm nearly straight and horizontal)
            is_arm_extended = left_angle > 150
            is_horizontal = abs(left_wrist[0] - left_elbow[0]) > abs(left_wrist[1] - left_elbow[1])
            wrist_left_of_elbow = left_wrist[0] < left_elbow[0]
            
            left_pointing = is_arm_extended and is_horizontal and wrist_left_of_elbow
        
        # Determine pose type
        pose_type = "Standing"
        is_pointing = False
        pointing_direction = None
        
        if right_pointing:
            pose_type = "Pointing Right"
            is_pointing = True
            pointing_direction = "right"
        elif left_pointing:
            pose_type = "Pointing Left"
            is_pointing = True
            pointing_direction = "left"
        
        # Create pose info
        pose_info = {
            "pose_type": pose_type,
            "confidence": confidence,
            "is_pointing": is_pointing,
            "pointing_direction": pointing_direction
        }
        
        # Add to recent poses and smooth
        self.recent_poses.append(pose_info)
        if len(self.recent_poses) > self.max_recent_poses:
            self.recent_poses.pop(0)
        
        return self._smooth_pose_classification()
    
    def _calculate_angle(self, a, b, c):
        """
        Calculate angle between three points (at point b)
        """
        # Convert to numpy arrays
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure within valid range
        
        # Calculate angle
        angle = np.arccos(cosine_angle)
        
        # Convert to degrees
        angle_degrees = np.degrees(angle)
        
        return angle_degrees
    
    def _smooth_pose_classification(self):
        """
        Smooth pose classifications to avoid flickering
        """
        if not self.recent_poses:
            return {
                "pose_type": "Unknown",
                "confidence": 0.0,
                "is_pointing": False,
                "pointing_direction": None
            }
        
        # Count pose types
        pose_counts = {}
        for pose in self.recent_poses:
            pose_type = pose["pose_type"]
            if pose_type not in pose_counts:
                pose_counts[pose_type] = 0
            pose_counts[pose_type] += 1
        
        # Get most common pose
        most_common_pose = max(pose_counts.items(), key=lambda x: x[1])[0]
        
        # Find most recent pose of this type
        for pose in reversed(self.recent_poses):
            if pose["pose_type"] == most_common_pose:
                return pose
        
        # Fallback to most recent
        return self.recent_poses[-1]
    
    def _add_pose_info_text(self, frame, pose_info):
        """
        Add pose information text to the frame
        """
        height, width, _ = frame.shape
        
        # Create background for text
        cv2.rectangle(frame, (10, 10), (330, 100), (0, 0, 0), -1)
        
        # Add pose type and confidence
        cv2.putText(frame, f"Pose: {pose_info['pose_type']}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {pose_info['confidence']:.1f}%", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add viewing recommendation
        recommendation = "Focus on Professor"
        if pose_info["is_pointing"]:
            if pose_info["pointing_direction"] == "right":
                recommendation = "Focus on Right Whiteboard"
            else:
                recommendation = "Focus on Left Whiteboard"
                
        cv2.putText(frame, f"Recommended View: {recommendation}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def process_video(video_path: str, output_path: str, visualize: bool = False, frame_skip: int = 5) -> None:
    """
    Process a video file for professor pose detection
    Args:
        video_path: Path to input video
        output_path: Path to save output JSON
        visualize: Whether to show visualization during processing
        frame_skip: Number of frames to skip between processing (1 = no skip, 2 = process every other frame, etc.)
    """
    print(f"Opening video file: {video_path}")
    
    # Initialize pose detector
    detector = ProfessorPoseDetector()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {total_frames}")
    print(f"Processing every {frame_skip} frames (estimated {total_frames//frame_skip} frames to process)")
    
    if width == 0 or height == 0 or fps == 0:
        print("Error: Invalid video properties")
        cap.release()
        return

    # Initialize results list
    results = []
    
    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    last_progress_time = start_time
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
            
            # Skip frames based on frame_skip parameter
            if frame_count % frame_skip != 0:
                # Skip reading frames we don't need
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + frame_skip - 1)
                continue
                
            processed_frames += 1
            
            try:
                # Process frame
                processed_frame, pose_info = detector.detect_pose(frame)
                
                # Get timestamp
                timestamp = frame_count / fps
                
                # Create frame result
                frame_result = {
                    'frame_index': frame_count,
                    'timestamp': timestamp,
                    'is_pointing': pose_info['is_pointing'],
                    'is_writing': False,  # OpenPose doesn't detect writing, so always False
                    'confidence': pose_info['confidence'],
                    'debug_info': {
                        'pose_type': pose_info['pose_type'],
                        'pointing_direction': pose_info['pointing_direction'],
                        'keypoints': [],  # Will be populated if needed
                        'bbox': None  # Will be populated if needed
                    }
                }
                results.append(frame_result)
                
                # Show visualization if enabled
                if visualize:
                    cv2.imshow('Pose Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress periodically (every 2 seconds)
                current_time = time.time()
                if current_time - last_progress_time >= 2.0:
                    elapsed = current_time - start_time
                    percentage = processed_frames / (total_frames / frame_skip) * 100
                    remaining = (elapsed / processed_frames) * ((total_frames / frame_skip) - processed_frames)
                    fps_processed = processed_frames / elapsed
                    
                    print(f"Progress: {percentage:.1f}% | {processed_frames}/{total_frames//frame_skip} frames | "
                          f"FPS: {fps_processed:.1f} | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
                    last_progress_time = current_time
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        
        # Save results to JSON
        output_data = {
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'video_path': video_path,
            'total_frames': processed_frames,
            'fps': fps,
            'frame_skip': frame_skip,
            'processing_stats': {
                'total_time': total_time,
                'average_fps': avg_fps,
                'frames_processed': processed_frames,
                'frames_skipped': total_frames - processed_frames
            },
            'results': results
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nProcessing completed!")
        print(f"Total frames processed: {processed_frames}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average processing speed: {avg_fps:.1f} FPS")
        print(f"Results saved to {output_path}")

def main():
    """
    Main function to process a specific video
    """
    # Set paths
    data_dir = Path("../../data")
    output_dir = Path("../../output")
    output_dir.mkdir(exist_ok=True)
    
    # Specific video to process
    video_file = "Lecture-Prof.mp4"
    video_path = data_dir / video_file
    
    if not video_path.exists():
        print(f"Error: Video file {video_path} does not exist!")
        return
        
    # Set output path
    output_path = output_dir / f"pose_{video_file}"
    
    print(f"\nProcessing {video_file}...")
    process_video(str(video_path), str(output_path))
    print(f"Saved processed video to {output_path}")

if __name__ == "__main__":
    print("Starting pose detection...")
    main()
    print("Pose detection completed.")
