import cv2
import numpy as np
import time
from pathlib import Path
import mediapipe as mp

class ProfessorPoseDetector:
    def __init__(self):
        """
        Initialize the pose detector with MediaPipe
        """
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For smoothing
        self.recent_poses = []
        self.max_recent_poses = 5
        
        # Define body parts we care about
        self.BODY_PARTS = {
            "LEFT_SHOULDER": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            "RIGHT_SHOULDER": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "LEFT_ELBOW": self.mp_pose.PoseLandmark.LEFT_ELBOW,
            "RIGHT_ELBOW": self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            "LEFT_WRIST": self.mp_pose.PoseLandmark.LEFT_WRIST,
            "RIGHT_WRIST": self.mp_pose.PoseLandmark.RIGHT_WRIST,
            "LEFT_HIP": self.mp_pose.PoseLandmark.LEFT_HIP,
            "RIGHT_HIP": self.mp_pose.PoseLandmark.RIGHT_HIP,
            "LEFT_KNEE": self.mp_pose.PoseLandmark.LEFT_KNEE,
            "RIGHT_KNEE": self.mp_pose.PoseLandmark.RIGHT_KNEE,
            "LEFT_ANKLE": self.mp_pose.PoseLandmark.LEFT_ANKLE,
            "RIGHT_ANKLE": self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }
    
    def detect_pose(self, frame):
        """
        Detect poses in the given frame
        Args:
            frame: Input video frame
        Returns:
            frame: Frame with detected pose
            pose_info: Dictionary containing pose information
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        # Create a copy for drawing
        annotated_frame = frame.copy()
        
        # Initialize pose info
        pose_info = {
            "pose_type": "Unknown",
            "confidence": 0.0,
            "is_pointing": False,
            "pointing_direction": None
        }
        
        if results.pose_landmarks:
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Analyze pose
            pose_info = self._analyze_pose(results.pose_landmarks.landmark, annotated_frame)
            
            # Add text with pose information
            self._add_pose_info_text(annotated_frame, pose_info)
        
        return annotated_frame, pose_info
    
    def _analyze_pose(self, landmarks, frame):
        """
        Analyze the detected pose and determine if the professor is pointing
        Args:
            landmarks: Detected keypoints
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
        
        # Check if we have the necessary landmarks
        if not all(key in landmarks for key in self.BODY_PARTS.values()):
            return pose_info
        
        # Calculate overall confidence (average of detected points)
        confidence = np.mean([landmarks[key].visibility for key in self.BODY_PARTS.values()]) * 100
        
        # Extract key points
        left_shoulder = landmarks[self.BODY_PARTS["LEFT_SHOULDER"]]
        right_shoulder = landmarks[self.BODY_PARTS["RIGHT_SHOULDER"]]
        left_elbow = landmarks[self.BODY_PARTS["LEFT_ELBOW"]]
        right_elbow = landmarks[self.BODY_PARTS["RIGHT_ELBOW"]]
        left_wrist = landmarks[self.BODY_PARTS["LEFT_WRIST"]]
        right_wrist = landmarks[self.BODY_PARTS["RIGHT_WRIST"]]
        
        # Check for pointing with right arm
        right_pointing = False
        if right_shoulder.visibility > 0.7 and right_elbow.visibility > 0.7 and right_wrist.visibility > 0.7:
            # Calculate angle
            right_angle = self._calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            )
            
            # Check if pointing (arm nearly straight and horizontal)
            is_arm_extended = right_angle > 150
            is_horizontal = abs(right_wrist.x - right_elbow.x) > abs(right_wrist.y - right_elbow.y)
            wrist_right_of_elbow = right_wrist.x > right_elbow.x
            
            right_pointing = is_arm_extended and is_horizontal and wrist_right_of_elbow
        
        # Check for pointing with left arm
        left_pointing = False
        if left_shoulder.visibility > 0.7 and left_elbow.visibility > 0.7 and left_wrist.visibility > 0.7:
            # Calculate angle
            left_angle = self._calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y],
                [left_wrist.x, left_wrist.y]
            )
            
            # Check if pointing (arm nearly straight and horizontal)
            is_arm_extended = left_angle > 150
            is_horizontal = abs(left_wrist.x - left_elbow.x) > abs(left_wrist.y - left_elbow.y)
            wrist_left_of_elbow = left_wrist.x < left_elbow.x
            
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

def process_video(video_path, output_path):
    """
    Process a video file for professor pose detection
    Args:
        video_path: Path to input video
        output_path: Path to save output video
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
    
    if width == 0 or height == 0 or fps == 0:
        print("Error: Invalid video properties")
        cap.release()
        return
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file {output_path}")
        cap.release()
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
                
            try:
                # Process frame
                processed_frame, pose_info = detector.detect_pose(frame)
                
                # Write frame
                out.write(processed_frame)
                
                frame_count += 1
                
                # Calculate and display FPS
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    processing_fps = frame_count / elapsed_time
                    print(f"Processing frame {frame_count}/{total_frames}, FPS: {processing_fps:.2f}")
                    
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
        out.release()
        detector.pose.close()
        cv2.destroyAllWindows()
        print(f"Processing completed. Processed {frame_count} frames.")

def main():
    """
    Main function to process a specific video
    """
    # Set paths
    data_dir = Path("../../data")
    output_dir = Path("../../output")
    output_dir.mkdir(exist_ok=True)
    
    # Specific video to process
    video_file = "Lecture1-Prof.mp4"
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
