import cv2
import numpy as np
import time
from pathlib import Path
import os

class PoseDetector:
    def __init__(self, proto_file=None, model_file=None):
        """
        Initialize the pose detector with OpenPose model
        Args:
            proto_file: Path to the .prototxt file containing the model architecture
            model_file: Path to the .caffemodel file containing the model weights
        """
        # Set default paths if not provided
        if proto_file is None:
            proto_file = "models/pose_estimation/pose_deploy_linevec.prototxt"
        if model_file is None:
            model_file = "models/pose_estimation/pose_iter_440000.caffemodel"
            
        self.net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
        self.in_height = 368
        self.in_width = 368
        self.threshold = 0.1
        
        # Define body parts and their connections
        self.BODY_PARTS = {
            0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
            5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
            10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
            15: "LEye", 16: "REar", 17: "LEar"
        }
        
        self.POSE_PAIRS = [
            [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
            [1,0], [0,14], [14,16], [0,15], [15,17]
        ]

    def detect_pose(self, frame):
        """
        Detect poses in the given frame
        Args:
            frame: Input video frame
        Returns:
            frame: Frame with detected poses
            poses: List of detected poses
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Prepare the frame for the network
        inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.in_width, self.in_height),
                                        (0, 0, 0), swapRB=False, crop=False)
        
        self.net.setInput(inp_blob)
        output = self.net.forward()
        
        H = output.shape[2]
        W = output.shape[3]
        
        # Empty list to store the detected keypoints
        points = []
        
        for i in range(len(self.BODY_PARTS)):
            # Confidence map of corresponding body's part
            prob_map = output[0, i, :, :]
            
            # Find global maxima of the prob_map
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            
            # Scale the point to fit on the original image
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H
            
            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        # Draw the detected poses
        for pair in self.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        
        # Analyze pose and add text description
        pose_description, confidence = self._analyze_pose(points)
        cv2.putText(frame, f"{pose_description} ({confidence:.1f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, points

    def _analyze_pose(self, points):
        """
        Analyze the detected pose and return a description and confidence
        Args:
            points: List of detected keypoints
        Returns:
            tuple: (description, confidence)
        """
        if not points[1] or not points[2] or not points[5]:  # Neck and shoulders
            return "Pose not clear", 0.0
            
        # Calculate angles and positions for analysis
        neck = points[1]
        r_shoulder = points[2]
        l_shoulder = points[5]
        
        # Calculate confidence based on number of detected points
        detected_points = sum(1 for p in points if p is not None)
        confidence = (detected_points / len(points)) * 100
        
        # Check if pointing towards board (right arm extended)
        if points[3] and points[4]:  # Right elbow and wrist
            arm_angle = self._calculate_angle(neck, r_shoulder, points[3])
            if 150 < arm_angle < 180:
                return "Pointing towards board", confidence
        
        # Check if writing on board
        if points[6] and points[7]:  # Left elbow and wrist
            arm_angle = self._calculate_angle(neck, l_shoulder, points[6])
            if 90 < arm_angle < 150:
                return "Writing on board", confidence
        
        # Check if standing straight
        if points[8] and points[11]:  # Hips
            hip_angle = self._calculate_angle(neck, points[8], points[11])
            if 170 < hip_angle < 190:
                return "Standing straight", confidence
        
        # Check if walking
        if points[9] and points[12]:  # Knees
            knee_angle = self._calculate_angle(points[8], points[9], points[10])
            if knee_angle < 150:
                return "Walking", confidence
        
        return "Normal standing pose", confidence

    def _calculate_angle(self, a, b, c):
        """
        Calculate angle between three points
        Args:
            a, b, c: Three points forming the angle
        Returns:
            float: Angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

def process_video(video_path, output_path):
    """
    Process a video file and detect poses
    Args:
        video_path: Path to input video
        output_path: Path to save output video
    """
    # Initialize pose detector
    detector = PoseDetector()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, _ = detector.detect_pose(frame)
        
        # Write frame
        out.write(processed_frame)
        
        frame_count += 1
        
        # Calculate and display FPS
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"Processing frame {frame_count}, FPS: {fps:.2f}")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_all_videos():
    """
    Process all videos in the data folder
    """
    data_dir = Path("../../data")
    output_dir = Path("../../output")
    output_dir.mkdir(exist_ok=True)
    
    # Debug: Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist!")
        return
    
    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = [f for f in data_dir.glob('**/*') if f.suffix.lower() in video_extensions]
    
    # Debug: Print number of videos found
    print(f"Found {len(video_files)} video files in {data_dir}")
    
    if not video_files:
        print(f"No video files found in {data_dir}. Please place videos in the data folder.")
        return
    
    for video_path in video_files:
        print(f"\nProcessing {video_path.name}...")
        try:
            # Debug: Check if video file exists and is readable
            if not video_path.exists():
                print(f"Error: Video file {video_path} does not exist!")
                continue
                
            output_path = output_dir / f"pose_{video_path.name}"
            process_video(str(video_path), str(output_path))
            print(f"Saved processed video to {output_path}")
        except Exception as e:
            print(f"Error processing {video_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting pose detection...")
    # Process all videos in the data folder
    process_all_videos()
    print("Pose detection completed.") 