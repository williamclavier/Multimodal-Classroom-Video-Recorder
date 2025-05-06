import cv2
import numpy as np
import mediapipe as mp
import time
import os
import json
from typing import Tuple, List, Any, Dict
from dataclasses import dataclass
from collections import deque

@dataclass
class BoardArea:
    x1: float
    y1: float
    x2: float
    y2: float
    
    def contains_point(self, x: float, y: float, margin: float = 0.05) -> bool:
        """Check if a point is within the board area with a margin"""
        return (self.x1 - margin <= x <= self.x2 + margin and 
                self.y1 - margin <= y <= self.y2 + margin)

class MotionTracker:
    def __init__(self, max_points: int = 10):
        self.positions = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        self.last_direction = None
        self.direction_changes = 0
    
    def add_point(self, x: float, y: float, timestamp: float):
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
        
        # Track direction changes
        if len(self.positions) >= 2:
            current_direction = np.arctan2(
                self.positions[-1][1] - self.positions[-2][1],
                self.positions[-1][0] - self.positions[-2][0]
            )
            if self.last_direction is not None:
                angle_diff = abs(current_direction - self.last_direction)
                if angle_diff > np.pi/4:  # Significant direction change
                    self.direction_changes += 1
            self.last_direction = current_direction
    
    def get_velocity(self) -> Tuple[float, float]:
        if len(self.positions) < 2:
            return 0.0, 0.0
        
        dx = self.positions[-1][0] - self.positions[0][0]
        dy = self.positions[-1][1] - self.positions[0][1]
        dt = self.timestamps[-1] - self.timestamps[0]
        
        if dt == 0:
            return 0.0, 0.0
        
        return float(dx/dt), float(dy/dt)  # Convert to Python float
    
    def get_acceleration(self) -> Tuple[float, float]:
        if len(self.positions) < 3:
            return 0.0, 0.0
        
        v1x, v1y = self.get_velocity()
        self.positions.rotate(-1)
        self.timestamps.rotate(-1)
        v2x, v2y = self.get_velocity()
        self.positions.rotate(1)
        self.timestamps.rotate(1)
        
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt == 0:
            return 0.0, 0.0
        
        return float((v2x - v1x)/dt), float((v2y - v1y)/dt)  # Convert to Python float
    
    def is_writing_motion(self) -> bool:
        if len(self.positions) < 3:
            return False
        
        # Calculate velocity and acceleration
        vx, vy = self.get_velocity()
        ax, ay = self.get_acceleration()
        
        # Calculate speed and acceleration magnitude
        speed = float(np.sqrt(vx*vx + vy*vy))  # Convert to Python float
        accel = float(np.sqrt(ax*ax + ay*ay))  # Convert to Python float
        
        # Writing typically has:
        # 1. Small but non-zero speed
        # 2. Small but non-zero acceleration
        # 3. Frequent direction changes
        # 4. More horizontal than vertical movement
        # 5. Arm should be raised (y position should be above shoulder level)
        horizontal_movement = abs(vx) > abs(vy)
        
        # Check if arm is raised by comparing y positions
        # If we have enough points, check if the arm is raised
        arm_raised = False
        if len(self.positions) >= 3:
            # Get the average y position of the last few points
            recent_y_positions = [p[1] for p in list(self.positions)[-3:]]
            avg_y = sum(recent_y_positions) / len(recent_y_positions)
            # Arm is considered raised if the average y position is in the upper half of the frame
            arm_raised = avg_y < 0.5  # Assuming y=0 is top of frame
        
        return (0.01 < speed < 0.5 and  # Speed threshold
                0.01 < accel < 0.3 and  # Acceleration threshold
                self.direction_changes >= 2 and  # At least 2 direction changes
                horizontal_movement and  # More horizontal than vertical movement
                arm_raised)  # Arm must be raised

def calibrate_board_area(frame: np.ndarray) -> BoardArea:
    """
    Let user select the board area by clicking and dragging
    """
    board_area = None
    start_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal board_area, start_point
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and start_point is not None:
            end_point = (x, y)
            # Convert to normalized coordinates
            h, w = frame.shape[:2]
            x1 = min(start_point[0], end_point[0]) / w
            y1 = min(start_point[1], end_point[1]) / h
            x2 = max(start_point[0], end_point[0]) / w
            y2 = max(start_point[1], end_point[1]) / h
            board_area = BoardArea(x1, y1, x2, y2)
            cv2.destroyWindow('Calibrate Board Area')
    
    cv2.namedWindow('Calibrate Board Area')
    cv2.setMouseCallback('Calibrate Board Area', mouse_callback)
    
    while board_area is None:
        display_frame = frame.copy()
        if start_point is not None:
            cv2.rectangle(display_frame, start_point, 
                         (cv2.getWindowImageRect('Calibrate Board Area')[2], 
                          cv2.getWindowImageRect('Calibrate Board Area')[3]), 
                         (0, 255, 0), 2)
        cv2.putText(display_frame, "Click and drag to select board area", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Calibrate Board Area', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return board_area

def draw_pose_landmarks(frame: np.ndarray, landmarks: List[Any], 
                       connections: List[Tuple[int, int]]) -> np.ndarray:
    """
    Draw pose landmarks and connections on the frame
    
    Args:
        frame: The video frame
        landmarks: List of pose landmarks
        connections: List of landmark connections to draw
        
    Returns:
        frame: Frame with landmarks and connections drawn
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        if start_point.visibility > 0.5 and end_point.visibility > 0.5:
            start_x = int(start_point.x * frame_width)
            start_y = int(start_point.y * frame_height)
            end_x = int(end_point.x * frame_width)
            end_y = int(end_point.y * frame_height)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Draw landmarks
    for landmark in landmarks:
        if landmark.visibility > 0.5:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    return frame

def detect_gesture(frame: np.ndarray, pose: mp.solutions.pose.Pose, 
                  board_area: BoardArea, motion_trackers: Dict[str, MotionTracker],
                  timestamp: float, visualize: bool = False) -> Tuple[bool, bool, float, np.ndarray, dict]:
    """
    Detect gestures in a frame using MediaPipe Pose
    
    Args:
        frame: The video frame
        pose: MediaPipe pose instance
        board_area: The area of the board
        motion_trackers: Dictionary of motion trackers for each hand
        timestamp: Current timestamp
        visualize: Whether to draw pose landmarks on the frame
        
    Returns:
        (is_pointing, is_writing, confidence, frame, debug_info): Boolean tuple, confidence score, 
        optionally visualized frame, and debug information
    """
    # Process frame with MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    debug_info = {
        'arm_angles': {'left': None, 'right': None},
        'shoulder_angles': {'left': None, 'right': None},
        'hand_positions': {'left': None, 'right': None},
        'visibility': {'left': None, 'right': None},
        'motion': {'left': None, 'right': None},
        'keypoints': [],  # Add keypoints list for main.py
        'bbox': None  # Add bbox for main.py
    }
    
    if results.pose_landmarks:
        # Get landmarks
        landmarks = results.pose_landmarks.landmark
        mp_pose = mp.solutions.pose
        
        # Get relevant landmarks
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate confidence based on visibility of key landmarks
        key_landmarks = [left_wrist, right_wrist, left_elbow, right_elbow, 
                        left_shoulder, right_shoulder, left_hip, right_hip]
        visibility_scores = [landmark.visibility for landmark in key_landmarks]
        confidence = sum(visibility_scores) / len(visibility_scores)
        
        # Add keypoints in the format expected by main.py
        frame_height, frame_width = frame.shape[:2]
        keypoints = []
        
        # Add wrist keypoints
        if left_wrist.visibility > 0.5:
            keypoints.append({
                'part': 'leftWrist',
                'position': (left_wrist.x * frame_width, left_wrist.y * frame_height),
                'score': left_wrist.visibility
            })
        if right_wrist.visibility > 0.5:
            keypoints.append({
                'part': 'rightWrist',
                'position': (right_wrist.x * frame_width, right_wrist.y * frame_height),
                'score': right_wrist.visibility
            })
            
        # Add other keypoints for visualization
        for landmark_idx, landmark in enumerate(landmarks):
            if landmark.visibility > 0.5:
                keypoints.append({
                    'part': mp_pose.PoseLandmark(landmark_idx).name,
                    'position': (landmark.x * frame_width, landmark.y * frame_height),
                    'score': landmark.visibility
                })
        
        debug_info['keypoints'] = keypoints
        
        # Calculate bounding box for the person
        x_coords = [kp['position'][0] for kp in keypoints]
        y_coords = [kp['position'][1] for kp in keypoints]
        if x_coords and y_coords:
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            # Add padding
            padding = 50
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame_width, x2 + padding)
            y2 = min(frame_height, y2 + padding)
            debug_info['bbox'] = (x1, y1, x2 - x1, y2 - y1)
        
        # Check for pointing gesture with improved criteria
        pointing = False
        pointing_confidence = 0.0
        
        # Calculate angles and positions for both arms
        for side in ['right', 'left']:
            wrist = right_wrist if side == 'right' else left_wrist
            elbow = right_elbow if side == 'right' else left_elbow
            shoulder = right_shoulder if side == 'right' else left_shoulder
            hip = right_hip if side == 'right' else left_hip
            
            if all(l.visibility > 0.5 for l in [wrist, elbow, shoulder, hip]):
                # Calculate arm extension angle
                arm_angle = calculate_angle(
                    (shoulder.x, shoulder.y),
                    (elbow.x, elbow.y),
                    (wrist.x, wrist.y)
                )
                
                # Calculate shoulder angle to determine if arm is raised
                shoulder_angle = calculate_angle(
                    (hip.x, hip.y),
                    (shoulder.x, shoulder.y),
                    (elbow.x, elbow.y)
                )
                
                # Store debug information
                debug_info['arm_angles'][side] = arm_angle
                debug_info['shoulder_angles'][side] = shoulder_angle
                debug_info['hand_positions'][side] = {
                    'x': wrist.x,
                    'y': wrist.y,
                    'relative_to_shoulder': {
                        'x': wrist.x - shoulder.x,
                        'y': wrist.y - shoulder.y
                    }
                }
                debug_info['visibility'][side] = {
                    'wrist': wrist.visibility,
                    'elbow': elbow.visibility,
                    'shoulder': shoulder.visibility
                }
                
                # Check if arm is extended and raised
                if arm_angle > 150 and shoulder_angle > 90:
                    # Calculate hand position relative to shoulder
                    hand_raised = wrist.y < shoulder.y
                    
                    # Additional criteria for pointing
                    hand_stable = abs(wrist.y - elbow.y) < 0.1  # Hand should be relatively stable
                    arm_straight = abs(arm_angle - 180) < 15  # Arm should be very straight
                    
                    if hand_raised and hand_stable and arm_straight:
                        pointing = True
                        pointing_confidence = min(wrist.visibility, elbow.visibility, shoulder.visibility)
                        break
        
        # Check for writing gesture with improved criteria
        writing = False
        writing_confidence = 0.0
        
        # Check both hands
        for side, wrist in [('left', left_wrist), ('right', right_wrist)]:
            if wrist.visibility > 0.5:
                # Update motion tracker
                motion_trackers[side].add_point(wrist.x, wrist.y, timestamp)
                
                # Check if hand is near the board
                near_board = board_area.contains_point(wrist.x, wrist.y)
                
                # Get motion information
                vx, vy = motion_trackers[side].get_velocity()
                ax, ay = motion_trackers[side].get_acceleration()
                
                # Convert numpy values to Python types for JSON serialization
                debug_info['motion'][side] = {
                    'velocity': (float(vx), float(vy)),
                    'acceleration': (float(ax), float(ay)),
                    'near_board': bool(near_board),
                    'is_writing_motion': bool(motion_trackers[side].is_writing_motion()),
                    'direction_changes': int(motion_trackers[side].direction_changes)
                }
                
                # Writing detection criteria:
                # 1. Hand must be near the board
                # 2. Motion must match writing pattern (small, frequent movements)
                if near_board and motion_trackers[side].is_writing_motion():
                    writing = True
                    writing_confidence = max(writing_confidence, wrist.visibility)
        
        # Update overall confidence
        confidence = max(confidence, pointing_confidence, writing_confidence)
        
        # Visualize if requested
        if visualize:
            frame = draw_pose_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw board area
            h, w = frame.shape[:2]
            cv2.rectangle(frame, 
                         (int(board_area.x1 * w), int(board_area.y1 * h)),
                         (int(board_area.x2 * w), int(board_area.y2 * h)),
                         (0, 255, 0), 2)
            
            # Add text overlay for gestures and debug info
            if pointing:
                cv2.putText(frame, "Pointing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if writing:
                cv2.putText(frame, "Writing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add debug information
            y_offset = 120
            for side in ['left', 'right']:
                if debug_info['arm_angles'][side] is not None:
                    cv2.putText(frame, f"{side.capitalize()} Arm: {debug_info['arm_angles'][side]:.1f}°", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y_offset += 25
                    cv2.putText(frame, f"{side.capitalize()} Shoulder: {debug_info['shoulder_angles'][side]:.1f}°", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y_offset += 25
                    
                    if debug_info['motion'][side] is not None:
                        motion = debug_info['motion'][side]
                        cv2.putText(frame, f"{side.capitalize()} Speed: {np.sqrt(motion['velocity'][0]**2 + motion['velocity'][1]**2):.3f}", 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        y_offset += 25
                        cv2.putText(frame, f"{side.capitalize()} Near Board: {motion['near_board']}", 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        y_offset += 25
        
        return pointing, writing, confidence, frame if visualize else None, debug_info
    
    return False, False, 0.0, frame if visualize else None, debug_info

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    
    Args:
        a: First point (shoulder)
        b: Second point (elbow)
        c: Third point (wrist)
        
    Returns:
        angle: Angle in degrees
    """
    # Convert to numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product and magnitudes
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clip to handle floating point issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def process_video(professor_video: str, output_path: str, visualize: bool = False, frame_skip: int = 5) -> None:
    """
    Process the professor video to detect poses and gestures
    
    Args:
        professor_video: Path to professor video
        output_path: Path to save the output JSON file
        visualize: Whether to show visualization during processing
        frame_skip: Number of frames to skip between processing (1 = no skip, 2 = process every other frame, etc.)
    """
    # Warn about frame skipping
    if frame_skip > 1:
        print(f"Warning: Frame skipping (frame_skip={frame_skip}) may reduce writing detection accuracy.")
        print("Consider using frame_skip=1 for more accurate writing detection.")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=2  # Use high complexity for better accuracy
    )
    
    # Open video file
    print(f"Opening video file: {professor_video}")
    prof_cap = cv2.VideoCapture(professor_video)
    
    if not prof_cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = prof_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(prof_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    print(f"- Processing every {frame_skip} frames")
    
    # Get first frame for board calibration
    ret, first_frame = prof_cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Calibrate board area
    print("Please select the board area by clicking and dragging...")
    board_area = calibrate_board_area(first_frame)
    if board_area is None:
        print("Error: Board area calibration failed")
        return
    
    # Initialize motion trackers
    motion_trackers = {
        'left': MotionTracker(),
        'right': MotionTracker()
    }
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize results list
    results = []
    
    # Process frames
    start_time = time.time()
    processed_frames = 0
    frame_count = 0
    
    # Reset video capture to start
    prof_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret_prof, prof_frame = prof_cap.read()
        
        if not ret_prof or processed_frames >= total_frames:
            break
        
        frame_count += 1
        
        # Skip frames based on frame_skip parameter
        if frame_count % frame_skip != 0:
            continue
        
        processed_frames += 1
        
        # Get timestamp
        timestamp = frame_count / fps
        
        # Detect gesture
        is_pointing, is_writing, confidence, vis_frame, debug_info = detect_gesture(
            prof_frame, pose, board_area, motion_trackers, timestamp, visualize
        )
        
        # Save results for this frame
        frame_result = {
            'frame_index': frame_count,
            'timestamp': timestamp,
            'is_pointing': is_pointing,
            'is_writing': is_writing,
            'confidence': confidence,
            'debug_info': debug_info
        }
        results.append(frame_result)
        
        # Show visualization if enabled
        if visualize and vis_frame is not None:
            cv2.imshow('Pose Estimation', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print progress periodically
        if processed_frames % 30 == 0:
            elapsed = time.time() - start_time
            percentage = processed_frames / (total_frames / frame_skip) * 100
            remaining = (elapsed / processed_frames) * ((total_frames / frame_skip) - processed_frames)
            print(f"Progress: {percentage:.1f}% | {processed_frames}/{total_frames//frame_skip} frames | "
                  f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
    
    # Save results to JSON
    output_data = {
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'video_path': professor_video,
        'total_frames': processed_frames,
        'fps': fps,
        'frame_skip': frame_skip,
        'board_area': {
            'x1': board_area.x1,
            'y1': board_area.y1,
            'x2': board_area.x2,
            'y2': board_area.y2
        },
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Release resources
    print("\nProcessing completed!")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print(f"Results saved to {output_path}")
    
    prof_cap.release()
    pose.close()
    if visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Process video with visualization and frame skipping
    process_video("demo.MOV", "output/pose_results.json", visualize=True, frame_skip=3)
