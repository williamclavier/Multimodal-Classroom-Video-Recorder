import cv2
import numpy as np
import mediapipe as mp
import time
import os
import json

def detect_gesture(frame, pose):
    """
    Detect gestures in a frame using MediaPipe Pose
    
    Args:
        frame: The video frame
        pose: MediaPipe pose instance
        
    Returns:
        (is_pointing, is_writing): Boolean tuple
    """
    # Process frame with MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
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
        
        # Check for pointing gesture (arm extension)
        pointing = False
        # Calculate angle for right arm
        if right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5 and right_wrist.visibility > 0.5:
            angle_right = calculate_angle(
                (right_shoulder.x, right_shoulder.y),
                (right_elbow.x, right_elbow.y),
                (right_wrist.x, right_wrist.y)
            )
            if angle_right > 150:  # Arm is extended
                pointing = True
                
        # Calculate angle for left arm
        if not pointing and left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5 and left_wrist.visibility > 0.5:
            angle_left = calculate_angle(
                (left_shoulder.x, left_shoulder.y),
                (left_elbow.x, left_elbow.y),
                (left_wrist.x, left_wrist.y)
            )
            if angle_left > 150:  # Arm is extended
                pointing = True
        
        # Check for writing gesture (hand near upper area of frame)
        writing = False
        writing_threshold = 0.3
        if (left_wrist.visibility > 0.5 and left_wrist.y < writing_threshold) or \
           (right_wrist.visibility > 0.5 and right_wrist.y < writing_threshold):
            writing = True
            
        return pointing, writing
    
    return False, False

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

def process_videos(professor_video, slides_video, output_path, save_video=False):
    """
    Process the professor and slides videos to detect poses and gestures
    
    Args:
        professor_video: Path to professor video
        slides_video: Path to slides video
        output_path: Path to save the output JSON file
        save_video: If True, also saves a composite video showing the processing
    """
    # Initialize MediaPipe Pose outside the loop
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=1  # Use medium complexity to balance speed and accuracy
    )
    
    # Open video files
    print(f"Opening video files...")
    prof_cap = cv2.VideoCapture(professor_video)
    slide_cap = cv2.VideoCapture(slides_video)
    
    if not prof_cap.isOpened() or not slide_cap.isOpened():
        print("Error: Could not open video files")
        return
    
    # Get video properties
    fps = prof_cap.get(cv2.CAP_PROP_FPS)
    width = int(slide_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(slide_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(prof_cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
                       int(slide_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    print(f"Video properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer if needed
    if save_video:
        video_output_path = output_path.replace('.json', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    # Initialize state variables
    current_state = 'default'  # default, pointing, writing
    state_frames = 0
    state_threshold = 10  # Number of frames to maintain a state
    
    # Initialize results list
    results = []
    
    # Process frames
    start_time = time.time()
    processed_frames = 0
    
    while True:
        ret_prof, prof_frame = prof_cap.read()
        ret_slide, slide_frame = slide_cap.read()
        
        if not (ret_prof and ret_slide) or processed_frames >= total_frames:
            break
        
        processed_frames += 1
        
        # Get timestamp
        timestamp = processed_frames / fps
        
        # Detect gesture
        is_pointing, is_writing = detect_gesture(prof_frame, pose)
        
        # Determine new state
        new_state = 'default'
        if is_pointing:
            new_state = 'pointing'
        elif is_writing:
            new_state = 'writing'
        
        # Update state with threshold
        if new_state != current_state:
            state_frames += 1
            if state_frames >= state_threshold:
                if current_state != new_state:
                    print(f"State changed from {current_state} to {new_state}")
                current_state = new_state
                state_frames = 0
        else:
            state_frames = 0
        
        # Save results for this frame
        frame_result = {
            'frame_index': processed_frames,
            'timestamp': timestamp,
            'is_pointing': is_pointing,
            'is_writing': is_writing,
            'state': current_state,
            'confidence': 1.0  # TODO: Add actual confidence calculation
        }
        results.append(frame_result)
        
        # Create output frame if saving video
        if save_video:
            if current_state == 'pointing':
                # Show only slides
                output_frame = slide_frame.copy()
            elif current_state == 'writing':
                # Show only professor
                output_frame = cv2.resize(prof_frame, (width, height))
            else:  # default state
                # Show slides with professor inset
                output_frame = slide_frame.copy()
                
                # Add professor inset in top-right corner
                inset_height = int(height * 0.3)
                inset_width = int(width * 0.3)
                prof_resized = cv2.resize(prof_frame, (inset_width, inset_height))
                
                # Place inset in top-right corner
                y_offset = 20
                x_offset = width - inset_width - 20
                
                # Create ROI
                roi = output_frame[y_offset:y_offset+inset_height, x_offset:x_offset+inset_width]
                
                # Create a mask for blending
                prof_gray = cv2.cvtColor(prof_resized, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(prof_gray, 0, 255, cv2.THRESH_BINARY)
                
                # Place professor inset on slides
                output_frame[y_offset:y_offset+inset_height, x_offset:x_offset+inset_width] = prof_resized
                
                # Add border around inset
                cv2.rectangle(output_frame, 
                             (x_offset, y_offset),
                             (x_offset+inset_width, y_offset+inset_height),
                             (255, 255, 255), 2)
            
            # Write the frame
            out.write(output_frame)
        
        # Print progress periodically
        if processed_frames % 30 == 0:
            elapsed = time.time() - start_time
            percentage = processed_frames / total_frames * 100
            remaining = (elapsed / processed_frames) * (total_frames - processed_frames)
            print(f"Progress: {percentage:.1f}% | {processed_frames}/{total_frames} frames | "
                  f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | State: {current_state}")
    
    # Save results to JSON
    output_data = {
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'video_path': professor_video,
        'total_frames': processed_frames,
        'fps': fps,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Release resources
    print("\nProcessing completed!")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print(f"Results saved to {output_path}")
    if save_video:
        print(f"Video saved to {video_output_path}")
    
    prof_cap.release()
    slide_cap.release()
    if save_video:
        out.release()
    pose.close()

if __name__ == "__main__":
    # Process videos
    process_videos("data/final-prof.mp4", "data/final-slide.mp4", "output/final_output.json", save_video=True)
