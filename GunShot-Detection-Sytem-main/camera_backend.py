#!/usr/bin/env python3
"""
Backend Camera Feed with Bounding Boxes
Captures video from camera, applies MediaPipe detection for head and hands,
and continuously updates the latest frame for frontend consumption.
"""

import cv2
import mediapipe as mp
import time
import threading
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Camera settings
CAMERA_INDEX = 0 # Default camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Output paths
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
LATEST_FRAME_PATH = TEMP_DIR / "camera_latest_frame.jpg"
CAMERA_STATUS_PATH = TEMP_DIR / "camera_status.json"

# MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

class CameraBackend:
    def __init__(self):
        self.running = False
        self.stop_flag = threading.Event()
        self.cap = None
        self.pose = None
        self.hands = None
        
        print("[Camera Backend] Initializing camera backend...")
        self.init_camera()
        self.init_mediapipe()
        
    def init_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
                
            print(f"[Camera Backend] Camera initialized: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
        except Exception as e:
            print(f"[Camera Backend] Camera initialization failed: {e}")
            raise
            
    def init_mediapipe(self):
        """Initialize MediaPipe solutions"""
        try:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print("[Camera Backend] MediaPipe initialized")
        except Exception as e:
            print(f"[Camera Backend] MediaPipe initialization failed: {e}")
            raise
            
    def draw_bounding_boxes(self, frame):
        """Draw bounding boxes for head and hands"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            hands_results = self.hands.process(rgb_frame)
        except Exception as e:
            print(f"[Camera Backend] MediaPipe processing error: {e}")
            return frame
            
        result_frame = frame.copy()
        h, w, _ = result_frame.shape
        
        # Draw head bounding box (GREEN)
        if pose_results and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Head landmarks
            head_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_EYE,
                mp_pose.PoseLandmark.RIGHT_EYE,
                mp_pose.PoseLandmark.LEFT_EAR,
                mp_pose.PoseLandmark.RIGHT_EAR,
                mp_pose.PoseLandmark.MOUTH_LEFT,
                mp_pose.PoseLandmark.MOUTH_RIGHT
            ]
            
            xs = []
            ys = []
            for landmark in head_landmarks:
                if hasattr(landmarks[landmark], 'x') and hasattr(landmarks[landmark], 'y'):
                    if 0 <= landmarks[landmark].x <= 1 and 0 <= landmarks[landmark].y <= 1:
                        xs.append(landmarks[landmark].x)
                        ys.append(landmarks[landmark].y)
            
            if xs and ys:
                x_min = int(max(0, min(xs) * w))
                x_max = int(min(w-1, max(xs) * w))
                y_min = int(max(0, min(ys) * h))
                y_max = int(min(h-1, max(ys) * h))
                
                # Add padding
                bw = max(1, x_max - x_min)
                bh = max(1, y_max - y_min)
                x_pad = int(bw * 0.8)
                y_pad_top = int(bh * 1.2)
                y_pad_bot = int(bh * 0.8)
                
                x_min = max(0, x_min - x_pad)
                x_max = min(w-1, x_max + x_pad)
                y_min = max(0, y_min - y_pad_top)
                y_max = min(h-1, y_max + y_pad_bot)
                
                cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(result_frame, "HEAD", (x_min, max(0, y_min-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw hand bounding boxes (RED)
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                
                x_min = int(max(0, min(xs) * w))
                x_max = int(min(w-1, max(xs) * w))
                y_min = int(max(0, min(ys) * h))
                y_max = int(min(h-1, max(ys) * h))
                
                # Add padding
                bw = max(1, x_max - x_min)
                bh = max(1, y_max - y_min)
                x_pad = int(bw * 1.5) + 20
                y_pad = int(bh * 1.5) + 20
                
                x_min = max(0, x_min - x_pad)
                x_max = min(w-1, x_max + x_pad)
                y_min = max(0, y_min - int(y_pad * 0.3))
                y_max = min(h-1, y_max + y_pad)
                
                cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                cv2.putText(result_frame, "HAND", (x_min, max(0, y_min-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result_frame
        
    def write_status(self, status):
        """Write camera status to JSON file"""
        status_data = {
            "status": status,
            "timestamp": time.time(),
            "camera_index": CAMERA_INDEX,
            "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "fps": FPS
        }
        try:
            with open(CAMERA_STATUS_PATH, "w") as f:
                json.dump(status_data, f)
        except Exception as e:
            print(f"[Camera Backend] Failed to write status: {e}")
            
    def run(self):
        """Main camera loop"""
        print("[Camera Backend] Starting camera feed...")
        self.running = True
        self.write_status("running")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while not self.stop_flag.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("[Camera Backend] Failed to read frame from camera")
                    break
                    
                # Apply bounding boxes
                frame_with_boxes = self.draw_bounding_boxes(frame)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame_with_boxes, timestamp, (10, FRAME_HEIGHT - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add "LIVE" indicator
                cv2.putText(frame_with_boxes, "LIVE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Save latest frame
                try:
                    cv2.imwrite(str(LATEST_FRAME_PATH), frame_with_boxes)
                except Exception as e:
                    print(f"[Camera Backend] Failed to save frame: {e}")
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    print(f"[Camera Backend] FPS: {current_fps:.1f}")
                
                # Control frame rate
                time.sleep(1.0 / FPS)
                
        except Exception as e:
            print(f"[Camera Backend] Camera loop error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        print("[Camera Backend] Cleaning up...")
        self.running = False
        self.write_status("stopped")
        
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
            
        print("[Camera Backend] Cleanup complete")

def main():
    """Main entry point"""
    try:
        camera = CameraBackend()
        camera.run()
    except KeyboardInterrupt:
        print("\n[Camera Backend] Interrupted by user")
    except Exception as e:
        print(f"[Camera Backend] Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
