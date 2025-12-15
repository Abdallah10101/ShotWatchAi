import os
import time
import threading
import math
from pathlib import Path
from datetime import datetime
from collections import deque
import csv

import numpy as np
import cv2
import torch
import torch.nn as nn
import pyaudio
import librosa
import warnings
import noisereduce as nr
from scipy import signal
import mediapipe as mp

warnings.filterwarnings('ignore')

# Audio parameters
SR = 22050 # 22050 audio samples can record in one second
CLIP_SEC = 2.0 # each audio clip is 2 seconds long
N_MELS = 128 # number of mel bands to generate
FMAX = 8000 # maximum frequency for mel spectrogram
CHUNK = 1024 # audio chunk size for pyaudio
FORMAT = pyaudio.paInt16 # 16-bit int format
CHANNELS = 1 # mono audio

# Enhanced audio processing parameters
TARGET_RMS = 0.05 # target RMS for normalization
NOISE_REDUCTION = True # enable noise reduction
PREEMPHASIS = True # enable pre-emphasis
DEEMPHASIS = True # enable de-emphasis

# Webcam and recording configuration
CAM_INDEX = 0 # default webcam index
RECORD_DIR = Path("gunshot_detections")
RECORD_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
AUTO_STOP_SEC = 10.0 # auto-stop after 10 seconds of silence
AUDIO_MIN_RMS = 0.01 # minimum RMS to consider audio for detection
AUDIO_CONF_THRESHOLD = 0.6  # confidence threshold for audio detection
AUDIO_COOLDOWN = 1.5 # seconds between detections

# Bounding box parameters
HEAD_X_PAD_MULT = 0.45 
HEAD_Y_PAD_TOP_MULT = 1.1
HEAD_Y_PAD_BOT_MULT = 0.45
HAND_X_PAD_MULT = 1.8
HAND_Y_PAD_MULT = 1.8

# Frame buffer configuration
FPS_EST = 20.0
FRAME_BUFFER_SECONDS = 4.0
FRAME_BUFFER_MAX = int(FPS_EST * FRAME_BUFFER_SECONDS)

# Mediapipe initialization
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose_sol = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_sol = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Helper functions
def day_ordinal(n):
    """Convert day number to ordinal (1st, 2nd, 3rd, etc.)"""
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def nice_filename_label(ts: float, label_text: str):
    """
    Generate filename in exact format: 
    '24th June 4 pm gunshot fired ; type of gun.jpg'
    """
    dt = datetime.fromtimestamp(ts)
    
    # Format components exactly as requested
    day = day_ordinal(dt.day)                    # "24th"
    month = dt.strftime("%B")                    # "June" 
    hour = dt.strftime("%I").lstrip("0") or "12" # "4" (not "04")
    ampm = dt.strftime("%p").lower()             # "pm"
    
    # Construct the exact format you requested
    filename = f"{day} {month} {hour} {ampm} gunshot fired ; {label_text}.jpg"
    
    return filename

# Model Architecture
class ResNet50LSTM(nn.Module):
    def __init__(self, n_classes: int, lstm_hidden: int = 512, lstm_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        import torchvision.models as models
        self.resnet50 = models.resnet50(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden, num_layers=lstm_layers,
                           batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(lstm_hidden * 2, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5), nn.Linear(512, n_classes))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        cnn_features = self.cnn_backbone(x)
        cnn_features = nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1))
        cnn_features = cnn_features.view(batch_size, 2048, -1)
        cnn_features = cnn_features.transpose(1, 2)
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden)

class GunDetectionSystem:
    def __init__(self, model_path, threshold=0.6, recordings_dir="gunshot_detections"):
        self.model_path = model_path
        self.threshold = threshold
        self.recordings_dir = Path(recordings_dir)
        self.model = None
        self.classes = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create recordings directory
        self.recordings_dir.mkdir(exist_ok=True)
        print(f"Detection images will be saved to: {self.recordings_dir}")
        
        # Audio processing state
        self.background_noise = None
        self.noise_floor = 0.001
        
        # Webcam and detection state
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_MAX)
        self.audio_flag = {"trigger": False, "ts": 0.0, "label": "unknown", "conf": 0.0}
        self.last_audio_save = 0.0
        self.saved_any = False
        self.stop_flag = threading.Event()
        self.last_sound_time = 0.0
        
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        print("Loading gun type detection model...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except:
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        self.classes = checkpoint['classes']
        print(f"Loaded {len(self.classes)} classes: {', '.join(self.classes)}")
        
        self.model = ResNet50LSTM(n_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def enhance_mic_audio(self, y, sr=SR):
        """Enhanced microphone audio processing"""
        # Remove DC offset
        y = y - np.mean(y)

        # Normalize amplitude
        y = y / (np.max(np.abs(y)) + 1e-9)

        # Apply de-emphasis
        if DEEMPHASIS:
            deemph_coef = 0.97
            y = np.convolve(y, [1], mode="same")
            for i in range(1, len(y)):
                y[i] = y[i] + deemph_coef * y[i - 1]

        # Apply pre-emphasis
        if PREEMPHASIS:
            y = librosa.effects.preemphasis(y, coef=0.97)

        # Noise reduction
        if NOISE_REDUCTION:
            try:
                y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)
            except Exception as e:
                print(f"[Warning] Noise reduction failed: {e}")

        # RMS normalization
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y *= (TARGET_RMS / (rms + 1e-9))

        return y

    def preprocess_file_audio(self, audio_data):
        """Convert audio to spectrogram for model input"""
        target_length = int(SR * CLIP_SEC)
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        else:
            audio_data = audio_data[:target_length]
        
        mel = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=SR, 
            n_mels=N_MELS, 
            fmax=FMAX, 
            hop_length=512, 
            win_length=1024
        )
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-9)
        
        mel_3channel = np.stack([mel, mel, mel])
        tensor_data = torch.tensor(mel_3channel, dtype=torch.float32).unsqueeze(0)
        
        return tensor_data

    def predict_audio(self, audio_data):
        """Predict gun type from audio data"""
        # Enhanced audio processing
        processed_audio = self.enhance_mic_audio(audio_data)
        
        # Convert to spectrogram
        input_tensor = self.preprocess_file_audio(processed_audio)
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_pred, predicted = torch.max(probabilities, 1)
            
            confidence_pred = confidence_pred.item()
            predicted_class = self.classes[predicted.item()]
            
            return predicted_class, confidence_pred

    def draw_bounding_boxes(self, frame):
        """Draw bounding boxes around head and hands using Mediapipe"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            p_res = pose_sol.process(rgb)
            h_res = hands_sol.process(rgb)
        except Exception:
            return frame

        result_frame = frame.copy()
        h, w, _ = result_frame.shape

        # Draw head bounding box
        if p_res and p_res.pose_landmarks:
            lm = p_res.pose_landmarks.landmark
            idxs = []
            for nm in ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT"]:
                try:
                    idxs.append(getattr(mp_pose.PoseLandmark, nm))
                except Exception:
                    pass
            
            xs = [lm[i].x for i in idxs if hasattr(lm[i], 'x')]
            ys = [lm[i].y for i in idxs if hasattr(lm[i], 'y')]
            
            if xs and ys:
                x_min = int(max(0, min(xs) * w))
                x_max = int(min(w-1, max(xs) * w))
                y_min = int(max(0, min(ys) * h))
                y_max = int(min(h-1, max(ys) * h))
                
                bw = max(1, x_max - x_min)
                bh = max(1, y_max - y_min)
                
                x_pad = int(bw * HEAD_X_PAD_MULT) + 30
                y_pad_top = int(bh * HEAD_Y_PAD_TOP_MULT) + 60
                y_pad_bot = int(bh * HEAD_Y_PAD_BOT_MULT) + 30
                
                x_min = max(0, x_min - x_pad)
                x_max = min(w-1, x_max + x_pad)
                y_min = max(0, y_min - y_pad_top)
                y_max = min(h-1, y_max + y_pad_bot)
                
                cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(result_frame, "HEAD", (x_min, max(0, y_min-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw hand bounding boxes
        if h_res and h_res.multi_hand_landmarks:
            for hand_landmarks in h_res.multi_hand_landmarks:
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                
                x_min = int(max(0, min(xs) * w))
                x_max = int(min(w-1, max(xs) * w))
                y_min = int(max(0, min(ys) * h))
                y_max = int(min(h-1, max(ys) * h))
                
                bw = max(1, x_max - x_min)
                bh = max(1, y_max - y_min)
                
                x_pad = int(bw * HAND_X_PAD_MULT) + 30
                y_pad = int(bh * HAND_Y_PAD_MULT) + 30
                
                x_min = max(0, x_min - x_pad)
                x_max = min(w-1, x_max + x_pad)
                y_min = max(0, y_min - int(y_pad * 0.15))
                y_max = min(h-1, y_max + y_pad)
                
                cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(result_frame, "HAND", (x_min, max(0, y_min-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return result_frame

    def audio_thread_func(self):
        """Audio processing thread"""
        p = pyaudio.PyAudio()
        try:
            # Find the best input device
            best_device = None
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:
                    if best_device is None or dev_info['defaultSampleRate'] == SR:
                        best_device = i
            
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SR,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=best_device
            )
            print(f"Audio recording started (Device: {best_device})")
        except Exception as e:
            print(f"ERROR: Cannot open microphone stream: {e}")
            self.stop_flag.set()
            return

        win_samps = int(CLIP_SEC * SR)
        ring = np.zeros(win_samps, dtype=np.float32)

        try:
            while not self.stop_flag.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Update ring buffer
                if len(arr) < win_samps:
                    ring = np.roll(ring, -len(arr))
                    ring[-len(arr):] = arr
                else:
                    ring = arr[-win_samps:]
                
                rms = float(np.sqrt(np.mean(arr**2)))
                print(f"Audio level: {rms:.4f}", end="\r")

                if rms < AUDIO_MIN_RMS:
                    time.sleep(0.1)
                    continue

                # Predict gun type using the enhanced model
                try:
                    predicted_class, confidence = self.predict_audio(ring)
                    
                    if confidence >= self.threshold:
                        now = time.time()
                        self.audio_flag.update({
                            "trigger": True, 
                            "ts": now, 
                            "label": predicted_class, 
                            "conf": confidence
                        })
                        self.last_sound_time = now
                        print(f"\n🎯 GUNSHOT DETECTED: {predicted_class} (confidence: {confidence:.1%})")
                        
                        # Cooldown period
                        time.sleep(AUDIO_COOLDOWN)
                    else:
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"\nAudio prediction error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"\nAudio thread stopped with error: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except Exception:
                pass
            self.stop_flag.set()

    def add_detection_annotation(self, frame, trigger_ts, label, confidence):
        """Add detection information annotation to the frame"""
        # Add timestamp
        timestamp_str = datetime.fromtimestamp(trigger_ts).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp_str}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection info
        detection_text = f"Gun Type: {label} ({confidence:.1%} confidence)"
        cv2.putText(frame, detection_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add bounding box labels
        cv2.putText(frame, "GREEN: Head detection", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "RED: Hand detection", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def save_detection_image(self, frame, trigger_ts, label, confidence):
        """Save image with your exact requested filename format"""
        if trigger_ts - self.last_audio_save < AUDIO_COOLDOWN:
            return

        # Generate filename in your exact requested format
        filename = nice_filename_label(trigger_ts, label)
        filepath = self.recordings_dir / filename
        
        # Draw bounding boxes and annotations
        annotated_frame = self.draw_bounding_boxes(frame)
        self.add_detection_annotation(annotated_frame, trigger_ts, label, confidence)
        
        try:
            cv2.imwrite(str(filepath), annotated_frame)
            print(f"CAPTURED: '{filename}'")
            self.last_audio_save = trigger_ts
            self.saved_any = True
        except Exception as e:
            print(f"Failed to save image: {e}")

    def run_detection_system(self):
        """Main detection system loop"""
        print("\n" + "="*60)
        print("GUNSHOT DETECTION SYSTEM ACTIVATED")
        print("="*60)
        print("Features:")
        print("• Real-time gun type detection from microphone")
        print("• Automatic webcam capture on detection")
        print("• Bounding boxes for head and hands")
        print(f"• Detection threshold: {self.threshold:.0%}")
        print(f"• File naming: '24th June 4 pm gunshot fired ; type of gun.jpg'")
        print("="*60)
        print("Press 'q' to quit, or wait for auto-stop after silence")
        print("="*60)

        # Start audio thread
        audio_t = threading.Thread(target=self.audio_thread_func, daemon=True)
        audio_t.start()

        # Initialize webcam
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera index {CAM_INDEX}")
            self.stop_flag.set()
            return

        try:
            while not self.stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
                
                # Add frame to buffer with timestamp
                ts = time.time()
                self.frame_buffer.appendleft((ts, frame.copy()))
                
                # Display live preview with bounding boxes
                preview_frame = self.draw_bounding_boxes(frame)
                cv2.imshow("Gun Detection System - Live Preview", preview_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[USER EXIT] 'q' pressed — stopping.")
                    self.stop_flag.set()
                    break
                
                # Handle audio triggers
                if self.audio_flag.get("trigger", False):
                    trigger_ts = self.audio_flag["ts"]
                    label = self.audio_flag["label"]
                    confidence = self.audio_flag["conf"]
                    
                    # Find the closest frame to the detection time
                    chosen_frame = None
                    if self.frame_buffer:
                        best_frame = min(self.frame_buffer, key=lambda x: abs(x[0] - trigger_ts))
                        if abs(best_frame[0] - trigger_ts) <= 1.0:  # Within 1 second
                            chosen_frame = best_frame[1].copy()
                    
                    if chosen_frame is None:
                        chosen_frame = frame.copy()
                    
                    # Save the detection image with your requested filename format
                    self.save_detection_image(chosen_frame, trigger_ts, label, confidence)
                    
                    # Reset trigger
                    self.audio_flag["trigger"] = False
                
                # Auto-stop after period of silence
                now = time.time()
                if self.saved_any and (now - self.last_sound_time) > AUTO_STOP_SEC:
                    print(f"\nNo gunshots detected for {AUTO_STOP_SEC} seconds — stopping automatically.")
                    self.stop_flag.set()
                    break
                    
        except Exception as e:
            print(f"Main loop error: {e}")
            self.stop_flag.set()
        finally:
            # Cleanup
            self.stop_flag.set()
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            try:
                pose_sol.close()
                hands_sol.close()
            except Exception:
                pass
            
            print(f"\nDetection system stopped.")
            print(f"Detection images saved to: {self.recordings_dir.resolve()}")
            if self.saved_any:
                print("Gunshots were successfully detected and captured!")
            else:
                print("No gunshots were detected during this session.")

def main():
    """Main function to run the integrated gun detection system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gunshot Detection System with Webcam Capture')
    parser.add_argument('--model', required=True, help='Path to trained gun type model')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold for detection')
    parser.add_argument('--recordings-dir', default='gunshot_detections', help='Directory to save detection images')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found!")
        return
    
    try:
        detection_system = GunDetectionSystem(
            model_path=args.model,
            threshold=args.threshold,
            recordings_dir=args.recordings_dir
        )
        
        detection_system.run_detection_system()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()