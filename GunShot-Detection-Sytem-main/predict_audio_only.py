#!/usr/bin/env python3
# State-of-the-Art Audio Gunshot Detection System
# Advanced Audio Processing + Visual Detection + Web Interface

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import pyaudio
import librosa
import warnings
import noisereduce as nr
import json
import torch
import torch.nn as nn
import torchvision.models as models

# Force OpenCV to use DirectShow instead of MSMF on Windows
if os.name == 'nt':  # Windows
    # Disable MSMF backend
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    # Boost DirectShow backend priority
    os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1000"
    print("[SYSTEM] Configured OpenCV to prioritize DirectShow backend on Windows")

import cv2
import mediapipe as mp
from scipy import signal
import wave
import re
import traceback
import subprocess
import sys

warnings.filterwarnings('ignore')

# Audio parameters (optimized for gunshot detection)
SR = 22050  # Increased sample rate for better frequency resolution
CLIP_SEC = 2.0
N_MELS = 128
FMAX = 10000  # Extended frequency range
CHUNK = 2048  # Larger chunk for better FFT
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Detection parameters
DETECTION_INTERVAL = 1.0  # Faster detection
ENHANCE_DETECTION = True

# Advanced audio processing parameters
TARGET_RMS = 0.08
NOISE_REDUCTION = True
PREEMPHASIS = True
DEEMPHASIS = True
ADAPTIVE_FILTERING = True
TRANSIENT_ENHANCE = True

# File system setup (always relative to project directory)
BASE_DIR = Path(__file__).resolve().parent
RECORD_DIR = BASE_DIR / "gunshot_detections"
RECORD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = BASE_DIR / "audio_recordings"
AUDIO_DIR.mkdir(exist_ok=True)

# Sensitivity settings (optimized)
AUDIO_MIN_RMS = 0.015
AUDIO_CONF_THRESHOLD = 0.7
AUDIO_COOLDOWN = 2.0

# Shared files for Next.js polling
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)
LATEST_FRAME_PATH = TEMP_DIR / "latest_frame.jpg"
LATEST_EVENT_PATH = TEMP_DIR / "latest_event.json"
SYSTEM_STATUS_PATH = TEMP_DIR / "system_status.json"
DETECTIONS_MANIFEST_PATH = TEMP_DIR / "detections.json"
MAX_DETECTIONS_MANIFEST = 30

def day_ordinal(n):
    """Convert day number to ordinal (1st, 2nd, 3rd, etc.)"""
    n = int(n)
    if 11 <= n <= 13:
        return f"{n}th"
    if n % 10 == 1:
        return f"{n}st"
    elif n % 10 == 2:
        return f"{n}nd"
    elif n % 10 == 3:
        return f"{n}rd"
    else:
        return f"{n}th"

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def sanitize_filename_text(text: str) -> str:
    """Replace filesystem-invalid characters and tidy whitespace."""
    sanitized = INVALID_FILENAME_CHARS.sub('-', text)
    sanitized = sanitized.replace(';', '-').replace(',', '-')
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized.rstrip('. ')


def nice_filename_label(ts: float, label_text: str):
    """Generate descriptive, filesystem-safe filename with timestamp"""
    dt = datetime.fromtimestamp(ts)
    
    day_str = day_ordinal(dt.day)
    month_str = dt.strftime("%B")
    hour_12 = dt.hour % 12
    if hour_12 == 0:
        hour_12 = 12
    hour_str = str(hour_12)
    minute_str = f"{dt.minute:02d}"
    second_str = f"{dt.second:02d}"
    ampm_str = "am" if dt.hour < 12 else "pm"

    timestamp_str = f"{hour_str}-{minute_str}-{second_str}"
    label_safe = sanitize_filename_text(label_text)
    base_name = f"{day_str} {month_str} {timestamp_str} {ampm_str} gunshot fired - {label_safe}"
    safe_name = sanitize_filename_text(base_name)
    return f"{safe_name}.jpg"

# Advanced Model Architecture
class ResNet50LSTM(nn.Module):
    def __init__(self, n_classes: int, lstm_hidden: int = 512, lstm_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        import torchvision.models as models
        self.resnet50 = models.resnet50(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden, num_layers=lstm_layers,
                           batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(lstm_hidden * 2, 1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7), 
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5), 
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        """Forward pass matching the training implementation.

        1) Resize mel-spectrograms to 224x224 for ResNet50.
        2) Extract CNN features via cnn_backbone.
        3) Apply global average pooling.
        4) Reshape to (batch, 1, 2048) and feed through bidirectional LSTM.
        5) Concatenate last forward/backward hidden states and classify.
        """

        batch_size = x.size(0)

        # Resize spectrogram to 224x224 for ResNet50
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # CNN feature extraction with ResNet50
        cnn_features = self.cnn_backbone(x)  # (batch, 2048, H, W)

        # Global average pooling and prepare for LSTM
        cnn_features = nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1))
        cnn_features = cnn_features.view(batch_size, 2048, -1)  # (batch, 2048, 1)
        cnn_features = cnn_features.transpose(1, 2)  # (batch, 1, 2048)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(cnn_features)

        # Use last hidden state from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_size * 2)

        # Classification
        out = self.classifier(hidden)
        return out

class StateOfTheArtGunDetectionSystem:
    def __init__(self, model_path, threshold=0.7, recordings_dir="gunshot_detections"):
        self.model_path = model_path
        self.threshold = threshold
        self.recordings_dir = Path(recordings_dir)
        self.audio_dir = AUDIO_DIR
        self.model = None
        self.classes = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        self.recordings_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        
        # Audio processing state
        self.background_noise = None
        self.audio_quality_history = []
        self.adaptive_threshold = AUDIO_MIN_RMS
        self.noise_profile = None
        
        # Detection state
        self.audio_flag = {"trigger": False, "ts": 0.0, "label": "unknown", "conf": 0.0, "quality": 0.0}
        self.last_audio_save = 0.0
        self.stop_flag = threading.Event()
        self.last_sound_time = 0.0
        self.detection_count = 0
        self.false_positive_count = 0
        
        # MediaPipe solutions
        self.pose_sol = None
        self.hands_sol = None
        self.face_sol = None
        
        # Camera for frame capture - will auto-detect OBSBOT camera
        self.camera = None
        # Start with index 0, will auto-detect and skip OBS Virtual Camera (index 1)
        self.camera_index = 0  # Will be updated by auto-detection
        self.camera_backend_code = None  # Will store the backend code that works
        self.use_ffmpeg = False  # Flag to use FFmpeg backend
        self.ffmpeg_process = None  # FFmpeg subprocess handle
        self.ffmpeg_camera_name = None  # DirectShow camera name for FFmpeg
        
        # Performance monitoring
        self.processing_times = []
        self.audio_quality_scores = []
        
        if os.name == 'nt':  # Only for Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)  # Set console input code page to UTF-8
            kernel32.SetConsoleOutputCP(65001)  # Set console output code page to UTF-8
        
        print("[SYSTEM] Initializing State-of-the-Art Detection System...")
        self.load_model()
        self.init_mediapipe()
        # Initialize camera for direct backend access
        self.init_camera()
        self.write_system_status("initializing")

    def append_detection_manifest(self, entry):
        """Persist detection metadata for frontend dashboards."""
        try:
            if DETECTIONS_MANIFEST_PATH.exists():
                with open(DETECTIONS_MANIFEST_PATH, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if not isinstance(manifest, list):
                    manifest = []
            else:
                manifest = []
        except Exception as e:
            print(f"[WARNING] Failed reading detections manifest: {e}")
            manifest = []

        manifest.append(entry)
        manifest = manifest[-MAX_DETECTIONS_MANIFEST:]

        try:
            with open(DETECTIONS_MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed writing detections manifest: {e}")

        
    def load_model(self):
        """Load the trained model with advanced verification"""
        print("[SYSTEM] Loading advanced gun type detection model...")
        
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # First try with weights_only=True
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"[WARNING] weights_only=True failed: {e}, trying with weights_only=False")
            # If that fails, try with weights_only=False (less secure but more compatible)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Verify checkpoint structure
        required_keys = ['classes', 'state_dict']
        for key in required_keys:
            if key not in checkpoint:
                print(f"[ERROR] Checkpoint missing required key: {key}")
                raise KeyError(f"Checkpoint missing key: {key}")
        
        self.classes = checkpoint['classes']
        print(f"[SYSTEM] Loaded {len(self.classes)} classes: {', '.join(self.classes)}")
        
        # Initialize model with correct number of classes
        self.model = ResNet50LSTM(n_classes=len(self.classes)).to(self.device)
        
        # Load state dict
        try:
            # First try direct load
            self.model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as e:
            print("[WARNING] Direct state dict loading failed, trying with strict=False")
            # If that fails, try with strict=False
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        self.model.eval()
        print("[SYSTEM] Model loaded successfully!")
    
    def init_mediapipe(self):
        """Initialize MediaPipe with optimized settings"""
        print("[SYSTEM] Initializing MediaPipe solutions...")
        try:
            # Optimized pose detection
            self.pose_sol = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,  # Higher complexity for better accuracy
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8,
                smooth_landmarks=True
            )
            
            # Optimized hand detection
            self.hands_sol = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                model_complexity=1
            )
            
            # Face detection for comprehensive analysis
            self.face_sol = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # Higher accuracy model
                min_detection_confidence=0.7
            )
            
            print("[SYSTEM] MediaPipe initialized successfully")
        except Exception as e:
            print(f"[ERROR] MediaPipe initialization failed: {e}")
            raise

    def get_camera_name(self, cap, index):
        """Try to get camera name/backend info"""
        try:
            # On Windows, try DirectShow backend to get device name
            if os.name == 'nt':
                # Try to get backend name
                backend = cap.getBackendName()
                return backend
        except:
            pass
        return None

    def detect_obsbot_camera(self):
        """Auto-detect OBSBOT camera by trying different indices and backends, skipping OBS Virtual Camera (index 1)"""
        print("[SYSTEM] Scanning for OBSBOT camera (skipping OBS Virtual Camera at index 1)...")
        print("[SYSTEM] Checking camera indices: 0, 2, 3, 4, 5, 6, 7...")
        
        # Try indices in order, but skip index 1 (OBS Virtual Camera)
        indices_to_try = [0, 2, 3, 4, 5, 6, 7]  # Skip index 1
        
        found_cameras = []
        
        for idx in indices_to_try:
            # Try both default backend and DirectShow on Windows
            backends_to_try = []
            if os.name == 'nt':  # Windows
                backends_to_try = [
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (None, "Default")
                ]
            else:
                backends_to_try = [(None, "Default")]
            
            for backend_code, backend_name in backends_to_try:
                try:
                    if backend_code is not None:
                        test_cap = cv2.VideoCapture(idx, backend_code)
                    else:
                        test_cap = cv2.VideoCapture(idx)
                    
                    if test_cap.isOpened():
                        # Give camera time to initialize
                        time.sleep(0.3)
                        
                        # Try to read frame multiple times
                        frame_read = False
                        test_frame = None
                        for attempt in range(3):
                            ret, test_frame = test_cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                frame_read = True
                                break
                            time.sleep(0.1)
                        
                        if frame_read:
                            backend_info = self.get_camera_name(test_cap, idx)
                            found_cameras.append((idx, backend_code, backend_name, backend_info))
                            print(f"[SYSTEM] ✓ Camera found at index {idx} using {backend_name} backend (resolution: {test_frame.shape[1]}x{test_frame.shape[0]})")
                            test_cap.release()
                            break  # Found working camera at this index, move to next
                        else:
                            print(f"[SYSTEM] ✗ Camera at index {idx} ({backend_name}) opened but cannot read frames")
                            test_cap.release()
                    else:
                        if backend_name == "Default":  # Only print error for last backend tried
                            print(f"[SYSTEM] ✗ Camera at index {idx} cannot be opened with any backend")
                except Exception as e:
                    if backend_name == "Default":  # Only print error for last backend tried
                        print(f"[SYSTEM] ✗ Error checking camera at index {idx}: {e}")
                    continue
        
        if found_cameras:
            # Use the first found camera (should be OBSBOT if it's the only one besides OBS)
            selected_idx, selected_backend, selected_backend_name, _ = found_cameras[0]
            print(f"[SYSTEM] Selected camera at index {selected_idx} using {selected_backend_name} backend")
            # Store the backend code for later use
            self.camera_backend_code = selected_backend
            return selected_idx
        
        print("[SYSTEM] No working camera found (excluding OBS Virtual Camera)")
        self.camera_backend_code = None
        return None

    def find_ffmpeg_camera_name(self):
        """Find OBSBOT camera name using FFmpeg - correctly handles stderr output"""
        try:
            print("[SYSTEM] Attempting to find camera using FFmpeg...")
            # FFmpeg prints device list to STDERR, not STDOUT
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-list_devices", "true",
                "-f", "dshow",
                "-i", "dummy",
            ]
            
            # Correct: capture stderr only, no capture_output=True
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=5)
            output = result.stderr
            
            obsbot_names = []
            
            for line in output.splitlines():
                # Example line: '  "OBSBOT Tiny 2 Lite StreamCamera"'
                if "OBSBOT" in line.upper() or "TINY" in line.upper() or "STREAMCAMERA" in line.upper():
                    # Extract quoted name
                    import re
                    quoted_names = re.findall(r'"([^"]+)"', line)
                    if quoted_names:
                        name = quoted_names[0]
                        obsbot_names.append(name)
                        print(f"[SYSTEM] Found OBSBOT camera via FFmpeg: {name}")
            
            if obsbot_names:
                return obsbot_names[0]  # Return first match
            
            # If OBSBOT not found, list all cameras for debugging
            print("[SYSTEM] OBSBOT not found in FFmpeg list. Available cameras:")
            in_video_section = False
            for line in output.splitlines():
                if 'DirectShow video devices' in line or 'video devices' in line.lower():
                    in_video_section = True
                    continue
                if in_video_section and '"' in line:
                    print(f"  {line.strip()}")
            
            return None
        except FileNotFoundError:
            print("[SYSTEM] FFmpeg not found. Install FFmpeg to use this backend.")
            return None
        except Exception as e:
            print(f"[SYSTEM] Error finding camera with FFmpeg: {e}")
            return None

    def init_ffmpeg_camera(self, camera_name, width=1280, height=720, fps=30):
        """Initialize camera using FFmpeg backend with OBSBOT Tiny 2 Lite StreamCamera settings"""
        try:
            print(f"[SYSTEM] Initializing FFmpeg camera: {camera_name}")
            print(f"[SYSTEM] Using OBSBOT settings: {width}x{height} MJPEG @ {fps} fps")
            print(f"[SYSTEM] FFmpeg will decode to raw BGR frames: {width}x{height}x3 = {width * height * 3} bytes per frame")

            # FFmpeg command:
            # - Input: DirectShow camera with MJPEG format
            # - Output: raw BGR24 frames to stdout
            cmd = [
                "ffmpeg",
                "-loglevel", "error",              # only show real errors
                "-f", "dshow",
                "-video_size", f"{width}x{height}",
                "-framerate", str(fps),
                "-i", f"video={camera_name}",
                "-pix_fmt", "bgr24",              # decode MJPEG → BGR24
                "-vcodec", "rawvideo",            # output raw video
                "-an",                             # no audio
                "-sn",                             # no subtitles
                "-f", "rawvideo",
                "-"                                # write to stdout
            ]

            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,           # we'll read errors once if it fails
                bufsize=0                         # unbuffered, let OS handle
            )

            self.ffmpeg_camera_name = camera_name
            self.ffmpeg_width = width
            self.ffmpeg_height = height

            # Give FFmpeg a moment to start
            time.sleep(0.5)

            # Try to grab one test frame
            test_frame = self.capture_ffmpeg_frame()
            if test_frame is not None:
                print(f"[SYSTEM] FFmpeg camera initialized successfully: {test_frame.shape}")
                return True
            else:
                print("[ERROR] FFmpeg camera opened but cannot read frames")
                # Dump a bit of stderr so we see the real error
                try:
                    stderr_output = self.ffmpeg_process.stderr.read(4000).decode("utf-8", errors="ignore")
                    if stderr_output:
                        print(f"[ERROR] FFmpeg stderr:\n{stderr_output}")
                except Exception:
                    pass
                self.cleanup_ffmpeg()
                return False

        except Exception as e:
            print(f"[ERROR] FFmpeg camera initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup_ffmpeg()
            return False

    def capture_ffmpeg_frame(self):
        """Capture a frame using FFmpeg - reads exactly width * height * 3 bytes"""
        if self.ffmpeg_process is None:
            return None

        try:
            frame_size = self.ffmpeg_width * self.ffmpeg_height * 3

            # Read until we either get a full frame or hit EOF
            raw = b""
            while len(raw) < frame_size:
                chunk = self.ffmpeg_process.stdout.read(frame_size - len(raw))
                if not chunk:
                    break  # EOF or ffmpeg died
                raw += chunk

            if len(raw) != frame_size:
                print(
                    f"[WARNING] FFmpeg read incomplete frame: "
                    f"got {len(raw)} bytes, expected {frame_size} bytes"
                )
                return None

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.ffmpeg_height, self.ffmpeg_width, 3)
            )
            return frame

        except Exception as e:
            print(f"[ERROR] FFmpeg frame capture error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup_ffmpeg(self):
        """Cleanup FFmpeg process"""
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
            self.ffmpeg_process = None
            self.ffmpeg_camera_name = None

    def init_camera(self):
        """
        Simple camera init for Windows + OBSBOT.
        Forces DirectShow backend on index 0 and skips FFmpeg.
        """
        import cv2
        import time

        print("[SYSTEM] Using simple OpenCV camera init (DirectShow index 0)")

        # Clean up any previous camera
        try:
            if getattr(self, "camera", None) is not None:
                self.camera.release()
        except Exception:
            pass

        # Make sure we are NOT in FFmpeg mode
        self.use_ffmpeg = False
        self.ffmpeg_process = None
        self.ffmpeg_camera_name = None

        # Open OBSBOT on index 0 with DirectShow
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Give the camera a moment to warm up
        time.sleep(0.5)

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] Simple camera init failed: could not grab frame from index 0")
            cap.release()
            self.camera = None
            self.camera_index = None
            return None

        self.camera = cap
        self.camera_index = 0

        h, w = frame.shape[:2]
        print("[SYSTEM] Camera initialized successfully at index 0")
        print(f"[SYSTEM] Camera resolution: {w}x{h}")
        print("[SYSTEM] Camera backend: DSHOW (forced)")

        return True

    def capture_camera_frame(self):
        """Capture a frame from the camera with retry logic"""
        # Use FFmpeg if enabled
        if self.use_ffmpeg:
            return self.capture_ffmpeg_frame()
        
        if self.camera is None:
            print("[ERROR] Camera is None - cannot capture frame")
            return None
        
        # Check if camera is still opened, if not try to reopen with the same backend
        if not self.camera.isOpened():
            print("[WARNING] Camera is not opened - attempting to reopen...")
            try:
                if self.camera_backend_code is not None:
                    self.camera = cv2.VideoCapture(self.camera_index, self.camera_backend_code)
                elif os.name == 'nt':  # Windows
                    self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                else:
                    self.camera = cv2.VideoCapture(self.camera_index)
                
                if not self.camera.isOpened():
                    print("[ERROR] Failed to reopen camera")
                    return None
                time.sleep(0.5)  # Give camera time to initialize
            except Exception as e:
                print(f"[ERROR] Exception while reopening camera: {e}")
                return None
        
        # Try to capture frame with retries
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                ret, frame = self.camera.read()
                
                if ret and frame is not None and frame.size > 0:
                    print(f"[CAMERA] Successfully captured frame: {frame.shape}")
                    return frame
                else:
                    if attempt < max_attempts - 1:
                        print(f"[WARNING] Frame capture attempt {attempt + 1} failed, retrying...")
                        time.sleep(0.1)
                    else:
                        print(f"[ERROR] Failed to capture frame after {max_attempts} attempts")
                        print(f"[ERROR] ret={ret}, frame is None={frame is None}, frame.size={frame.size if frame is not None else 'N/A'}")
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"[WARNING] Exception on attempt {attempt + 1}: {e}, retrying...")
                    time.sleep(0.1)
                else:
                    print(f"[ERROR] Exception while capturing camera frame after {max_attempts} attempts: {e}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        return None

    def analyze_audio_characteristics(self, audio_data):
        """Comprehensive audio analysis for quality assessment"""
        features = {}
        
        # Time-domain features
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        features['peak'] = np.max(np.abs(audio_data))
        features['crest_factor'] = features['peak'] / (features['rms'] + 1e-8)
        features['dynamic_range'] = 20 * np.log10(features['peak'] / (features['rms'] + 1e-8))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=SR)[0]
        features['spectral_centroid'] = np.mean(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=SR, roll_percent=0.85)[0]
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=SR)[0]
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        # Temporal features
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        features['onset_strength'] = np.mean(librosa.onset.onset_strength(y=audio_data, sr=SR))
        
        # MFCC features (first 5 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=SR, n_mfcc=13)
        for i in range(5):
            features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
        
        return features

    def calculate_audio_quality_score(self, features):
        """Calculate confidence that audio contains gunshot-like characteristics"""
        score = 0.0
        max_score = 0.0
        
        # RMS level (gunshots are typically loud)
        if 0.02 <= features['rms'] <= 0.5:
            score += 0.25
            max_score += 0.25
        
        # Spectral centroid (gunshots typically 800Hz-8kHz)
        if 800 <= features['spectral_centroid'] <= 8000:
            score += 0.20
            max_score += 0.20
        
        # Onset strength (gunshots have strong attacks)
        if features['onset_strength'] > 0.3:
            score += 0.15
            max_score += 0.15
        
        # Crest factor (gunshots have high peak-to-RMS ratio)
        if features['crest_factor'] > 4.0:
            score += 0.15
            max_score += 0.15
        
        # Dynamic range
        if features['dynamic_range'] > 15:
            score += 0.15
            max_score += 0.15
        
        # Spectral rolloff (gunshots have energy in higher frequencies)
        if features['spectral_rolloff'] > 3000:
            score += 0.10
            max_score += 0.10
        
        return score / max_score if max_score > 0 else 0.0

    def advanced_audio_enhancement(self, y, sr=SR):
        """State-of-the-art audio enhancement pipeline"""
        start_time = time.time()
        
        # 1. DC offset removal and basic normalization
        y = y - np.mean(y)
        y = y / (np.max(np.abs(y)) + 1e-9)
        
        # 2. De-emphasis filter (if enabled)
        if DEEMPHASIS:
            deemph_coef = 0.97
            y_deemph = np.zeros_like(y)
            y_deemph[0] = y[0]
            for i in range(1, len(y)):
                y_deemph[i] = y[i] + deemph_coef * y_deemph[i-1]
            y = y_deemph
        
        # 3. Pre-emphasis for high-frequency enhancement
        if PREEMPHASIS:
            y = librosa.effects.preemphasis(y, coef=0.97)
        
        # 4. Advanced noise reduction
        if NOISE_REDUCTION:
            try:
                if self.noise_profile is None:
                    # Estimate noise from first 100ms
                    noise_sample = y[:int(0.1 * sr)] if len(y) > int(0.1 * sr) else y
                    self.noise_profile = nr.spectral_gating(y, sr, prop_decrease=0.8)
                
                y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.85, stationary=False)
            except Exception as e:
                print(f"[WARNING] Noise reduction failed: {e}")
        
        # 5. Adaptive high-pass filtering
        if ADAPTIVE_FILTERING:
            features = self.analyze_audio_characteristics(y)
            if features['spectral_centroid'] > 2500:
                cutoff = 40.0  # Lower cutoff for high-frequency content
            else:
                cutoff = 60.0
            
            nyquist = sr / 2
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
            y = signal.filtfilt(b, a, y)
        
        # 6. Transient enhancement
        if TRANSIENT_ENHANCE:
            # High-pass for transient detection
            b_hf, a_hf = signal.butter(4, 2000/(sr/2), btype='high')
            high_freq = signal.filtfilt(b_hf, a_hf, y)
            
            # Detect transients
            transient_energy = np.convolve(high_freq**2, np.ones(50)/50, mode='same')
            transient_threshold = np.percentile(transient_energy, 75)
            transient_mask = transient_energy > transient_threshold
            
            # Apply emphasis around transients
            kernel = np.exp(-np.linspace(-2, 2, 150)**2)
            emphasis_weights = np.convolve(transient_mask.astype(float), kernel, mode='same')
            emphasis_weights = np.clip(emphasis_weights * 0.3 + 1, 1, 1.3)
            y = y * emphasis_weights
        
        # 7. Spectral balancing
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Gunshot-specific spectral shaping
        weights = np.ones_like(freqs)
        mid_band = (freqs >= 500) & (freqs <= 5000)
        weights[mid_band] = 1.4  # Boost mid frequencies
        
        high_band = freqs > 7000
        weights[high_band] = 0.8  # Reduce very high frequencies
        
        balanced_magnitude = magnitude * weights[:, np.newaxis]
        balanced_stft = balanced_magnitude * np.exp(1j * phase)
        y = librosa.istft(balanced_stft, hop_length=512)
        
        # 8. Final RMS normalization with soft limiting
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            gain = TARGET_RMS / (rms + 1e-9)
            gain = np.clip(gain, 0.1, 10.0)  # Reasonable gain limits
            y = y * gain
            
            # Soft clipping to prevent distortion
            clip_threshold = 0.9
            peaks = np.abs(y) > clip_threshold
            if np.any(peaks):
                y[peaks] = np.sign(y[peaks]) * clip_threshold
        
        # Update processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return y

    def preprocess_audio_for_model(self, audio_data):
        """Convert enhanced audio to model input format"""
        target_length = int(SR * CLIP_SEC)
        
        # Ensure correct length
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        else:
            audio_data = audio_data[:target_length]
        
        # Generate mel spectrogram with optimized parameters
        mel = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=SR, 
            n_mels=N_MELS, 
            fmax=FMAX,
            hop_length=256,  # Higher resolution
            win_length=1024,
            n_fft=2048
        )
        
        # Convert to dB scale with noise floor
        mel_db = librosa.power_to_db(mel, ref=np.max, amin=1e-6)
        
        # Advanced normalization
        mel_normalized = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
        
        # Create 3-channel input (compatible with ResNet)
        mel_3channel = np.stack([mel_normalized, mel_normalized, mel_normalized])
        tensor_data = torch.tensor(mel_3channel, dtype=torch.float32).unsqueeze(0)
        
        return tensor_data

    def predict_audio(self, audio_data):
        """Advanced audio prediction with quality assessment"""
        # Analyze original audio
        original_features = self.analyze_audio_characteristics(audio_data)
        quality_score = self.calculate_audio_quality_score(original_features)
        
        print(f"[AUDIO] Audio Analysis: RMS={original_features['rms']:.4f}, "
              f"Centroid={original_features['spectral_centroid']:.0f}Hz, "
              f"Quality={quality_score:.2f}")
        
        # Only process if audio quality is sufficient
        if quality_score < 0.3 and ENHANCE_DETECTION:
            return "low_quality", 0.0, quality_score, original_features
        
        # Apply advanced enhancement
        enhanced_audio = self.advanced_audio_enhancement(audio_data)
        
        # Extract features from enhanced audio
        enhanced_features = self.analyze_audio_characteristics(enhanced_audio)
        enhanced_quality = self.calculate_audio_quality_score(enhanced_features)
        
        # Convert to model input
        input_tensor = self.preprocess_audio_for_model(enhanced_audio)
        
        # Model prediction
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_pred, predicted = torch.max(probabilities, 1)
            
            confidence_pred = confidence_pred.item()
            predicted_class = self.classes[predicted.item()]
            
            # Adjust confidence based on audio quality
            adjusted_confidence = confidence_pred * min(enhanced_quality, 1.0)
            
            return predicted_class, adjusted_confidence, enhanced_quality, enhanced_features

    def save_audio_recording(self, audio_data, trigger_ts, label, confidence, quality):
        """Save audio recording for later analysis"""
        current_time = time.time()
        if current_time - self.last_audio_save < 1.0:  # Minimum 1 second between saves
            return
        
        try:
            # Convert to int16 for WAV file
            audio_data_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create descriptive filename
            timestamp = datetime.fromtimestamp(trigger_ts).strftime("%Y%m%d_%H%M%S")
            safe_class = "".join(c if c.isalnum() else "_" for c in label)
            confidence_pct = int(confidence * 100)
            quality_pct = int(quality * 100)
            
            filename = f"gunshot_{timestamp}_{safe_class}_{confidence_pct}pc_{quality_pct}q.wav"
            filepath = self.audio_dir / filename
            
            # Save as WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SR)
                wav_file.writeframes(audio_data_int16.tobytes())
            
            print(f"[AUDIO] Audio saved: {filename}")
            
        except Exception as e:
            print(f"[ERROR] Error saving audio: {e}")

    def add_detection_overlay(self, frame, trigger_ts, label, confidence, quality, features):
        """Add detection information overlay to a camera frame"""
        h, w, _ = frame.shape
        timestamp_str = datetime.fromtimestamp(trigger_ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Main detection banner
        status_color = (0, 0, 255) if confidence >= self.threshold else (0, 165, 255)  # Red or Orange
        status_text = "HIGH CONFIDENCE DETECTION" if confidence >= self.threshold else "TENTATIVE DETECTION"
        
        # Add semi-transparent background for text (SMALLER BOX)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 240), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "GUNSHOT DETECTION SYSTEM", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, status_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Detection details
        cv2.putText(frame, f"Gun Type: {label}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Audio Quality: {quality:.1%}", (20, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Timestamp: {timestamp_str}", (20, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Audio analytics (condensed)
        cv2.putText(frame, "Audio Analytics:", (20, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        rms = features.get('rms', 0.0)
        spectral_centroid = features.get('spectral_centroid', 0.0)
        onset_strength = features.get('onset_strength', 0.0)
        
        cv2.putText(frame, f"RMS: {rms:.4f}", (20, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        cv2.putText(frame, f"Spectral Centroid: {spectral_centroid:.0f} Hz", (20, 225), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        
        return frame

    def draw_advanced_bounding_boxes(self, frame):
        """Advanced bounding box detection with confidence scores"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame = frame.copy()
            h, w, _ = result_frame.shape
            
            detection_info = []
            
            # Pose detection for head and body
            if self.pose_sol:
                pose_results = self.pose_sol.process(rgb)
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    # Head detection
                    head_indices = [
                        mp.solutions.pose.PoseLandmark.NOSE,
                        mp.solutions.pose.PoseLandmark.LEFT_EYE, mp.solutions.pose.PoseLandmark.RIGHT_EYE,
                        mp.solutions.pose.PoseLandmark.LEFT_EAR, mp.solutions.pose.PoseLandmark.RIGHT_EAR,
                        mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER, mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER,
                        mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER, mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER,
                        mp.solutions.pose.PoseLandmark.MOUTH_LEFT, mp.solutions.pose.PoseLandmark.MOUTH_RIGHT
                    ]
                    
                    xs, ys = [], []
                    for idx in head_indices:
                        landmark = landmarks[idx]
                        if landmark.visibility > 0.6:
                            xs.append(landmark.x)
                            ys.append(landmark.y)
                    
                    if xs and ys:
                        x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                        y_min, y_max = int(min(ys) * h), int(max(ys) * h)
                        
                        # Adaptive padding
                        padding_x = int((x_max - x_min) * 0.4)
                        padding_y = int((y_max - y_min) * 0.6)
                        
                        x_min = max(0, x_min - padding_x)
                        x_max = min(w, x_max + padding_x)
                        y_min = max(0, y_min - padding_y)
                        y_max = min(h, y_max + padding_y)
                        
                        # Draw head bounding box
                        cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                        cv2.putText(result_frame, "HEAD", (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        detection_info.append(f"Head: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            # Hand detection
            if self.hands_sol:
                hand_results = self.hands_sol.process(rgb)
                if hand_results.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]
                        
                        # Get raw hand coordinates
                        hand_x_min, hand_x_max = int(min(xs) * w), int(max(xs) * w)
                        hand_y_min, hand_y_max = int(min(ys) * h), int(max(ys) * h)
                        
                        # Draw GUN bounding box FIRST (PINK) - larger outer box
                        gun_padding = 60
                        gun_x_min = max(0, hand_x_min - gun_padding)
                        gun_x_max = min(w, hand_x_max + gun_padding)
                        gun_y_min = max(0, hand_y_min - gun_padding)
                        gun_y_max = min(h, hand_y_max + gun_padding)
                        
                        # Pink color in BGR format: (180, 105, 255) = Bright Pink/Magenta
                        cv2.rectangle(result_frame, (gun_x_min, gun_y_min), (gun_x_max, gun_y_max), (180, 105, 255), 4)
                        cv2.putText(result_frame, "GUN", (gun_x_min, gun_y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 105, 255), 3)
                        
                        # Calculate hand box with smaller padding
                        hand_padding = 25
                        x_min = max(0, hand_x_min - hand_padding)
                        x_max = min(w, hand_x_max + hand_padding)
                        y_min = max(0, hand_y_min - hand_padding)
                        y_max = min(h, hand_y_max + hand_padding)
                        
                        # Draw hand bounding box (RED) - inner box
                        cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                        cv2.putText(result_frame, f"HAND {i+1}", (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        detection_info.append(f"Gun {i+1}: [{gun_x_min}, {gun_y_min}, {gun_x_max}, {gun_y_max}]")
                        detection_info.append(f"Hand {i+1}: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            # Face detection
            if self.face_sol:
                face_results = self.face_sol.process(rgb)
                if face_results.detections:
                    for i, detection in enumerate(face_results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        x_min = int(bbox.xmin * w)
                        y_min = int(bbox.ymin * h)
                        x_max = int((bbox.xmin + bbox.width) * w)
                        y_max = int((bbox.ymin + bbox.height) * h)
                        
                        # Add padding
                        padding = 15
                        x_min = max(0, x_min - padding)
                        x_max = min(w, x_max + padding)
                        y_min = max(0, y_min - padding)
                        y_max = min(h, y_max + padding)
                        
                        # Draw face bounding box
                        cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.putText(result_frame, "FACE", (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        detection_info.append(f"Face {i+1}: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            if detection_info:
                print(f"[DETECTION] Detections: {', '.join(detection_info)}")
            else:
                print("[DETECTION] No detections found")
                
            return result_frame
            
        except Exception as e:
            print(f"[ERROR] Bounding box error: {e}")
            return frame

    def create_advanced_detection_frame(self, trigger_ts, label, confidence, quality, features):
        """Create sophisticated detection frame with analytics"""
        # Create professional-looking frame
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw detailed person silhouette
        self.draw_detailed_person(frame)
        
        # Apply bounding boxes
        frame_with_boxes = self.draw_advanced_bounding_boxes(frame)
        
        # Add comprehensive information overlay
        timestamp_str = datetime.fromtimestamp(trigger_ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Main detection banner
        status_color = (0, 0, 255) if confidence >= self.threshold else (0, 165, 255)  # Red or Orange
        status_text = "HIGH CONFIDENCE DETECTION" if confidence >= self.threshold else "TENTATIVE DETECTION"
        
        cv2.putText(frame_with_boxes, "GUNSHOT DETECTION SYSTEM", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
        cv2.putText(frame_with_boxes, status_text, (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # Detection details
        cv2.putText(frame_with_boxes, f"Gun Type: {label}", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame_with_boxes, f"Confidence: {confidence:.1%}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame_with_boxes, f"Audio Quality: {quality:.1%}", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame_with_boxes, f"Timestamp: {timestamp_str}", (50, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Audio analytics with safe dictionary access
        cv2.putText(frame_with_boxes, "Audio Analytics:", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        rms = features.get('rms', 0.0)
        spectral_centroid = features.get('spectral_centroid', 0.0)
        onset_strength = features.get('onset_strength', 0.0)
        
        cv2.putText(frame_with_boxes, f"RMS: {rms:.4f}", (50, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        cv2.putText(frame_with_boxes, f"Spectral Centroid: {spectral_centroid:.0f} Hz", (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        cv2.putText(frame_with_boxes, f"Onset Strength: {onset_strength:.3f}", (50, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        # System performance
        avg_processing = np.mean(self.processing_times) if self.processing_times else 0
        cv2.putText(frame_with_boxes, f"Avg Processing: {avg_processing*1000:.1f}ms", (50, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        cv2.putText(frame_with_boxes, f"Total Detections: {self.detection_count}", (50, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        
        return frame_with_boxes

    def draw_detailed_person(self, frame):
        """Draw detailed person silhouette for better detection"""
        h, w, _ = frame.shape
        
        # Colors
        skin_color = (200, 180, 180)
        body_color = (160, 160, 160)
        outline_color = (0, 0, 0)
        
        # Center coordinates
        center_x, center_y = w // 2, h // 2
        
        # Head (larger for better detection)
        head_radius = 80
        head_center = (center_x, center_y - 180)
        cv2.circle(frame, head_center, head_radius, skin_color, -1)
        cv2.circle(frame, head_center, head_radius, outline_color, 2)
        
        # Body
        body_top = (head_center[0], head_center[1] + head_radius)
        body_bottom = (body_top[0], body_top[1] + 220)
        body_width = 120
        cv2.rectangle(frame, 
                     (body_top[0] - body_width//2, body_top[1]),
                     (body_top[0] + body_width//2, body_bottom[1]),
                     body_color, -1)
        cv2.rectangle(frame, 
                     (body_top[0] - body_width//2, body_top[1]),
                     (body_top[0] + body_width//2, body_bottom[1]),
                     outline_color, 2)
        
        # Arms (extended for hand detection)
        arm_length = 140
        arm_thickness = 40
        # Left arm
        cv2.rectangle(frame,
                     (body_top[0] - body_width//2 - arm_length, body_top[1] + 50),
                     (body_top[0] - body_width//2, body_top[1] + 50 + arm_thickness),
                     skin_color, -1)
        # Left hand
        left_hand_center = (body_top[0] - body_width//2 - arm_length, body_top[1] + 50 + arm_thickness//2)
        cv2.circle(frame, left_hand_center, 30, skin_color, -1)
        cv2.circle(frame, left_hand_center, 30, outline_color, 2)
        
        # Right arm
        cv2.rectangle(frame,
                     (body_top[0] + body_width//2, body_top[1] + 50),
                     (body_top[0] + body_width//2 + arm_length, body_top[1] + 50 + arm_thickness),
                     skin_color, -1)
        # Right hand
        right_hand_center = (body_top[0] + body_width//2 + arm_length, body_top[1] + 50 + arm_thickness//2)
        cv2.circle(frame, right_hand_center, 30, skin_color, -1)
        cv2.circle(frame, right_hand_center, 30, outline_color, 2)
        
        # Legs
        leg_width = 30
        leg_spacing = 60
        cv2.rectangle(frame,
                     (body_top[0] - leg_spacing//2 - leg_width//2, body_bottom[1]),
                     (body_top[0] - leg_spacing//2 + leg_width//2, body_bottom[1] + 160),
                     body_color, -1)
        cv2.rectangle(frame,
                     (body_top[0] + leg_spacing//2 - leg_width//2, body_bottom[1]),
                     (body_top[0] + leg_spacing//2 + leg_width//2, body_bottom[1] + 160),
                     body_color, -1)
        
        # Facial features (improves head detection)
        cv2.circle(frame, (head_center[0] - 25, head_center[1] - 15), 10, outline_color, -1)  # Left eye
        cv2.circle(frame, (head_center[0] + 25, head_center[1] - 15), 10, outline_color, -1)  # Right eye
        cv2.ellipse(frame, (head_center[0], head_center[1] + 20), (35, 15), 0, 0, 180, outline_color, 3)  # Mouth

    def write_shared_frame(self, frame):
        """Write frame for frontend consumption"""
        if frame is not None:
            try:
                cv2.imwrite(str(LATEST_FRAME_PATH), frame)
            except Exception as e:
                print(f"[ERROR] Frame write failed: {e}")

    def write_shared_event(self, trigger_ts, label, confidence, quality, features):
        """Write comprehensive detection event"""
        event = {
            "timestamp": trigger_ts,
            "datetime": datetime.fromtimestamp(trigger_ts).isoformat(),
            "label": label,
            "confidence": confidence,
            "quality": quality,
            "features": features,
            "filename": nice_filename_label(trigger_ts, label),
            "detection_id": self.detection_count,
            "system_status": "active"
        }
        try:
            with open(LATEST_EVENT_PATH, "w") as f:
                json.dump(event, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Event write failed: {e}")

    def write_system_status(self, status, message=""):
        """Write system status for monitoring"""
        status_data = {
            "status": status,
            "message": message,
            "timestamp": time.time(),
            "detection_count": self.detection_count,
            "false_positives": self.false_positive_count,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "audio_quality_avg": np.mean(self.audio_quality_scores) if self.audio_quality_scores else 0
        }
        try:
            with open(SYSTEM_STATUS_PATH, "w") as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Status write failed: {e}")

    def save_detection(self, trigger_ts, label, confidence, quality, features, audio_data):
        """Complete detection saving procedure"""
        if trigger_ts - self.last_audio_save < AUDIO_COOLDOWN:
            return
        
        print(f"[DETECTION] GUNSHOT DETECTED - Saving complete detection package...")
        
        # Generate filename
        filename = nice_filename_label(trigger_ts, label)
        filepath = TEMP_DIR / filename
        
        # Create detection frame
        detection_frame = self.create_advanced_detection_frame(trigger_ts, label, confidence, quality, features)
        
        saved_path = None

        # Save image in temp so frontend can access it directly
        try:
            cv2.imwrite(str(filepath), detection_frame)
            print(f"[IMAGE] Image saved: {filename}")
            saved_path = str(filepath)
        except Exception as e:
            print(f"[ERROR] Image save failed: {e}")
            saved_path = None
        
        # Save audio
        self.save_audio_recording(audio_data, trigger_ts, label, confidence, quality)
        
        # Update shared files
        self.write_shared_frame(detection_frame)
        self.write_shared_event(trigger_ts, label, confidence, quality, features)
        
        # Update state
        self.last_audio_save = trigger_ts
        self.detection_count += 1
        
        print(f"[SYSTEM] Detection #{self.detection_count} saved: {label} ({confidence:.1%} confidence)")

    def save_detection_from_frame(self, trigger_ts, label, confidence, quality, features, frame, lat=None, lng=None, accuracy=None):
        """Save a detection using an externally provided frame (e.g. from webcam).

        This avoids creating a synthetic frame and does not attempt to save
        audio, making it suitable for the WebSocket-based frontend that
        already sends captured video frames.
        
        Args:
            trigger_ts: Detection timestamp
            label: Weapon type label
            confidence: Detection confidence (0-1)
            quality: Detection quality score
            features: Additional feature dictionary
            frame: Camera frame to save
            lat: GPS latitude (optional)
            lng: GPS longitude (optional)
            accuracy: GPS accuracy in meters (optional)
        """
        
        location_info = f", location=({lat}, {lng})" if lat and lng else ""
        print(f"[SAVE] save_detection_from_frame called: trigger_ts={trigger_ts}, label={label}, confidence={confidence}{location_info}")
        print(f"[SAVE] Frame shape: {frame.shape if frame is not None else 'None'}")
        print(f"[SAVE] Last audio save: {self.last_audio_save}, cooldown: {AUDIO_COOLDOWN}")
        print(f"[SAVE] Time since last save: {trigger_ts - self.last_audio_save:.2f}s")

        if trigger_ts - self.last_audio_save < AUDIO_COOLDOWN:
            print(f"[SAVE] Skipping save due to cooldown (need {AUDIO_COOLDOWN}s between saves)")
            return None

        print(f"[DETECTION] GUNSHOT DETECTED - Saving complete detection package (from frame)...")
        print(f"[DETECTION] Frame received from frontend: shape {frame.shape if frame is not None else 'None'}")
        
        if frame is None:
            print("[ERROR] Frame is None, cannot save")
            return None

        # Generate filename
        filename = nice_filename_label(trigger_ts, label)
        filepath = TEMP_DIR / filename
        
        print(f"[SAVE] Saving to: {filepath}")
        print(f"[SAVE] TEMP_DIR: {TEMP_DIR}")
        print(f"[SAVE] TEMP_DIR exists: {TEMP_DIR.exists()}")

        # Add detection overlay to the real camera frame
        try:
            detection_frame = self.add_detection_overlay(
                frame.copy(), trigger_ts, label, confidence, quality, features
            )
            print(f"[SAVE] Added detection overlay, frame shape: {detection_frame.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to add overlay: {e}")
            import traceback
            traceback.print_exc()
            detection_frame = frame  # Use frame without overlay if overlay fails
        
        saved_path = None

        # Save image
        try:
            success = cv2.imwrite(str(filepath), detection_frame)
            if success:
                print(f"[IMAGE] Image saved successfully: {filepath}")
                print(f"[IMAGE] File exists: {filepath.exists()}")
                print(f"[IMAGE] File size: {filepath.stat().st_size if filepath.exists() else 0} bytes")
                saved_path = str(filepath)
            else:
                print(f"[ERROR] cv2.imwrite returned False - file may not have been saved")
        except Exception as e:
            print(f"[ERROR] Image save failed: {e}")
            import traceback
            traceback.print_exc()

        # Update shared files for frontend polling - use the processed frame with overlay
        try:
            self.write_shared_frame(detection_frame)
            print(f"[SAVE] Updated latest_frame.jpg")
        except Exception as e:
            print(f"[ERROR] Failed to write shared frame: {e}")
            
        try:
            self.write_shared_event(trigger_ts, label, confidence, quality, features)
            print(f"[SAVE] Updated latest_event.json")
        except Exception as e:
            print(f"[ERROR] Failed to write shared event: {e}")

        # Update state
        self.last_audio_save = trigger_ts
        self.detection_count += 1

        print(f"[SYSTEM] Detection #{self.detection_count} saved: {label} ({confidence:.1%} confidence)")

        if saved_path:
            try:
                detection_data = {
                    "timestamp": trigger_ts,
                    "label": label,
                    "confidence": confidence,
                    "quality": quality,
                    "filename": filename,
                    "path": saved_path,
                    "detection_id": self.detection_count,
                    "source": "websocket",
                }
                
                # Add location data if available
                if lat is not None and lng is not None:
                    detection_data["lat"] = lat
                    detection_data["lng"] = lng
                    if accuracy is not None:
                        detection_data["accuracy"] = accuracy
                
                self.append_detection_manifest(detection_data)
                print(f"[SAVE] Added to detection manifest (with location: {lat}, {lng})" if lat and lng else "[SAVE] Added to detection manifest")
            except Exception as e:
                print(f"[ERROR] Failed to append to manifest: {e}")

        return saved_path

    def audio_thread_func(self):
        """Advanced real-time audio processing thread"""
        print("[SYSTEM] Starting advanced audio processing thread...")
        
        # Initialize audio system
        p = pyaudio.PyAudio()
        
        # Find best audio device
        best_device = None
        print("[AUDIO] Available audio devices:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                print(f"  {i}: {dev_info['name']} (SR: {dev_info['defaultSampleRate']})")
                if best_device is None or dev_info['defaultSampleRate'] == SR:
                    best_device = i
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SR,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=best_device
            )
            print(f"[AUDIO] Audio recording started (Device {best_device}, SR: {SR})")
        except Exception as e:
            print(f"[ERROR] Cannot open audio stream: {e}")
            self.stop_flag.set()
            return

        # Audio buffer setup
        win_samps = int(SR * CLIP_SEC)
        audio_buffer = np.zeros(win_samps, dtype=np.float32)
        buffer_pos = 0

        try:
            last_detection_time = 0.0
            consecutive_low_quality = 0
            
            # Initial frame - only write if no frame exists yet
            # Don't overwrite real frames from frontend
            if not LATEST_FRAME_PATH.exists():
                initial_frame = self.create_advanced_detection_frame(
                    time.time(), "Waiting for detection...", 0.0, 0.0, {}
                )
                self.write_shared_frame(initial_frame)
            self.write_system_status("active", "Monitoring for gunshots...")
            
            while not self.stop_flag.is_set():
                # Read audio data
                data = stream.read(CHUNK, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Update buffer
                if buffer_pos + len(arr) <= win_samps:
                    audio_buffer[buffer_pos:buffer_pos+len(arr)] = arr
                    buffer_pos += len(arr)
                else:
                    remaining = win_samps - buffer_pos
                    audio_buffer[buffer_pos:] = arr[:remaining]
                    audio_buffer[:len(arr)-remaining] = arr[remaining:]
                    buffer_pos = len(arr) - remaining
                
                current_time = time.time()
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    last_detection_time = current_time
                    
                    audio_chunk = audio_buffer.copy()
                    rms = float(np.sqrt(np.mean(audio_chunk**2)))
                    
                    print(f"[AUDIO] Analyzing... RMS: {rms:.4f} (threshold: {self.adaptive_threshold:.4f})")
                    
                    if rms >= self.adaptive_threshold:
                        try:
                            predicted_class, confidence, quality, features = self.predict_audio(audio_chunk)
                            
                            if predicted_class == "low_quality":
                                consecutive_low_quality += 1
                                if consecutive_low_quality <= 2:
                                    print(f"[AUDIO] Low quality audio - skipping detection")
                                # Adaptive threshold adjustment
                                self.adaptive_threshold = min(self.adaptive_threshold * 1.1, 0.1)
                            else:
                                consecutive_low_quality = 0
                                self.adaptive_threshold = max(self.adaptive_threshold * 0.95, AUDIO_MIN_RMS)
                                
                                print(f"[PREDICTION] Prediction: {predicted_class} (confidence: {confidence:.2f}, quality: {quality:.2f})")
                                
                                if confidence >= self.threshold:
                                    now = time.time()

                                    # Capture frame from camera directly
                                    print(f"[DETECTION] GUNSHOT DETECTED: {predicted_class} ({confidence:.1%} confidence)")
                                    print(f"[DETECTION] Attempting to capture frame from camera index {self.camera_index}...")
                                    
                                    camera_frame = self.capture_camera_frame()
                                    
                                    if camera_frame is not None:
                                        print(f"[DETECTION] Camera frame captured successfully: shape {camera_frame.shape}")
                                        # Apply face and hand detection on real camera frame
                                        frame_with_detections = self.draw_advanced_bounding_boxes(camera_frame.copy())
                                        
                                        # Add detection information overlay to the frame
                                        frame = self.add_detection_overlay(
                                            frame_with_detections, now, predicted_class, confidence, quality, features
                                        )
                                        
                                        print(f"[DETECTION] Successfully processed real camera frame with detections")
                                        
                                        # Save the frame in temp folder
                                        self.save_detection_from_frame(now, predicted_class, confidence, quality, features, frame)
                                        
                                        # Also update temp/latest frame
                                        self.write_shared_frame(frame)
                                        
                                        self.write_system_status("detection", f"Gunshot detected: {predicted_class}")
                                    else:
                                        # Camera capture failed - log error but don't skip detection
                                        print("[WARNING] Camera capture failed, but detection will still be logged")
                                        print("[WARNING] Frame will not be saved, but detection event will be recorded")
                                        
                                        # Still update event and status
                                        self.write_shared_event(now, predicted_class, confidence, quality, features)
                                        self.write_system_status("detection", f"Gunshot detected: {predicted_class} (no frame)")
                                        
                                        # Update state
                                        self.last_audio_save = now
                                        self.detection_count += 1
                                else:
                                    print(f"[AUDIO] Below confidence threshold: {confidence:.2f} < {self.threshold}")
                                    self.false_positive_count += 1
                                
                                # Update quality history
                                self.audio_quality_scores.append(quality)
                                if len(self.audio_quality_scores) > 50:
                                    self.audio_quality_scores.pop(0)
                                    
                        except Exception as e:
                            print(f"[ERROR] Prediction error: {e}")
                    else:
                        print(f"[AUDIO] Audio too quiet (RMS: {rms:.4f} < {self.adaptive_threshold:.4f})")
                
                time.sleep(0.005)  # Reduced sleep for more responsive processing
                
        except Exception as e:
            print(f"[ERROR] Audio thread error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except:
                pass
            self.stop_flag.set()

    def run_detection_system(self):
        """Main system execution"""
        print("\n" + "="*80)
        print("STATE-OF-THE-ART GUNSHOT DETECTION SYSTEM")
        print("="*80)
        print("• Advanced audio processing with quality assessment")
        print("• Real-time MediaPipe bounding boxes (Head, Hands, Face)")
        print("• Comprehensive detection analytics and logging")
        print("• Web interface integration via JSON/Image polling")
        print("• Dual output: Annotated images + Audio recordings")
        print("• Adaptive thresholding and noise reduction")
        print("="*80)
        print("Press Ctrl+C to stop the system")
        print("="*80)
        
        # Start audio processing thread
        audio_thread = threading.Thread(target=self.audio_thread_func, daemon=True)
        audio_thread.start()
        
        try:
            print("\n[SYSTEM] System monitoring active...")
            print("[AUDIO] Listening for gunshot sounds...")
            print("[WEB] Frontend available via temp/latest_frame.jpg and latest_event.json")
            
            # Main monitoring loop
            while not self.stop_flag.is_set():
                time.sleep(1)
                
                # Periodic status update
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    avg_quality = np.mean(self.audio_quality_scores) if self.audio_quality_scores else 0
                    print(f"[SYSTEM] System Status: Detections={self.detection_count}, "
                          f"Avg Quality={avg_quality:.2f}, "
                          f"Adaptive Threshold={self.adaptive_threshold:.4f}")
                    self.write_system_status("active", f"Monitoring - {self.detection_count} detections")
                    
        except KeyboardInterrupt:
            print(f"\n[SYSTEM] User requested shutdown...")
            self.stop_flag.set()
        except Exception as e:
            print(f"[ERROR] System error: {e}")
            self.stop_flag.set()
        finally:
            self.stop_flag.set()
            # Cleanup camera
            if self.use_ffmpeg:
                self.cleanup_ffmpeg()
                print("[SYSTEM] FFmpeg camera released")
            elif self.camera is not None:
                try:
                    self.camera.release()
                    print("[SYSTEM] Camera released")
                except Exception as e:
                    print(f"[WARNING] Error releasing camera: {e}")
            self.write_system_status("stopped", "System shutdown complete")
            print(f"\n[SYSTEM] System stopped.")
            print(f"[SYSTEM] Summary: {self.detection_count} detections, {self.false_positive_count} false positives")
            print(f"[SYSTEM] Output: {self.recordings_dir.resolve()}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='State-of-the-Art Gunshot Detection System')
    parser.add_argument('--model', required=True, help='Path to trained gun type model')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for detection')
    parser.add_argument('--recordings-dir', default='gunshot_detections', help='Directory to save detection images')
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"[ERROR] Model file {args.model} not found!")
        return
    
    try:
        detection_system = StateOfTheArtGunDetectionSystem(
            model_path=args.model,
            threshold=args.threshold,
            recordings_dir=args.recordings_dir
        )
        detection_system.run_detection_system()
    except Exception as e:
        print(f"[ERROR] Error starting detection system: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()