import argparse
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import librosa
import pyaudio
from pathlib import Path
import warnings
from scipy import signal
import wave
import datetime
import os
import noisereduce as nr
warnings.filterwarnings('ignore')

# Audio parameters (must match training)
SR = 22050
CLIP_SEC = 2.0
N_MELS = 128
FMAX = 8000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Enhanced audio processing parameters
TARGET_RMS = 0.05       # RMS normalization target
NOISE_REDUCTION = True  # Enable spectral gating
PREEMPHASIS = True      # Apply pre-emphasis filter
DEEMPHASIS = True       # Apply deemphasis correction (reverse EQ curve)

class ImprovedGunTypePredictor:
    def __init__(self, model_path, threshold=0.6, save_recordings=True, recordings_dir="recordings"):
        self.model_path = model_path
        self.threshold = threshold
        self.save_recordings = save_recordings
        self.recordings_dir = Path(recordings_dir)
        self.model = None
        self.classes = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create recordings directory if needed
        if self.save_recordings:
            self.recordings_dir.mkdir(exist_ok=True)
            print(f"📁 Recordings will be saved to: {self.recordings_dir}")
        
        # Audio processing state
        self.background_noise = None
        self.noise_floor = 0.001
        self.audio_quality_history = []
        self.adaptive_threshold = 0.02
        
        # Recording state
        self.recording_counter = 0
        self.last_saved_time = 0
        self.min_save_interval = 2.0  # Minimum seconds between saves
        
        self.load_model()
        
        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.raw_audio_buffer = np.array([], dtype=np.int16)  # For saving raw audio
        self.buffer_lock = threading.Lock()
        self.is_recording = False
        
    def load_model(self):
        """Load the trained model"""
        print("Loading model...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except:
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        self.classes = checkpoint['classes']
        print(f"Loaded {len(self.classes)} classes")
        
        self.model = ResNet50LSTM(n_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def _save_audio_recording(self, audio_data_float, predicted_class="unknown", confidence=0.0):
        """Save audio recording to WAV file"""
        if not self.save_recordings:
            return
            
        current_time = time.time()
        if current_time - self.last_saved_time < self.min_save_interval:
            return
            
        try:
            # Convert float audio to int16 for WAV file
            audio_data_int16 = (audio_data_float * 32767).astype(np.int16)
            
            # Create filename with timestamp and prediction info
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_class = "".join(c if c.isalnum() else "_" for c in predicted_class)
            confidence_pct = int(confidence * 100)
            
            filename = f"gunshot_{timestamp}_{safe_class}_{confidence_pct}pc.wav"
            filepath = self.recordings_dir / filename
            
            # Save as WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(SR)
                wav_file.writeframes(audio_data_int16.tobytes())
            
            self.recording_counter += 1
            self.last_saved_time = current_time
            print(f"💾 Saved recording: {filename}")
            
        except Exception as e:
            print(f"Error saving recording: {e}")
    
    def _analyze_audio_characteristics(self, audio_data):
        """Analyze audio to understand its characteristics"""
        features = {}
        
        # Basic stats
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        features['peak'] = np.max(np.abs(audio_data))
        features['dynamic_range'] = features['peak'] / (features['rms'] + 1e-8)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=SR)[0]
        features['spectral_centroid'] = np.mean(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=SR)[0]
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        # Zero crossing rate
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Onset strength
        features['onset_strength'] = np.mean(librosa.onset.onset_strength(y=audio_data, sr=SR))
        
        return features
    
    def enhance_mic_audio(self, y, sr=SR):
        """Enhanced microphone audio processing from first script"""
        # 1️⃣ Remove DC offset
        y = y - np.mean(y)

        # 2️⃣ Normalize amplitude
        y = y / (np.max(np.abs(y)) + 1e-9)

        # 3️⃣ Apply de-emphasis (manual inverse of pre-emphasis)
        if DEEMPHASIS:
            deemph_coef = 0.97
            # simple inverse IIR filter (de-emphasis)
            y = np.convolve(y, [1], mode="same")  # start clean
            for i in range(1, len(y)):
                y[i] = y[i] + deemph_coef * y[i - 1]

        # 4️⃣ Apply pre-emphasis to enhance clarity of transients
        if PREEMPHASIS:
            y = librosa.effects.preemphasis(y, coef=0.97)

        # 5️⃣ Noise reduction using spectral gating
        if NOISE_REDUCTION:
            try:
                y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)
            except Exception as e:
                print(f"[Warning] Noise reduction failed: {e}")

        # 6️⃣ RMS normalization
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y *= (TARGET_RMS / (rms + 1e-9))

        return y

    def _adaptive_high_pass_filter(self, audio_data):
        """Adaptive high-pass filter based on content"""
        features = self._analyze_audio_characteristics(audio_data)
        
        # Use lower cutoff for gunshots (preserve low frequencies)
        if features['spectral_centroid'] > 2000:  # High frequency content
            cutoff = 50.0  # Lower cutoff to preserve gunshot characteristics
        else:
            cutoff = 80.0  # Normal cutoff
            
        nyquist = SR / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(3, normal_cutoff, btype='high', analog=False)
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def _spectral_balance_correction(self, audio_data):
        """Correct spectral balance to match training data characteristics"""
        # Gunshots typically have strong mid-frequency content
        # Boost frequencies between 300Hz and 4000Hz
        
        stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
        
        # Create frequency weighting curve
        weights = np.ones_like(freqs)
        
        # Boost mid frequencies (where gunshots have most energy)
        mid_band = (freqs >= 300) & (freqs <= 4000)
        weights[mid_band] = 1.3
        
        # Slight reduction of very high frequencies (often noise)
        high_band = freqs > 6000
        weights[high_band] = 0.9
        
        # Apply spectral shaping
        balanced_magnitude = magnitude * weights[:, np.newaxis]
        
        # Reconstruct
        balanced_stft = balanced_magnitude * np.exp(1j * phase)
        balanced_audio = librosa.istft(balanced_stft, hop_length=512)
        
        # Match length
        min_len = min(len(audio_data), len(balanced_audio))
        return balanced_audio[:min_len]
    
    def _smart_noise_gate(self, audio_data, threshold_ratio=2.0):
        """Intelligent noise gate that preserves transients"""
        # Calculate envelope
        envelope = np.abs(signal.hilbert(audio_data))
        envelope_smooth = np.convolve(envelope, np.ones(100)/100, mode='same')
        
        # Adaptive threshold based on background
        if self.background_noise is None:
            self.background_noise = np.percentile(envelope_smooth, 30)
        
        threshold = self.background_noise * threshold_ratio
        
        # Create gate mask (preserve attack portions)
        gate_mask = envelope_smooth > threshold
        
        # Expand mask to include attack transients
        kernel = np.ones(50)  # 50 samples ~ 2.3ms
        gate_mask_expanded = np.convolve(gate_mask.astype(float), kernel, mode='same') > 0
        
        gated_audio = audio_data * gate_mask_expanded
        
        return gated_audio
    
    def _transient_preservation(self, audio_data):
        """Specifically preserve and enhance transient sounds"""
        # Use wavelet-like approach for transient detection
        # High-pass filtered version for transient detection
        b, a = signal.butter(4, 1000/(SR/2), btype='high')
        high_freq = signal.filtfilt(b, a, audio_data)
        
        # Detect transients using high-frequency energy
        transient_energy = np.convolve(high_freq**2, np.ones(20)/20, mode='same')
        transient_mask = transient_energy > np.percentile(transient_energy, 70)
        
        # Create emphasis around transients
        kernel = np.exp(-np.linspace(-2, 2, 100)**2)  # Gaussian kernel
        emphasis_weights = np.convolve(transient_mask.astype(float), kernel, mode='same')
        emphasis_weights = np.clip(emphasis_weights * 0.5 + 1, 1, 1.5)  # 1.0-1.5x boost
        
        enhanced_audio = audio_data * emphasis_weights
        
        return enhanced_audio
    
    def _level_matching(self, audio_data, target_rms=0.12):
        """Advanced level matching with transient preservation"""
        current_rms = np.sqrt(np.mean(audio_data**2))
        
        if current_rms < 1e-8:
            return audio_data
        
        # Calculate gain with careful limiting
        desired_gain = target_rms / (current_rms + 1e-8)
        
        # Soft knee compression
        if desired_gain > 4.0:
            desired_gain = 4.0
        elif desired_gain < 0.25:
            desired_gain = 0.25
            
        normalized_audio = audio_data * desired_gain
        
        # Gentle soft clipping to preserve dynamics
        clip_threshold = 0.8
        over_threshold = np.abs(normalized_audio) > clip_threshold
        if np.any(over_threshold):
            # Only clip the peaks, preserve the rest
            peaks = normalized_audio[over_threshold]
            clipped_peaks = np.clip(peaks, -clip_threshold, clip_threshold)
            normalized_audio[over_threshold] = clipped_peaks
        
        return normalized_audio
    
    def _calculate_confidence_score(self, audio_data):
        """Calculate confidence that audio contains a gunshot-like sound"""
        features = self._analyze_audio_characteristics(audio_data)
        score = 0.0
        
        # RMS level (gunshots are typically loud)
        if 0.05 <= features['rms'] <= 0.4:
            score += 0.3
        elif 0.02 <= features['rms'] < 0.05:
            score += 0.1
            
        # Spectral centroid (gunshots typically 1k-5kHz)
        if 800 <= features['spectral_centroid'] <= 5000:
            score += 0.3
        elif 500 <= features['spectral_centroid'] < 800:
            score += 0.15
            
        # Onset strength (gunshots have strong attacks)
        if features['onset_strength'] > 0.5:
            score += 0.2
        elif features['onset_strength'] > 0.2:
            score += 0.1
            
        # Dynamic range (gunshots have high dynamic range)
        if features['dynamic_range'] > 5:
            score += 0.2
            
        return score
    
    def preprocess_microphone_audio(self, audio_data):
        """Advanced microphone preprocessing pipeline using enhanced processing"""
        original_features = self._analyze_audio_characteristics(audio_data)
        
        print(f"🎤 Input: RMS={original_features['rms']:.4f}, "
              f"Centroid={original_features['spectral_centroid']:.0f}Hz, "
              f"Onset={original_features['onset_strength']:.3f}")
        
        # Use the enhanced audio processing from the first script
        audio_data = self.enhance_mic_audio(audio_data)
        
        # Additional processing steps for gunshot detection
        audio_data = self._adaptive_high_pass_filter(audio_data)
        audio_data = self._smart_noise_gate(audio_data, threshold_ratio=2.5)
        audio_data = self._spectral_balance_correction(audio_data)
        audio_data = self._transient_preservation(audio_data)
        audio_data = self._level_matching(audio_data, target_rms=0.15)
        
        processed_features = self._analyze_audio_characteristics(audio_data)
        confidence = self._calculate_confidence_score(audio_data)
        
        print(f"Processed: RMS={processed_features['rms']:.4f}, "
              f"Centroid={processed_features['spectral_centroid']:.0f}Hz, "
              f"Confidence={confidence:.2f}")
        
        return audio_data, confidence
    
    def preprocess_file_audio(self, audio_data):
        """Original file processing (unchanged)"""
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
    
    def predict(self, audio_data, is_mic_input=False):
        """Predict with confidence-based processing"""
        if is_mic_input:
            # Advanced processing for microphone
            processed_audio, confidence = self.preprocess_microphone_audio(audio_data)
            
            if confidence < 0.3:
                return f"Unknown - Low Quality (score: {confidence:.2f})", 0.0, np.zeros(len(self.classes))
            
            # Convert to spectrogram
            input_tensor = self.preprocess_file_audio(processed_audio)
        else:
            # Original file processing
            input_tensor = self.preprocess_file_audio(audio_data)
            confidence = 1.0
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_pred, predicted = torch.max(probabilities, 1)
            
            confidence_pred = confidence_pred.item()
            predicted_class = self.classes[predicted.item()]
            all_probs = probabilities.cpu().numpy()[0]
            
            # Adjust confidence based on audio quality
            adjusted_confidence = confidence_pred * min(confidence, 1.0)
            
            return predicted_class, adjusted_confidence, all_probs
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Simple background noise estimation
            if self.background_noise is None:
                self.background_noise = np.sqrt(np.mean(audio_data**2))
            else:
                # Update background noise estimate slowly
                current_noise = np.sqrt(np.mean(audio_data**2))
                self.background_noise = 0.99 * self.background_noise + 0.01 * current_noise
            
            with self.buffer_lock:
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
                # Also keep raw int16 data for saving
                self.raw_audio_buffer = np.concatenate([self.raw_audio_buffer, np.frombuffer(in_data, dtype=np.int16)])
                
                max_buffer_length = SR * 5
                if len(self.audio_buffer) > max_buffer_length:
                    self.audio_buffer = self.audio_buffer[-max_buffer_length:]
                if len(self.raw_audio_buffer) > max_buffer_length:
                    self.raw_audio_buffer = self.raw_audio_buffer[-max_buffer_length:]
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        self.audio = pyaudio.PyAudio()
        
        # Try to find the best input device
        best_device = None
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                if best_device is None or dev_info['defaultSampleRate'] == SR:
                    best_device = i
                print(f"  {i}: {dev_info['name']} (SR: {dev_info['defaultSampleRate']})")
        
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SR,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.audio_callback,
            input_device_index=best_device
        )
        self.is_recording = True
        self.stream.start_stream()
        print(f"Recording started (Device: {best_device})")
    
    def stop_recording(self):
        if hasattr(self, 'stream'):
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def get_recent_audio(self, duration=CLIP_SEC):
        with self.buffer_lock:
            required_length = int(SR * duration)
            if len(self.audio_buffer) >= required_length:
                return self.audio_buffer[-required_length:]
            return None
    
    def real_time_prediction(self, update_interval=1.0):
        """Run real-time prediction with quality monitoring"""
        print("\nStarting improved real-time prediction...")
        print("Make sharp, impulsive sounds (like clapping or tapping) for testing")
        print(f"Recordings will be saved to: {self.recordings_dir}")
        print("Press Ctrl+C to stop\n")
        
        self.start_recording()
        
        try:
            consecutive_low_quality = 0
            
            while True:
                time.sleep(update_interval)
                audio_chunk = self.get_recent_audio()
                
                if audio_chunk is not None:
                    predicted_class, confidence, all_probs = self.predict(audio_chunk, is_mic_input=True)
                    
                    if predicted_class.startswith("Unknown"):
                        consecutive_low_quality += 1
                        if consecutive_low_quality <= 3:  # Only show first few
                            print(f"🔇 {predicted_class}")
                    else:
                        consecutive_low_quality = 0
                        self.display_prediction(predicted_class, confidence, all_probs)
                        
                        # Save the recording if confidence is above threshold
                        if confidence >= self.threshold * 0.5:  # Lower threshold for saving
                            self._save_audio_recording(audio_chunk, predicted_class, confidence)
                else:
                    print("Collecting audio...")
                    
        except KeyboardInterrupt:
            print(f"\nStopping... Total recordings saved: {self.recording_counter}")
        finally:
            self.stop_recording()
    
    def display_prediction(self, predicted_class, confidence, all_probs):
        print("\n" + "="*70)
        print(f"GUN TYPE CLASSIFICATION")
        print("="*70)
        
        if confidence >= self.threshold:
            print(f"PREDICTION: {predicted_class}")
            print(f"CONFIDENCE: {confidence:.1%}")
            print("RELIABLE DETECTION")
        else:
            print(f"TENTATIVE: {predicted_class}")
            print(f"CONFIDENCE: {confidence:.1%} (needs {self.threshold:.0%})")
            print("Try making a sharper, louder sound")
        
        print(f"\nPROBABILITY DISTRIBUTION:")
        top_indices = np.argsort(all_probs)[-5:][::-1]  # Top 5
        for i in top_indices:
            class_name = self.classes[i]
            prob = all_probs[i]
            bar = "█" * int(prob * 40)
            marker = "" if i == np.argmax(all_probs) else "  "
            print(f"  {marker} {class_name:<25}: {prob:.1%} {bar}")
        
        print("="*70)

# Model architecture
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

def main():
    parser = argparse.ArgumentParser(description='Improved gun type prediction with recording save')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--file', help='Predict from audio file')
    parser.add_argument('--interval', type=float, default=1.0, help='Update interval')
    parser.add_argument('--no-save', action='store_true', help='Disable saving recordings')
    parser.add_argument('--recordings-dir', default='recordings', help='Directory to save recordings')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found!")
        return
    
    try:
        predictor = ImprovedGunTypePredictor(
            args.model, 
            args.threshold,
            save_recordings=not args.no_save,
            recordings_dir=args.recordings_dir
        )
        
        if args.file:
            if not Path(args.file).exists():
                print(f"Error: Audio file {args.file} not found!")
                return
            
            print(f"Loading audio file: {args.file}")
            audio_data, sr = librosa.load(args.file, sr=SR)
            if sr != SR:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SR)
            
            predicted_class, confidence, all_probs = predictor.predict(audio_data, is_mic_input=False)
            predictor.display_prediction(predicted_class, confidence, all_probs)
        else:
            predictor.real_time_prediction(update_interval=args.interval)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()