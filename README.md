# Gunshot Detection System

A real-time gunshot detection system with webcam integration and gun type classification.

## Features

- **Real-time Audio Detection**: Detects gunshots every 2 seconds
- **Gun Type Classification**: Identifies specific gun types (13 classes)
- **Webcam Integration**: Captures frames on detection with bounding boxes
- **MediaPipe Detection**: 
  - 🔵 BLUE boxes: Face detection
  - 🟢 GREEN boxes: Head detection  
  - 🔴 RED boxes: Hand detection
- **Frontend Display**: Live webcam feed on Next.js interface
- **Automatic Saving**: Saves detection images as "Gun Name - YYYY-MM-DD HH-MM-SS.jpg"


## Presentation
- **Video Link**: (https://youtu.be/EBYnU8mtW4Y)

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Camera permissions enabled
- Ensure Only 1 camera (Preferably with PTZ functionality)
- Microphone access

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### Running the System


#### **Manual Start**
```bash
# Terminal 1: Start By running
python .\start_server.py

#Terminal 2: Start Next.js Frontend
npm run dev

# Terminal 3: Back End (Threshold Can be changed to whatever number u want)
python predict_audio_only.py --model guntype_resnet50.pth --threshold 0.6
```

### Access Points
- **Frontend**: http://localhost:3000
- **Detection Images**: `gunshot_detections/` folder
- **Live Feed**: `temp/latest_frame.jpg` (for frontend polling)

## Camera Setup

If camera doesn't work, run the permission fix:
```bash
fix_camera_permissions.bat
```

This will:
1. Open Windows Camera Settings
2. Open Device Manager
3. Test camera functionality

## File Structure

```
├── predict_web_integrated.py    # Main detection system
├── guntype_resnet50.pth         # Trained model
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js configuration
├── app/                         # Next.js frontend
├── temp/                        # Shared files for frontend
├── gunshot_detections/          # Saved detection images
└── fix_camera_permissions.bat   # Camera permission fix
```

## Detection Classes

The system can detect 13 gun types:
- 38s&ws Dot38 Caliber
- AK-12, AK-47
- Glock 17 9mm Caliber
- IMI Desert Eagle
- M16, M249, M4
- MG-42, MP5
- Remington 870 12 Gauge
- Ruger AR 556 Dot223 Caliber
- Zastava M92

## How It Works

1. **Audio Monitoring**: System continuously monitors microphone every 2 seconds
2. **Sound Analysis**: Processes audio to detect gunshot patterns
3. **Gun Classification**: Identifies specific gun type with confidence score
4. **Frame Capture**: Captures webcam frame at detection moment
5. **Bounding Boxes**: Adds MediaPipe detection boxes for face/head/hands
6. **Save & Display**: Saves annotated image and updates frontend

## Troubleshooting

### Camera Issues
- Run `fix_camera_permissions.bat`
- Check Windows privacy settings
- Close other camera apps (Zoom, Teams)

### Audio Issues
- Check microphone permissions
- Ensure external mic is connected
- Test with `python -c "import pyaudio; print('PyAudio OK')"`

### Model Loading Issues
- Verify `guntype_resnet50.pth` exists
- Check PyTorch installation
- Ensure sufficient RAM available

## Development

### Environment Setup
```bash
# Create virtual environment
setup_venv.bat

# Install dependencies  
install_dependencies.bat
```

### Configuration
Edit `predict_web_integrated.py` to adjust:
- Detection thresholds
- Camera index
- Audio sensitivity
- Detection intervals

## License

© 2025 Gunshot Detection System
