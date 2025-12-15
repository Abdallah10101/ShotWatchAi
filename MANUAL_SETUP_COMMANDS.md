# Manual Setup Commands for Python 3.10

Run these commands one by one in Command Prompt (cmd) or PowerShell.

## Step 1: Navigate to Project Directory

```cmd
cd "C:\Users\sarma\Desktop\GunShot-Detection-Sytem-main\GunShot-Detection-Sytem-main"
```

## Step 2: Remove Old Virtual Environment (if exists)

```cmd
rmdir /s /q venv
```

Or in PowerShell:
```powershell
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
```

## Step 3: Create Virtual Environment with Python 3.10

```cmd
py -3.10 -m venv venv
```

## Step 4: Activate Virtual Environment

**In Command Prompt (cmd):**
```cmd
venv\Scripts\activate.bat
```

**In PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

If you get an execution policy error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 5: Verify Python Version

```cmd
python --version
```

Should show: `Python 3.10.x`

## Step 6: Upgrade pip

```cmd
python -m pip install --upgrade pip
```

## Step 7: Install PyTorch (CPU version)

This will take 5-10 minutes:

```cmd
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If that fails, try:
```cmd
python -m pip install torch torchvision torchaudio
```

## Step 8: Install Other Dependencies

```cmd
python -m pip install opencv-python mediapipe librosa pyaudio noisereduce soundfile numpy scipy
```

Or install from requirements.txt:
```cmd
python -m pip install -r requirements.txt
```

## Step 9: Verify Installation

```cmd
python -c "import torch; import cv2; import librosa; import mediapipe; print('All dependencies installed successfully!')"
```

## Step 10: Deactivate (when done)

```cmd
deactivate
```

---

## Quick Copy-Paste (All Commands)

```cmd
cd "C:\Users\sarma\Desktop\GunShot-Detection-Sytem-main\GunShot-Detection-Sytem-main"
rmdir /s /q venv
py -3.10 -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import torch; import cv2; import librosa; import mediapipe; print('All dependencies installed successfully!')"
```

---

## Troubleshooting

### If PyAudio installation fails on Windows:

1. Download the wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
2. Download the version matching your Python (e.g., `PyAudio-0.2.11-cp310-cp310-win_amd64.whl` for Python 3.10)
3. Install it:
   ```cmd
   python -m pip install PyAudio-0.2.11-cp310-cp310-win_amd64.whl
   ```

### If MediaPipe installation fails:

```cmd
python -m pip install mediapipe --no-deps
python -m pip install numpy opencv-python
```

### If you get "No module named pip":

```cmd
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

