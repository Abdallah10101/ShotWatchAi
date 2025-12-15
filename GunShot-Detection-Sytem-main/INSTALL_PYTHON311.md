# Installing Python 3.11

## Option 1: Download and Install Manually

1. **Download Python 3.11:**
   - Visit: https://www.python.org/downloads/release/python-3110/
   - Download "Windows installer (64-bit)" for Python 3.11.0 or later

2. **Install Python 3.11:**
   - Run the installer
   - **Important:** Check "Add Python 3.11 to PATH" during installation
   - Click "Install Now"

3. **Verify Installation:**
   ```bash
   py -3.11 --version
   ```

4. **Run the setup script:**
   ```bash
   setup_venv_python311.bat
   ```

## Option 2: Use Windows Package Manager (winget)

```bash
winget install Python.Python.3.11
```

Then run:
```bash
setup_venv_python311.bat
```

## Option 3: Use Python 3.10 (Already Installed)

If you don't want to install Python 3.11, you can use Python 3.10 which is already installed:

```bash
py -3.10 -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## After Installation

Once Python 3.11 is installed, run:
```bash
setup_venv_python311.bat
```

This will create a virtual environment with Python 3.11 and install all dependencies.

