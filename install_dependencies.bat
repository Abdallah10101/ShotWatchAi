@echo off
echo ========================================
echo Installing Python Dependencies
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo Using venv Python...
venv\Scripts\python.exe --version
echo.

echo Installing pip...
python -m pip install --target venv\Lib\site-packages pip setuptools wheel --quiet
echo.

echo Installing PyTorch (CPU version)...
echo This may take several minutes...
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo WARNING: PyTorch installation failed. Trying default installation...
    venv\Scripts\python.exe -m pip install torch torchvision torchaudio
)
echo.

echo Installing other dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.
pause

