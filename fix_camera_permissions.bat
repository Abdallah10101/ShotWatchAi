@echo off
echo Camera Permission Fix for Gunshot Detection System
echo =====================================================
echo.
echo This will help you fix camera permission issues on Windows.
echo.

echo Step 1: Opening Windows Camera Settings...
echo Please check these settings in the window that opens:
echo - Camera access: On
echo - Allow apps to access your camera: On  
echo - Allow desktop apps to access your camera: On
echo.
start ms-settings:privacy-webcam

echo Step 2: Opening Device Manager...
echo Check if your camera is enabled and has no errors:
echo - Expand Cameras or Imaging devices
echo - Right-click camera -^> Enable device
echo - Right-click camera -^> Update driver
echo.
start devmgmt.msc

echo Step 3: Checking for camera-using applications...
echo Please close these applications if running:
echo - Zoom, Microsoft Teams, Skype
echo - Discord, Slack
echo - OBS Studio, Streamlabs
echo - Windows Camera app
echo.

echo Step 4: Testing camera access...
echo Testing basic camera functionality...
python test_camera.py

echo.
echo If camera is still not working, try:
echo 1. Restart your computer
echo 2. Unplug and replug external camera
echo 3. Try different USB port
echo 4. Run as Administrator
echo.

echo After fixing permissions, try:
echo - npm run dev-full (for full system)
echo - npm run dev-audio-only (if camera still fails)
echo.

pause
