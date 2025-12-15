"""
Flask server for camera stream endpoint
Provides MJPEG stream from the detection system's camera
"""
from flask import Flask, Response
import cv2
import threading
import time
import numpy as np
from predict_audio_only import StateOfTheArtGunDetectionSystem

app = Flask(__name__)

# Global detection system instance (will be set by main)
system = None

@app.route("/camera-stream")
def camera_stream():
    """MJPEG stream endpoint for camera feed - works with both OpenCV and FFmpeg backends"""
    def gen():
        # Wait for system and camera to be initialized
        max_wait = 30  # Wait up to 30 seconds
        wait_count = 0
        while system is None:
            if wait_count >= max_wait * 10:  # 30 seconds at 0.1s intervals
                # Send error frame
                error_frame = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + error_frame + b"\r\n"
                )
                return
            time.sleep(0.1)
            wait_count += 1
        
        # Wait for camera to be initialized (either OpenCV or FFmpeg)
        wait_count = 0
        while True:
            camera_ready = False
            if system is not None:
                # Check if OpenCV camera is ready
                if system.camera is not None and system.camera.isOpened():
                    camera_ready = True
                # Check if FFmpeg camera is ready
                elif system.use_ffmpeg and system.ffmpeg_process is not None:
                    camera_ready = True
            
            if camera_ready:
                break
            
            if wait_count >= max_wait * 10:  # 30 seconds
                # Send error frame
                error_frame = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + error_frame + b"\r\n"
                )
                return
            
            time.sleep(0.1)
            wait_count += 1
        
        # Now stream frames
        while True:
            if system is None:
                time.sleep(0.1)
                continue
            
            # Use existing capture method (works with both OpenCV and FFmpeg)
            frame = system.capture_camera_frame()
            if frame is None:
                time.sleep(0.033)  # ~30 fps fallback
                continue

            # Encode frame as JPEG
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            jpg_bytes = buffer.tobytes()
            # MJPEG chunk
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
            
            # Control frame rate (~30 fps)
            time.sleep(0.033)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    """Health check endpoint"""
    camera_available = False
    if system is not None:
        # Check if camera is available (either OpenCV or FFmpeg)
        camera_available = (system.camera is not None) or (system.use_ffmpeg and system.ffmpeg_process is not None)
    
    return {"status": "ok", "camera_available": camera_available}

def run_flask_server(host='0.0.0.0', port=5000):
    """Run Flask server in a separate thread"""
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    # For standalone testing
    print("[CAMERA STREAM] Starting Flask server on http://localhost:5000/camera-stream")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

