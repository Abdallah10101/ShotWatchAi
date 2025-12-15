import asyncio
import threading
from websocket_server import WebSocketServer
from predict_audio_only import StateOfTheArtGunDetectionSystem
from camera_stream_server import run_flask_server
import camera_stream_server

# Global detection system instance (shared with Flask server)
detection_system = None

def run_websocket_server():
    server = WebSocketServer()
    asyncio.run(server.start())

def run_detection_system():
    global detection_system
    detection_system = StateOfTheArtGunDetectionSystem(
        model_path="guntype_resnet50.pth",
        threshold=0.7
    )
    # Make system available to Flask app
    camera_stream_server.system = detection_system
    detection_system.run_detection_system()

if __name__ == "__main__":
    # Start detection system first (this initializes the camera)
    print("[SYSTEM] Initializing detection system and camera...")
    detection_system = StateOfTheArtGunDetectionSystem(
        model_path="guntype_resnet50.pth",
        threshold=0.7
    )
    # Make system available to Flask app
    camera_stream_server.system = detection_system
    
    # Initialize camera before starting Flask server
    print("[SYSTEM] Initializing camera...")
    detection_system.init_camera()
    
    # Wait a moment for camera to be ready
    import time
    time.sleep(1)
    
    # Start Flask camera stream server in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True, kwargs={'host': '0.0.0.0', 'port': 5000})
    flask_thread.start()
    print("[SYSTEM] Camera stream server started on http://localhost:5000/camera-stream")
    
    # Start WebSocket server in a separate thread
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()
    print("[SYSTEM] WebSocket server started on ws://localhost:8765")
    
    # Start detection system in the main thread (this runs the audio processing)
    detection_system.run_detection_system()
