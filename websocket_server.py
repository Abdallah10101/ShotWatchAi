import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from predict_audio_only import StateOfTheArtGunDetectionSystem
import threading
import time

class WebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.detection_system = StateOfTheArtGunDetectionSystem(
            model_path="guntype_resnet50.pth",
            threshold=0.7
        )
        self.is_running = False

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        print(f"New client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'audio':
                    # Process audio data
                    frame_bytes = base64.b64decode(data['data'])
                    int16_audio = np.frombuffer(frame_bytes, dtype=np.int16)
                    audio_data = int16_audio.astype(np.float32) / 32768.0

                    # predict_audio returns (predicted_class, adjusted_confidence, enhanced_quality, enhanced_features)
                    label, confidence, quality, features = self.detection_system.predict_audio(audio_data)

                    if confidence > self.detection_system.threshold:
                        # Request frame from client
                        await websocket.send(json.dumps({
                            'type': 'request_frame',
                            'detection': {
                                'label': label,
                                'confidence': float(confidence),
                                'timestamp': time.time()
                            }
                        }))
                
                elif data.get('type') == 'frame':
                    # Save frame with detection
                    print("[WEBSOCKET] ========================================")
                    print("[WEBSOCKET] Received frame from frontend")
                    print(f"[WEBSOCKET] Frame data length: {len(data.get('frame', ''))}")
                    
                    try:
                        frame_data = base64.b64decode(data['frame'])
                        print(f"[WEBSOCKET] Decoded base64, size: {len(frame_data)} bytes")
                    except Exception as e:
                        print(f"[ERROR] Failed to decode base64: {e}")
                        continue
                    
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        print("[ERROR] Failed to decode frame from base64 - cv2.imdecode returned None")
                        print(f"[ERROR] Frame data preview: {frame_data[:100] if len(frame_data) > 100 else frame_data}")
                        continue
                    
                    print(f"[WEBSOCKET] Decoded frame successfully: shape {frame.shape}")
                    
                    # Process frame with MediaPipe for bounding boxes (face and hands)
                    try:
                        frame_with_boxes = self.detection_system.draw_advanced_bounding_boxes(frame)
                        print(f"[WEBSOCKET] Applied MediaPipe detections to frame: {frame_with_boxes.shape}")
                    except Exception as e:
                        print(f"[ERROR] MediaPipe processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                        frame_with_boxes = frame  # Use original frame if MediaPipe fails
                    
                    # Save the frame using the provided image instead of generating a synthetic one
                    detection_info = data.get('detection', {})
                    trigger_ts = detection_info.get('timestamp', time.time())
                    label = detection_info.get('label', 'unknown')
                    confidence = float(detection_info.get('confidence', 0.0))
                    lat = detection_info.get('lat')
                    lng = detection_info.get('lng')
                    accuracy = detection_info.get('accuracy')
                    
                    location_str = f", location=({lat}, {lng})" if lat and lng else ""
                    print(f"[WEBSOCKET] Detection info: label={label}, confidence={confidence}, timestamp={trigger_ts}{location_str}")

                    saved_path = None
                    try:
                        print(f"[WEBSOCKET] Calling save_detection_from_frame...")
                        saved_path = self.detection_system.save_detection_from_frame(
                            trigger_ts=trigger_ts,
                            label=label,
                            confidence=confidence,
                            quality=1.0,
                            features={},
                            frame=frame_with_boxes,  # Frame already has bounding boxes
                            lat=lat,
                            lng=lng,
                            accuracy=accuracy
                        )
                        if saved_path:
                            print(f"[WEBSOCKET] ✓ Frame processed and saved successfully: {saved_path}")
                        else:
                            print(f"[WEBSOCKET] ⚠ save_detection_from_frame returned None (may be cooldown)")
                    except Exception as e:
                        print(f"[ERROR] save_detection_from_frame failed: {e}")
                        import traceback
                        traceback.print_exc()
                    print("[WEBSOCKET] ========================================")

                    # Notify client that the image has been saved
                    try:
                        await websocket.send(json.dumps({
                            'type': 'image_saved',
                            'detection': {
                                'label': label,
                                'confidence': confidence,
                                'timestamp': trigger_ts,
                            },
                            # Only include a simple filename if we have one
                            'file': str(saved_path) if saved_path else None,
                        }))
                    except Exception as e:
                        print(f"Failed to send image_saved event: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Remaining clients: {len(self.clients)}")

    async def start(self):
        self.is_running = True
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            while self.is_running:
                await asyncio.sleep(1)

    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    server = WebSocketServer()
    asyncio.get_event_loop().run_until_complete(server.start())
