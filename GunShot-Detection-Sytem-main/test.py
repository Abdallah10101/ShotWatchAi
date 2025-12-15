import cv2
import pyaudio
import threading

class OBSBOTController:
    def __init__(self):
        self.video_source = 1
        self.audio_device_index = None
        self.setup_camera()
        self.setup_audio()
    
    def setup_camera(self):
        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print("Camera initialized")
    
    def setup_audio(self):
        self.p = pyaudio.PyAudio()
        # Find OBSBOT audio device
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if 'obsbot' in info['name'].lower():
                self.audio_device_index = i
                print(f"Found OBSBOT audio: {info['name']}")
                break
        
        if self.audio_device_index is not None:
            self.audio_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=self.audio_device_index,
                frames_per_buffer=1024
            )
    
    def start_stream(self):
        video_thread = threading.Thread(target=self.video_stream)
        video_thread.daemon = True
        video_thread.start()
        
        if hasattr(self, 'audio_stream'):
            audio_thread = threading.Thread(target=self.audio_stream)
            audio_thread.daemon = True
            audio_thread.start()
        
        video_thread.join()
    
    def video_stream(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('OBSBOT Tiny 2 Lite - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def audio_stream(self):
        print("Audio streaming started...")
        while True:
            try:
                data = self.audio_stream.read(1024)
                # Process audio data here
                pass
            except:
                break

# Run the camera
controller = OBSBOTController()
controller.start_stream()