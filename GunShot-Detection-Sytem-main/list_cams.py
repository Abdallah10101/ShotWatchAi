import cv2

print("Scanning camera indices 0–5 using default backend...")
for i in range(6):
    cap = cv2.VideoCapture(i)  # 🔹 no CAP_DSHOW
    opened = cap.isOpened()
    print(f"Index {i}: opened = {opened}")
    if opened:
        ret, frame = cap.read()
        print(f"  frame_ok = {ret}")
        cap.release()
