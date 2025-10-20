import cv2

print("🔍 Scanning available cameras...")
for i in range(5):  # test first 5 indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera found at index {i}")
        else:
            print(f"⚠️  Camera at index {i} opened but no frames returned")
        cap.release()
    else:
        print(f"❌ No camera at index {i}")
g