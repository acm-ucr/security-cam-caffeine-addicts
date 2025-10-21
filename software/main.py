from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # <-- your trained model file

# Open the security camera (0 = default webcam, or replace with your RTSP/USB camera link)
cap = cv2.VideoCapture(2)  # e.g. "rtsp://username:password@ip_address:port/stream"

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, stream=True)

    # Display results with bounding boxes
    for r in results:
        annotated_frame = r.plot()  # Draw bounding boxes and labels
        cv2.imshow("Security Camera - YOLOv8", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
