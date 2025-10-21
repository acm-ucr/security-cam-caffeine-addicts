from ultralytics import YOLO
import cv2
from sender import send_alert  # import the MQTT sender function

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # <-- your trained model file

# Open the security camera (0 = default webcam, 2 = external)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("✅ YOLO running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 inference
    results = model(frame, stream=True)

    # Process and display results
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("Security Camera - YOLOv8", annotated_frame)

        # Loop through detected objects
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            # Send MQTT alert for specific detections
            if label in ["person", "car"]:  # modify as needed
                send_alert(label, confidence)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
