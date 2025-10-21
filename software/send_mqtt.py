from ultralytics import YOLO
import cv2
import paho.mqtt.client as mqtt
import json

# MQTT setup
broker = "broker.emqx.io"
port = 1883
topic = "security/alert"

client = mqtt.Client()
client.connect(broker, port, 60)

# Load YOLO model
model = YOLO("yolov8s.pt")  # or your trained model
cap = cv2.VideoCapture(2)   # adjust camera index

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("✅ YOLO camera running and MQTT connected. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("Security Camera - YOLOv8", annotated_frame)

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # Publish detections for specific objects
            if label in ["person", "car"]:
                msg = {"object": label, "confidence": round(conf, 2)}
                client.publish(topic, json.dumps(msg))
                print(f"📡 Sent: {msg}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()
