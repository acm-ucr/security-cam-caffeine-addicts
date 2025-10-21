import paho.mqtt.client as mqtt
import json

# MQTT Configuration
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "security/alert"

# Initialize and connect to broker
client = mqtt.Client()
client.connect(BROKER, PORT, 60)

def send_alert(object_label, confidence):
    """Publish a detection alert message to the MQTT broker."""
    message = {
        "object": object_label,
        "confidence": round(confidence, 2)
    }
    client.publish(TOPIC, json.dumps(message))
    print(f"📡 Alert sent: {message}")

# Optional test (run this file alone to send a test message)
if __name__ == "__main__":
    send_alert("test_object", 0.99)
    client.disconnect()
