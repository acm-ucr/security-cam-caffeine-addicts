import paho.mqtt.client as mqtt
import json

# MQTT setup
broker = "broker.emqx.io"
port = 1883
topic = "security/alert"

# Called when connected to broker
def on_connect(client, userdata, flags, rc):
    print("✅ Connected to broker.")
    client.subscribe(topic)

# Called when a message is received
def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        data = json.loads(payload)
        print(f"🚨 ALERT: {data['object']} detected with {data['confidence']*100:.1f}% confidence")
    except Exception:
        print("📩 Received:", payload)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port, 60)
print(f"Listening on topic: {topic}")

client.loop_forever()
