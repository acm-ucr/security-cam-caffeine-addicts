from ultralytics import YOLO
import cv2
from sender import send_alert  # import the MQTT sender function
import tkinter as tk
from PIL import Image, ImageTk, Image
import threading
import time

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # <-- your trained model file


class CleanSecurityUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Security Camera")
        self.root.geometry("900x700")
        self.root.configure(bg="white")

        # ---- Title ----
        header = tk.Label(
            root,
            text="Live Camera Feed",
            font=("Arial", 26, "bold"),
            bg="white",
            fg="black"
        )
        header.pack(pady=15)

        # ---- Video Area ----
        self.video_frame = tk.Frame(root, bg="white", width=800, height=500)
        self.video_frame.pack(pady=10)

        self.video_label = tk.Label(self.video_frame, bg="white")
        self.video_label.pack()

        # ---- "Screenshotted" Notification ----
        self.ss_label = tk.Label(
            root,
            text="",
            font=("Arial", 16),
            bg="white",
            fg="green"
        )
        self.ss_label.pack()

        # ---- Buttons ----
        button_frame = tk.Frame(root, bg="white")
        button_frame.pack(pady=20)

        self.start_btn = tk.Button(
            button_frame,
            text="Show Camera Feed",
            font=("Arial", 14),
            bg="#3498db",
            fg="black",
            width=18,
            height=2,
            command=self.start_camera
        )
        self.start_btn.pack(side="left", padx=20)

        self.screenshot_btn = tk.Button(
            button_frame,
            text="Screenshot",
            font=("Arial", 14),
            bg="#2ECC71",
            fg="black",
            width=14,
            height=2,
            command=self.take_screenshot
        )
        self.screenshot_btn.pack(side="left", padx=20)

        self.stop_btn = tk.Button(
            button_frame,
            text="Stop",
            font=("Arial", 14),
            bg="#E74C3C",
            fg="black",
            width=12,
            height=2,
            command=self.stop_camera
        )
        self.stop_btn.pack(side="left", padx=20)

        self.running = False
        self.cap = None
        self.last_frame = None  # store last YOLO frame for screenshot

    def start_camera(self):
        if not self.running:
            self.running = True

            # Open the security camera (0 = default webcam, 1 or 2 = external)
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                return

            print("\u2705 YOLO running...")

            threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        self.ss_label.config(text="")

    def take_screenshot(self):
        if self.last_frame is not None:
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.last_frame)

            # Show "Screenshotted" message
            self.ss_label.config(text="Screenshotted ✔️")

            # Hide after 2 seconds
            self.root.after(2000, lambda: self.ss_label.config(text=""))

    def camera_loop(self):
        target_w = 800
        target_h = 500

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Run YOLOv8 inference
            results = model(frame, stream=True)

            for r in results:
                annotated_frame = r.plot()

                # Save last frame for screenshot
                self.last_frame = annotated_frame.copy()

                # ---- KEEP ASPECT RATIO ----
                h, w, _ = annotated_frame.shape
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                resized = cv2.resize(annotated_frame, (new_w, new_h))

                # ---- Center inside 800×500 ----
                canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
                pil_frame = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

                x = (target_w - new_w) // 2
                y = (target_h - new_h) // 2
                canvas.paste(pil_frame, (x, y))

                imgtk = ImageTk.PhotoImage(canvas)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)


# ---- Run App ----
root = tk.Tk()
app = CleanSecurityUI(root)
root.mainloop()
