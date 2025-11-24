from ultralytics import YOLO
import cv2
from sender import send_alert  # import the MQTT sender function
import customtkinter as ctk
from PIL import Image, ImageTk, Image
import threading
import time
import os
import platform
import subprocess
from collections import deque

# Directories
SCREENSHOT_DIR = "screenshots"
CLIP_DIR = "clips"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)

# CustomTkinter global appearance
ctk.set_appearance_mode("light")         # or "dark"
ctk.set_default_color_theme("blue")      # built-in theme

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # <-- your trained model file


class CleanSecurityUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Security Camera")
        self.root.geometry("1000x700")  # slightly wider for preview/gallery

        # Default target size (used if we can't read widget sizes yet)
        self.target_w = 800
        self.target_h = 600

        # For video clips
        self.clip_seconds = 30
        self.frame_buffer = deque()  # will store (timestamp, frame)

        # Layout: 2 columns (video left, preview/gallery right)
        self.root.grid_rowconfigure(1, weight=1)   # middle row (video + side panel)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)

        # ---- Title ----
        header = ctk.CTkLabel(
            root,
            text="Live Camera Feed",
            font=("Arial", 26, "bold"),
            text_color="black"
        )
        header.grid(row=0, column=0, columnspan=2, pady=15, sticky="nwe")

        # ---- Video Area (left) ----
        self.video_frame = ctk.CTkFrame(
            root,
            fg_color="white",
            corner_radius=20
        )
        self.video_frame.grid(
            row=1, column=0,
            padx=20, pady=10,
            sticky="nsew"
        )

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # ---- Right-side tabs: Preview & Gallery ----
        self.side_tabs = ctk.CTkTabview(root)
        self.side_tabs.grid(
            row=1, column=1,
            padx=(0, 20), pady=10,
            sticky="nsew"
        )

        preview_tab = self.side_tabs.add("Preview")
        gallery_tab = self.side_tabs.add("Gallery")

        clips_tab = self.side_tabs.add("Clips")

        self.clips_scroll = ctk.CTkScrollableFrame(
            clips_tab,
            label_text="Saved Clips"
        )
        self.clips_scroll.pack(expand=True, fill="both", padx=10, pady=10)

        # ---- Preview tab content ----
        preview_title = ctk.CTkLabel(
            preview_tab,
            text="Last Screenshot",
            font=("Arial", 16, "bold")
        )
        preview_title.pack(pady=10)

        self.ss_preview_label = ctk.CTkLabel(
            preview_tab,
            text="No screenshot yet",
        )
        self.ss_preview_label.pack(padx=10, pady=10)

        # ---- Gallery tab content ----
        self.gallery_scroll = ctk.CTkScrollableFrame(
            gallery_tab,
            label_text="Screenshots"
        )
        self.gallery_scroll.pack(expand=True, fill="both", padx=10, pady=(0, 10))

        self.gallery_back_btn = ctk.CTkButton(
            gallery_tab,
            text="Back to Preview",
            fg_color="#7f8c8d",
            hover_color="#636e72",
            text_color="white",
            corner_radius=20,
            command=self.back_to_preview
        )
        self.gallery_back_btn.pack(pady=(0, 10))

        # ---- "Screenshotted" Notification ----
        self.ss_label = ctk.CTkLabel(
            root,
            text="",
            font=("Arial", 16),
            text_color="green"
        )
        self.ss_label.grid(row=2, column=0, columnspan=2, pady=(5, 0))

        # ---- Detections Text ----
        self.detections_label = ctk.CTkLabel(
            root,
            text="Detections: (none)",
            font=("Arial", 14),
            text_color="black"
        )
        self.detections_label.grid(row=3, column=0, columnspan=2, pady=(0, 10))

        # ---- Buttons ----
        button_frame = ctk.CTkFrame(root, fg_color="transparent")
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)

        self.start_btn = ctk.CTkButton(
            button_frame,
            text="Show Camera Feed",
            font=("Arial", 14),
            fg_color="#3498db",
            hover_color="#2980b9",
            text_color="white",
            corner_radius=30,
            width=180,
            height=40,
            command=self.start_camera
        )
        self.start_btn.pack(side="left", padx=15)

        self.screenshot_btn = ctk.CTkButton(
            button_frame,
            text="Screenshot",
            font=("Arial", 14),
            fg_color="#2ECC71",
            hover_color="#27ae60",
            text_color="white",
            corner_radius=30,
            width=140,
            height=40,
            command=self.take_screenshot
        )
        self.screenshot_btn.pack(side="left", padx=15)

        # Clip last 30s button
        self.clip_btn = ctk.CTkButton(
            button_frame,
            text="Clip Last 30s",
            font=("Arial", 14),
            fg_color="#f39c12",
            hover_color="#d68910",
            text_color="white",
            corner_radius=30,
            width=150,
            height=40,
            command=self.clip_last_30s
        )
        self.clip_btn.pack(side="left", padx=15)

        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="Stop",
            font=("Arial", 14),
            fg_color="#E74C3C",
            hover_color="#c0392b",
            text_color="white",
            corner_radius=30,
            width=120,
            height=40,
            command=self.stop_camera
        )
        self.stop_btn.pack(side="left", padx=15)

        self.running = False
        self.cap = None
        self.last_frame = None
        self.camera_thread = None

        # store preview image reference
        self.ss_preview_image = None
        self.refresh_gallery()
        self.refresh_clips()

    # ---------- Camera controls ----------
    def start_camera(self):
        if self.running:
            return

        # Open the security camera (0 = default webcam, 1 or 2 = external)
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            self.cap = None
            return

        print("YOLO running...")
        self.running = True

        self.camera_thread = threading.Thread(
            target=self.camera_loop,
            daemon=True
        )
        self.camera_thread.start()

    def stop_camera(self):
        # tells loop to stop
        self.running = False

        # release camera if open
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # wait for thread to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)

        # clear UI
        self.video_label.configure(image=None)
        self.video_label.image = None
        self.ss_label.configure(text="")
        self.detections_label.configure(text="Detections: (stopped)")

    # ---------- Screenshot logic ----------

    def take_screenshot(self):
        if self.last_frame is None:
            print("[Screenshot] No frame available yet. Make sure the camera feed is running.")
            self.ss_label.configure(text="No frame yet ❌")
            self.root.after(2000, lambda: self.ss_label.configure(text=""))
            return

        filename = os.path.join(
            SCREENSHOT_DIR,
            f"screenshot_{int(time.time())}.jpg"
        )

        success = cv2.imwrite(filename, self.last_frame)

        if success:
            print(f"[Screenshot] Saved to {filename}")
            self.ss_label.configure(text=f"Screenshotted ✔️\n{os.path.basename(filename)}")

            # ---- Update small square preview on the Preview tab ----
            preview_size = 200
            h, w, _ = self.last_frame.shape
            scale = min(preview_size / w, preview_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(self.last_frame, (new_w, new_h))
            canvas = Image.new("RGB", (preview_size, preview_size), (240, 240, 240))
            pil_resized = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            x = (preview_size - new_w) // 2
            y = (preview_size - new_h) // 2
            canvas.paste(pil_resized, (x, y))

            tk_img = ImageTk.PhotoImage(canvas)
            self.ss_preview_label.configure(image=tk_img, text="")
            self.ss_preview_label.image = tk_img
            self.ss_preview_image = tk_img

            # refresh gallery thumbnails if we're on the Gallery tab
            self.refresh_gallery()
        else:
            print(f"[Screenshot] Failed to save to {filename}")
            self.ss_label.configure(text="Screenshot failed ❌")

        # Hide text after 2 seconds
        self.root.after(2000, lambda: self.ss_label.configure(text=""))

    # ---------- Gallery logic (same window) ----------

    def back_to_preview(self):
        self.side_tabs.set("Preview")

    def refresh_gallery(self):
        # Clear old children
        for child in self.gallery_scroll.winfo_children():
            child.destroy()

        # Keep references for thumbnails
        if not hasattr(self, "gallery_thumbs"):
            self.gallery_thumbs = []
        self.gallery_thumbs.clear()

        files = sorted(
            f for f in os.listdir(SCREENSHOT_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        if not files:
            no_label = ctk.CTkLabel(self.gallery_scroll, text="No screenshots yet.")
            no_label.pack(pady=20)
            return

        thumb_size = 160
        cols = 2
        row = 0
        col = 0

        for fname in files:
            path = os.path.join(SCREENSHOT_DIR, fname)

            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((thumb_size, thumb_size))

            tk_thumb = ImageTk.PhotoImage(pil_img)
            self.gallery_thumbs.append(tk_thumb)

            frame = ctk.CTkFrame(self.gallery_scroll, fg_color="white", corner_radius=10)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="n")

            img_label = ctk.CTkLabel(frame, image=tk_thumb, text="")
            img_label.pack(padx=5, pady=5)

            name_label = ctk.CTkLabel(frame, text=fname, font=("Arial", 10))
            name_label.pack(pady=(0, 5))

            col += 1
            if col >= cols:
                col = 0
                row += 1

    # ---------- Clip last 30 seconds of video ----------

    def clip_last_30s(self):
        if not self.frame_buffer:
            print("[Clip] No frames in buffer yet.")
            self.ss_label.configure(text="No frames to clip ❌")
            self.root.after(2000, lambda: self.ss_label.configure(text=""))
            return

        # Take the frames currently in the buffer
        frames = [f for (_, f) in self.frame_buffer]
        h, w, _ = frames[0].shape

        # Estimate FPS from how many frames we've collected
        fps_est = max(1, int(len(frames) / self.clip_seconds))  # rough estimate

        filename = os.path.join(
            CLIP_DIR,
            f"clip_{int(time.time())}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, fps_est, (w, h))

        for f in frames:
            out.write(f)

        out.release()

        print(f"[Clip] Saved last ~{self.clip_seconds}s to {filename}")
        self.ss_label.configure(text=f"Saved clip: {os.path.basename(filename)}")

        # Refresh Clips tab list
        self.refresh_clips()

        self.root.after(2000, lambda: self.ss_label.configure(text=""))


    def open_clip(self, path: str):
        """Open the given video file with the OS default player."""
        if not os.path.exists(path):
            print(f"[Clip] File not found: {path}")
            return

        system = platform.system()
        try:
            if system == "Darwin":          # macOS
                subprocess.Popen(["open", path])
            elif system == "Windows":
                os.startfile(path)          # type: ignore[attr-defined]
            else:                           # Linux / others
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            print(f"[Clip] Failed to open {path}: {e}")

    def refresh_clips(self):
        # Clear old clip widgets
        for child in self.clips_scroll.winfo_children():
            child.destroy()

        clips = sorted(
            f for f in os.listdir(CLIP_DIR)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        )

        if not clips:
            label = ctk.CTkLabel(self.clips_scroll, text="No clips saved yet.")
            label.pack(pady=20)
            return

        for fname in clips:
            path = os.path.join(CLIP_DIR, fname)

            row_frame = ctk.CTkFrame(self.clips_scroll, fg_color="white", corner_radius=10)
            row_frame.pack(fill="x", padx=10, pady=5)

            name_label = ctk.CTkLabel(
                row_frame,
                text=fname,
                anchor="w"
            )
            name_label.pack(side="left", padx=10, pady=10, expand=True, fill="x")

            play_btn = ctk.CTkButton(
                row_frame,
                text="Play",
                width=80,
                fg_color="#2ecc71",
                hover_color="#27ae60",
                command=lambda p=path: self.open_clip(p)
            )
            play_btn.pack(side="right", padx=10, pady=10)

    # ---------- Camera loop ----------
    def camera_loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # ---- Run YOLOv8 inference (single result) ----
            results = model(frame)[0]  # get first result
            annotated_frame = results.plot()
            self.last_frame = annotated_frame.copy()

            # Add to rolling buffer for clips
            now = time.time()
            self.frame_buffer.append((now, annotated_frame.copy()))
            cutoff = now - self.clip_seconds
            # remove frames older than clip_seconds
            while self.frame_buffer and self.frame_buffer[0][0] < cutoff:
                self.frame_buffer.popleft()

            # ---- Collect detection names ----
            all_detected_names = []
            if results.boxes is not None and len(results.boxes) > 0:
                cls_ids = results.boxes.cls.tolist()
                for cls_id in cls_ids:
                    name = model.names[int(cls_id)]
                    all_detected_names.append(name)

            if all_detected_names:
                unique_names = sorted(set(all_detected_names))
                detections_text = "Detections: " + ", ".join(unique_names)
            else:
                detections_text = "Detections: (none)"

            # ---- Get current frame size to be responsive ----
            frame_w = self.video_frame.winfo_width() or self.target_w
            frame_h = self.video_frame.winfo_height() or self.target_h

            # ---- Resize and center while keeping aspect ratio ----
            h, w, _ = annotated_frame.shape
            scale = min(frame_w / w, frame_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(annotated_frame, (new_w, new_h))
            canvas = Image.new("RGB", (frame_w, frame_h), (255, 255, 255))
            pil_frame = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            x = (frame_w - new_w) // 2
            y = (frame_h - new_h) // 2
            canvas.paste(pil_frame, (x, y))

            imgtk = ImageTk.PhotoImage(canvas)

            # ---- UI update scheduled on main thread ----
            self.root.after(
                0,
                lambda img=imgtk, text=detections_text: self.update_ui(img, text)
            )

            time.sleep(0.01)  # avoid maxing out CPU

    def update_ui(self, imgtk, detections_text):
        # keep reference to image to prevent GC
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk
        self.detections_label.configure(text=detections_text)


# ---- Run App ----
if __name__ == "__main__":
    root = ctk.CTk()
    app = CleanSecurityUI(root)
    root.mainloop()
