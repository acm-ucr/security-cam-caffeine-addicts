from ultralytics import YOLO    # model inference 
import cv2                      # video handling
from sender import send_alert   # TODO: import the MQTT sender function (not implemented)
import customtkinter as ctk     # prettier Tkinter UI with dark mode
from PIL import Image, ImageTk  # convert OpenCV frames to Tk friendly images
import threading                # run camera loop in BG so UI doesn't freeze
import time                     # timestamps for filenames
import os                       # filesystem
from collections import deque   # rolling buffer for last 30 seconds of frames
import platform                 # open clips in OS default media player
import subprocess               # open clips using mac / linux
import os                       # operating system

# Directory list
SCREENSHOT_DIR = "screenshots"
CLIP_DIR = "clips"
os.makedirs(SCREENSHOT_DIR, exist_ok=True) # creates screenshot directory
os.makedirs(CLIP_DIR, exist_ok=True)       # creates clip directory

# CustomTkinter global appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_BG        = "#1e1e1e"   # dark gray background
CARD_BG       = "#2b2b2b"   # slightly lighter for cards/frames
HOVER_CARD_BG = "#3a3a3a"   # card color on hover / focus

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # load custom RoboFlow model

class CleanSecurityUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Security Camera")
        self.root.geometry("1000x700")

        # Default target size (fallback if widget size = 0)
        self.target_w = 800
        self.target_h = 600

        # For video clips (rolling buffer of last 30 seconds)
        self.clip_seconds = 30
        self.frame_buffer = deque()  # (timestamp, frame)

        # Holds gallery thumbnails
        self.gallery_thumbs = []

        # Root layout: header + main area
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ---- Title ----
        self.header_label = ctk.CTkLabel(
            root,
            text="Live Camera Feed", 
            font=("Arial", 26, "bold"), 
            text_color="white"
        )
        self.header_label.grid(row=0, column=0, pady=(15, 15), sticky="nwe")

        # ---- Main container (stack: camera view / gallery view) ----
        self.main_container = ctk.CTkFrame(root, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # CAMERA VIEW (default)
        self.camera_view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.camera_view.grid(row=0, column=0, sticky="nsew")

        # GALLERY VIEW (hidden initially)
        self.gallery_view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.gallery_view.grid(row=0, column=0, sticky="nsew")
        self.gallery_view.grid_remove()  # start hidden

        # ---------- Build camera view ----------
        self._build_camera_view()

        # ---------- Build gallery view ----------
        self._build_gallery_view()
        self.running = False
        self.cap = None
        self.last_frame = None
        self.camera_thread = None

    # ============================================================
    # UI BUILDERS
    # ============================================================
    def _build_camera_view(self):
        cv = self.camera_view

        # Layout for camera view: 4 rows
        cv.grid_rowconfigure(0, weight=1)   # video + preview
        cv.grid_rowconfigure(1, weight=0)   # status
        cv.grid_rowconfigure(2, weight=0)   # detections
        cv.grid_rowconfigure(3, weight=0)   # buttons
        cv.grid_columnconfigure(0, weight=3)
        cv.grid_columnconfigure(1, weight=1)

        # ---- Video Area (left) ----
        self.video_frame = ctk.CTkFrame(cv, fg_color=CARD_BG, corner_radius=20)
        self.video_frame.grid(row=0, column=0,padx=20, pady=0,sticky="nsew")

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=0, pady=0)

        # ---- Right: Screenshot Preview ----
        self.preview_frame = ctk.CTkFrame(
            cv,
            fg_color=CARD_BG,
            corner_radius=20
        )

        self.preview_frame.grid(
            row=0, 
            column=1,
            padx=(0, 20), 
            pady=0,sticky="nsew"
        )

        preview_title = ctk.CTkLabel(
            self.preview_frame,
            text="Last Screenshot",
            font=("Arial", 16, "bold")
        )
        preview_title.pack(pady=(10, 5))

        self.ss_preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="No screenshot yet",
        )
        self.ss_preview_label.pack(padx=10, pady=(5, 10))

        # ---- Status / Notifications ----
        self.ss_label = ctk.CTkLabel(
            cv,
            text="",
            font=("Arial", 16),
            text_color="green"
        )
        self.ss_label.grid(row=1, column=0, columnspan=2, pady=(2, 0))

        # ---- Detections Text ----
        self.detections_label = ctk.CTkLabel(
            cv,
            text="Detections: (none)",
            font=("Arial", 14),
            text_color="white"
        )
        self.detections_label.grid(row=2, column=0, columnspan=2, pady=(0, 5))

        # ---- Buttons ----
        button_frame = ctk.CTkFrame(cv, fg_color="transparent")
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 20))

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

        self.gallery_btn = ctk.CTkButton(
            button_frame,
            text="Open Gallery",
            font=("Arial", 14),
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            text_color="white",
            corner_radius=30,
            width=150,
            height=40,
            command=self.show_gallery_view
        )
        self.gallery_btn.pack(side="left", padx=15)

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

    def _build_gallery_view(self):
        gv = self.gallery_view

        gv.grid_rowconfigure(1, weight=1)
        gv.grid_columnconfigure(0, weight=1)

        # Header (inside gallery view)
        header_frame = ctk.CTkFrame(gv, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        title_label = ctk.CTkLabel(
            header_frame,
            text="Gallery (Screenshots & Clips)",
            font=("Arial", 20, "bold")
        )
        title_label.pack(side="left", padx=10)

        back_btn = ctk.CTkButton(
            header_frame,
            text="Back to Camera",
            fg_color="#7f8c8d",
            hover_color="#636e72",
            text_color="white",
            corner_radius=20,
            width=140,
            command=self.show_camera_view
        )
        back_btn.pack(side="right", padx=10)

        # Scrollable gallery content
        self.gallery_scroll = ctk.CTkScrollableFrame(gv, fg_color=APP_BG)
        self.gallery_scroll.grid(
            row=1, column=0,
            sticky="nsew",
            padx=10, pady=(0, 10)
        )

    # ============================================================
    # VIEW SWITCHING
    # ============================================================
    def show_gallery_view(self):
        self.header_label.configure(text="Gallery")
        self.camera_view.grid_remove()
        self.gallery_view.grid()
        self.populate_gallery(self.gallery_scroll)

    def show_camera_view(self):
        self.header_label.configure(text="Live Camera Feed")
        self.gallery_view.grid_remove()
        self.camera_view.grid()

    # ============================================================
    # CAMERA CONTROLS
    # ============================================================
    def start_camera(self):
        if self.running:
            return

        # Open the security camera (0 = default webcam, 1 or 2 = external)
        self.cap = cv2.VideoCapture(0)
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
        self.running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)

        self.video_label.configure(image=None)
        self.video_label.image = None
        self.ss_label.configure(text="")
        self.detections_label.configure(text="Detections: (stopped)")

    # ============================================================
    # SCREENSHOT
    # ============================================================
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

            # Small preview
            preview_size = 160
            h, w, _ = self.last_frame.shape
            scale = min(preview_size / w, preview_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(self.last_frame, (new_w, new_h))
            canvas = Image.new("RGB", (preview_size, preview_size), (30, 30, 30))
            pil_resized = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            x = (preview_size - new_w) // 2
            y = (preview_size - new_h) // 2
            canvas.paste(pil_resized, (x, y))

            tk_img = ImageTk.PhotoImage(canvas)
            self.ss_preview_label.configure(image=tk_img, text="")
            self.ss_preview_label.image = tk_img
        else:
            print(f"[Screenshot] Failed to save to {filename}")
            self.ss_label.configure(text="Screenshot failed ❌")

        self.root.after(2000, lambda: self.ss_label.configure(text=""))

    # ============================================================
    # CLIPS: SAVE + PLAY
    # ============================================================
    def clip_last_30s(self):
        if not self.frame_buffer:
            print("[Clip] No frames in buffer yet.")
            self.ss_label.configure(text="No frames to clip ❌")
            self.root.after(2000, lambda: self.ss_label.configure(text=""))
            return

        frames = [f for (_, f) in self.frame_buffer]
        h, w, _ = frames[0].shape

        fps_est = max(1, int(len(frames) / self.clip_seconds))

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
        self.root.after(2000, lambda: self.ss_label.configure(text=""))


    # ============================================================
    # OPEN WITH MEDIA PLAYER
    # ============================================================
    def open_with_system_player(self, path):
        """Open video using OS default media player."""
        if not os.path.exists(path):
            print("File not found:", path)
            return

        system = platform.system()

        if system == "Darwin":     # macOS
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:                     # Linux / Other
            subprocess.Popen(["xdg-open", path])

    # ============================================================
    # OPEN FULLSCREEN - ***NOT USED***
    # ============================================================
    def open_clip_fullscreen(self, path: str):
        """Play clip fullscreen in a new CustomTkinter window."""
        if not os.path.exists(path):
            print(f"[Clip] File not found: {path}")
            return

        player = ctk.CTkToplevel(self.root)
        player.title(os.path.basename(path))

        sw = player.winfo_screenwidth()
        sh = player.winfo_screenheight()
        player.geometry(f"{sw}x{sh}+0+0")
        player.attributes("-fullscreen", True)
        player.bind("<Escape>", lambda e: player.destroy())

        video_label = ctk.CTkLabel(player, text="")
        video_label.pack(expand=True, fill="both")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[Clip] Could not open {path}")
            player.destroy()
            return

        def update_frame():
            if not cap.isOpened():
                player.destroy()
                return

            ret, frame = cap.read()
            if not ret:
                cap.release()
                player.destroy()
                return

            h, w, _ = frame.shape
            max_w, max_h = sw, sh
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
            pil_frame = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            x = (max_w - new_w) // 2
            y = (max_h - new_h) // 2
            canvas.paste(pil_frame, (x, y))

            tk_img = ImageTk.PhotoImage(canvas)
            video_label.configure(image=tk_img)
            video_label.image = tk_img

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            delay = int(1000 / fps)
            player.after(delay, update_frame)

        def on_close():
            if cap.isOpened():
                cap.release()
            player.destroy()

        player.protocol("WM_DELETE_WINDOW", lambda: on_close())
        update_frame()

    # ============================================================
    # FULLSCREEN IMAGE - ***NOT USED***
    # ============================================================
    def open_fullscreen_image(self, path: str):
        """Open a screenshot fullscreen."""
        if not os.path.exists(path):
            print(f"[Image] File not found: {path}")
            return

        viewer = ctk.CTkToplevel(self.root)
        viewer.title(os.path.basename(path))

        sw = viewer.winfo_screenwidth()
        sh = viewer.winfo_screenheight()
        viewer.geometry(f"{sw}x{sh}+0+0")
        viewer.attributes("-fullscreen", True)
        viewer.bind("<Escape>", lambda e: viewer.destroy())

        label = ctk.CTkLabel(viewer, text="")
        label.pack(expand=True, fill="both")

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            label.configure(text="Failed to load image")
            return

        h, w, _ = img_bgr.shape
        scale = min(sw / w, sh / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img_bgr, (new_w, new_h))
        canvas = Image.new("RGB", (sw, sh), (0, 0, 0))
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        x = (sw - new_w) // 2
        y = (sh - new_h) // 2
        canvas.paste(pil_img, (x, y))

        tk_img = ImageTk.PhotoImage(canvas)
        label.configure(image=tk_img)
        label.image = tk_img

    # ============================================================
    # GALLERY (INSIDE MAIN WINDOW)
    # ============================================================
    def populate_gallery(self, parent):
        """Populate gallery with both screenshots and clips."""
        # Clear existing
        for child in parent.winfo_children():
            child.destroy()

        items = []

        # Screenshots
        for fname in os.listdir(SCREENSHOT_DIR):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(SCREENSHOT_DIR, fname)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    mtime = 0
                items.append({
                    "type": "screenshot",
                    "name": fname,
                    "path": path,
                    "mtime": mtime
                })

        # Clips
        for fname in os.listdir(CLIP_DIR):
            if fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                path = os.path.join(CLIP_DIR, fname)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    mtime = 0
                items.append({
                    "type": "clip",
                    "name": fname,
                    "path": path,
                    "mtime": mtime
                })

        # Newest first
        items.sort(key=lambda x: x["mtime"], reverse=True)

        if not items:
            no_label = ctk.CTkLabel(parent, text="No screenshots or clips yet.")
            no_label.pack(pady=20)
            return

        self.gallery_thumbs.clear()

        for item in items:
            media_path = item["path"]

            # Outer card: add border + more vertical spacing between items
            outer = ctk.CTkFrame(
                parent,
                fg_color=CARD_BG,
                corner_radius=10,
                border_width=1,
                border_color="#444444",
            )
            outer.pack(fill="x", padx=12, pady=10)

            inner = ctk.CTkFrame(outer, fg_color=CARD_BG)
            inner.pack(fill="x", padx=10, pady=10)

            # Thumbnail
            if item["type"] == "screenshot":
                thumb_label = self._build_screenshot_thumb(inner, media_path)
            else:
                thumb_label = self._build_clip_thumb(inner, media_path)

            thumb_label.pack(side="left", padx=(0, 15))

            # Info label
            info_text = f"{item['name']} ({item['type']})"
            info_label = ctk.CTkLabel(inner, text=info_text, anchor="w")
            info_label.pack(side="left", expand=True, fill="x")

            # Buttons frame (only delete)
            btn_frame = ctk.CTkFrame(inner, fg_color=CARD_BG)
            btn_frame.pack(side="right")

            delete_btn = ctk.CTkButton(
                btn_frame,
                text="Delete",
                width=70,
                fg_color="#e74c3c",
                hover_color="#c0392b",
                bg_color="transparent",
                command=lambda p=media_path, parent=parent: self.delete_media_and_refresh(p, parent)
            )
            delete_btn.pack(side="top", padx=5, pady=2)

            # ---- Hover effect: change card color on enter/leave ----
            def on_enter(e, o=outer, i=inner):
                o.configure(fg_color=HOVER_CARD_BG)
                i.configure(fg_color=HOVER_CARD_BG)

            def on_leave(e, o=outer, i=inner):
                o.configure(fg_color=CARD_BG)
                i.configure(fg_color=CARD_BG)

            for w in (outer, inner, thumb_label, info_label):
                w.bind("<Enter>", on_enter)
                w.bind("<Leave>", on_leave)

            # ---- Make the WHOLE card open with system player on double-click ----
            def bind_open(widget, p=media_path):
                widget.bind(
                    "<Double-Button-1>",
                    lambda e, path=p: self.open_with_system_player(path)
                )

            for w in (outer, inner, thumb_label, info_label):
                bind_open(w)

    # ============================================================
    # SCREENSHOT THUMBNAIL 
    # ============================================================
    def _build_screenshot_thumb(self, parent, path):
        thumb_size = 80
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                raise ValueError("Failed to load image")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((thumb_size, thumb_size))

            tk_thumb = ImageTk.PhotoImage(pil_img)
            self.gallery_thumbs.append(tk_thumb)

            lbl = ctk.CTkLabel(parent, image=tk_thumb, text="")
            return lbl
        except Exception as e:
            print(f"[Gallery] Error loading screenshot {path}: {e}")
            return ctk.CTkLabel(parent, text="Img")

    # ============================================================
    # CLIP THUMBNAIL 
    # ============================================================
    def _build_clip_thumb(self, parent, path):
        thumb_size = 80
        try:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise ValueError("Failed to read frame")

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((thumb_size, thumb_size))

            tk_thumb = ImageTk.PhotoImage(pil_img)
            self.gallery_thumbs.append(tk_thumb)

            lbl = ctk.CTkLabel(parent, image=tk_thumb, text="")
            return lbl
        except Exception as e:
            print(f"[Gallery] Error loading clip thumb {path}: {e}")
            return ctk.CTkLabel(parent, text="🎥", font=("Arial", 24))

    # ============================================================
    # DELETE MEDIA 
    # ============================================================
    def delete_media_and_refresh(self, path, parent):
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[Gallery] Deleted {path}")
        except Exception as e:
            print(f"[Gallery] Failed to delete {path}: {e}")

        self.populate_gallery(parent)

    # ============================================================
    # UPDATE UI (helper for camera loop)
    # ============================================================
    def update_ui(self, imgtk, detections_text):
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk
        self.detections_label.configure(text=detections_text)

    # ============================================================
    # CAMERA LOOP
    # ============================================================
    def camera_loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Run YOLOv8 inference
            results = model(frame)[0]
            annotated_frame = results.plot()
            self.last_frame = annotated_frame.copy()

            # Add frame to clip buffer
            now = time.time()
            self.frame_buffer.append((now, annotated_frame.copy()))
            cutoff = now - self.clip_seconds
            while self.frame_buffer and self.frame_buffer[0][0] < cutoff:
                self.frame_buffer.popleft()

            # Detections
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

            # Fit to video frame area
            frame_w = self.video_frame.winfo_width() or self.target_w
            frame_h = self.video_frame.winfo_height() or self.target_h

            h, w, _ = annotated_frame.shape
            scale = min(frame_w / w, frame_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(annotated_frame, (new_w, new_h))
            canvas = Image.new("RGB", (frame_w, frame_h), (30, 30, 30))
            pil_frame = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            x = (frame_w - new_w) // 2
            y = (frame_h - new_h) // 2
            canvas.paste(pil_frame, (x, y))

            imgtk = ImageTk.PhotoImage(canvas)

            self.root.after(
                0,
                lambda img=imgtk, text=detections_text: self.update_ui(img, text)
            )

            time.sleep(0.01)
# ============================================================
# END of CleanSecurityUI class
# ============================================================

# ---- Run App ----
if __name__ == "__main__":
    root = ctk.CTk()
    root.configure(fg_color=APP_BG)
    app = CleanSecurityUI(root)
    root.mainloop()
