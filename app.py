"""
Max Headroom Digitizer - Desktop Application
Full VTuber streaming application with WebSocket OBS integration
"""
import sys
import os
import cv2
import numpy as np
import time
import json
import threading
import socket
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

# Try to import GUI libraries
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    import websocket
except ImportError:
    try:
        import ws as websocket
    except ImportError:
        websocket = None

VERSION = "3.0.0"

@dataclass
class AppConfig:
    """Application configuration."""
    ws_host = "localhost"
    ws_port = 30000
    camera_index = 0
    target_fps = 30
    resolution = (640, 480)
    smoothing = 0.8
    enable_websocket = True
    enable_obs = False
    obs_ndi_name = "MaxHeadroom"
    test_mode = False
    glitch_intensity = 0.15
    hologram_enabled = True

@dataclass
class FaceData:
    """Current face tracking data."""
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_pose: Dict[str, List[float]] = field(default_factory=lambda: {"rotation": [0, 0, 0], "translation": [0, 0, 1.5]})
    landmarks: List[Tuple[int, int]] = field(default_factory=list)
    timestamp: float = 0.0

class MaxHeadroomApp:
    """Main desktop application."""
    
    WINDOW_NAME = "Max Headroom Digitizer v3.0"
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        
        # Core components
        self.cap = None
        self.cascade = None
        self.ws = None
        self.ws_connected = False
        
        # State
        self.running = False
        self.frame_count = 0
        self.sent_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Tracking data
        self.current_data = FaceData()
        self.smoothed_data = FaceData()
        
        # Smoothing buffers
        self.blendshape_buffer = {}
        
        # OBS
        self.obs_socket = None
        self.obs_connected = False
        
        # GUI
        self.root = None
        self.canvas = None
        self.info_label = None
    
    def init(self) -> bool:
        """Initialize all components."""
        print("[Init] Loading face detector...")
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.cascade.empty():
            print("[Init] ERROR: Failed to load detector")
            return False
        print("[Init] Face detector loaded")
        
        if not self.config.test_mode:
            print(f"[Init] Opening camera {self.config.camera_index}...")
            self.cap = cv2.VideoCapture(self.config.camera_index)
            
            if not self.cap.isOpened():
                print("[Init] Camera unavailable - TEST MODE")
                self.config.test_mode = True
            else:
                w, h = self.config.resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                print(f"[Init] Camera: {w}x{h}")
        
        return True
    
    def connect_websocket(self) -> bool:
        """Connect to WebSocket server."""
        if not self.config.enable_websocket or not websocket:
            return False
        
        try:
            print(f"[WS] Connecting to {self.config.ws_host}:{self.config.ws_port}...")
            self.ws = websocket.create_connection(
                f"ws://{self.config.ws_host}:{self.config.ws_port}",
                timeout=3
            )
            self.ws_connected = True
            print("[WS] Connected!")
            return True
        except Exception as e:
            print(f"[WS] Connection failed: {e}")
            self.ws_connected = False
            return False
    
    def connect_obs(self) -> bool:
        """Connect to OBS via WebSocket."""
        if not self.config.enable_obs:
            return False
        
        try:
            print(f"[OBS] Connecting to OBS on port {5556}...")
            self.obs_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.obs_socket.connect(("localhost", 5556))
            self.obs_connected = True
            print("[OBS] Connected!")
            return True
        except Exception as e:
            print(f"[OBS] Connection failed: {e}")
            return False
    
    def send_websocket(self, data: Dict) -> bool:
        """Send data via WebSocket."""
        if not self.ws or not self.ws_connected:
            return False
        try:
            self.ws.send(json.dumps(data))
            self.sent_count += 1
            return True
        except:
            self.ws = None
            self.ws_connected = False
            return False
    
    def send_obs(self, data: Dict) -> bool:
        """Send data to OBS."""
        if not self.obs_socket or not self.obs_connected:
            return False
        try:
            self.obs_socket.send(json.dumps(data).encode())
            return True
        except:
            self.obs_socket = None
            self.obs_connected = False
            return False
    
    def detect_face(self, gray):
        """Detect face in grayscale image."""
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        return tuple(faces[0]) if len(faces) > 0 else None
    
    def calculate_blendshapes(self, face_rect: Tuple, time: float) -> Dict[str, float]:
        """Calculate ARKit blendshapes from face bounding box."""
        if face_rect is None:
            return self._test_blendshapes(time)
        
        x, y, w, h = face_rect
        
        mouth_open = min(1.0, (h / w) * 0.5)
        smile = 0.3 + 0.2 * np.sin(time * 3)
        blink = 0.0
        
        blends = {
            "jawOpen": mouth_open,
            "mouthSmile_L": smile,
            "mouthSmile_R": smile,
            "eyeBlink_L": blink,
            "eyeBlink_R": blink,
            "browUp_L": 0.1,
            "browUp_R": 0.1,
            "cheekPuff": smile * 0.5,
            "mouthClose": 1.0 - mouth_open,
            "mouthFunnel": mouth_open * 0.2,
            "mouthPucker": mouth_open * 0.3,
        }
        
        return blends
    
    def _test_blendshapes(self, t: float) -> Dict[str, float]:
        """Generate test blendshapes."""
        return {
            "jawOpen": 0.2 + 0.15 * np.sin(t * 2),
            "mouthSmile_L": 0.3 + 0.1 * np.sin(t * 3),
            "mouthSmile_R": 0.3 + 0.1 * np.sin(t * 3 + 0.5),
            "eyeBlink_L": 0.0,
            "eyeBlink_R": 0.0,
            "browUp_L": 0.1 * (1 + np.sin(t * 1.5)),
            "browUp_R": 0.1 * (1 + np.sin(t * 1.5 + 0.3)),
            "cheekPuff": 0.2 + 0.1 * np.sin(t * 2.5),
            "mouthClose": 0.8 - 0.1 * np.sin(t * 2),
            "mouthFunnel": 0.1,
            "mouthPucker": 0.1,
        }
    
    def calculate_pose(self, face_rect: Tuple, frame_shape: Tuple) -> Dict[str, List[float]]:
        """Calculate head pose."""
        if face_rect is None:
            t = time.time()
            return {
                "rotation": [5 * np.sin(t * 0.5), 10 * np.sin(t * 0.3), 0],
                "translation": [0.1 * np.sin(t * 0.5), 0.05 * np.sin(t * 0.7), 1.5]
            }
        
        x, y, w, h = face_rect
        fh, fw = frame_shape
        
        center_x = x + w / 2
        center_y = y + h / 2
        
        norm_x = (center_x - fw / 2) / fw * 2
        norm_y = (center_y - fh / 2) / fh * 2
        
        return {
            "rotation": [norm_y * 15, norm_x * 20, 0],
            "translation": [norm_x, -norm_y, max(0.5, min(3.0, 200 / w))]
        }
    
    def smooth_blendshapes(self, blends: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing."""
        s = self.config.smoothing
        smoothed = {}
        
        for name, value in blends.items():
            if name in self.blendshape_buffer:
                prev = self.blendshape_buffer[name]
                smoothed[name] = prev * s + value * (1 - s)
            else:
                smoothed[name] = value
            self.blendshape_buffer[name] = smoothed[name]
        
        return smoothed
    
    def process_frame(self, frame) -> FaceData:
        """Process single frame."""
        t = time.time()
        
        if self.config.test_mode or frame is None:
            blends = self._test_blendshapes(t)
            pose = self.calculate_pose(None, (480, 640))
            return FaceData(
                blendshapes=blends,
                head_pose=pose,
                landmarks=[],
                timestamp=t
            )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = self.detect_face(gray)
        
        blends = self.calculate_blendshapes(face_rect, t)
        blends = self.smooth_blendshapes(blends)
        
        pose = self.calculate_pose(face_rect, gray.shape)
        
        return FaceData(
            blendshapes=blends,
            head_pose=pose,
            landmarks=[],
            timestamp=t
        )
    
    def draw_hologram_overlay(self, frame, data: FaceData):
        """Draw Max Headroom holographic overlay."""
        h, w = frame.shape[:2]
        
        # Scanlines
        for y in range(0, h, 3):
            cv2.line(frame, (0, y), (w, y), (0, 50, 0), 1)
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 20, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 2), (0, 255, 0), 2)
        
        # Title with glitch effect
        title = "MAX HEADROOM"
        offset = int(np.sin(time.time() * 10) * 3) if self.config.glitch_intensity > 0 else 0
        cv2.putText(frame, title, (15 + offset, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # FPS
        fps_text = f"FPS: {self.fps}"
        cv2.putText(frame, fps_text, (w - 90, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Test mode indicator
        if self.config.test_mode:
            cv2.putText(frame, "TEST MODE", (w//2 - 60, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Blendshape bars
        y_pos = 70
        for name in list(data.blendshapes.keys())[:8]:
            val = data.blendshapes.get(name, 0)
            bar_w = int(val * 120)
            
            # Bar background
            cv2.rectangle(frame, (10, y_pos), (130, y_pos + 18), (30, 30, 30), -1)
            # Bar fill
            cv2.rectangle(frame, (10, y_pos), (10 + bar_w, y_pos + 18), (0, 255, 0), -1)
            # Bar outline
            cv2.rectangle(frame, (10, y_pos), (130, y_pos + 18), (0, 100, 0), 1)
            
            # Label
            cv2.putText(frame, f"{name}: {val:.2f}", (140, y_pos + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            
            y_pos += 22
        
        # Status indicators
        status_y = h - 25
        
        # WebSocket status
        ws_stat = "WS: ON" if self.ws_connected else "WS: OFF"
        ws_color = (0, 255, 0) if self.ws_connected else (0, 0, 255)
        cv2.putText(frame, ws_stat, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ws_color, 2)
        
        # OBS status if enabled
        if self.config.enable_obs:
            obs_stat = "OBS: ON" if self.obs_connected else "OBS: OFF"
            obs_color = (0, 255, 0) if self.obs_connected else (100, 100, 100)
            cv2.putText(frame, obs_stat, (100, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, obs_color, 2)
        
        # Hologram effect on edges
        for i in range(0, w, 50):
            frame[0:5, i:i+3] = frame[0:5, i:i+3] * 0.8 + np.array([0, 30, 0], dtype=np.uint8)
            frame[h-5:h, i:i+3] = frame[h-5:h, i:i+3] * 0.8 + np.array([0, 30, 0], dtype=np.uint8)
        
        # RGB split effect
        if self.config.glitch_intensity > 0.1:
            split = int(self.config.glitch_intensity * 15)
            if split > 0:
                frame_r = frame.copy()
                frame_b = frame.copy()
                frame[:, split:] = frame_r[:, :-split]
                frame[:, :-split] = frame_b[:, split:]
        
        return frame
    
    def run_gui(self):
        """Run GUI application."""
        if not GUI_AVAILABLE:
            print("ERROR: tkinter not available")
            self.run_cli()
            return
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(self.WINDOW_NAME)
        self.root.geometry("800x700")
        self.root.configure(bg="black")
        
        # Title
        title_label = tk.Label(
            self.root,
            text="MAX HEADROOM DIGITIZER",
            font=("Consolas", 18, "bold"),
            fg="#00FF00",
            bg="black"
        )
        title_label.pack(pady=10)
        
        # Canvas for camera
        canvas_frame = tk.Frame(self.root, bg="black")
        canvas_frame.pack(pady=5)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=640,
            height=480,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Controls
        control_frame = tk.Frame(self.root, bg="black")
        control_frame.pack(pady=10)
        
        # Server config
        tk.Label(control_frame, text="WebSocket:", fg="#00FF00", bg="black").grid(row=0, column=0, padx=5)
        self.ws_host_entry = tk.Entry(control_frame, width=12, fg="#00FF00", bg="#001100")
        self.ws_host_entry.insert(0, self.config.ws_host)
        self.ws_host_entry.grid(row=0, column=1, padx=5)
        
        self.ws_port_entry = tk.Entry(control_frame, width=6, fg="#00FF00", bg="#001100")
        self.ws_port_entry.insert(0, str(self.config.ws_port))
        self.ws_port_entry.grid(row=0, column=2, padx=5)
        
        # Buttons
        self.connect_btn = tk.Button(
            control_frame,
            text="CONNECT",
            fg="#00FF00",
            bg="#001100",
            activebackground="#003300",
            command=self.connect_websocket
        )
        self.connect_btn.grid(row=0, column=3, padx=10)
        
        # Status label
        self.info_label = tk.Label(
            self.root,
            text="Ready - Press START to begin",
            font=("Consolas", 10),
            fg="#00FF00",
            bg="black"
        )
        self.info_label.pack(pady=5)
        
        # Start/Stop buttons
        btn_frame = tk.Frame(self.root, bg="black")
        btn_frame.pack(pady=10)
        
        start_btn = tk.Button(
            btn_frame,
            text="START",
            font=("Consolas", 12, "bold"),
            fg="#00FF00",
            bg="#002200",
            activebackground="#004400",
            width=10,
            command=self.start_tracking
        )
        start_btn.pack(side=tk.LEFT, padx=10)
        
        stop_btn = tk.Button(
            btn_frame,
            text="STOP",
            font=("Consolas", 12, "bold"),
            fg="#FF0000",
            bg="#220000",
            activebackground="#440000",
            width=10,
            command=self.stop_tracking
        )
        stop_btn.pack(side=tk.LEFT, padx=10)
        
        # Test mode checkbox
        self.test_var = tk.BooleanVar(value=self.config.test_mode)
        tk.Checkbutton(
            self.root,
            text="Test Mode (No Camera)",
            variable=self.test_var,
            fg="#00FF00",
            bg="black",
            selectcolor="#003300",
            command=self.toggle_test_mode
        ).pack(pady=5)
        
        # Glitch slider
        tk.Label(self.root, text="Hologram Intensity:", fg="#00FF00", bg="black").pack()
        glitch_scale = tk.Scale(
            self.root,
            from_=0,
            to=100,
            variable=tk.IntVar(value=int(self.config.glitch_intensity * 100)),
            orient=tk.HORIZONTAL,
            fg="#00FF00",
            bg="black",
            troughcolor="#002200",
            highlightthickness=0,
            command=self.set_glitch
        )
        glitch_scale.pack(pady=5)
        
        # Start GUI loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update()
        self.root.mainloop()
    
    def toggle_test_mode(self):
        self.config.test_mode = self.test_var.get()
        self.update_status(f"Test mode: {'ON' if self.config.test_mode else 'OFF'}")
    
    def set_glitch(self, value):
        self.config.glitch_intensity = int(value) / 100
    
    def connect_websocket(self):
        self.config.ws_host = self.ws_host_entry.get()
        self.config.ws_port = int(self.ws_port_entry.get())
        
        if self.connect_websocket():
            self.update_status("WebSocket connected!")
        else:
            self.update_status("WebSocket connection failed")
    
    def start_tracking(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._thread.start()
            self.update_status("Tracking started!")
    
    def stop_tracking(self):
        self.running = False
        self.update_status("Tracking stopped")
    
    def update_status(self, text: str):
        if self.info_label:
            self.info_label.config(text=text)
    
    def _tracking_loop(self):
        """Main tracking loop."""
        while self.running:
            frame = None
            
            if self.config.test_mode:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.rectangle(frame, (50, 50), (590, 430), (0, 150, 0), 3)
            else:
                if self.cap:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
            
            # Process frame
            data = self.process_frame(frame)
            self.current_data = data
            
            # Draw overlay
            frame = self.draw_hologram_overlay(frame, data)
            
            # Send to WebSocket
            if self.ws_connected:
                ws_data = {
                    "type": "face_data",
                    "blendshapes": data.blendshapes,
                    "head_pose": data.head_pose,
                    "timestamp": data.timestamp,
                }
                self.send_websocket(ws_data)
            
            # Update GUI
            if self.root and self.canvas:
                self.root.after(0, self.update_canvas, frame)
            
            # FPS
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
            
            time.sleep(1 / self.config.target_fps)
    
    def update_canvas(self, frame):
        """Update canvas with current frame."""
        if not self.canvas or frame is None:
            return
        
        # Convert frame to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Convert to tkinter compatible format
        img = tk.PhotoImage(width=w, height=h)
        
        # Fill image data
        import PIL.Image as Image
        import PIL.ImageTk as ImageTk
        
        pil_img = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def on_close(self):
        """Handle window close."""
        self.running = False
        if self.root:
            self.root.destroy()
    
    def run_cli(self):
        """Run command-line version."""
        if not self.init():
            return
        
        self.connect_websocket()
        
        self.running = True
        print("[Main] Starting - press Ctrl+C to stop")
        
        while self.running:
            frame = None
            
            if self.config.test_mode:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.rectangle(frame, (50, 50), (590, 430), (0, 150, 0), 3)
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            data = self.process_frame(frame)
            frame = self.draw_hologram_overlay(frame, data)
            
            if self.ws_connected:
                self.send_websocket({
                    "type": "face_data",
                    "blendshapes": data.blendshapes,
                    "head_pose": data.head_pose,
                    "timestamp": data.timestamp,
                })
            
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
                print(f"FPS: {self.fps} | Sent: {self.sent_count}")
            
            delay = int(1000 / self.config.target_fps)
            cv2.imshow(self.WINDOW_NAME, frame)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        self.running = False
        self.stop()
    
    def stop(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        if self.ws:
            self.ws.close()
        cv2.destroyAllWindows()
        print(f"[Main] Stopped - processed {self.frame_count} frames")

def main():
    parser = argparse.ArgumentParser(description="Max Headroom Digitizer")
    parser.add_argument("--gui", action="store_true", help="Use GUI (default if tkinter available)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--ws-host", default="localhost", help="WebSocket host")
    parser.add_argument("--ws-port", type=int, default=30000, help="WebSocket port")
    parser.add_argument("--test", action="store_true", help="Start in test mode")
    parser.add_argument("--glitch", type=int, default=15, help="Hologram glitch intensity (0-100)")
    args = parser.parse_args()
    
    config = AppConfig()
    config.camera_index = args.camera
    config.target_fps = args.fps
    config.ws_host = args.ws_host
    config.ws_port = args.ws_port
    config.test_mode = args.test
    config.glitch_intensity = args.glitch / 100
    
    app = MaxHeadroomApp(config)
    
    if args.gui or GUI_AVAILABLE:
        app.run_gui()
    else:
        app.run_cli()

if __name__ == "__main__":
    main()