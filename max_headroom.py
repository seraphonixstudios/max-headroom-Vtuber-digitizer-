#!/usr/bin/env python3
"""
Max Headroom Digitizer - Desktop Application
Complete VTuber streaming application with WebSocket OBS integration
Author: CRACKED-DEV-Ω
Version: 3.0.0
License: MIT
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

# Version
VERSION = "3.0.0"

# Check for GUI libraries
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("tkinter not available - using CLI mode")

# Check for WebSocket
try:
    import websocket
except ImportError:
    try:
        import ws as websocket
    except ImportError:
        websocket = None
        print("warning: websocket-client not installed")

@dataclass
class AppConfig:
    """Application configuration"""
    ws_host: str = "localhost"
    ws_port: int = 30000
    camera_index: int = 0
    target_fps: int = 30
    resolution_w: int = 640
    resolution_h: int = 480
    smoothing: float = 0.8
    enable_websocket: bool = True
    enable_obs: bool = False
    obs_ndi_name: str = "MaxHeadroom"
    test_mode: bool = False
    glitch_intensity: float = 0.15
    hologram_enabled: bool = True

@dataclass
class FaceTrackingData:
    """Face tracking data structure"""
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_pose: Dict[str, List[float]] = field(default_factory=dict)
    landmarks: List[Tuple[int, int]] = field(default_factory=list)
    timestamp: float = 0.0

class FaceDetector:
    """Face detection using Haar Cascade"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade")
    
    def detect(self, gray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        return tuple(faces[0]) if len(faces) > 0 else None

class BlendShapeCalculator:
    """Calculate ARKit-compatible blendshapes"""
    
    SHAPES = [
        "browDown_L", "browDown_R", "browUp_L", "browUp_R",
        "cheekPuff", "cheekSquint_L", "cheekSquint_R",
        "eyeBlink_L", "eyeBlink_R",
        "eyeLookDown_L", "eyeLookDown_R",
        "eyeLookUp_L", "eyeLookUp_R",
        "eyeSquint_L", "eyeSquint_R",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimple_L", "mouthDimple_R",
        "mouthFunnel", "mouthLeft", "mouthPucker",
        "mouthRight", "mouthSmile_L", "mouthSmile_R",
        "mouthUpperUp_L", "mouthUpperUp_R",
        "noseSneer_L", "noseSneer_R",
    ]
    
    def calculate(self, face_rect: Optional[Tuple], time_val: float) -> Dict[str, float]:
        if face_rect is None:
            return self._test_shapes(time_val)
        
        x, y, w, h = face_rect
        aspect = h / w if w > 0 else 1
        
        return {
            "jawOpen": min(1.0, aspect * 0.4),
            "mouthSmile_L": 0.35 + 0.15 * np.sin(time_val * 3),
            "mouthSmile_R": 0.35 + 0.15 * np.sin(time_val * 3 + 0.5),
            "eyeBlink_L": 0.0,
            "eyeBlink_R": 0.0,
            "browUp_L": 0.1 * (1 + np.sin(time_val * 1.5)),
            "browUp_R": 0.1 * (1 + np.sin(time_val * 1.5 + 0.3)),
            "cheekPuff": 0.2 + 0.1 * np.sin(time_val * 2.5),
            "noseSneer_L": max(0, 0.2 - 0.1 * np.sin(time_val)),
            "noseSneer_R": max(0, 0.2 - 0.1 * np.sin(time_val)),
            "mouthClose": 0.8,
            "mouthFunnel": 0.1,
            "mouthPucker": 0.1,
            "mouthLeft": 0.0,
            "mouthRight": 0.0,
            "mouthDimple_L": 0.0,
            "mouthDimple_R": 0.0,
            "mouthUpperUp_L": 0.0,
            "mouthUpperUp_R": 0.0,
            "browDown_L": 0.0,
            "browDown_R": 0.0,
            "eyeLookDown_L": 0.0,
            "eyeLookDown_R": 0.0,
            "eyeLookUp_L": 0.0,
            "eyeLookUp_R": 0.0,
            "eyeSquint_L": 0.0,
            "eyeSquint_R": 0.0,
            "jawForward": 0.0,
            "jawLeft": 0.0,
            "jawRight": 0.0,
            "cheekSquint_L": 0.0,
            "cheekSquint_R": 0.0,
        }
    
    def _test_shapes(self, t: float) -> Dict[str, float]:
        return {
            "jawOpen": 0.2 + 0.15 * np.sin(t * 2),
            "mouthSmile_L": 0.3 + 0.1 * np.sin(t * 3),
            "mouthSmile_R": 0.3 + 0.1 * np.sin(t * 3 + 0.5),
            "eyeBlink_L": 0.0,
            "eyeBlink_R": 0.0,
            "browUp_L": 0.1 * (1 + np.sin(t * 1.5)),
            "browUp_R": 0.1 * (1 + np.sin(t * 1.5 + 0.3)),
            "cheekPuff": 0.2 + 0.1 * np.sin(t * 2.5),
            "noseSneer_L": 0.1 * (1 + np.sin(t)),
            "noseSneer_R": 0.1 * (1 + np.sin(t)),
            "mouthClose": 0.8 - 0.1 * np.sin(t * 2),
            "mouthFunnel": 0.1,
            "mouthPucker": 0.1,
            "mouthLeft": 0.0,
            "mouthRight": 0.0,
            "mouthDimple_L": 0.0,
            "mouthDimple_R": 0.0,
            "mouthUpperUp_L": 0.0,
            "mouthUpperUp_R": 0.0,
            "browDown_L": 0.0,
            "browDown_R": 0.0,
            "eyeLookDown_L": 0.0,
            "eyeLookDown_R": 0.0,
            "eyeLookUp_L": 0.0,
            "eyeLookUp_R": 0.0,
            "eyeSquint_L": 0.0,
            "eyeSquint_R": 0.0,
            "jawForward": 0.0,
            "jawLeft": 0.0,
            "jawRight": 0.0,
            "cheekSquint_L": 0.0,
            "cheekSquint_R": 0.0,
        }

class MaxHeadroomApp:
    """Main application"""
    
    WINDOW_NAME = "Max Headroom Digitizer"
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        
        self.cap = None
        self.detector = None
        self.blendshape_calc = BlendShapeCalculator()
        
        self.ws = None
        self.ws_connected = False
        
        self.running = False
        self.frame_count = 0
        self.sent_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        self.current_data = None
        self.smoothed_data = None
        self.blendshape_buffer = {}
        
        self.root = None
        self.canvas = None
        self.info_label = None
        
        # Filter system
        self.filter_manager = None
        self._init_filters()
    
    def _init_filters(self):
        """Initialize filter system for desktop app."""
        try:
            from filters import FilterManager
            self.filter_manager = FilterManager()
        except Exception as e:
            print(f"[Init] Filter system not available: {e}")
    
    def init(self) -> bool:
        print("[Init] Loading face detector...")
        self.detector = FaceDetector()
        print("[Init] Face detector loaded")
        
        if not self.config.test_mode:
            print(f"[Init] Opening camera {self.config.camera_index}...")
            self.cap = cv2.VideoCapture(self.config.camera_index)
            
            if not self.cap.isOpened():
                print("[Init] Camera unavailable - using TEST MODE")
                self.config.test_mode = True
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution_w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution_h)
                actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"[Init] Camera: {int(actual_w)}x{int(actual_h)}")
        
        return True
    
    def connect_websocket(self) -> bool:
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
    
    def send_websocket(self, data: Dict) -> bool:
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
    
    def process_frame(self, frame, time_val: float) -> FaceTrackingData:
        face_rect = None
        
        if not self.config.test_mode and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rect = self.detector.detect(gray)
        
        blends = self.blendshape_calc.calculate(face_rect, time_val)
        
        if self.config.smoothing > 0:
            blends = self._smooth_blendshapes(blends)
        
        pose = self._calculate_pose(face_rect, frame.shape[:2] if frame is not None else (480, 640), time_val)
        
        return FaceTrackingData(
            blendshapes=blends,
            head_pose=pose,
            landmarks=[],
            timestamp=time_val
        )
    
    def _smooth_blendshapes(self, blends: Dict[str, float]) -> Dict[str, float]:
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
    
    def _calculate_pose(self, face_rect, frame_shape, time_val) -> Dict[str, List[float]]:
        if face_rect is None:
            return {
                "rotation": [5 * np.sin(time_val * 0.5), 10 * np.sin(time_val * 0.3), 0],
                "translation": [0.1 * np.sin(time_val * 0.5), 0.05 * np.sin(time_val * 0.7), 1.5]
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
    
    def draw_hologram(self, frame, data: FaceTrackingData) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Scanlines
        for y in range(0, h, 3):
            cv2.line(frame, (0, y), (w, y), (0, 50, 0), 1)
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 20, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 2), (0, 255, 0), 2)
        
        # Title with glitch
        offset = int(np.sin(time.time() * 10) * self.config.glitch_intensity * 20) if self.config.glitch_intensity > 0 else 0
        cv2.putText(frame, "MAX HEADROOM", (15 + offset, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps}", (w - 90, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Test mode
        if self.config.test_mode:
            cv2.putText(frame, "TEST MODE", (w//2 - 60, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Blendshape bars
        y_pos = 70
        for name in list(data.blendshapes.keys())[:8]:
            val = data.blendshapes.get(name, 0)
            bar_w = int(val * 120)
            
            cv2.rectangle(frame, (10, y_pos), (130, y_pos + 18), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, y_pos), (10 + bar_w, y_pos + 18), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, y_pos), (130, y_pos + 18), (0, 100, 0), 1)
            
            cv2.putText(frame, f"{name}: {val:.2f}", (140, y_pos + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            y_pos += 22
        
        # Status
        ws_status = "WS: ON" if self.ws_connected else "WS: OFF"
        ws_color = (0, 255, 0) if self.ws_connected else (0, 0, 255)
        cv2.putText(frame, ws_status, (10, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ws_color, 2)
        
        # Filter status display
        if self.filter_manager:
            active = self.filter_manager.get_all_status()
            enabled = [f["name"] for f in active if f["enabled"]]
            if enabled:
                filt_text = "FILTERS: " + ", ".join(enabled[:3])
                cv2.putText(frame, filt_text, (10, h - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # RGB split
        if self.config.glitch_intensity > 0.1:
            split = int(self.config.glitch_intensity * 15)
            if split > 0:
                frame[:, split:] = frame[:, :-split]
        
        return frame
    
    def run_gui(self):
        if not GUI_AVAILABLE:
            return self.run_cli()
        
        self.root = tk.Tk()
        self.root.title(self.WINDOW_NAME)
        self.root.geometry("800x700")
        self.root.configure(bg="black")
        
        # Title
        title = tk.Label(self.root, text="MAX HEADROOM DIGITIZER", font=("Consolas", 18, "bold"),
                       fg="#00FF00", bg="black")
        title.pack(pady=10)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black", highlightthickness=0)
        self.canvas.pack(pady=5)
        
        # Controls
        controls = tk.Frame(self.root, bg="black")
        controls.pack(pady=10)
        
        tk.Label(controls, text="Host:", fg="#00FF00", bg="black").grid(row=0, column=0)
        self.host_entry = tk.Entry(controls, width=12, fg="#00FF00", bg="#001100")
        self.host_entry.insert(0, self.config.ws_host)
        self.host_entry.grid(row=0, column=1)
        
        tk.Label(controls, text="Port:", fg="#00FF00", bg="black").grid(row=0, column=2)
        self.port_entry = tk.Entry(controls, width=6, fg="#00FF00", bg="#001100")
        self.port_entry.insert(0, str(self.config.ws_port))
        self.port_entry.grid(row=0, column=3)
        
        tk.Button(controls, text="CONNECT", fg="#00FF00", bg="#001100",
                 command=self.on_connect).grid(row=0, column=4, padx=10)
        
        # Info
        self.info_label = tk.Label(self.root, text="Ready", font=("Consolas", 10),
                                  fg="#00FF00", bg="black")
        self.info_label.pack(pady=5)
        
        # Buttons
        buttons = tk.Frame(self.root, bg="black")
        buttons.pack(pady=10)
        
        tk.Button(buttons, text="START", font=("Consolas", 12, "bold"), fg="#00FF00", bg="#002200",
                width=10, command=self.on_start).pack(side=tk.LEFT, padx=10)
        tk.Button(buttons, text="STOP", font=("Consolas", 12, "bold"), fg="#FF0000", bg="#220000",
                width=10, command=self.on_stop).pack(side=tk.LEFT, padx=10)
        
        # Checkboxes
        self.test_var = tk.BooleanVar(value=self.config.test_mode)
        tk.Checkbutton(self.root, text="Test Mode", variable=self.test_var, fg="#00FF00", bg="black",
                    selectcolor="#003300", command=self.on_toggle_test).pack()
        
        # Glitch slider
        tk.Label(self.root, text="Hologram Intensity:", fg="#00FF00", bg="black").pack()
        self.glitch_scale = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL,
                                  fg="#00FF00", bg="black", troughcolor="#002200",
                                  highlightthickness=0,
                                  command=self.on_glitch_change)
        self.glitch_scale.set(int(self.config.glitch_intensity * 100))
        self.glitch_scale.pack(pady=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.running = True
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()
        
        self.root.mainloop()
    
    def on_connect(self):
        self.config.ws_host = self.host_entry.get()
        self.config.ws_port = int(self.port_entry.get())
        self.connect_websocket()
        if self.ws_connected:
            self.info_label.config(text="WebSocket Connected!")
        else:
            self.info_label.config(text="Connection Failed")
    
    def on_start(self):
        if not self.running:
            self.running = True
            self.info_label.config(text="Tracking Started")
    
    def on_stop(self):
        self.running = False
        self.info_label.config(text="Tracking Stopped")
    
    def on_toggle_test(self):
        self.config.test_mode = self.test_var.get()
    
    def on_glitch_change(self, value):
        self.config.glitch_intensity = int(value) / 100
    
    def on_close(self):
        self.running = False
        if self.root:
            self.root.destroy()
    
    def _tracking_loop(self):
        while True:
            if not self.running:
                time.sleep(0.1)
                continue
            
            frame = None
            
            if self.config.test_mode:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.rectangle(frame, (50, 50), (590, 430), (0, 150, 0), 3)
            else:
                if self.cap:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
            
            t = time.time()
            data = self.process_frame(frame, t)
            
            # Apply filters
            if self.filter_manager and frame is not None:
                frame = self.filter_manager.process(
                    frame,
                    blendshapes=data.blendshapes,
                    head_pose=data.head_pose,
                    frame_id=self.frame_count
                )
            
            frame = self.draw_hologram(frame, data)
            
            # Build payload with filter status
            payload = {
                "type": "face_data",
                "blendshapes": data.blendshapes,
                "head_pose": data.head_pose,
                "timestamp": data.timestamp,
            }
            if self.filter_manager:
                active = [f.name for f in self.filter_manager.filters if f.enabled]
                if active:
                    payload["filter_status"] = {"active": active}
            
            if self.ws_connected:
                self.send_websocket(payload)
            
            if self.canvas and frame is not None:
                try:
                    from PIL import Image, ImageTk
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(pil_img)
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                except:
                    pass
            
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
            
            time.sleep(1 / self.config.target_fps)
    
    def run_cli(self):
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
            
            t = time.time()
            data = self.process_frame(frame, t)
            
            # Apply filters
            if self.filter_manager and frame is not None:
                frame = self.filter_manager.process(
                    frame,
                    blendshapes=data.blendshapes,
                    head_pose=data.head_pose,
                    frame_id=self.frame_count
                )
            
            frame = self.draw_hologram(frame, data)
            
            # Build payload with filter status
            payload = {
                "type": "face_data",
                "blendshapes": data.blendshapes,
                "head_pose": data.head_pose,
                "timestamp": data.timestamp,
            }
            if self.filter_manager:
                active = [f.name for f in self.filter_manager.filters if f.enabled]
                if active:
                    payload["filter_status"] = {"active": active}
            
            if self.ws_connected:
                self.send_websocket(payload)
            
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
                print(f"FPS: {self.fps} | Sent: {self.sent_count}")
            
            delay = int(1000 / self.config.target_fps)
            cv2.imshow(self.WINDOW_NAME, frame)
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            if self.filter_manager:
                if key == ord('d') or key == ord('D'):
                    self.filter_manager.toggle_filter("Max Headroom")
                    print("[App] Max Headroom filter toggled")
                if key == ord('b') or key == ord('B'):
                    self.filter_manager.toggle_filter("Skin Smoothing")
                if key == ord('c') or key == ord('C'):
                    self.filter_manager.toggle_filter("Color Grading")
                if key == ord('r') or key == ord('R'):
                    self.filter_manager.reset()
                    print("[App] All filters reset")
        
        self.running = False
        if self.cap:
            self.cap.release()
        if self.ws:
            self.ws.close()
        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.ws:
            self.ws.close()

def main():
    parser = argparse.ArgumentParser(description="Max Headroom Digitizer")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--glitch", type=int, default=15)
    args = parser.parse_args()
    
    print(f"Max Headroom Digitizer v{VERSION}")
    
    config = AppConfig()
    config.camera_index = args.camera
    config.target_fps = args.fps
    config.ws_port = args.port
    config.ws_host = args.host
    config.test_mode = args.test
    config.glitch_intensity = args.glitch / 100
    
    app = MaxHeadroomApp(config)
    
    if args.gui or GUI_AVAILABLE:
        app.run_gui()
    else:
        app.run_cli()

if __name__ == "__main__":
    main()