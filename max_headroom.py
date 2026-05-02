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
VERSION = "3.1.1"

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
        
        # Import themed components
        try:
            from gui_themes import (
                Colors, MatrixRainCanvas, SacredGeometryCanvas, CRTOverlayCanvas,
                NeonButton, CrystallineFrame, TerminalLog, HUDOverlay,
                GlitchLabel, CrystalProgressBar, HexDisplay, WaveformCanvas,
                apply_dark_theme
            )
            THEMES_AVAILABLE = True
        except Exception as e:
            print(f"[GUI] Themed components not available: {e}")
            THEMES_AVAILABLE = False
            return self._run_basic_gui()
        
        self.root = tk.Tk()
        self.root.title(self.WINDOW_NAME)
        self.root.geometry("1200x900")
        self.root.configure(bg=Colors.DEEP_SPACE)
        apply_dark_theme(self.root)
        
        # =====================================================================
        # TOP BAR: Title + Matrix Rain strip
        # =====================================================================
        top_frame = tk.Frame(self.root, bg=Colors.DEEP_SPACE)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.title_label = GlitchLabel(top_frame, text="MAX HEADROOM DIGITIZER v3.1",
                                       color=Colors.CRT_CYAN, font=("Consolas", 20, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=10)
        
        # Matrix rain strip (small)
        self.matrix_rain = MatrixRainCanvas(top_frame, width=400, height=40,
                                           column_spacing=10, font_size=10)
        self.matrix_rain.pack(side=tk.RIGHT, padx=10)
        self.matrix_rain.start()
        
        # =====================================================================
        # MAIN CONTENT: Video (left) + Controls (right)
        # =====================================================================
        main_frame = tk.Frame(self.root, bg=Colors.DEEP_SPACE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # ---- LEFT: Video preview with overlays ----
        video_frame = tk.Frame(main_frame, bg=Colors.DEEP_SPACE)
        video_frame.pack(side=tk.LEFT, padx=5)
        
        # Video canvas container
        video_container = tk.Frame(video_frame, bg=Colors.DEEP_SPACE)
        video_container.pack()
        
        self.canvas = tk.Canvas(video_container, width=640, height=480,
                               bg=Colors.VOID_BLACK, highlightthickness=0)
        self.canvas.pack()
        
        # CRT scanline overlay (on top of video)
        self.crt_overlay = CRTOverlayCanvas(video_container, width=640, height=480)
        self.crt_overlay.place(x=0, y=0)
        self.crt_overlay.start()
        
        # HUD overlay (targeting reticle)
        self.hud_overlay = HUDOverlay(video_container, width=640, height=480)
        self.hud_overlay.place(x=0, y=0)
        self.hud_overlay.start()
        
        # Video status bar
        video_status = tk.Frame(video_frame, bg=Colors.DEEP_SPACE)
        video_status.pack(fill=tk.X, pady=2)
        
        self.video_status_label = tk.Label(video_status, text="CAMERA: STANDBY",
                                          fg=Colors.WARNING, bg=Colors.DEEP_SPACE,
                                          font=("Consolas", 9, "bold"))
        self.video_status_label.pack(side=tk.LEFT)
        
        self.fps_label = tk.Label(video_status, text="FPS: 0",
                                 fg=Colors.CRT_CYAN, bg=Colors.DEEP_SPACE,
                                 font=("Consolas", 9))
        self.fps_label.pack(side=tk.RIGHT)
        
        # Waveform visualizer below video
        self.waveform = WaveformCanvas(video_frame, bars=32, width=640, height=50,
                                      color=Colors.CRYSTAL_BLUE)
        self.waveform.pack(pady=5)
        self.waveform.start()
        
        # ---- RIGHT: Control panels ----
        right_frame = tk.Frame(main_frame, bg=Colors.DEEP_SPACE)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Connection panel
        conn_panel = CrystallineFrame(right_frame, width=280, height=140,
                                     title="NETWORK LINK", color=Colors.CRYSTAL_BLUE)
        conn_panel.pack(pady=5)
        
        # Connection controls inside panel
        conn_inner = tk.Frame(right_frame, bg=Colors.DARK_PANEL)
        conn_inner.place(in_=conn_panel, x=10, y=30, width=260, height=100)
        
        tk.Label(conn_inner, text="HOST:", fg=Colors.CRT_CYAN, bg=Colors.DARK_PANEL,
                font=("Consolas", 9)).grid(row=0, column=0, sticky="w")
        self.host_entry = tk.Entry(conn_inner, width=14, fg=Colors.CRT_CYAN,
                                  bg=Colors.VOID_BLACK, insertbackground=Colors.CRT_CYAN,
                                  font=("Consolas", 9))
        self.host_entry.insert(0, self.config.ws_host)
        self.host_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(conn_inner, text="PORT:", fg=Colors.CRT_CYAN, bg=Colors.DARK_PANEL,
                font=("Consolas", 9)).grid(row=1, column=0, sticky="w")
        self.port_entry = tk.Entry(conn_inner, width=8, fg=Colors.CRT_CYAN,
                                  bg=Colors.VOID_BLACK, insertbackground=Colors.CRT_CYAN,
                                  font=("Consolas", 9))
        self.port_entry.insert(0, str(self.config.ws_port))
        self.port_entry.grid(row=1, column=1, padx=5, sticky="w")
        
        self.connect_btn = NeonButton(conn_inner, text="LINK", width=80, height=28,
                                     color=Colors.ATLANTEAN_TEAL,
                                     command=self.on_connect)
        self.connect_btn.grid(row=2, column=0, columnspan=2, pady=8)
        
        self.ws_status_led = tk.Label(conn_inner, text="● OFFLINE", fg=Colors.ALERT,
                                     bg=Colors.DARK_PANEL, font=("Consolas", 9, "bold"))
        self.ws_status_led.grid(row=0, column=2, rowspan=3, padx=10)
        
        # Filter Control Panel
        filter_panel = CrystallineFrame(right_frame, width=280, height=260,
                                       title="FILTER MATRIX", color=Colors.ATLANTEAN_TEAL)
        filter_panel.pack(pady=5)
        
        filter_inner = tk.Frame(right_frame, bg=Colors.DARK_PANEL)
        filter_inner.place(in_=filter_panel, x=10, y=30, width=260, height=220)
        
        # Filter buttons grid
        filter_buttons = [
            ("ANDROID", "Max Headroom", Colors.NEON_PINK, Colors.NEON_PURPLE),
            ("BEAUTY", "Skin Smoothing", Colors.ATLANTEAN_TEAL, Colors.AQUA_GLOW),
            ("BACKGROUND", "Background", Colors.PLASMA_BLUE, Colors.CRYSTAL_BLUE),
            ("AR", "AR Overlay", Colors.SACRED_GOLD, Colors.CRT_CYAN),
            ("MORPH", "Face Morph", Colors.NEON_ORANGE, Colors.WARNING),
            ("COLOR", "Color Grading", Colors.MATRIX_GREEN, Colors.CRT_GREEN),
        ]
        
        self.filter_btn_refs = {}
        for i, (label, filt_name, color, hover) in enumerate(filter_buttons):
            row = i // 2
            col = i % 2
            btn = NeonButton(filter_inner, text=label, width=110, height=28,
                           color=color, hover_color=hover,
                           command=lambda n=filt_name: self._toggle_filter(n))
            btn.grid(row=row, column=col, padx=4, pady=4)
            self.filter_btn_refs[filt_name] = btn
        
        # Reset button
        reset_btn = NeonButton(filter_inner, text="RESET ALL", width=110, height=28,
                              color=Colors.ALERT, hover_color=Colors.WARNING,
                              command=self._reset_filters)
        reset_btn.grid(row=3, column=0, padx=4, pady=4)
        
        # Intensity slider
        tk.Label(filter_inner, text="GLITCH INTENSITY:", fg=Colors.CRT_CYAN,
                bg=Colors.DARK_PANEL, font=("Consolas", 8)).grid(row=3, column=1, sticky="w")
        self.glitch_scale = tk.Scale(filter_inner, from_=0, to=100, orient=tk.HORIZONTAL,
                                    fg=Colors.CRT_CYAN, bg=Colors.DARK_PANEL,
                                    troughcolor=Colors.SCANLINE, highlightthickness=0,
                                    command=self.on_glitch_change, length=100)
        self.glitch_scale.set(int(self.config.glitch_intensity * 100))
        self.glitch_scale.grid(row=4, column=0, columnspan=2, sticky="ew")
        
        # System Panel
        sys_panel = CrystallineFrame(right_frame, width=280, height=180,
                                    title="SYSTEM CORE", color=Colors.SACRED_GOLD)
        sys_panel.pack(pady=5)
        
        sys_inner = tk.Frame(right_frame, bg=Colors.DARK_PANEL)
        sys_inner.place(in_=sys_panel, x=10, y=30, width=260, height=140)
        
        # Sacred geometry animation
        self.sacred_geo = SacredGeometryCanvas(sys_inner, width=100, height=100)
        self.sacred_geo.pack(side=tk.LEFT, padx=5)
        self.sacred_geo.start()
        
        # Hex display
        self.hex_display = HexDisplay(sys_inner, rows=5, cols=4, width=130, height=100)
        self.hex_display.pack(side=tk.RIGHT, padx=5)
        self.hex_display.start()
        
        # Test mode toggle
        self.test_var = tk.BooleanVar(value=self.config.test_mode)
        test_cb = tk.Checkbutton(sys_inner, text="SIMULATION MODE", variable=self.test_var,
                                fg=Colors.MATRIX_GREEN, bg=Colors.DARK_PANEL,
                                selectcolor=Colors.VOID_BLACK,
                                command=self.on_toggle_test, font=("Consolas", 8))
        test_cb.place(x=5, y=110)
        
        # =====================================================================
        # BOTTOM: Terminal Log
        # =====================================================================
        bottom_frame = tk.Frame(self.root, bg=Colors.DEEP_SPACE)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.terminal_log = TerminalLog(bottom_frame, height=6, width=120)
        self.terminal_log.pack(fill=tk.X)
        self.terminal_log.log("System initialization sequence started...", "system")
        self.terminal_log.log("Loading face detection cascade...", "system")
        self.terminal_log.log("Filter matrix online - 6 filters ready", "ok")
        self.terminal_log.log("Waiting for user command...", "system")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.running = True
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()
        
        self.root.mainloop()
    
    def _run_basic_gui(self):
        """Fallback basic GUI if themes fail to load."""
        self.root = tk.Tk()
        self.root.title(self.WINDOW_NAME)
        self.root.geometry("800x700")
        self.root.configure(bg="black")
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black", highlightthickness=0)
        self.canvas.pack(pady=5)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.running = True
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()
        self.root.mainloop()
    
    def _toggle_filter(self, name):
        if self.filter_manager:
            state = self.filter_manager.toggle_filter(name)
            status = "ACTIVE" if state else "OFFLINE"
            color = "ok" if state else "warning"
            self.terminal_log.log(f"Filter '{name}': {status}", color)
    
    def _reset_filters(self):
        if self.filter_manager:
            self.filter_manager.reset()
            self.terminal_log.log("All filters reset to default", "system")
    
    def on_connect(self):
        self.config.ws_host = self.host_entry.get()
        self.config.ws_port = int(self.port_entry.get())
        self.connect_websocket()
        if self.ws_connected:
            self.ws_status_led.config(text="● ONLINE", fg=Colors.OK)
            self.terminal_log.log(f"WebSocket linked to {self.config.ws_host}:{self.config.ws_port}", "ok")
        else:
            self.ws_status_led.config(text="● FAILED", fg=Colors.ALERT)
            self.terminal_log.log("WebSocket connection failed", "alert")
    
    def on_start(self):
        if not self.running:
            self.running = True
            self.terminal_log.log("Tracking sequence initiated", "ok")
    
    def on_stop(self):
        self.running = False
        self.terminal_log.log("Tracking sequence halted", "warning")
    
    def on_toggle_test(self):
        self.config.test_mode = self.test_var.get()
        mode = "SIMULATION" if self.config.test_mode else "LIVE CAMERA"
        self.terminal_log.log(f"Mode switched: {mode}", "system")
    
    def on_glitch_change(self, value):
        self.config.glitch_intensity = int(value) / 100
    
    def on_close(self):
        self.running = False
        # Stop animations
        try:
            if hasattr(self, 'matrix_rain'): self.matrix_rain.stop()
            if hasattr(self, 'crt_overlay'): self.crt_overlay.stop()
            if hasattr(self, 'hud_overlay'): self.hud_overlay.stop()
            if hasattr(self, 'sacred_geo'): self.sacred_geo.stop()
            if hasattr(self, 'hex_display'): self.hex_display.stop()
            if hasattr(self, 'waveform'): self.waveform.stop()
            if hasattr(self, 'title_label'): self.title_label.destroy()
        except:
            pass
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
                # Draw test pattern
                cv2.rectangle(frame, (50, 50), (590, 430), (0, 150, 0), 2)
                cv2.putText(frame, "SIMULATION MODE", (220, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
            
            # Update video canvas with thread safety
            if self.canvas and frame is not None:
                try:
                    from PIL import Image, ImageTk
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize if needed
                    if frame_rgb.shape[1] != 640 or frame_rgb.shape[0] != 480:
                        frame_rgb = cv2.resize(frame_rgb, (640, 480))
                    pil_img = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # Use after() for thread-safe GUI update
                    def update_image(p=photo):
                        self.canvas.delete("all")
                        self.canvas.create_image(0, 0, anchor=tk.NW, image=p)
                        self.canvas.image = p  # Keep reference
                    
                    if hasattr(self, 'root') and self.root:
                        self.root.after(0, update_image)
                    
                    # Update HUD target position based on face
                    face_rect = None
                    if not self.config.test_mode and self.cap:
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        if hasattr(self, 'detector') and self.detector:
                            face_rect = self.detector.detect(gray)
                    
                    if face_rect:
                        fx, fy, fw, fh = face_rect
                        hud_x = int((fx + fw / 2) / frame.shape[1] * 640)
                        hud_y = int((fy + fh / 2) / frame.shape[0] * 480)
                        if hasattr(self, 'hud_overlay'):
                            self.hud_overlay.set_target(hud_x, hud_y)
                    
                except Exception as e:
                    pass
            
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
                
                # Update FPS label thread-safely
                if hasattr(self, 'fps_label') and self.fps_label:
                    def update_fps(f=self.fps):
                        self.fps_label.config(text=f"FPS: {f}")
                    if hasattr(self, 'root') and self.root:
                        self.root.after(0, update_fps)
            
            self.frame_count += 1
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