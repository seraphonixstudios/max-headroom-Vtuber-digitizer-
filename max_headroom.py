#!/usr/bin/env python3
"""
Max Headroom Digitizer - Desktop Application
Complete VTuber streaming application with WebSocket OBS integration
Author: CRACKED-DEV-Ω
Version: 3.2.0
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
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

# Version
VERSION = "3.2.0"

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

# Optional theme colors for module-level access
try:
    from gui_themes import Colors
except ImportError:
    Colors = None

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
        self.ws_reconnect_attempts = 0
        self.ws_last_reconnect = 0
        
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
        
        # Thread-safe frame queue for GUI updates
        self._frame_queue = deque(maxlen=2)
        self._ui_lock = threading.Lock()
        self._ui_pending = False
        
        # Camera recovery
        self._camera_fail_count = 0
        self._camera_last_retry = 0
        
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
        try:
            self.detector = FaceDetector()
            print("[Init] Face detector loaded")
        except Exception as e:
            print(f"[Init] Face detector failed: {e}")
            self.detector = None
        
        if not self.config.test_mode:
            self._open_camera()
        
        return True
    
    def _open_camera(self):
        """Open camera with retry logic."""
        print(f"[Init] Opening camera {self.config.camera_index}...")
        self.cap = cv2.VideoCapture(self.config.camera_index)
        
        if not self.cap.isOpened():
            print("[Init] Camera unavailable - using TEST MODE")
            self.config.test_mode = True
            self._camera_fail_count += 1
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution_h)
            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"[Init] Camera: {int(actual_w)}x{int(actual_h)}")
            self._camera_fail_count = 0
    
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
            self.ws_reconnect_attempts = 0
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
        
        if not self.config.test_mode and frame is not None and self.detector is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rect = self.detector.detect(gray)
            except Exception as e:
                self._try_log(f"Face detection error: {e}", "warning")
        
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
        
        # Initialize camera BEFORE building GUI
        self.init()
        
        # Import themed components
        try:
            from gui_themes import (
                Colors, MatrixRainCanvas, SacredGeometryCanvas, CRTOverlayCanvas,
                NeonButton, CrystallineFrame, TerminalLog, HUDOverlay,
                GlitchLabel, CrystalProgressBar, HexDisplay, WaveformCanvas,
                StatusIndicator, BlendshapeBars, apply_dark_theme
            )
            THEMES_AVAILABLE = True
        except Exception as e:
            print(f"[GUI] Themed components not available: {e}")
            THEMES_AVAILABLE = False
            return self._run_basic_gui()
        
        # =====================================================================
        # ROOT WINDOW SETUP
        # =====================================================================
        self.root = tk.Tk()
        self.root.title(f"{self.WINDOW_NAME} v{VERSION}")
        self.root.geometry("1366x900")
        self.root.configure(bg=Colors.VOID_BLACK)
        self.root.minsize(1100, 750)
        apply_dark_theme(self.root)
        
        # =====================================================================
        # STYLES
        # =====================================================================
        PANEL_BG = "#0a0e1a"
        BORDER_COLOR = "#1a2a3a"
        ACCENT_CYAN = "#00d4ff"
        TEXT_DIM = "#557788"
        TEXT_BRIGHT = "#aaddff"
        
        # =====================================================================
        # TOP BAR - Studio Header
        # =====================================================================
        top_bar = tk.Frame(self.root, bg=Colors.VOID_BLACK, height=36)
        top_bar.pack(fill=tk.X, padx=0, pady=0)
        top_bar.pack_propagate(False)
        
        # Left: Title
        title_frame = tk.Frame(top_bar, bg=Colors.VOID_BLACK)
        title_frame.pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(title_frame, text="MAX HEADROOM", fg=ACCENT_CYAN, bg=Colors.VOID_BLACK,
                font=("Consolas", 14, "bold")).pack(side=tk.LEFT)
        tk.Label(title_frame, text=f" STUDIO  v{VERSION}", fg=TEXT_DIM, bg=Colors.VOID_BLACK,
                font=("Consolas", 10)).pack(side=tk.LEFT, padx=(4, 0))
        
        # Center: Mode indicator
        self.mode_label = tk.Label(top_bar, text="STANDBY", fg=Colors.WARNING, bg=Colors.VOID_BLACK,
                                  font=("Consolas", 10, "bold"))
        self.mode_label.pack(side=tk.LEFT, padx=40)
        
        # Right: Status indicators
        indicators = tk.Frame(top_bar, bg=Colors.VOID_BLACK)
        indicators.pack(side=tk.RIGHT, padx=12)
        
        self.cam_indicator = StatusIndicator(indicators, color=Colors.ALERT)
        self.cam_indicator.pack(side=tk.LEFT, padx=2)
        tk.Label(indicators, text="CAM", fg=TEXT_DIM, bg=Colors.VOID_BLACK,
                font=("Consolas", 7)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.track_indicator = StatusIndicator(indicators, color=Colors.ALERT)
        self.track_indicator.pack(side=tk.LEFT, padx=2)
        tk.Label(indicators, text="TRACK", fg=TEXT_DIM, bg=Colors.VOID_BLACK,
                font=("Consolas", 7)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.ws_indicator = StatusIndicator(indicators, color=Colors.ALERT)
        self.ws_indicator.pack(side=tk.LEFT, padx=2)
        tk.Label(indicators, text="NET", fg=TEXT_DIM, bg=Colors.VOID_BLACK,
                font=("Consolas", 7)).pack(side=tk.LEFT, padx=(0, 0))
        
        # =====================================================================
        # MAIN WORKSPACE - OBS Style: Preview Left, Dock Right
        # =====================================================================
        workspace = tk.Frame(self.root, bg=Colors.VOID_BLACK)
        workspace.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # ---- LEFT: Preview Monitor ----
        preview_frame = tk.Frame(workspace, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                                highlightthickness=1)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        # Preview header bar
        preview_header = tk.Frame(preview_frame, bg=PANEL_BG, height=24)
        preview_header.pack(fill=tk.X, padx=0, pady=0)
        preview_header.pack_propagate(False)
        tk.Label(preview_header, text="PREVIEW", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8)).pack(side=tk.LEFT, padx=8)
        self.preview_res_label = tk.Label(preview_header, text="640x480", fg=TEXT_DIM,
                                         bg=PANEL_BG, font=("Consolas", 8))
        self.preview_res_label.pack(side=tk.RIGHT, padx=8)
        
        # Video canvas with fixed aspect container
        video_container = tk.Frame(preview_frame, bg=Colors.VOID_BLACK)
        video_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.canvas = tk.Canvas(video_container, bg=Colors.VOID_BLACK, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Corner bracket overlay (broadcast monitor style)
        self._draw_preview_brackets()
        
        # CRT overlay
        self.crt_overlay = CRTOverlayCanvas(video_container, width=640, height=480)
        self.crt_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self.crt_overlay.start()
        
        # HUD overlay
        self.hud_overlay = HUDOverlay(video_container, width=640, height=480)
        self.hud_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self.hud_overlay.start()
        
        # ---- RIGHT: Control Dock ----
        dock = tk.Frame(workspace, bg=Colors.VOID_BLACK, width=340)
        dock.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        dock.pack_propagate(False)
        
        # --- SCENES Panel ---
        scenes_panel = tk.Frame(dock, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                               highlightthickness=1)
        scenes_panel.pack(fill=tk.X, pady=(0, 4))
        
        scenes_header = tk.Frame(scenes_panel, bg=PANEL_BG, height=22)
        scenes_header.pack(fill=tk.X)
        scenes_header.pack_propagate(False)
        tk.Label(scenes_header, text="SCENES", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=8)
        
        scenes_list = tk.Frame(scenes_panel, bg=PANEL_BG)
        scenes_list.pack(fill=tk.X, padx=4, pady=4)
        
        scene_items = [
            ("Default", None),
            ("Android Mode", lambda: self._activate_scene("android")),
            ("Beauty Mode", lambda: self._activate_scene("beauty")),
            ("Color Grade", lambda: self._activate_scene("color")),
        ]
        self.scene_btn_refs = {}
        for name, cmd in scene_items:
            lbl = tk.Label(scenes_list, text=f"  {name}", fg=TEXT_BRIGHT, bg=PANEL_BG,
                          font=("Consolas", 9), anchor="w", cursor="hand2")
            lbl.pack(fill=tk.X, pady=1)
            if cmd:
                lbl.bind("<Button-1>", lambda e, c=cmd: c())
            self.scene_btn_refs[name] = lbl
        self._highlight_scene("Default")
        
        # --- SOURCES Panel (Filter Toggles) ---
        sources_panel = tk.Frame(dock, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                                highlightthickness=1)
        sources_panel.pack(fill=tk.X, pady=(0, 4))
        
        sources_header = tk.Frame(sources_panel, bg=PANEL_BG, height=22)
        sources_header.pack(fill=tk.X)
        sources_header.pack_propagate(False)
        tk.Label(sources_header, text="SOURCES", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=8)
        
        sources_list = tk.Frame(sources_panel, bg=PANEL_BG)
        sources_list.pack(fill=tk.X, padx=4, pady=4)
        
        filter_items = [
            ("Max Headroom", "MH", Colors.NEON_PINK),
            ("Skin Smoothing", "SKIN", Colors.ATLANTEAN_TEAL),
            ("Background", "BG", Colors.PLASMA_BLUE),
            ("AR Overlay", "AR", Colors.SACRED_GOLD),
            ("Face Morph", "MORPH", Colors.NEON_ORANGE),
            ("Color Grading", "COLOR", Colors.MATRIX_GREEN),
        ]
        self.filter_toggle_refs = {}
        for name, abbr, color in filter_items:
            row = tk.Frame(sources_list, bg=PANEL_BG)
            row.pack(fill=tk.X, pady=2)
            
            eye = tk.Label(row, text="●", fg=TEXT_DIM, bg=PANEL_BG, font=("Consolas", 10))
            eye.pack(side=tk.LEFT, padx=(4, 4))
            
            lbl = tk.Label(row, text=name, fg=TEXT_BRIGHT, bg=PANEL_BG,
                          font=("Consolas", 9), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            abbr_lbl = tk.Label(row, text=abbr, fg=color, bg=PANEL_BG,
                               font=("Consolas", 8, "bold"))
            abbr_lbl.pack(side=tk.RIGHT, padx=4)
            
            # Click to toggle
            for widget in (row, eye, lbl, abbr_lbl):
                widget.bind("<Button-1>", lambda e, n=name: self._toggle_filter(n))
            
            self.filter_toggle_refs[name] = {"eye": eye, "label": lbl, "abbr": abbr_lbl, "color": color}
        
        # Glitch intensity slider
        slider_row = tk.Frame(sources_panel, bg=PANEL_BG)
        slider_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(slider_row, text="GLITCH:", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8)).pack(side=tk.LEFT)
        self.glitch_scale = tk.Scale(slider_row, from_=0, to=100, orient=tk.HORIZONTAL,
                                    fg=ACCENT_CYAN, bg=PANEL_BG, troughcolor=BORDER_COLOR,
                                    highlightthickness=0, command=self.on_glitch_change,
                                    length=180, showvalue=0)
        self.glitch_scale.set(int(self.config.glitch_intensity * 100))
        self.glitch_scale.pack(side=tk.LEFT, padx=4)
        
        # --- AUDIO MIXER (Blendshapes) ---
        mixer_panel = tk.Frame(dock, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                              highlightthickness=1)
        mixer_panel.pack(fill=tk.X, pady=(0, 4), expand=True)
        
        mixer_header = tk.Frame(mixer_panel, bg=PANEL_BG, height=22)
        mixer_header.pack(fill=tk.X)
        mixer_header.pack_propagate(False)
        tk.Label(mixer_header, text="MIXER", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=8)
        
        mixer_inner = tk.Frame(mixer_panel, bg=PANEL_BG)
        mixer_inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.bs_bars = BlendshapeBars(mixer_inner, width=320, height=240, max_bars=14)
        self.bs_bars.pack(fill=tk.BOTH, expand=True)
        
        # --- TRANSITIONS / CONTROLS ---
        trans_panel = tk.Frame(dock, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                              highlightthickness=1)
        trans_panel.pack(fill=tk.X, pady=(0, 4))
        
        trans_inner = tk.Frame(trans_panel, bg=PANEL_BG)
        trans_inner.pack(fill=tk.X, padx=8, pady=6)
        
        self.connect_btn = tk.Button(trans_inner, text="LINK NETWORK", bg=BORDER_COLOR,
                                    fg=ACCENT_CYAN, font=("Consolas", 9, "bold"),
                                    activebackground=ACCENT_CYAN, activeforeground=Colors.VOID_BLACK,
                                    relief=tk.FLAT, cursor="hand2", command=self.on_connect)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 4))
        
        self.test_btn = tk.Button(trans_inner, text="TEST CAMERA", bg=BORDER_COLOR,
                                 fg=TEXT_BRIGHT, font=("Consolas", 9, "bold"),
                                 activebackground=Colors.OK, activeforeground=Colors.VOID_BLACK,
                                 relief=tk.FLAT, cursor="hand2", command=self._test_camera)
        self.test_btn.pack(side=tk.LEFT, padx=4)
        
        self.ws_status_text = tk.Label(trans_inner, text="OFFLINE", fg=Colors.ALERT,
                                      bg=PANEL_BG, font=("Consolas", 9, "bold"))
        self.ws_status_text.pack(side=tk.RIGHT)
        
        # Host/port entries
        net_row = tk.Frame(trans_panel, bg=PANEL_BG)
        net_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(net_row, text="HOST:", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8)).pack(side=tk.LEFT)
        self.host_entry = tk.Entry(net_row, width=12, fg=TEXT_BRIGHT,
                                  bg=Colors.VOID_BLACK, insertbackground=ACCENT_CYAN,
                                  font=("Consolas", 9), relief=tk.FLAT)
        self.host_entry.insert(0, self.config.ws_host)
        self.host_entry.pack(side=tk.LEFT, padx=4)
        tk.Label(net_row, text="PORT:", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8)).pack(side=tk.LEFT, padx=(8, 0))
        self.port_entry = tk.Entry(net_row, width=6, fg=TEXT_BRIGHT,
                                  bg=Colors.VOID_BLACK, insertbackground=ACCENT_CYAN,
                                  font=("Consolas", 9), relief=tk.FLAT)
        self.port_entry.insert(0, str(self.config.ws_port))
        self.port_entry.pack(side=tk.LEFT, padx=4)
        
        # =====================================================================
        # BOTTOM DOCK - Audio Mixer + Stats
        # =====================================================================
        bottom_dock = tk.Frame(self.root, bg=Colors.VOID_BLACK, height=140)
        bottom_dock.pack(fill=tk.X, padx=8, pady=(4, 8))
        bottom_dock.pack_propagate(False)
        
        # Left: Head Pose
        pose_frame = tk.Frame(bottom_dock, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                             highlightthickness=1)
        pose_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        pose_header = tk.Frame(pose_frame, bg=PANEL_BG, height=22)
        pose_header.pack(fill=tk.X)
        pose_header.pack_propagate(False)
        tk.Label(pose_header, text="HEAD POSE", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=8)
        
        pose_inner = tk.Frame(pose_frame, bg=PANEL_BG)
        pose_inner.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        self.pose_rot_label = tk.Label(pose_inner, text="ROT:  0.00   0.00   0.00",
                                      fg=TEXT_BRIGHT, bg=PANEL_BG, font=("Consolas", 10))
        self.pose_rot_label.pack(anchor="w", pady=2)
        self.pose_trans_label = tk.Label(pose_inner, text="POS:  0.00   0.00   0.00",
                                        fg=TEXT_BRIGHT, bg=PANEL_BG, font=("Consolas", 10))
        self.pose_trans_label.pack(anchor="w", pady=2)
        
        # Center: System Core
        sys_frame = tk.Frame(bottom_dock, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                            highlightthickness=1)
        sys_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        sys_header = tk.Frame(sys_frame, bg=PANEL_BG, height=22)
        sys_header.pack(fill=tk.X)
        sys_header.pack_propagate(False)
        tk.Label(sys_header, text="SYSTEM CORE", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=8)
        
        sys_inner = tk.Frame(sys_frame, bg=PANEL_BG)
        sys_inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        sys_top = tk.Frame(sys_inner, bg=PANEL_BG)
        sys_top.pack(fill=tk.X)
        self.sacred_geo = SacredGeometryCanvas(sys_top, width=70, height=70)
        self.sacred_geo.pack(side=tk.LEFT, padx=4)
        self.sacred_geo.start()
        self.hex_display = HexDisplay(sys_top, rows=4, cols=5, width=140, height=70)
        self.hex_display.pack(side=tk.RIGHT, padx=4)
        self.hex_display.start()
        
        self.test_var = tk.BooleanVar(value=self.config.test_mode)
        test_cb = tk.Checkbutton(sys_inner, text="SIMULATION MODE", variable=self.test_var,
                                fg=Colors.MATRIX_GREEN, bg=PANEL_BG,
                                selectcolor=Colors.VOID_BLACK, activebackground=PANEL_BG,
                                command=self.on_toggle_test, font=("Consolas", 8))
        test_cb.pack(anchor="w", pady=(4, 0))
        
        # Right: Terminal Log
        log_frame = tk.Frame(bottom_dock, bg=PANEL_BG, highlightbackground=BORDER_COLOR,
                            highlightthickness=1)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_header = tk.Frame(log_frame, bg=PANEL_BG, height=22)
        log_header.pack(fill=tk.X)
        log_header.pack_propagate(False)
        tk.Label(log_header, text="CONSOLE", fg=TEXT_DIM, bg=PANEL_BG,
                font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=8)
        
        self.terminal_log = TerminalLog(log_frame, height=6, width=60)
        self.terminal_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # =====================================================================
        # STATUS BAR
        # =====================================================================
        status_bar = tk.Frame(self.root, bg=PANEL_BG, height=24)
        status_bar.pack(fill=tk.X, padx=0, pady=0)
        status_bar.pack_propagate(False)
        
        self.video_status_label = tk.Label(status_bar, text="CAMERA: STANDBY",
                                          fg=Colors.WARNING, bg=PANEL_BG,
                                          font=("Consolas", 9, "bold"))
        self.video_status_label.pack(side=tk.LEFT, padx=12)
        
        self.fps_label = tk.Label(status_bar, text="FPS: 0", fg=ACCENT_CYAN,
                                 bg=PANEL_BG, font=("Consolas", 9))
        self.fps_label.pack(side=tk.LEFT, padx=12)
        
        self.packets_label = tk.Label(status_bar, text="PKT: 0", fg=TEXT_DIM,
                                     bg=PANEL_BG, font=("Consolas", 9))
        self.packets_label.pack(side=tk.LEFT, padx=12)
        
        self.frame_time_label = tk.Label(status_bar, text="FRAME: 0ms", fg=TEXT_DIM,
                                        bg=PANEL_BG, font=("Consolas", 9))
        self.frame_time_label.pack(side=tk.LEFT, padx=12)
        
        tk.Label(status_bar, text="D:Android B:Beauty C:Color G:BG A:AR M:Morph R:Reset",
                fg=TEXT_DIM, bg=PANEL_BG, font=("Consolas", 8)).pack(side=tk.RIGHT, padx=12)
        
        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        self.terminal_log.log(f"Studio initialization v{VERSION}", "system")
        if self.config.test_mode:
            self.terminal_log.log("Camera unavailable - running in simulation mode", "warning")
        else:
            self.terminal_log.log("Camera initialized", "ok")
        self.terminal_log.log("Filter matrix online - 6 sources ready", "ok")
        self.terminal_log.log("Press TEST CAMERA to verify video feed", "system")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Key>", self._on_keypress)
        
        self.running = True
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()
        
        self.root.mainloop()
    
    def _draw_preview_brackets(self):
        """Draw broadcast monitor corner brackets on canvas."""
        pass  # Drawn dynamically in _update_video_canvas
    
    def _activate_scene(self, scene_name):
        """Activate a scene preset."""
        self._highlight_scene(scene_name)
        if scene_name == "android":
            self.filter_manager.reset() if self.filter_manager else None
            self.filter_manager.enable_filter("Max Headroom") if self.filter_manager else None
        elif scene_name == "beauty":
            self.filter_manager.reset() if self.filter_manager else None
            self.filter_manager.enable_filter("Skin Smoothing") if self.filter_manager else None
        elif scene_name == "color":
            self.filter_manager.reset() if self.filter_manager else None
            self.filter_manager.enable_filter("Color Grading") if self.filter_manager else None
        self._update_filter_toggles()
        self.terminal_log.log(f"Scene activated: {scene_name}", "ok")
    
    def _highlight_scene(self, scene_name):
        """Highlight active scene in list."""
        for name, lbl in getattr(self, 'scene_btn_refs', {}).items():
            if name == scene_name:
                lbl.config(fg=Colors.CRT_CYAN, bg="#1a2a3a")
            else:
                lbl.config(fg=TEXT_DIM, bg=PANEL_BG)
    
    def _update_filter_toggles(self):
        """Update source list eye icons to show active state."""
        if not self.filter_manager:
            return
        for filt in self.filter_manager.filters:
            ref = self.filter_toggle_refs.get(filt.name)
            if ref:
                if filt.enabled:
                    ref["eye"].config(fg=ref["color"])
                    ref["label"].config(fg=Colors.CRT_CYAN)
                else:
                    ref["eye"].config(fg=TEXT_DIM)
                    ref["label"].config(fg=TEXT_BRIGHT)
    
    def _test_camera(self):
        """Test camera and show diagnostic info."""
        self.terminal_log.log("Running camera diagnostics...", "system")
        
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.terminal_log.log(f"Camera {i}: {w}x{h} - AVAILABLE", "ok")
                ret, frame = cap.read()
                if ret:
                    self.terminal_log.log(f"Camera {i}: Frame read OK", "ok")
                else:
                    self.terminal_log.log(f"Camera {i}: Cannot read frames", "alert")
            else:
                self.terminal_log.log(f"Camera {i}: Not available", "warning")
            cap.release()
        
        self.terminal_log.log("Diagnostics complete. Use --camera N to select.", "system")
    
    def _toggle_filter(self, name):
        if self.filter_manager:
            state = self.filter_manager.toggle_filter(name)
            status = "ACTIVE" if state else "OFFLINE"
            color = "ok" if state else "warning"
            self.terminal_log.log(f"Source '{name}': {status}", color)
            self._update_filter_toggles()
    
    def _reset_filters(self):
        if self.filter_manager:
            self.filter_manager.reset()
            self.terminal_log.log("All sources reset", "system")
            self._update_filter_toggles()
    
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
    
    def _on_keypress(self, event):
        """Handle keyboard shortcuts."""
        key = event.char.upper()
        if key == 'D':
            self._toggle_filter("Max Headroom")
        elif key == 'B':
            self._toggle_filter("Skin Smoothing")
        elif key == 'C':
            self._toggle_filter("Color Grading")
        elif key == 'G':
            self._toggle_filter("Background")
        elif key == 'A':
            self._toggle_filter("AR Overlay")
        elif key == 'M':
            self._toggle_filter("Face Morph")
        elif key == 'R':
            self._reset_filters()
        elif key == 'Q':
            self.on_close()
    
    def _update_status_indicators(self, tracking_active: bool):
        """Update LED status indicators based on system state."""
        if Colors is None:
            return
        if hasattr(self, 'cam_indicator'):
            if self.config.test_mode:
                self.cam_indicator.set_color(Colors.WARNING)
            elif self.cap and self.cap.isOpened():
                self.cam_indicator.set_color(Colors.OK)
            else:
                self.cam_indicator.set_color(Colors.ALERT)
        
        if hasattr(self, 'track_indicator'):
            self.track_indicator.set_color(Colors.OK if tracking_active else Colors.WARNING)
        
        if hasattr(self, 'ws_indicator'):
            self.ws_indicator.set_color(Colors.OK if self.ws_connected else Colors.ALERT)
    
    def on_connect(self):
        self.config.ws_host = self.host_entry.get()
        try:
            self.config.ws_port = int(self.port_entry.get())
        except ValueError:
            self.terminal_log.log("Invalid port number", "alert")
            return
        self.connect_websocket()
        if self.ws_connected:
            self.ws_status_text.config(text="ONLINE", fg=Colors.OK)
            self.terminal_log.log(f"WebSocket linked to {self.config.ws_host}:{self.config.ws_port}", "ok")
        else:
            self.ws_status_text.config(text="FAILED", fg=Colors.ALERT)
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
        self._update_status_indicators(True)
    
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
        """Main tracking and rendering loop with error recovery."""
        while True:
            if not self.running:
                time.sleep(0.05)
                continue
            
            frame_start = time.time()
            frame = None
            
            # ---- Camera acquisition with recovery ----
            if self.config.test_mode:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.rectangle(frame, (50, 50), (590, 430), (0, 150, 0), 2)
                cv2.putText(frame, "SIMULATION MODE", (220, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                if self.cap is None or not self.cap.isOpened():
                    now = time.time()
                    if now - self._camera_last_retry > 3.0:
                        self._camera_last_retry = now
                        self._try_log("Camera lost - attempting recovery...", "warning")
                        self._open_camera()
                        self._update_status_indicators(False)
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    self._camera_fail_count += 1
                    if self._camera_fail_count > 10:
                        self._try_log("Camera read failure - triggering recovery", "alert")
                        self._camera_fail_count = 0
                        self._camera_last_retry = 0
                    continue
                else:
                    self._camera_fail_count = 0
            
            # ---- Process frame ----
            t = time.time()
            try:
                data = self.process_frame(frame, t)
            except Exception as e:
                self._try_log(f"Processing error: {e}", "alert")
                continue
            
            # ---- Apply filters ----
            if self.filter_manager and frame is not None:
                try:
                    frame = self.filter_manager.process(
                        frame,
                        blendshapes=data.blendshapes,
                        head_pose=data.head_pose,
                        frame_id=self.frame_count
                    )
                except Exception as e:
                    self._try_log(f"Filter error: {e}", "warning")
            
            frame = self.draw_hologram(frame, data)
            
            # ---- WebSocket with auto-reconnect ----
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
            
            sent = self.send_websocket(payload)
            if not sent and self.config.enable_websocket and websocket:
                now = time.time()
                if now - self.ws_last_reconnect > 5.0:
                    self.ws_last_reconnect = now
                    self.ws_reconnect_attempts += 1
                    backoff = min(30, 2 ** self.ws_reconnect_attempts)
                    self._try_log(f"WS reconnect in {backoff}s (attempt {self.ws_reconnect_attempts})...", "warning")
                    if self.ws_reconnect_attempts <= 5:
                        time.sleep(min(backoff, 2))
                        if self.connect_websocket():
                            self._try_log("WebSocket reconnected", "ok")
                            if hasattr(self, 'ws_status_text'):
                                self.ws_status_text.config(text="ONLINE", fg=Colors.OK)
            
            # ---- Thread-safe GUI update ----
            if self.canvas and frame is not None and self.root:
                frame_time = (time.time() - frame_start) * 1000
                self._schedule_ui_update(frame, data, frame_time)
            
            # ---- FPS counter ----
            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = now
            
            self.frame_count += 1
            elapsed = time.time() - frame_start
            sleep_time = max(0, (1.0 / self.config.target_fps) - elapsed)
            time.sleep(sleep_time)
    
    def _try_log(self, message, level="system"):
        """Thread-safe logging that works before GUI is ready."""
        print(f"[{level.upper()}] {message}")
        if hasattr(self, 'terminal_log') and self.terminal_log:
            try:
                self.terminal_log.log(message, level)
            except:
                pass
    
    def _schedule_ui_update(self, frame, data, frame_time_ms):
        """Schedule GUI update from tracking thread with frame skip logic."""
        with self._ui_lock:
            # Replace queued frame (keep only latest)
            self._frame_queue.clear()
            self._frame_queue.append((frame.copy(), data, frame_time_ms))
            
            if self._ui_pending:
                return
            self._ui_pending = True
        
        def do_update():
            try:
                with self._ui_lock:
                    if not self._frame_queue:
                        self._ui_pending = False
                        return
                    frame, data, ft = self._frame_queue.popleft()
                    self._ui_pending = False
                
                self._update_video_canvas(frame)
                self._update_hud(data, frame)
                self._update_blendshape_bars(data)
                self._update_pose_labels(data)
                self._update_status_indicators(True)
                self._update_fps_display(ft)
            except Exception:
                self._ui_pending = False
        
        try:
            self.root.after(0, do_update)
        except:
            pass
    
    def _update_video_canvas(self, frame):
        """Update video canvas with latest frame."""
        try:
            from PIL import Image, ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Scale to canvas size maintaining aspect ratio
            h, w = frame_rgb.shape[:2]
            canvas_w = max(640, self.canvas.winfo_width())
            canvas_h = max(480, self.canvas.winfo_height())
            if canvas_w > 1 and canvas_h > 1:
                scale = min(canvas_w / w, canvas_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                if new_w != w or new_h != h:
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            pil_img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
        except Exception:
            pass
    
    def _update_hud(self, data, frame):
        """Update HUD overlay with face position."""
        if not self.config.test_mode and self.cap and frame is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.detector:
                    face_rect = self.detector.detect(gray)
                    if face_rect:
                        fx, fy, fw, fh = face_rect
                        # Get actual canvas size
                        cw = max(640, self.canvas.winfo_width())
                        ch = max(480, self.canvas.winfo_height())
                        hud_x = int((fx + fw / 2) / frame.shape[1] * cw)
                        hud_y = int((fy + fh / 2) / frame.shape[0] * ch)
                        if hasattr(self, 'hud_overlay'):
                            self.hud_overlay.set_target(hud_x, hud_y)
            except:
                pass
    
    def _update_blendshape_bars(self, data):
        """Update blendshape visualization with real data."""
        if hasattr(self, 'bs_bars') and data and data.blendshapes:
            self.bs_bars.update_values(data.blendshapes)
    
    def _update_pose_labels(self, data):
        """Update head pose display."""
        if hasattr(self, 'pose_rot_label') and data and data.head_pose:
            rot = data.head_pose.get("rotation", [0, 0, 0])
            trans = data.head_pose.get("translation", [0, 0, 0])
            self.pose_rot_label.config(text=f"ROT: {rot[0]:5.1f} {rot[1]:5.1f} {rot[2]:5.1f}")
            self.pose_trans_label.config(text=f"POS: {trans[0]:5.2f} {trans[1]:5.2f} {trans[2]:5.2f}")
    
    def _update_fps_display(self, frame_time_ms):
        """Update FPS and performance labels."""
        if hasattr(self, 'fps_label'):
            self.fps_label.config(text=f"FPS: {self.fps}")
        if hasattr(self, 'packets_label'):
            self.packets_label.config(text=f"PKT: {self.sent_count}")
        if hasattr(self, 'frame_time_label'):
            self.frame_time_label.config(text=f"FRAME: {frame_time_ms:.1f}ms")
        if hasattr(self, 'video_status_label'):
            mode = "SIMULATION" if self.config.test_mode else "LIVE"
            cam_ok = self.config.test_mode or (self.cap and self.cap.isOpened())
            self.video_status_label.config(text=f"CAMERA: {mode}", fg=Colors.OK if cam_ok else Colors.ALERT)

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
