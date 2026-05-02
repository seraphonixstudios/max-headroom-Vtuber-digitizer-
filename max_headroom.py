#!/usr/bin/env python3
"""
Max Headroom Digitizer v3.3 - Professional Broadcast Studio
Complete VTuber streaming application with FrameBus architecture.
Author: CRACKED-DEV-Omega
Version: 3.3.0
License: MIT
"""
import sys, os, cv2, numpy as np, time, json, threading, socket, argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

VERSION = "3.3.0"

# GUI check
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# WebSocket check
try:
    import websocket
except ImportError:
    websocket = None

# Optional imports
try:
    from gui_themes import Colors
except ImportError:
    Colors = None

try:
    from camera_manager import CameraManager, CameraInfo
except ImportError:
    CameraManager = None

@dataclass
class AppConfig:
    ws_host: str = "localhost"
    ws_port: int = 30000
    camera_index: int = 0
    target_fps: int = 30
    resolution_w: int = 640
    resolution_h: int = 480
    smoothing: float = 0.8
    enable_websocket: bool = True
    test_mode: bool = False
    glitch_intensity: float = 0.15

@dataclass
class FaceTrackingData:
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_pose: Dict[str, List[float]] = field(default_factory=dict)
    landmarks: List[Tuple[int, int]] = field(default_factory=list)
    timestamp: float = 0.0

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade")
    
    def detect(self, gray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        return tuple(faces[0]) if len(faces) > 0 else None

class BlendShapeCalculator:
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
            "eyeBlink_L": 0.0, "eyeBlink_R": 0.0,
            "browUp_L": 0.1 * (1 + np.sin(time_val * 1.5)),
            "browUp_R": 0.1 * (1 + np.sin(time_val * 1.5 + 0.3)),
            "cheekPuff": 0.2 + 0.1 * np.sin(time_val * 2.5),
            "noseSneer_L": max(0, 0.2 - 0.1 * np.sin(time_val)),
            "noseSneer_R": max(0, 0.2 - 0.1 * np.sin(time_val)),
            "mouthClose": 0.8, "mouthFunnel": 0.1, "mouthPucker": 0.1,
            "mouthLeft": 0.0, "mouthRight": 0.0,
            "mouthDimple_L": 0.0, "mouthDimple_R": 0.0,
            "mouthUpperUp_L": 0.0, "mouthUpperUp_R": 0.0,
            "browDown_L": 0.0, "browDown_R": 0.0,
            "eyeLookDown_L": 0.0, "eyeLookDown_R": 0.0,
            "eyeLookUp_L": 0.0, "eyeLookUp_R": 0.0,
            "eyeSquint_L": 0.0, "eyeSquint_R": 0.0,
            "jawForward": 0.0, "jawLeft": 0.0, "jawRight": 0.0,
            "cheekSquint_L": 0.0, "cheekSquint_R": 0.0,
        }
    
    def _test_shapes(self, t: float) -> Dict[str, float]:
        return {
            "jawOpen": 0.2 + 0.15 * np.sin(t * 2),
            "mouthSmile_L": 0.3 + 0.1 * np.sin(t * 3),
            "mouthSmile_R": 0.3 + 0.1 * np.sin(t * 3 + 0.5),
            "eyeBlink_L": 0.0, "eyeBlink_R": 0.0,
            "browUp_L": 0.1 * (1 + np.sin(t * 1.5)),
            "browUp_R": 0.1 * (1 + np.sin(t * 1.5 + 0.3)),
            "cheekPuff": 0.2 + 0.1 * np.sin(t * 2.5),
            "noseSneer_L": 0.1 * (1 + np.sin(t)),
            "noseSneer_R": 0.1 * (1 + np.sin(t)),
            "mouthClose": 0.8 - 0.1 * np.sin(t * 2),
            "mouthFunnel": 0.1, "mouthPucker": 0.1,
            "mouthLeft": 0.0, "mouthRight": 0.0,
            "mouthDimple_L": 0.0, "mouthDimple_R": 0.0,
            "mouthUpperUp_L": 0.0, "mouthUpperUp_R": 0.0,
            "browDown_L": 0.0, "browDown_R": 0.0,
            "eyeLookDown_L": 0.0, "eyeLookDown_R": 0.0,
            "eyeLookUp_L": 0.0, "eyeLookUp_R": 0.0,
            "eyeSquint_L": 0.0, "eyeSquint_R": 0.0,
            "jawForward": 0.0, "jawLeft": 0.0, "jawRight": 0.0,
            "cheekSquint_L": 0.0, "cheekSquint_R": 0.0,
        }

class MaxHeadroomApp:
    WINDOW_NAME = "Max Headroom Digitizer"
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.cam_mgr = None
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
        self.blendshape_buffer = {}
        self.filter_manager = None
        self._init_filters()
        self._frame_queue = deque(maxlen=2)
        self._ui_lock = threading.Lock()
        self._ui_pending = False
        self._last_log_time = 0
    
    def _init_filters(self):
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
        
        if not self.config.test_mode and CameraManager is not None:
            self.cam_mgr = CameraManager(timeout=3.0)
            ok = self.cam_mgr.open(self.config.camera_index, self.config.resolution_w, self.config.resolution_h)
            if not ok:
                print("[Init] Camera failed - using TEST MODE")
                self.config.test_mode = True
        return True
    
    def connect_websocket(self) -> bool:
        if not self.config.enable_websocket or not websocket:
            return False
        try:
            self.ws = websocket.create_connection(
                f"ws://{self.config.ws_host}:{self.config.ws_port}", timeout=3)
            self.ws_connected = True
            return True
        except Exception as e:
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
            self.ws_connected = False
            return False
    
    def _smooth_blendshapes(self, blends: Dict[str, float]) -> Dict[str, float]:
        s = self.config.smoothing
        smoothed = {}
        for name, value in blends.items():
            if name in self.blendshape_buffer:
                smoothed[name] = self.blendshape_buffer[name] * s + value * (1 - s)
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
        cx, cy = x + w / 2, y + h / 2
        nx, ny = (cx - fw / 2) / fw * 2, (cy - fh / 2) / fh * 2
        return {
            "rotation": [ny * 15, nx * 20, 0],
            "translation": [nx, -ny, max(0.5, min(3.0, 200 / w))]
        }
    
    def _try_log(self, msg, level="system"):
        now = time.time()
        if now - self._last_log_time > 0.1:  # Rate limit logs
            self._last_log_time = now
            print(f"[{level.upper()}] {msg}")
        if hasattr(self, 'terminal_log') and self.terminal_log:
            try:
                self.terminal_log.log(msg, level)
            except:
                pass
    
    # ========================================================================
    # GUI - Professional Broadcast Studio
    # ========================================================================
    def run_gui(self):
        if not GUI_AVAILABLE:
            return self.run_cli()
        
        self.init()
        
        try:
            from gui_themes import (
                Colors, MatrixRainCanvas, SacredGeometryCanvas, CRTOverlayCanvas,
                NeonButton, CrystallineFrame, TerminalLog, HUDOverlay,
                GlitchLabel, CrystalProgressBar, HexDisplay, WaveformCanvas,
                StatusIndicator, BlendshapeBars, apply_dark_theme
            )
            THEMES_AVAILABLE = True
        except Exception as e:
            print(f"[GUI] Themes not available: {e}")
            return self._run_basic_gui()
        
        # Color palette
        C = Colors
        PANEL = "#0d1117"
        BORDER = "#21262d"
        ACCENT = "#00d4ff"
        TEXT = "#c9d1d9"
        TEXT_DIM = "#8b949e"
        
        self.root = tk.Tk()
        self.root.title(f"{self.WINDOW_NAME} v{VERSION}")
        self.root.geometry("1400x900")
        self.root.configure(bg=C.VOID_BLACK)
        self.root.minsize(1200, 800)
        apply_dark_theme(self.root)
        
        # ===================================================================
        # TOP HEADER
        # ===================================================================
        header = tk.Frame(self.root, bg=C.VOID_BLACK, height=40)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        tk.Label(header, text="MAX HEADROOM", fg=ACCENT, bg=C.VOID_BLACK,
                font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(header, text=f"STUDIO  v{VERSION}", fg=TEXT_DIM, bg=C.VOID_BLACK,
                font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 20))
        
        self.mode_label = tk.Label(header, text="STANDBY", fg=C.WARNING, bg=C.VOID_BLACK,
                                  font=("Segoe UI", 10, "bold"))
        self.mode_label.pack(side=tk.LEFT, padx=40)
        
        ind_frame = tk.Frame(header, bg=C.VOID_BLACK)
        ind_frame.pack(side=tk.RIGHT, padx=16)
        self.cam_indicator = StatusIndicator(ind_frame, color=C.ALERT)
        self.cam_indicator.pack(side=tk.LEFT, padx=2)
        tk.Label(ind_frame, text="CAM", fg=TEXT_DIM, bg=C.VOID_BLACK, font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=(0, 10))
        self.track_indicator = StatusIndicator(ind_frame, color=C.ALERT)
        self.track_indicator.pack(side=tk.LEFT, padx=2)
        tk.Label(ind_frame, text="TRACK", fg=TEXT_DIM, bg=C.VOID_BLACK, font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=(0, 10))
        self.ws_indicator = StatusIndicator(ind_frame, color=C.ALERT)
        self.ws_indicator.pack(side=tk.LEFT, padx=2)
        tk.Label(ind_frame, text="NET", fg=TEXT_DIM, bg=C.VOID_BLACK, font=("Segoe UI", 7)).pack(side=tk.LEFT)
        
        # ===================================================================
        # MAIN WORKSPACE
        # ===================================================================
        workspace = tk.Frame(self.root, bg=C.VOID_BLACK)
        workspace.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # ---- LEFT: Preview ----
        preview = tk.Frame(workspace, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        preview_hdr = tk.Frame(preview, bg=PANEL, height=28)
        preview_hdr.pack(fill=tk.X)
        preview_hdr.pack_propagate(False)
        tk.Label(preview_hdr, text="PREVIEW", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        self.rec_label = tk.Label(preview_hdr, text="", fg=C.ALERT, bg=PANEL, font=("Segoe UI", 9, "bold"))
        self.rec_label.pack(side=tk.LEFT, padx=8)
        self.res_label = tk.Label(preview_hdr, text="640x480", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8))
        self.res_label.pack(side=tk.RIGHT, padx=10)
        
        video_container = tk.Frame(preview, bg=C.VOID_BLACK)
        video_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.canvas = tk.Canvas(video_container, bg=C.VOID_BLACK, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.crt_overlay = CRTOverlayCanvas(video_container, width=640, height=480)
        self.crt_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self.crt_overlay.start()
        
        self.hud_overlay = HUDOverlay(video_container, width=640, height=480)
        self.hud_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self.hud_overlay.start()
        
        # ---- RIGHT: Dock ----
        dock = tk.Frame(workspace, bg=C.VOID_BLACK, width=360)
        dock.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        dock.pack_propagate(False)
        
        # --- SCENES ---
        scenes = tk.Frame(dock, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        scenes.pack(fill=tk.X, pady=(0, 4))
        tk.Frame(scenes, bg=PANEL, height=26).pack(fill=tk.X)
        scenes_hdr = tk.Frame(scenes, bg=PANEL)
        scenes_hdr.place(x=0, y=0, relwidth=1, height=26)
        tk.Label(scenes_hdr, text="SCENES", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        
        scenes_list = tk.Frame(scenes, bg=PANEL)
        scenes_list.pack(fill=tk.X, padx=6, pady=4)
        self.scene_refs = {}
        for name in ["Default", "Android Mode", "Beauty Mode", "Color Grade"]:
            lbl = tk.Label(scenes_list, text=f"  {name}", fg=TEXT, bg=PANEL,
                          font=("Consolas", 10), anchor="w", cursor="hand2")
            lbl.pack(fill=tk.X, pady=1)
            lbl.bind("<Button-1>", lambda e, n=name: self._activate_scene(n))
            self.scene_refs[name] = lbl
        self._highlight_scene("Default")
        
        # --- SOURCES ---
        sources = tk.Frame(dock, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        sources.pack(fill=tk.X, pady=(0, 4))
        tk.Frame(sources, bg=PANEL, height=26).pack(fill=tk.X)
        src_hdr = tk.Frame(sources, bg=PANEL)
        src_hdr.place(x=0, y=0, relwidth=1, height=26)
        tk.Label(src_hdr, text="SOURCES", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        
        src_list = tk.Frame(sources, bg=PANEL)
        src_list.pack(fill=tk.X, padx=6, pady=4)
        self.filter_toggle_refs = {}
        filter_items = [
            ("Max Headroom", "MH", C.NEON_PINK),
            ("Skin Smoothing", "SKIN", C.ATLANTEAN_TEAL),
            ("Background", "BG", C.PLASMA_BLUE),
            ("AR Overlay", "AR", C.SACRED_GOLD),
            ("Face Morph", "MORPH", C.NEON_ORANGE),
            ("Color Grading", "COLOR", C.MATRIX_GREEN),
        ]
        for name, abbr, color in filter_items:
            row = tk.Frame(src_list, bg=PANEL)
            row.pack(fill=tk.X, pady=2)
            eye = tk.Label(row, text="●", fg=TEXT_DIM, bg=PANEL, font=("Consolas", 10))
            eye.pack(side=tk.LEFT, padx=(4, 4))
            lbl = tk.Label(row, text=name, fg=TEXT, bg=PANEL, font=("Consolas", 10), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ab = tk.Label(row, text=abbr, fg=color, bg=PANEL, font=("Consolas", 8, "bold"))
            ab.pack(side=tk.RIGHT, padx=4)
            for w in (row, eye, lbl, ab):
                w.bind("<Button-1>", lambda e, n=name: self._toggle_filter(n))
            self.filter_toggle_refs[name] = {"eye": eye, "label": lbl, "color": color}
        
        # Glitch slider
        slider_row = tk.Frame(sources, bg=PANEL)
        slider_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(slider_row, text="GLITCH:", fg=TEXT_DIM, bg=PANEL, font=("Consolas", 8)).pack(side=tk.LEFT)
        self.glitch_scale = tk.Scale(slider_row, from_=0, to=100, orient=tk.HORIZONTAL,
                                    fg=ACCENT, bg=PANEL, troughcolor=BORDER,
                                    highlightthickness=0, command=self.on_glitch_change,
                                    length=220, showvalue=0)
        self.glitch_scale.set(int(self.config.glitch_intensity * 100))
        self.glitch_scale.pack(side=tk.LEFT, padx=4)
        
        # --- MIXER ---
        mixer = tk.Frame(dock, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        mixer.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
        tk.Frame(mixer, bg=PANEL, height=26).pack(fill=tk.X)
        mix_hdr = tk.Frame(mixer, bg=PANEL)
        mix_hdr.place(x=0, y=0, relwidth=1, height=26)
        tk.Label(mix_hdr, text="MIXER", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        
        mix_inner = tk.Frame(mixer, bg=PANEL)
        mix_inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.bs_bars = BlendshapeBars(mix_inner, width=340, height=260, max_bars=14)
        self.bs_bars.pack(fill=tk.BOTH, expand=True)
        
        # --- TRANSITIONS ---
        trans = tk.Frame(dock, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        trans.pack(fill=tk.X, pady=(0, 4))
        tk.Frame(trans, bg=PANEL, height=26).pack(fill=tk.X)
        tr_hdr = tk.Frame(trans, bg=PANEL)
        tr_hdr.place(x=0, y=0, relwidth=1, height=26)
        tk.Label(tr_hdr, text="CONTROLS", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        
        tr_inner = tk.Frame(trans, bg=PANEL)
        tr_inner.pack(fill=tk.X, padx=10, pady=8)
        
        self.connect_btn = tk.Button(tr_inner, text="LINK", bg=BORDER, fg=ACCENT,
                                    font=("Consolas", 9, "bold"), relief=tk.FLAT,
                                    cursor="hand2", command=self.on_connect)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 4))
        
        self.test_btn = tk.Button(tr_inner, text="TEST CAM", bg=BORDER, fg=TEXT,
                                 font=("Consolas", 9, "bold"), relief=tk.FLAT,
                                 cursor="hand2", command=self._test_camera)
        self.test_btn.pack(side=tk.LEFT, padx=4)
        
        self.ws_status_text = tk.Label(tr_inner, text="OFFLINE", fg=C.ALERT,
                                      bg=PANEL, font=("Consolas", 9, "bold"))
        self.ws_status_text.pack(side=tk.RIGHT)
        
        net_row = tk.Frame(trans, bg=PANEL)
        net_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(net_row, text="HOST:", fg=TEXT_DIM, bg=PANEL, font=("Consolas", 8)).pack(side=tk.LEFT)
        self.host_entry = tk.Entry(net_row, width=12, fg=TEXT, bg=C.VOID_BLACK,
                                  insertbackground=ACCENT, font=("Consolas", 9), relief=tk.FLAT)
        self.host_entry.insert(0, self.config.ws_host)
        self.host_entry.pack(side=tk.LEFT, padx=4)
        tk.Label(net_row, text="PORT:", fg=TEXT_DIM, bg=PANEL, font=("Consolas", 8)).pack(side=tk.LEFT, padx=(8, 0))
        self.port_entry = tk.Entry(net_row, width=6, fg=TEXT, bg=C.VOID_BLACK,
                                  insertbackground=ACCENT, font=("Consolas", 9), relief=tk.FLAT)
        self.port_entry.insert(0, str(self.config.ws_port))
        self.port_entry.pack(side=tk.LEFT, padx=4)
        
        # ===================================================================
        # BOTTOM DOCK
        # ===================================================================
        bottom = tk.Frame(self.root, bg=C.VOID_BLACK, height=150)
        bottom.pack(fill=tk.X, padx=8, pady=(4, 8))
        bottom.pack_propagate(False)
        
        # Head Pose
        pose_f = tk.Frame(bottom, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        pose_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Frame(pose_f, bg=PANEL, height=24).pack(fill=tk.X)
        ph = tk.Frame(pose_f, bg=PANEL)
        ph.place(x=0, y=0, relwidth=1, height=24)
        tk.Label(ph, text="HEAD POSE", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        pi = tk.Frame(pose_f, bg=PANEL)
        pi.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.pose_rot_label = tk.Label(pi, text="ROT:   0.00   0.00   0.00", fg=TEXT, bg=PANEL, font=("Consolas", 10))
        self.pose_rot_label.pack(anchor="w", pady=2)
        self.pose_trans_label = tk.Label(pi, text="POS:   0.00   0.00   0.00", fg=TEXT, bg=PANEL, font=("Consolas", 10))
        self.pose_trans_label.pack(anchor="w", pady=2)
        
        # System Core
        sys_f = tk.Frame(bottom, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        sys_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Frame(sys_f, bg=PANEL, height=24).pack(fill=tk.X)
        sh = tk.Frame(sys_f, bg=PANEL)
        sh.place(x=0, y=0, relwidth=1, height=24)
        tk.Label(sh, text="SYSTEM CORE", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        si = tk.Frame(sys_f, bg=PANEL)
        si.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        st = tk.Frame(si, bg=PANEL)
        st.pack(fill=tk.X)
        self.sacred_geo = SacredGeometryCanvas(st, width=60, height=60)
        self.sacred_geo.pack(side=tk.LEFT, padx=4)
        self.sacred_geo.start()
        self.hex_display = HexDisplay(st, rows=4, cols=5, width=120, height=60)
        self.hex_display.pack(side=tk.RIGHT, padx=4)
        self.hex_display.start()
        self.test_var = tk.BooleanVar(value=self.config.test_mode)
        tk.Checkbutton(si, text="SIMULATION", variable=self.test_var,
                      fg=C.MATRIX_GREEN, bg=PANEL, selectcolor=C.VOID_BLACK,
                      command=self.on_toggle_test, font=("Consolas", 8)).pack(anchor="w", pady=(4, 0))
        
        # Console
        log_f = tk.Frame(bottom, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        log_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Frame(log_f, bg=PANEL, height=24).pack(fill=tk.X)
        lh = tk.Frame(log_f, bg=PANEL)
        lh.place(x=0, y=0, relwidth=1, height=24)
        tk.Label(lh, text="CONSOLE", fg=TEXT_DIM, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=10)
        self.terminal_log = TerminalLog(log_f, height=6, width=60)
        self.terminal_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # ===================================================================
        # STATUS BAR
        # ===================================================================
        status = tk.Frame(self.root, bg=PANEL, height=28)
        status.pack(fill=tk.X, padx=0, pady=0)
        status.pack_propagate(False)
        self.video_status_label = tk.Label(status, text="CAMERA: STANDBY", fg=C.WARNING, bg=PANEL, font=("Consolas", 9, "bold"))
        self.video_status_label.pack(side=tk.LEFT, padx=12)
        self.fps_label = tk.Label(status, text="FPS: 0", fg=ACCENT, bg=PANEL, font=("Consolas", 9))
        self.fps_label.pack(side=tk.LEFT, padx=12)
        self.packets_label = tk.Label(status, text="PKT: 0", fg=TEXT_DIM, bg=PANEL, font=("Consolas", 9))
        self.packets_label.pack(side=tk.LEFT, padx=12)
        self.frame_time_label = tk.Label(status, text="FRAME: 0ms", fg=TEXT_DIM, bg=PANEL, font=("Consolas", 9))
        self.frame_time_label.pack(side=tk.LEFT, padx=12)
        tk.Label(status, text="D:Android B:Beauty C:Color G:BG A:AR M:Morph R:Reset",
                fg=TEXT_DIM, bg=PANEL, font=("Consolas", 8)).pack(side=tk.RIGHT, padx=12)
        
        # Init log
        self.terminal_log.log(f"Studio v{VERSION} initialized", "system")
        if self.config.test_mode:
            self.terminal_log.log("Simulation mode active - no camera required", "warning")
        else:
            self.terminal_log.log("Camera ready", "ok")
        self.terminal_log.log("Press TEST CAM to scan for cameras", "system")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Key>", self._on_keypress)
        
        self.running = True
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()
        self.root.mainloop()
    
    def _run_basic_gui(self):
        self.root = tk.Tk()
        self.root.title(self.WINDOW_NAME)
        self.root.geometry("800x700")
        self.root.configure(bg="black")
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black", highlightthickness=0)
        self.canvas.pack(pady=5)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.running = True
        threading.Thread(target=self._tracking_loop, daemon=True).start()
        self.root.mainloop()
    
    def _on_keypress(self, event):
        key = event.char.upper()
        if key == 'D': self._toggle_filter("Max Headroom")
        elif key == 'B': self._toggle_filter("Skin Smoothing")
        elif key == 'C': self._toggle_filter("Color Grading")
        elif key == 'G': self._toggle_filter("Background")
        elif key == 'A': self._toggle_filter("AR Overlay")
        elif key == 'M': self._toggle_filter("Face Morph")
        elif key == 'R': self._reset_filters()
        elif key == 'Q': self.on_close()
    
    def _toggle_filter(self, name):
        if self.filter_manager:
            state = self.filter_manager.toggle_filter(name)
            status = "ON" if state else "OFF"
            col = "ok" if state else "warning"
            self.terminal_log.log(f"{name}: {status}", col)
            self._update_filter_toggles()
    
    def _reset_filters(self):
        if self.filter_manager:
            self.filter_manager.reset()
            self.terminal_log.log("All sources reset", "system")
            self._update_filter_toggles()
    
    def _update_filter_toggles(self):
        if not self.filter_manager:
            return
        for filt in self.filter_manager.filters:
            ref = self.filter_toggle_refs.get(filt.name)
            if ref:
                if filt.enabled:
                    ref["eye"].config(fg=ref["color"])
                    ref["label"].config(fg=Colors.CRT_CYAN if Colors else "cyan")
                else:
                    ref["eye"].config(fg="#8b949e")
                    ref["label"].config(fg="#c9d1d9")
    
    def _activate_scene(self, scene_name):
        self._highlight_scene(scene_name)
        if not self.filter_manager:
            return
        self.filter_manager.reset()
        if scene_name == "Android Mode":
            self.filter_manager.enable_filter("Max Headroom")
        elif scene_name == "Beauty Mode":
            self.filter_manager.enable_filter("Skin Smoothing")
        elif scene_name == "Color Grade":
            self.filter_manager.enable_filter("Color Grading")
        self._update_filter_toggles()
        self.terminal_log.log(f"Scene: {scene_name}", "ok")
    
    def _highlight_scene(self, scene_name):
        for name, lbl in getattr(self, 'scene_refs', {}).items():
            if name == scene_name:
                lbl.config(fg=Colors.CRT_CYAN if Colors else "cyan", bg="#1a2a3a")
            else:
                lbl.config(fg="#c9d1d9", bg="#0d1117")
    
    def _test_camera(self):
        self.terminal_log.log("Scanning cameras...", "system")
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if ret:
                    self.terminal_log.log(f"Camera {i}: {w}x{h} OK", "ok")
                else:
                    self.terminal_log.log(f"Camera {i}: open but no frames", "warning")
            else:
                self.terminal_log.log(f"Camera {i}: not available", "warning")
            cap.release()
        self.terminal_log.log("Scan complete", "system")
    
    def on_connect(self):
        self.config.ws_host = self.host_entry.get()
        try:
            self.config.ws_port = int(self.port_entry.get())
        except ValueError:
            self.terminal_log.log("Invalid port", "alert")
            return
        self.connect_websocket()
        if self.ws_connected:
            self.ws_status_text.config(text="ONLINE", fg=Colors.OK if Colors else "green")
            self.terminal_log.log(f"WS connected to {self.config.ws_host}:{self.config.ws_port}", "ok")
        else:
            self.ws_status_text.config(text="FAILED", fg=Colors.ALERT if Colors else "red")
            self.terminal_log.log("WS connection failed", "alert")
    
    def on_toggle_test(self):
        self.config.test_mode = self.test_var.get()
        mode = "SIMULATION" if self.config.test_mode else "LIVE"
        self.terminal_log.log(f"Mode: {mode}", "system")
        self._update_status(True)
    
    def on_glitch_change(self, value):
        self.config.glitch_intensity = int(value) / 100
    
    def on_close(self):
        self.running = False
        try:
            for attr in ['crt_overlay', 'hud_overlay', 'sacred_geo', 'hex_display']:
                obj = getattr(self, attr, None)
                if obj and hasattr(obj, 'stop'):
                    obj.stop()
        except:
            pass
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
    
    def _update_status(self, tracking_active: bool):
        if not Colors:
            return
        if hasattr(self, 'cam_indicator'):
            if self.config.test_mode:
                self.cam_indicator.set_color(Colors.WARNING)
            elif self.cam_mgr and self.cam_mgr.is_opened():
                self.cam_indicator.set_color(Colors.OK)
            else:
                self.cam_indicator.set_color(Colors.ALERT)
        if hasattr(self, 'track_indicator'):
            self.track_indicator.set_color(Colors.OK if tracking_active else Colors.WARNING)
        if hasattr(self, 'ws_indicator'):
            self.ws_indicator.set_color(Colors.OK if self.ws_connected else Colors.ALERT)
    
    # ========================================================================
    # TRACKING LOOP
    # ========================================================================
    def _tracking_loop(self):
        while True:
            if not self.running:
                time.sleep(0.05)
                continue
            
            t_start = time.time()
            frame = None
            
            # Get frame
            if self.config.test_mode:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.rectangle(frame, (50, 50), (590, 430), (0, 150, 0), 2)
                cv2.putText(frame, "SIMULATION MODE", (180, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                if self.cam_mgr:
                    frame = self.cam_mgr.read()
                if frame is None and self.cam_mgr and not self.cam_mgr.is_opened():
                    # Try recovery
                    time.sleep(0.1)
                    continue
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process
            t = time.time()
            try:
                face_rect = None
                if not self.config.test_mode and self.detector is not None:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        face_rect = self.detector.detect(gray)
                    except Exception as e:
                        self._try_log(f"Detection error: {e}", "warning")
                
                blends = self.blendshape_calc.calculate(face_rect, t)
                if self.config.smoothing > 0:
                    blends = self._smooth_blendshapes(blends)
                
                pose = self._calculate_pose(face_rect, frame.shape[:2], t)
                data = FaceTrackingData(blendshapes=blends, head_pose=pose, timestamp=t)
                
                # Filters
                if self.filter_manager:
                    try:
                        frame = self.filter_manager.process(frame, blendshapes=blends,
                                                            head_pose=pose, frame_id=self.frame_count)
                    except Exception as e:
                        self._try_log(f"Filter error: {e}", "warning")
                
                # Draw overlay
                frame = self._draw_overlay(frame, data)
                
                # WebSocket
                payload = {
                    "type": "face_data",
                    "blendshapes": blends,
                    "head_pose": pose,
                    "timestamp": t,
                }
                if self.filter_manager:
                    active = [f.name for f in self.filter_manager.filters if f.enabled]
                    if active:
                        payload["filter_status"] = {"active": active}
                self.send_websocket(payload)
                
                # Schedule GUI update
                if hasattr(self, 'canvas') and self.root:
                    ft = (time.time() - t_start) * 1000
                    self._schedule_ui(frame, data, ft)
                
                # FPS
                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                
                self.frame_count += 1
                
            except Exception as e:
                self._try_log(f"Processing error: {e}", "alert")
            
            elapsed = time.time() - t_start
            sleep_t = max(0, (1.0 / self.config.target_fps) - elapsed)
            time.sleep(sleep_t)
    
    def _draw_overlay(self, frame, data):
        h, w = frame.shape[:2]
        # Scanlines
        for y in range(0, h, 3):
            cv2.line(frame, (0, y), (w, y), (0, 50, 0), 1)
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 20, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 2), (0, 255, 0), 2)
        # Title
        offset = int(np.sin(time.time() * 10) * self.config.glitch_intensity * 20) if self.config.glitch_intensity > 0 else 0
        cv2.putText(frame, "MAX HEADROOM", (15 + offset, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # FPS
        cv2.putText(frame, f"FPS: {self.fps}", (w - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Test mode
        if self.config.test_mode:
            cv2.putText(frame, "TEST MODE", (w // 2 - 80, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Blendshape bars
        y_pos = 55
        for name in list(data.blendshapes.keys())[:8]:
            val = data.blendshapes.get(name, 0)
            bar_w = int(val * 100)
            cv2.rectangle(frame, (10, y_pos), (110, y_pos + 14), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, y_pos), (10 + bar_w, y_pos + 14), (0, 255, 0), -1)
            cv2.putText(frame, f"{name}: {val:.2f}", (120, y_pos + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            y_pos += 18
        # WS status
        ws_color = (0, 255, 0) if self.ws_connected else (0, 0, 255)
        cv2.putText(frame, "WS: ON" if self.ws_connected else "WS: OFF", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ws_color, 2)
        # Filters
        if self.filter_manager:
            active = [f["name"] for f in self.filter_manager.get_all_status() if f["enabled"]]
            if active:
                cv2.putText(frame, "FILTERS: " + ", ".join(active[:3]), (10, h - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        # RGB split
        if self.config.glitch_intensity > 0.1:
            split = int(self.config.glitch_intensity * 15)
            if split > 0:
                frame[:, split:] = frame[:, :-split]
        return frame
    
    def _schedule_ui(self, frame, data, frame_time_ms):
        with self._ui_lock:
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
                
                self._update_canvas(frame)
                self._update_hud(frame)
                self.bs_bars.update_values(data.blendshapes)
                self.pose_rot_label.config(text=f"ROT:  {data.head_pose.get('rotation', [0,0,0])[0]:6.1f}  {data.head_pose.get('rotation', [0,0,0])[1]:6.1f}  {data.head_pose.get('rotation', [0,0,0])[2]:6.1f}")
                self.pose_trans_label.config(text=f"POS:  {data.head_pose.get('translation', [0,0,0])[0]:6.2f}  {data.head_pose.get('translation', [0,0,0])[1]:6.2f}  {data.head_pose.get('translation', [0,0,0])[2]:6.2f}")
                self._update_status(True)
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.packets_label.config(text=f"PKT: {self.sent_count}")
                self.frame_time_label.config(text=f"FRAME: {ft:.1f}ms")
                mode = "SIM" if self.config.test_mode else "LIVE"
                cam_ok = self.config.test_mode or (self.cam_mgr and self.cam_mgr.is_opened())
                self.video_status_label.config(text=f"CAMERA: {mode}", fg=Colors.OK if Colors and cam_ok else "green")
            except Exception:
                self._ui_pending = False
        
        try:
            self.root.after(0, do_update)
        except:
            pass
    
    def _update_canvas(self, frame):
        try:
            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            cw = max(640, self.canvas.winfo_width())
            ch = max(480, self.canvas.winfo_height())
            if cw > 1 and ch > 1:
                scale = min(cw / w, ch / h)
                nw, nh = int(w * scale), int(h * scale)
                if nw != w or nh != h:
                    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pil = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            self.res_label.config(text=f"{w}x{h}")
        except Exception:
            pass
    
    def _update_hud(self, frame):
        if not self.config.test_mode and self.detector and frame is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rect = self.detector.detect(gray)
                if face_rect:
                    fx, fy, fw, fh = face_rect
                    cw = max(640, self.canvas.winfo_width())
                    ch = max(480, self.canvas.winfo_height())
                    hx = int((fx + fw / 2) / frame.shape[1] * cw)
                    hy = int((fy + fh / 2) / frame.shape[0] * ch)
                    if hasattr(self, 'hud_overlay'):
                        self.hud_overlay.set_target(hx, hy)
            except:
                pass
    
    # ========================================================================
    # CLI MODE
    # ========================================================================
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
                if self.cam_mgr:
                    frame = self.cam_mgr.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
            
            t = time.time()
            face_rect = None
            if not self.config.test_mode and self.detector:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_rect = self.detector.detect(gray)
                except:
                    pass
            
            blends = self.blendshape_calc.calculate(face_rect, t)
            if self.config.smoothing > 0:
                blends = self._smooth_blendshapes(blends)
            pose = self._calculate_pose(face_rect, frame.shape[:2], t)
            
            if self.filter_manager:
                frame = self.filter_manager.process(frame, blendshapes=blends,
                                                    head_pose=pose, frame_id=self.frame_count)
            
            frame = self._draw_overlay(frame, FaceTrackingData(blendshapes=blends, head_pose=pose, timestamp=t))
            
            payload = {"type": "face_data", "blendshapes": blends, "head_pose": pose, "timestamp": t}
            if self.filter_manager:
                active = [f.name for f in self.filter_manager.filters if f.enabled]
                if active:
                    payload["filter_status"] = {"active": active}
            self.send_websocket(payload)
            
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
                print(f"FPS: {self.fps} | Sent: {self.sent_count}")
            
            self.frame_count += 1
            cv2.imshow(self.WINDOW_NAME, frame)
            key = cv2.waitKey(int(1000 / self.config.target_fps)) & 0xFF
            if key == ord('q'):
                break
            if self.filter_manager:
                if key == ord('d') or key == ord('D'):
                    self.filter_manager.toggle_filter("Max Headroom")
                if key == ord('b') or key == ord('B'):
                    self.filter_manager.toggle_filter("Skin Smoothing")
                if key == ord('c') or key == ord('C'):
                    self.filter_manager.toggle_filter("Color Grading")
                if key == ord('r') or key == ord('R'):
                    self.filter_manager.reset()
        
        self.running = False
        if self.cam_mgr:
            self.cam_mgr.close()
        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False
        if self.cam_mgr:
            self.cam_mgr.close()
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
