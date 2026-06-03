#!/usr/bin/env python3
"""
Max Headroom Digitizer v3.4 - Professional VTuber Studio
Drastically improved UI, filter quality, and background removal.
Version: 3.4.1
"""
import sys, os, cv2, numpy as np, time, json, threading, socket, argparse, random, math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

VERSION = "3.4.1"

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    import websocket
except ImportError:
    websocket = None

try:
    from gui_themes import Colors
except ImportError:
    Colors = None

try:
    from camera_manager import CameraManager
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
    def detect(self, gray):
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        return tuple(faces[0]) if len(faces) > 0 else None

class BlendShapeCalculator:
    SHAPES = [
        "browDown_L", "browDown_R", "browUp_L", "browUp_R",
        "cheekPuff", "cheekSquint_L", "cheekSquint_R",
        "eyeBlink_L", "eyeBlink_R", "eyeLookDown_L", "eyeLookDown_R",
        "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimple_L", "mouthDimple_R",
        "mouthFunnel", "mouthLeft", "mouthPucker",
        "mouthRight", "mouthSmile_L", "mouthSmile_R",
        "mouthUpperUp_L", "mouthUpperUp_R",
        "noseSneer_L", "noseSneer_R",
    ]
    def calculate(self, face_rect, time_val):
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
    def _test_shapes(self, t):
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
    
    def __init__(self, config=None):
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
        # GUI vars (initialized even without GUI for test compatibility)
        self.bg_mode_var = None
        self.cam_var = None
        self.quality_scale = None
        # Dev console tracking
        self._drawn_count = 0
        self._last_dev_err = "none"
        self._pipeline_active = False
    
    def _init_filters(self):
        try:
            from filters import FilterManager
            self.filter_manager = FilterManager()
        except Exception as e:
            print(f"[Init] Filters not available: {e}")
    
    def init(self):
        print("[Init] Loading face detector...")
        try:
            self.detector = FaceDetector()
            print("[Init] Face detector loaded")
        except Exception as e:
            print(f"[Init] Face detector failed: {e}")
            self.detector = None
        if not self.config.test_mode and CameraManager is not None:
            self.cam_mgr = CameraManager(timeout=3.0)
            # Try configured index first, then scan for any available camera
            ok = self.cam_mgr.open(self.config.camera_index, self.config.resolution_w, self.config.resolution_h)
            if not ok:
                print(f"[Init] Camera {self.config.camera_index} failed, scanning...")
                cams = CameraManager.discover(max_index=5)
                if cams:
                    idx = cams[0].index
                    print(f"[Init] Found camera at index {idx}")
                    ok = self.cam_mgr.open(idx, self.config.resolution_w, self.config.resolution_h)
                    if ok:
                        self.config.camera_index = idx
                if not ok:
                    print("[Init] No camera found - using TEST MODE")
                    self.config.test_mode = True
        return True
    
    def connect_websocket(self):
        if not self.config.enable_websocket or not websocket:
            return False
        try:
            self.ws = websocket.create_connection(f"ws://{self.config.ws_host}:{self.config.ws_port}", timeout=3)
            self.ws_connected = True
            return True
        except:
            self.ws_connected = False
            return False
    
    def send_websocket(self, data):
        if not self.ws or not self.ws_connected:
            return False
        try:
            self.ws.send(json.dumps(data))
            self.sent_count += 1
            return True
        except:
            self.ws_connected = False
            return False
    
    def _smooth_blendshapes(self, blends):
        s = self.config.smoothing
        smoothed = {}
        for name, value in blends.items():
            if name in self.blendshape_buffer:
                smoothed[name] = self.blendshape_buffer[name] * s + value * (1 - s)
            else:
                smoothed[name] = value
            self.blendshape_buffer[name] = smoothed[name]
        return smoothed
    
    def _calculate_pose(self, face_rect, frame_shape, time_val):
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
        if now - self._last_log_time > 0.1:
            self._last_log_time = now
            print(f"[{level.upper()}] {msg}")
        if hasattr(self, 'terminal_log') and self.terminal_log:
            try:
                self.terminal_log.log(msg, level)
            except:
                pass
    
    # ========================================================================
    # GUI v3.4 - DRASTICALLY IMPROVED
    # ========================================================================
    def run_gui(self):
        if not GUI_AVAILABLE:
            return self.run_cli()
        self.init()
        try:
            from gui_themes import (
                Colors, MatrixRainCanvas, SacredGeometryCanvas, NeonButton,
                CrystallineFrame, TerminalLog, GlitchLabel, HexDisplay,
                WaveformCanvas, StatusIndicator, BlendshapeBars, apply_dark_theme
            )
        except Exception as e:
            print(f"[GUI] Themes not available: {e}")
            return self._run_basic_gui()
        
        C = Colors
        BG = "#0a0a0f"
        PANEL = "#111118"
        PANEL_HOVER = "#1a1a25"
        BORDER = "#222230"
        ACCENT = "#00d4ff"
        ACCENT_DIM = "#0088aa"
        TEXT = "#e0e0e8"
        TEXT_DIM = "#666688"
        TEXT_MUTE = "#444455"
        GREEN = "#00cc66"
        RED = "#ff3355"
        ORANGE = "#ffaa33"
        
        self.root = tk.Tk()
        self.root.title(f"MAX HEADROOM STUDIO  v{VERSION}")
        self.root.geometry("1500x950")
        self.root.configure(bg=BG)
        self.root.minsize(1300, 850)
        apply_dark_theme(self.root)
        
        # Make all fonts sharper
        self.root.option_add("*Font", "{Segoe UI} 9")
        
        # Pre-create fonts (tuples with spaces in family name fail on some tkinter builds)
        try:
            import tkinter.font as tkfont
            self.f7  = tkfont.Font(family="Segoe UI", size=7)
            self.f8  = tkfont.Font(family="Segoe UI", size=8)
            self.f8b = tkfont.Font(family="Segoe UI", size=8, weight="bold")
            self.f9  = tkfont.Font(family="Segoe UI", size=9)
            self.f9b = tkfont.Font(family="Segoe UI", size=9, weight="bold")
            self.f10  = tkfont.Font(family="Segoe UI", size=10)
            self.f10b = tkfont.Font(family="Segoe UI", size=10, weight="bold")
            self.f11  = tkfont.Font(family="Segoe UI", size=11)
            self.f16b = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        except Exception:
            self.f7 = self.f8 = self.f8b = self.f9 = self.f9b = self.f10 = self.f10b = self.f11 = self.f16b = None
        
        # ===================================================================
        # HEADER
        # ===================================================================
        header = tk.Frame(self.root, bg=BG, height=44)
        header.pack(fill=tk.X, padx=16, pady=(12, 8))
        header.pack_propagate(False)
        
        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side=tk.LEFT)
        tk.Label(title_box, text="MAX HEADROOM", fg=ACCENT, bg=BG,
                font=self.f16b).pack(side=tk.LEFT)
        tk.Label(title_box, text=f"  STUDIO", fg=TEXT_DIM, bg=BG,
                font=self.f11).pack(side=tk.LEFT)
        
        # Center: Recording + Mode
        center_box = tk.Frame(header, bg=BG)
        center_box.pack(side=tk.LEFT, padx=40)
        self.rec_dot = tk.Label(center_box, text="", fg=RED, bg=BG, font=self.f10)
        self.rec_dot.pack(side=tk.LEFT)
        self.mode_text = tk.Label(center_box, text="STANDBY", fg=ORANGE, bg=BG,
                                 font=self.f10b)
        self.mode_text.pack(side=tk.LEFT, padx=8)
        
        # Right: Indicators
        ind = tk.Frame(header, bg=BG)
        ind.pack(side=tk.RIGHT)
        for label, color_attr in [("CAM", "cam_indicator"), ("TRACK", "track_indicator"), ("NET", "ws_indicator")]:
            box = tk.Frame(ind, bg=BG)
            box.pack(side=tk.LEFT, padx=8)
            ind_obj = StatusIndicator(box, color=RED if label == "CAM" else RED, width=14, height=14)
            ind_obj.pack(side=tk.LEFT)
            setattr(self, color_attr, ind_obj)
            tk.Label(box, text=label, fg=TEXT_DIM, bg=BG, font=self.f7).pack(side=tk.LEFT, padx=(4, 0))
        
        # ===================================================================
        # MAIN SPLIT
        # ===================================================================
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)
        
        # ---- LEFT: Preview + Scenes ----
        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Preview
        preview_frame = tk.Frame(left, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        preview_hdr = tk.Frame(preview_frame, bg=PANEL, height=32)
        preview_hdr.pack(fill=tk.X)
        preview_hdr.pack_propagate(False)
        tk.Label(preview_hdr, text="PREVIEW", fg=TEXT_DIM, bg=PANEL,
                font=self.f9b).pack(side=tk.LEFT, padx=12)
        self.preview_info = tk.Label(preview_hdr, text="640x480 @ 30fps", fg=TEXT_MUTE,
                                    bg=PANEL, font=self.f9)
        self.preview_info.pack(side=tk.RIGHT, padx=12)
        
        vid_container = tk.Frame(preview_frame, bg=BG)
        vid_container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        self.canvas = tk.Canvas(vid_container, bg=BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Overlays drawn directly on video canvas (bg="" canvases cause TclError on this tkinter build)
        self._crt_flicker = 0
        self._crt_after_id = None
        self._hud_target = None
        self._frame_w = 640
        self._frame_h = 480
        self._start_crt_animation()
        
        # Scene presets under preview
        scenes_bar = tk.Frame(left, bg=BG, height=60)
        scenes_bar.pack(fill=tk.X, pady=(8, 0))
        scenes_bar.pack_propagate(False)
        
        self.scene_refs = {}
        scene_data = [
            ("Default", "#333344", None),
            ("Android", C.NEON_PINK if C else "#ff00aa", "android"),
            ("Beauty", C.ATLANTEAN_TEAL if C else "#00e5ff", "beauty"),
            ("Color", C.SACRED_GOLD if C else "#ffd700", "color"),
            ("Remove BG", GREEN, "remove_bg"),
            ("Blur BG", C.PLASMA_BLUE if C else "#4a90d9", "blur_bg"),
        ]
        for name, color, action in scene_data:
            btn = tk.Frame(scenes_bar, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=(0, 6), fill=tk.Y, expand=True)
            lbl = tk.Label(btn, text=name, fg=TEXT, bg=PANEL, font=self.f9b)
            lbl.pack(expand=True)
            indicator = tk.Label(btn, text="", bg=color, height=1)
            indicator.pack(fill=tk.X, side=tk.BOTTOM)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=PANEL_HOVER))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=PANEL))
            btn.bind("<Button-1>", lambda e, a=action, n=name: self._activate_scene(a, n))
            lbl.bind("<Button-1>", lambda e, a=action, n=name: self._activate_scene(a, n))
            self.scene_refs[name] = {"frame": btn, "label": lbl, "indicator": indicator, "color": color}
        self._highlight_scene("Default")
        
        # ---- RIGHT: Tabbed Control Panel ----
        right = tk.Frame(main, bg=BG, width=440)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)
        
        # Notebook style
        style = ttk.Style()
        style.configure("Studio.TNotebook", background=BG, borderwidth=0)
        style.configure("Studio.TNotebook.Tab", background=PANEL, foreground=TEXT_DIM,
                        padding=[14, 4], font=("Segoe UI", 9))
        style.map("Studio.TNotebook.Tab", background=[("selected", "#1a1a25"), ("active", PANEL_HOVER)],
                  foreground=[("selected", ACCENT), ("active", TEXT)])
        style.layout("Studio.TNotebook", [("Studio.TNotebook.client", {"sticky": "nswe"})])
        
        notebook = ttk.Notebook(right, style="Studio.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # ========== TAB 1: CAMERA ==========
        cam_tab = tk.Frame(notebook, bg=PANEL)
        notebook.add(cam_tab, text="  CAMERA  ")
        ci = tk.Frame(cam_tab, bg=PANEL)
        ci.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(ci, text="Device:", fg=TEXT_DIM, bg=PANEL, font=self.f9).pack(anchor="w")
        self.cam_var = tk.StringVar(value=f"Camera {self.config.camera_index}")
        self.cam_menu = ttk.Combobox(ci, textvariable=self.cam_var,
                                     values=["Camera 0", "Camera 1", "Camera 2", "Test Pattern"],
                                     state="readonly", width=20)
        self.cam_menu.pack(fill=tk.X, pady=4)
        self.cam_menu.bind("<<ComboboxSelected>>", self._on_camera_change)
        
        cam_btn_row = tk.Frame(ci, bg=PANEL)
        cam_btn_row.pack(fill=tk.X, pady=(8, 0))
        for txt, cmd in [("SCAN", self._test_camera), ("REFRESH", self._refresh_camera)]:
            btn = tk.Button(cam_btn_row, text=txt, bg=BORDER, fg=TEXT, relief=tk.FLAT,
                           font=self.f9b, cursor="hand2", command=cmd)
            btn.pack(side=tk.LEFT, padx=(0, 6), fill=tk.X, expand=True)
        
        # Quick Stats
        tk.Label(ci, text="", bg=PANEL, height=1).pack()
        self.cam_info_label = tk.Label(ci, text="Resolution: --  FPS: --", fg=TEXT_DIM,
                                      bg=PANEL, font=self.f9, anchor="w")
        self.cam_info_label.pack(fill=tk.X, pady=(8, 0))
        
        # ========== TAB 2: FILTERS ==========
        filt_tab = tk.Frame(notebook, bg=PANEL)
        notebook.add(filt_tab, text="  FILTERS  ")
        
        # Scrollable filter area
        filt_canvas = tk.Canvas(filt_tab, bg=PANEL, highlightthickness=0)
        filt_scroll = tk.Frame(filt_canvas, bg=PANEL)
        filt_vbar = tk.Scrollbar(filt_tab, orient=tk.VERTICAL, command=filt_canvas.yview)
        filt_canvas.configure(yscrollcommand=filt_vbar.set)
        filt_vbar.pack(side=tk.RIGHT, fill=tk.Y)
        filt_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        filt_canvas_window = filt_canvas.create_window((0, 0), window=filt_scroll, anchor="nw")
        
        def _configure_filt_scroll(event):
            filt_canvas.configure(scrollregion=filt_canvas.bbox("all"))
            filt_canvas.itemconfig(filt_canvas_window, width=filt_canvas.winfo_width())
        filt_scroll.bind("<Configure>", _configure_filt_scroll)
        
        fi = filt_scroll
        self.filter_toggle_refs = {}
        self.filter_intensity_vars = {}
        filter_data = [
            ("MH", C.NEON_PINK if C else "#ff00aa", "Max Headroom"),
            ("SKIN", C.ATLANTEAN_TEAL if C else "#00e5ff", "Skin Smoothing"),
            ("CLR", C.MATRIX_GREEN if C else "#00cc66", "Color Grading"),
            ("MORPH", C.NEON_ORANGE if C else "#ff8800", "Face Morph"),
            ("BG", C.PLASMA_BLUE if C else "#4a90d9", "Background"),
            ("AR", C.SACRED_GOLD if C else "#ffd700", "AR Overlay"),
            ("MOCAP", C.CRYSTAL_BLUE if C else "#00bbff", "MoCap Viz"),
        ]
        for abbr, color, filt_name in filter_data:
            row = tk.Frame(fi, bg=PANEL)
            row.pack(fill=tk.X, padx=8, pady=4)
            # Eye toggle
            eye = tk.Label(row, text="●", fg=TEXT_MUTE, bg=PANEL, font=("Consolas", 12), cursor="hand2")
            eye.pack(side=tk.LEFT, padx=(0, 6))
            eye.bind("<Button-1>", lambda e, fn=filt_name: self._toggle_filter(fn))
            # Label
            lbl = tk.Label(row, text=filt_name, fg=TEXT, bg=PANEL, font=self.f10, anchor="w", cursor="hand2")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl.bind("<Button-1>", lambda e, fn=filt_name: self._toggle_filter(fn))
            # Badge
            badge = tk.Label(row, text=abbr, fg=color, bg=PANEL, font=self.f8b)
            badge.pack(side=tk.RIGHT, padx=2)
            # Up/down reorder
            up_btn = tk.Label(row, text="▲", fg=TEXT_MUTE, bg=PANEL, font=("Consolas", 8), cursor="hand2")
            up_btn.pack(side=tk.RIGHT, padx=1)
            up_btn.bind("<Button-1>", lambda e, fn=filt_name: self._move_filter(fn, -1))
            down_btn = tk.Label(row, text="▼", fg=TEXT_MUTE, bg=PANEL, font=("Consolas", 8), cursor="hand2")
            down_btn.pack(side=tk.RIGHT, padx=1)
            down_btn.bind("<Button-1>", lambda e, fn=filt_name: self._move_filter(fn, 1))
            # Hover
            for w in (row, lbl, eye):
                w.bind("<Enter>", lambda e, r=row: r.config(bg=PANEL_HOVER) if not r.cget("bg") == "#1a1a25" else None)
                w.bind("<Leave>", lambda e, r=row: r.config(bg=PANEL))
            # Intensity slider row
            isl_row = tk.Frame(fi, bg=PANEL)
            isl_row.pack(fill=tk.X, padx=20, pady=(0, 6))
            isl_var = tk.DoubleVar(value=0.5)
            self.filter_intensity_vars[filt_name] = isl_var
            isl = tk.Scale(isl_row, from_=0, to=100, orient=tk.HORIZONTAL, bg=PANEL,
                          fg=color, troughcolor=BORDER, highlightthickness=0, sliderlength=14,
                          length=260, showvalue=0, variable=isl_var,
                          command=lambda v, fn=filt_name: self._on_filter_intensity(fn, v))
            isl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            # Percentage label
            pct_lbl = tk.Label(isl_row, text="50%", fg=TEXT_DIM, bg=PANEL, font=("Consolas", 8), width=4, anchor="w")
            pct_lbl.pack(side=tk.LEFT, padx=(4, 0))
            isl.pct_lbl = pct_lbl
            # Store refs
            self.filter_toggle_refs[filt_name] = {"eye": eye, "label": lbl, "badge": badge,
                                                  "color": color, "row": row, "isl": isl, "pct_label": pct_lbl}
        
        # Global controls
        sep = tk.Frame(fi, bg="#1a1a25", height=1)
        sep.pack(fill=tk.X, padx=8, pady=4)
        global_row = tk.Frame(fi, bg=PANEL)
        global_row.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(global_row, text="Global Quality:", fg=TEXT_DIM, bg=PANEL, font=self.f9).pack(side=tk.LEFT)
        self.quality_scale = tk.Scale(global_row, from_=1, to=100, orient=tk.HORIZONTAL,
                                      bg=PANEL, fg=ACCENT, troughcolor=BORDER, highlightthickness=0,
                                      sliderlength=14, length=160, showvalue=0,
                                      command=self._on_quality_change)
        self.quality_scale.set(75)
        self.quality_scale.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        self.quality_label = tk.Label(global_row, text="High", fg=ACCENT, bg=PANEL, font=self.f8)
        self.quality_label.pack(side=tk.LEFT)
        
        glitch_row = tk.Frame(fi, bg=PANEL)
        glitch_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(glitch_row, text="Glitch:", fg=TEXT_DIM, bg=PANEL, font=self.f9).pack(side=tk.LEFT)
        self.glitch_scale = tk.Scale(glitch_row, from_=0, to=100, orient=tk.HORIZONTAL,
                                     bg=PANEL, fg=ORANGE, troughcolor=BORDER, highlightthickness=0,
                                     sliderlength=14, length=200, showvalue=0,
                                     command=self.on_glitch_change)
        self.glitch_scale.set(int(self.config.glitch_intensity * 100))
        self.glitch_scale.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        
        # Reset all
        reset_btn = tk.Button(fi, text="RESET ALL FILTERS", bg="#332222", fg=RED, relief=tk.FLAT,
                             font=self.f9b, cursor="hand2", command=self._reset_filters)
        reset_btn.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # ========== TAB 3: COLOR ==========
        color_tab = tk.Frame(notebook, bg=PANEL)
        notebook.add(color_tab, text="  COLOR  ")
        ci2 = tk.Frame(color_tab, bg=PANEL)
        ci2.pack(fill=tk.X, padx=10, pady=10)
        
        # Background section
        tk.Label(ci2, text="BACKGROUND", fg=TEXT_DIM, bg=PANEL, font=self.f9b).pack(anchor="w")
        self.bg_mode_var = tk.StringVar(value="remove")
        bg_row = tk.Frame(ci2, bg=PANEL)
        bg_row.pack(fill=tk.X, pady=4)
        for mode, label in [("remove", "Remove"), ("blur", "Blur"), ("color", "Color"), ("replace", "Image")]:
            rb = tk.Radiobutton(bg_row, text=label, variable=self.bg_mode_var, value=mode,
                               fg=TEXT, bg=PANEL, selectcolor=BG, activebackground=PANEL,
                               font=self.f9, command=self._on_bg_mode_change)
            rb.pack(side=tk.LEFT, padx=(0, 8))
        
        # Color preset section
        sep2 = tk.Frame(ci2, bg="#1a1a25", height=1)
        sep2.pack(fill=tk.X, pady=8)
        tk.Label(ci2, text="COLOR PRESETS", fg=TEXT_DIM, bg=PANEL, font=self.f9b).pack(anchor="w")
        preset_grid = tk.Frame(ci2, bg=PANEL)
        preset_grid.pack(fill=tk.X, pady=4)
        presets = [("None", None), ("Warm", "warm"), ("Cool", "cool"),
                   ("Cyberpunk", "cyberpunk"), ("Vintage", "vintage"),
                   ("Noir", "noir"), ("Matrix", "matrix")]
        for i, (plabel, pval) in enumerate(presets):
            bg_color = "#1a1a25"
            fg_color = TEXT
            if plabel == "None":
                bg_color = "#222230"
                fg_color = ACCENT
            elif plabel == "Warm":
                bg_color = "#332211"
                fg_color = "#ffaa44"
            elif plabel == "Cool":
                bg_color = "#112244"
                fg_color = "#4488ff"
            elif plabel == "Cyberpunk":
                bg_color = "#221133"
                fg_color = "#ff44ff"
            elif plabel == "Matrix":
                bg_color = "#112211"
                fg_color = "#44ff44"
            btn = tk.Label(preset_grid, text=plabel, bg=bg_color, fg=fg_color,
                          font=self.f8b, padx=8, pady=4, cursor="hand2")
            if pval is None:
                btn.bind("<Button-1>", lambda e: self._set_color_preset(None))
            else:
                btn.bind("<Button-1>", lambda e, v=pval: self._set_color_preset(v))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # ========== TAB 4: NETWORK ==========
        net_tab = tk.Frame(notebook, bg=PANEL)
        notebook.add(net_tab, text="  NETWORK  ")
        ni = tk.Frame(net_tab, bg=PANEL)
        ni.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(ni, text="Host:", fg=TEXT_DIM, bg=PANEL, font=self.f9).pack(anchor="w")
        self.host_entry = tk.Entry(ni, fg=TEXT, bg=BG, insertbackground=ACCENT,
                                  relief=tk.FLAT, font=self.f10)
        self.host_entry.insert(0, self.config.ws_host)
        self.host_entry.pack(fill=tk.X, pady=2)
        
        tk.Label(ni, text="Port:", fg=TEXT_DIM, bg=PANEL, font=self.f9).pack(anchor="w", pady=(8, 0))
        self.port_entry = tk.Entry(ni, fg=TEXT, bg=BG, insertbackground=ACCENT,
                                  relief=tk.FLAT, font=self.f10)
        self.port_entry.insert(0, str(self.config.ws_port))
        self.port_entry.pack(fill=tk.X, pady=2)
        
        self.link_btn = tk.Button(ni, text="CONNECT", bg=ACCENT_DIM, fg="white", relief=tk.FLAT,
                                 font=self.f9b, cursor="hand2", command=self.on_connect)
        self.link_btn.pack(fill=tk.X, pady=(8, 4))
        self.ws_status_text = tk.Label(ni, text="OFFLINE", fg=RED, bg=PANEL, font=self.f10b)
        self.ws_status_text.pack()
        
        # Simulation mode
        self.test_var = tk.BooleanVar(value=self.config.test_mode)
        tk.Checkbutton(ni, text="Simulation Mode", variable=self.test_var, fg=GREEN, bg=PANEL,
                      selectcolor=BG, activebackground=PANEL, command=self.on_toggle_test,
                      font=self.f9).pack(anchor="w", pady=(12, 0))
        
        # ========== TAB 5: MIXER ==========
        mix_tab = tk.Frame(notebook, bg=PANEL)
        notebook.add(mix_tab, text="  MIXER  ")
        mi = tk.Frame(mix_tab, bg=PANEL)
        mi.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.bs_bars = BlendshapeBars(mi, width=380, height=300, max_bars=12)
        self.bs_bars.pack(fill=tk.BOTH, expand=True)
        
        # VU meters
        vu_frame = tk.Frame(mi, bg=PANEL)
        vu_frame.pack(fill=tk.X, pady=(8, 0))
        self.sacred_geo = SacredGeometryCanvas(vu_frame, width=80, height=80)
        self.sacred_geo.pack(side=tk.LEFT, padx=4)
        self.sacred_geo.start()
        self.hex_display = HexDisplay(vu_frame, rows=3, cols=6, width=160, height=80)
        self.hex_display.pack(side=tk.RIGHT, padx=4)
        self.hex_display.start()
        
        # ===================================================================
        # BOTTOM BAR
        # ===================================================================
        bottom = tk.Frame(self.root, bg=BG, height=170)
        bottom.pack(fill=tk.X, padx=16, pady=(8, 12))
        bottom.pack_propagate(False)
        
        # Head Pose
        pose_f = self._make_panel(bottom, "HEAD POSE", width=320)
        pose_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        pi = tk.Frame(pose_f, bg=PANEL)
        pi.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.pose_rot_label = tk.Label(pi, text="ROT    0.00   0.00   0.00", fg=TEXT, bg=PANEL, font=("Consolas", 11))
        self.pose_rot_label.pack(anchor="w", pady=2)
        self.pose_trans_label = tk.Label(pi, text="POS    0.00   0.00   0.00", fg=TEXT, bg=PANEL, font=("Consolas", 11))
        self.pose_trans_label.pack(anchor="w", pady=2)
        
        # Waveform
        wav_f = self._make_panel(bottom, "WAVEFORM", width=280)
        wav_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.waveform_canvas = WaveformCanvas(wav_f, bars=32, width=280, height=140, color=ACCENT)
        self.waveform_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.waveform_canvas.start()
        
        # Console
        log_f = self._make_panel(bottom, "CONSOLE", width=360)
        log_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.terminal_log = TerminalLog(log_f, height=7, width=60)
        self.terminal_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # DEV Console
        dev_f = self._make_panel(bottom, "DEV", width=260)
        dev_f.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 0))
        di = tk.Frame(dev_f, bg=PANEL)
        di.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        # Debug metrics
        self.dev_labels = {}
        dev_metrics = [
            ("cam_idx",    "Cam",    ""),
            ("cam_size",   "Size",   ""),
            ("frames_rx",  "Rx Frm", "0"),
            ("frames_ok",  "Drawn",   "0"),
            ("frm_time",   "Frm ms", "0"),
            ("canvas_sz",  "Canvas", ""),
            ("pipeline",   "Pipe",   "idle"),
            ("last_err",   "Error",  "none"),
        ]
        for key, label, default in dev_metrics:
            row = tk.Frame(di, bg=PANEL)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label+":", fg=TEXT_DIM, bg=PANEL, font=self.f8,
                    width=7, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text=default, fg=TEXT, bg=PANEL, font=("Consolas", 8),
                          anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.dev_labels[key] = lbl
        
        # ===================================================================
        # FOOTER
        # ===================================================================
        footer = tk.Frame(self.root, bg=PANEL, height=32)
        footer.pack(fill=tk.X, padx=0, pady=0)
        footer.pack_propagate(False)
        self.video_status_label = tk.Label(footer, text="CAMERA: STANDBY", fg=ORANGE, bg=PANEL,
                                          font=self.f9b)
        self.video_status_label.pack(side=tk.LEFT, padx=16)
        self.fps_label = tk.Label(footer, text="FPS: 0", fg=ACCENT, bg=PANEL, font=self.f9)
        self.fps_label.pack(side=tk.LEFT, padx=16)
        self.packets_label = tk.Label(footer, text="PKT: 0", fg=TEXT_DIM, bg=PANEL, font=self.f9)
        self.packets_label.pack(side=tk.LEFT, padx=16)
        self.frame_time_label = tk.Label(footer, text="FRAME: 0ms", fg=TEXT_DIM, bg=PANEL, font=self.f9)
        self.frame_time_label.pack(side=tk.LEFT, padx=16)
        tk.Label(footer, text="D:Android  B:Beauty  C:Color  G:BG  A:AR  M:Morph  R:Reset  Q:Quit",
                fg=TEXT_MUTE, bg=PANEL, font=self.f8).pack(side=tk.RIGHT, padx=16)
        
        # Init
        self.terminal_log.log(f"Studio v{VERSION} initialized", "system")
        self.terminal_log.log("Background removal ready", "ok")
        self.terminal_log.log("Select REMOVE mode in Background panel", "system")
        if self.config.test_mode:
            self.terminal_log.log("Running in simulation mode", "warning")
        else:
            self.terminal_log.log("Camera connected", "ok")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Key>", self._on_keypress)
        
        self.running = True
        threading.Thread(target=self._tracking_loop, daemon=True).start()
        self.root.mainloop()
    
    def _make_section(self, parent, title):
        """Create a collapsible-looking section panel."""
        frame = tk.Frame(parent, bg="#111118", highlightbackground="#222230", highlightthickness=1)
        frame.pack(fill=tk.X, pady=(0, 8))
        hdr = tk.Frame(frame, bg="#1a1a25", height=30)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text=title, fg="#666688", bg="#1a1a25", font=self.f9b).pack(side=tk.LEFT, padx=12)
        return frame
    
    def _make_panel(self, parent, title, width=None):
        frame = tk.Frame(parent, bg="#111118", highlightbackground="#222230", highlightthickness=1, width=width)
        if width:
            frame.pack_propagate(False)
        hdr = tk.Frame(frame, bg="#1a1a25", height=28)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text=title, fg="#666688", bg="#1a1a25", font=self.f9b).pack(side=tk.LEFT, padx=12)
        return frame
    
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
            self._highlight_scene("Default")
    
    def _update_filter_toggles(self):
        if not self.filter_manager:
            return
        for filt in self.filter_manager.filters:
            ref = self.filter_toggle_refs.get(filt.name)
            if ref:
                if filt.enabled:
                    ref["eye"].config(fg=ref["color"])
                    ref["label"].config(fg="#e0e0e8")
                    ref["isl"].config(fg=ref["color"])
                    ref["pct_label"].config(fg=ref["color"])
                else:
                    ref["eye"].config(fg="#444455")
                    ref["label"].config(fg="#666688")
                    ref["isl"].config(fg="#444455")
                    ref["pct_label"].config(fg="#666688")
    
    def _on_filter_intensity(self, name, value):
        v = float(value)
        ref = self.filter_toggle_refs.get(name)
        if ref:
            ref["pct_label"].config(text=f"{int(v)}%")
        if self.filter_manager:
            f = self.filter_manager.get_filter(name)
            if f:
                f.set_param("intensity", v / 100.0)
    
    def _move_filter(self, name, delta):
        if not self.filter_manager:
            return
        filters = self.filter_manager.filters
        idx = next((i for i, f in enumerate(filters) if f.name == name), None)
        if idx is None:
            return
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(filters):
            return
        filters[idx].priority, filters[new_idx].priority = filters[new_idx].priority, filters[idx].priority
        self.terminal_log.log(f"Moved {name} {'up' if delta < 0 else 'down'}", "system")
    
    def _set_color_preset(self, preset_name):
        if not self.filter_manager:
            return
        cf = self.filter_manager.get_filter("Color Grading")
        if cf:
            cf.set_param("preset", preset_name or "none")
            cf.enable() if preset_name else cf.disable()
            self.terminal_log.log(f"Color preset: {preset_name or 'None'}", "system")
    
    def _activate_scene(self, action, name):
        if not self.filter_manager:
            return
        self.filter_manager.reset()
        self._highlight_scene(name)
        if action == "android":
            self.filter_manager.enable_filter("Max Headroom")
        elif action == "beauty":
            self.filter_manager.enable_filter("Skin Smoothing")
        elif action == "color":
            self.filter_manager.enable_filter("Color Grading")
        elif action == "remove_bg":
            self.filter_manager.enable_filter("Background")
            f = self.filter_manager.get_filter("Background")
            if f:
                f.set_mode("remove")
                self.bg_mode_var.set("remove")
        elif action == "blur_bg":
            self.filter_manager.enable_filter("Background")
            f = self.filter_manager.get_filter("Background")
            if f:
                f.set_mode("blur")
                self.bg_mode_var.set("blur")
        self._update_filter_toggles()
        self.terminal_log.log(f"Scene: {name}", "ok")
    
    def _highlight_scene(self, scene_name):
        for name, ref in self.scene_refs.items():
            if name == scene_name:
                ref["indicator"].config(bg=ref["color"])
                ref["frame"].config(highlightbackground=ref["color"], highlightthickness=1)
            else:
                ref["indicator"].config(bg="#222230")
                ref["frame"].config(highlightbackground="#222230", highlightthickness=1)
    
    def _on_camera_change(self, event):
        choice = self.cam_var.get()
        if choice == "Test Pattern":
            self.config.test_mode = True
            if self.cam_mgr:
                self.cam_mgr.close()
            self.terminal_log.log("Switched to test pattern", "system")
        else:
            try:
                idx = int(choice.replace("Camera ", ""))
                self.config.camera_index = idx
                self.config.test_mode = False
                if self.cam_mgr:
                    self.cam_mgr.close()
                self.cam_mgr = CameraManager(timeout=3.0)
                ok = self.cam_mgr.open(idx)
                if ok:
                    self.terminal_log.log(f"Camera {idx} active", "ok")
                else:
                    self.terminal_log.log(f"Camera {idx} failed, scanning...", "system")
                    cams = CameraManager.discover(max_index=5)
                    if cams:
                        idx2 = cams[0].index
                        ok = self.cam_mgr.open(idx2)
                        if ok:
                            self.config.camera_index = idx2
                            self.terminal_log.log(f"Camera {idx2} active", "ok")
                            self.cam_var.set(f"Camera {idx2}")
                    if not ok:
                        self.terminal_log.log("No camera found", "alert")
            except:
                pass
    
    def _refresh_camera(self):
        if self.cam_mgr:
            self.cam_mgr.close()
        if not self.config.test_mode:
            self.cam_mgr = CameraManager(timeout=3.0)
            ok = self.cam_mgr.open(self.config.camera_index)
            if not ok:
                self.terminal_log.log(f"Camera {self.config.camera_index} failed, scanning...", "system")
                cams = CameraManager.discover(max_index=5)
                if cams:
                    idx = cams[0].index
                    ok = self.cam_mgr.open(idx)
                    if ok:
                        self.config.camera_index = idx
            if ok:
                self.terminal_log.log("Camera refreshed", "ok")
            else:
                self.terminal_log.log("No camera found", "alert")
    
    def _test_camera(self):
        self.terminal_log.log("Scanning cameras...", "system")
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                status = "OK" if ret else "NO FRAMES"
                color = "ok" if ret else "warning"
                self.terminal_log.log(f"Camera {i}: {w}x{h} - {status}", color)
            else:
                self.terminal_log.log(f"Camera {i}: not available", "warning")
            cap.release()
        self.terminal_log.log("Scan complete", "system")
    
    def _on_bg_mode_change(self):
        mode = self.bg_mode_var.get()
        f = self.filter_manager.get_filter("Background") if self.filter_manager else None
        if f:
            f.set_mode(mode)
            self.terminal_log.log(f"Background mode: {mode}", "system")
    
    def _on_quality_change(self, value):
        v = int(value)
        label = "Low" if v < 33 else "Medium" if v < 66 else "High"
        self.quality_label.config(text=label)
        if self.filter_manager:
            for filt in self.filter_manager.filters:
                if "quality" in filt.params:
                    filt.set_param("quality", label.lower())
    
    def on_connect(self):
        self.config.ws_host = self.host_entry.get()
        try:
            self.config.ws_port = int(self.port_entry.get())
        except ValueError:
            self.terminal_log.log("Invalid port", "alert")
            return
        self.connect_websocket()
        if self.ws_connected:
            self.ws_status_text.config(text="ONLINE", fg="#00cc66")
            self.terminal_log.log(f"WS connected", "ok")
        else:
            self.ws_status_text.config(text="FAILED", fg="#ff3355")
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
        if self._crt_after_id:
            try:
                self.canvas.after_cancel(self._crt_after_id)
            except:
                pass
        try:
            for attr in ['sacred_geo', 'hex_display', 'waveform_canvas']:
                obj = getattr(self, attr, None)
                if obj and hasattr(obj, 'stop'):
                    obj.stop()
        except:
            pass
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
    
    def _update_status(self, tracking_active):
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
    # TRACKING
    # ========================================================================
    def _tracking_loop(self):
        while True:
            if not self.running:
                time.sleep(0.05)
                continue
            t_start = time.time()
            frame = None
            if self.config.test_mode:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.rectangle(frame, (50, 50), (590, 430), (0, 150, 0), 2)
                cv2.putText(frame, "SIMULATION MODE", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                if self.cam_mgr:
                    frame = self.cam_mgr.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
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
                if self.filter_manager:
                    try:
                        self._pipeline_active = len([f for f in self.filter_manager.filters if f.enabled]) > 0
                        frame = self.filter_manager.process(frame, blendshapes=blends, head_pose=pose,
                                                            face_rect=face_rect, frame_id=self.frame_count)
                    except Exception as e:
                        self._try_log(f"Filter error: {e}", "warning")
                        self._pipeline_active = False
                else:
                    self._pipeline_active = False
                frame = self._draw_overlay(frame, data)
                payload = {"type": "face_data", "blendshapes": blends, "head_pose": pose, "timestamp": t}
                if self.filter_manager:
                    active = [f.name for f in self.filter_manager.filters if f.enabled]
                    if active:
                        payload["filter_status"] = {"active": active}
                self.send_websocket(payload)
                if hasattr(self, 'canvas') and self.root:
                    ft = (time.time() - t_start) * 1000
                    self._schedule_ui(frame, data, ft)
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
        for y in range(0, h, 3):
            cv2.line(frame, (0, y), (w, y), (0, 50, 0), 1)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 20, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 2), (0, 255, 0), 2)
        offset = int(np.sin(time.time() * 10) * self.config.glitch_intensity * 20) if self.config.glitch_intensity > 0 else 0
        cv2.putText(frame, "MAX HEADROOM", (15 + offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.config.test_mode:
            cv2.putText(frame, "TEST MODE", (w // 2 - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_pos = 55
        for name in list(data.blendshapes.keys())[:8]:
            val = data.blendshapes.get(name, 0)
            bar_w = int(val * 100)
            cv2.rectangle(frame, (10, y_pos), (110, y_pos + 14), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, y_pos), (10 + bar_w, y_pos + 14), (0, 255, 0), -1)
            cv2.putText(frame, f"{name}: {val:.2f}", (120, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            y_pos += 18
        ws_color = (0, 255, 0) if self.ws_connected else (0, 0, 255)
        cv2.putText(frame, "WS: ON" if self.ws_connected else "WS: OFF", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ws_color, 2)
        if self.filter_manager:
            active = [f["name"] for f in self.filter_manager.get_all_status() if f["enabled"]]
            if active:
                cv2.putText(frame, "FILTERS: " + ", ".join(active[:3]), (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
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
                self.pose_rot_label.config(text=f"ROT    {data.head_pose.get('rotation', [0,0,0])[0]:6.1f}   {data.head_pose.get('rotation', [0,0,0])[1]:6.1f}   {data.head_pose.get('rotation', [0,0,0])[2]:6.1f}")
                self.pose_trans_label.config(text=f"POS    {data.head_pose.get('translation', [0,0,0])[0]:6.2f}   {data.head_pose.get('translation', [0,0,0])[1]:6.2f}   {data.head_pose.get('translation', [0,0,0])[2]:6.2f}")
                self._update_status(True)
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.packets_label.config(text=f"PKT: {self.sent_count}")
                self.frame_time_label.config(text=f"FRAME: {ft:.1f}ms")
                mode = "SIM" if self.config.test_mode else "LIVE"
                cam_ok = self.config.test_mode or (self.cam_mgr and self.cam_mgr.is_opened())
                self.video_status_label.config(text=f"CAMERA: {mode}", fg="#00cc66" if cam_ok else "#ffaa33")
                # Update DEV console
                if hasattr(self, 'dev_labels'):
                    fh, fw2 = frame.shape[:2] if hasattr(frame, 'shape') else (0, 0)
                    csz = f"{getattr(self, '_frame_w', 0)}x{getattr(self, '_frame_h', 0)}"
                    cam_idx_s = str(self.config.camera_index)
                    cam_info = f"{cam_idx_s} {'OK' if cam_ok else '--'}"
                    rx = str(self.frame_count)
                    drawn = getattr(self, '_drawn_count', 0)
                    pipe_status = "active" if getattr(self, '_pipeline_active', False) else "idle"
                    err_text = getattr(self, '_last_dev_err', 'none')
                    self.dev_labels['cam_idx'].config(text=cam_info)
                    self.dev_labels['cam_size'].config(text=f"{fw2}x{fh}")
                    self.dev_labels['frames_rx'].config(text=rx)
                    self.dev_labels['frames_ok'].config(text=str(drawn))
                    self.dev_labels['frm_time'].config(text=f"{ft:.1f}")
                    self.dev_labels['canvas_sz'].config(text=csz)
                    self.dev_labels['pipeline'].config(text=pipe_status)
                    self.dev_labels['last_err'].config(text=err_text, fg=RED if err_text != 'none' else TEXT)
            except Exception:
                self._ui_pending = False
        try:
            self.root.after(0, do_update)
        except:
            pass
    
    def _start_crt_animation(self):
        """Animate CRT scanlines overlay on the video canvas via after()."""
        def _tick():
            if not self.running:
                return
            try:
                w = max(640, self.canvas.winfo_width())
                h = max(480, self.canvas.winfo_height())
                # Draw scanlines + flicker on top of existing canvas content
                self._crt_flicker = 0.5 + random.random() * 0.5
                # Add occasional flicker band
                if random.random() < 0.05 and hasattr(self, 'canvas'):
                    band_y = random.randint(0, h - 20)
                    self.canvas.create_rectangle(0, band_y, w, band_y + 2,
                                                fill="#00ff0044", outline="")
            except Exception:
                pass
            self._crt_after_id = self.canvas.after(80, _tick)
        self.canvas.after(80, _tick)
    
    def _update_canvas(self, frame):
        try:
            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self._frame_w, self._frame_h = w, h
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
            # Draw CRT scanlines overlay
            ch2 = max(480, self.canvas.winfo_height())
            cw2 = max(640, self.canvas.winfo_width())
            flicker = getattr(self, '_crt_flicker', 0.5)
            for y in range(0, ch2, 3):
                alpha = int(30 * flicker)
                self.canvas.create_line(0, y, cw2, y, fill=f"#00{alpha:02x}00", width=1)
            # Draw HUD if target available
            if getattr(self, '_hud_target', None):
                hx, hy = self._hud_target
                self._draw_hud_reticle(cw2, ch2, hx, hy)
            self.preview_info.config(text=f"{w}x{h} @ {self.config.target_fps}fps")
            self._drawn_count += 1
            self._last_dev_err = "none"
        except Exception as e:
            self._last_dev_err = str(e)[:30]
            self._try_log(f"Canvas error: {e}", "warning")
    
    def _draw_hud_reticle(self, cw, ch, tx, ty):
        """Draw targeting reticle at given coordinates."""
        try:
            r = 30
            self.canvas.create_oval(tx - r, ty - r, tx + r, ty + r,
                                   outline="#00ff0088", width=1)
            self.canvas.create_line(tx - r - 10, ty, tx - r + 5, ty,
                                   fill="#00ff0088", width=1)
            self.canvas.create_line(tx + r - 5, ty, tx + r + 10, ty,
                                   fill="#00ff0088", width=1)
            self.canvas.create_line(tx, ty - r - 10, tx, ty - r + 5,
                                   fill="#00ff0088", width=1)
            self.canvas.create_line(tx, ty + r - 5, tx, ty + r + 10,
                                   fill="#00ff0088", width=1)
            # Rotating data ring
            t = time.time()
            ring_r = 45
            for i in range(8):
                a = t * 0.5 + i * 3.14159 / 4
                px = tx + int(ring_r * math.cos(a))
                py = ty + int(ring_r * math.sin(a))
                self.canvas.create_oval(px - 2, py - 2, px + 2, py + 2,
                                       fill="#00ff0044", outline="")
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
                    self._hud_target = (hx, hy)
            except:
                pass
    
    # ========================================================================
    # CLI
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
                frame = self.filter_manager.process(frame, blendshapes=blends, head_pose=pose, frame_id=self.frame_count)
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
