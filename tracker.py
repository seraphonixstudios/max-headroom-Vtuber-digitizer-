"""
Max Headroom Digitizer - Advanced Face Tracker
Real-time webcam face tracking with MediaPipe/landmark detection and WebSocket output
"""
import cv2
import numpy as np
import time
import json
import sys
import argparse
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

VERSION = "3.1.1"

# Logging setup
try:
    from logging_utils import LOG, get_logger
    LOG = get_logger("Tracker")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.Tracker")

try:
    import websocket
except ImportError:
    try:
        import ws as websocket
    except ImportError:
        LOG.error("websocket-client not installed. Run: pip install websocket-client")
        sys.exit(1)

class Config:
    """Tracker configuration."""
    ws_host = "localhost"
    ws_port = 30000
    camera_index = 0
    target_fps = 30
    resolution = (640, 480)
    smoothing = 0.7
    enable_ws = True
    test_mode = False
    digital_mode = True
    glitch_intensity = 0.15
    eye_glow = False

@dataclass
class BlendShapeConfig:
    """Configuration for each blendshape."""
    min_val: float = 0.0
    max_val: float = 1.0
    smoothing: float = 0.7

class FaceDetector:
    """Face detection using Haar Cascade or MediaPipe."""
    
    LANDMARK_INDICES = {
        "jaw": list(range(0, 17)),
        "left_eyebrow": list(range(17, 22)),
        "right_eyebrow": list(range(22, 27)),
        "nose_bridge": list(range(27, 31)),
        "nose_tip": list(range(31, 36)),
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "outer_lips": list(range(48, 60)),
        "inner_lips": list(range(60, 68)),
    }
    
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_points = None
        
    def detect(self, gray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face and return bounding box."""
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        return tuple(faces[0]) if len(faces) > 0 else None
    
    def detect_landmarks(self, gray, face_rect) -> Optional[List[Tuple[int, int]]]:
        """Detect facial landmarks using the face region."""
        if face_rect is None:
            return None
            
        x, y, w, h = face_rect
        
        try:
            landmarks = []
            
            face_roi = gray[y:y+h, x:x+w]
            
            landmarks.append((x + w//2, y + h))          # Chin
            landmarks.append((x + w//2, y))              # Forehead
            
            for i in range(17):
                lx = x + int((w / 16) * i)
                landmarks.append((lx, y + h - 5))
            
            left_eye_x = x + w // 4
            left_eye_y = y + h // 3
            right_eye_x = x + 3 * w // 4
            right_eye_y = y + h // 3
            
            for i in range(6):
                landmarks.append((left_eye_x - 10 + i * 4, left_eye_y))
            for i in range(6):
                landmarks.append((right_eye_x - 10 + i * 4, right_eye_y))
            
            nose_x = x + w // 2
            nose_y = y + h // 2
            for _ in range(8):
                landmarks.append((nose_x, nose_y))
            for _ in range(4):
                landmarks.append((nose_x, nose_y + 10))
            
            mouth_y = y + 2 * h // 3
            mouth_left = x + w // 4
            mouth_right = x + 3 * w // 4
            for i in range(12):
                mx = mouth_left + int((mouth_right - mouth_left) / 11 * i)
                landmarks.append((mx, mouth_y))
            for i in range(8):
                mx = mouth_left + 10 + int((mouth_right - mouth_left - 20) / 6 * i)
                landmarks.append((mx, mouth_y + 10))
            
            self.face_points = landmarks
            return landmarks
            
        except Exception as e:
            return None

class BlendShapeCalculator:
    """Calculate ARKit-compatible blendshapes from face landmarks."""
    
    ARKIT_BLENDSHAPES = [
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
    
    LANDMARK_INDICES = {
        "jaw": list(range(0, 17)),
        "left_eyebrow": list(range(17, 22)),
        "right_eyebrow": list(range(22, 27)),
        "nose": list(range(27, 36)),
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "outer_lip": list(range(48, 60)),
        "inner_lip": list(range(60, 68)),
    }
    
    def __init__(self):
        self.previous_values = {name: 0.0 for name in self.ARKIT_BLENDSHAPES}
        self.smoothing = 0.7
        self.eye_gaze_prev = {"L": (0, 0), "R": (0, 0)}
    
    def _get_region(self, landmarks: List, name: str) -> List:
        """Get landmarks for a facial region."""
        indices = self.LANDMARK_INDICES.get(name, [])
        return [landmarks[i] for i in indices if i < len(landmarks)]
    
    def _calculate_eye_gaze(self, landmarks: List, face_rect: Tuple) -> Dict[str, float]:
        """Calculate eye gaze direction from pupil landmarks."""
        x, y, w, h = face_rect
        
        left_eye = self._get_region(landmarks, "left_eye")
        right_eye = self._get_region(landmarks, "right_eye")
        
        gaze = {"eyeLookUp_L": 0.0, "eyeLookDown_L": 0.0, "eyeLookUp_R": 0.0, "eyeLookDown_R": 0.0}
        
        if len(left_eye) >= 6:
            left_pupil = ((left_eye[0][0] + left_eye[3][0]) / 2, (left_eye[0][1] + left_eye[3][1]) / 2)
            left_center = ((left_eye[1][0] + left_eye[4][0]) / 2, (left_eye[1][1] + left_eye[4][1]) / 2)
            
            dx = left_pupil[0] - left_center[0]
            dy = left_pupil[1] - left_center[1]
            
            gaze["eyeLookUp_L"] = max(0, min(1.0, -dy / 10))
            gaze["eyeLookDown_L"] = max(0, min(1.0, dy / 10))
            gaze["eyeLookDown_R"] = gaze["eyeLookDown_L"]
            gaze["eyeLookUp_R"] = gaze["eyeLookUp_L"]
        
        return gaze
    
    def _calculate_mouth_corner(self, landmarks: List, face_rect: Tuple, corner_idx: int, side: str) -> float:
        """Calculate mouth corner elevation (dimple)."""
        x, y, w, h = face_rect
        
        lip_left = landmarks[48]
        lip_right = landmarks[54]
        
        corner_y = lip_left[1] if side == "L" else lip_right[1]
        mouth_center_y = (lip_left[1] + lip_right[1]) / 2
        
        elevation = (mouth_center_y - corner_y) / h * 2
        return max(0, min(1.0, elevation))
    
    def _calculate_cheek_squint(self, landmarks: List, face_rect: Tuple, side: str) -> float:
        """Calculate cheek squint from eye to cheek distance."""
        x, y, w, h = face_rect
        
        eye_idx = 36 if side == "L" else 42
        cheek_idx = 50 if side == "L" else 52
        
        if len(landmarks) > cheek_idx:
            eye_point = landmarks[eye_idx + 3]
            cheek_point = landmarks[cheek_idx]
            
            dist = abs(cheek_point[1] - eye_point[1])
            return max(0, min(1.0, dist / 15))
        
        return 0.0
    
    def calculate(self, landmarks: List[Tuple[int, int]], face_rect: Tuple[int, int, int, int], time: float) -> Dict[str, float]:
        """Calculate enhanced blendshape values from landmarks."""
        if landmarks is None or face_rect is None:
            return self.previous_values.copy()
        
        x, y, w, h = face_rect
        blendshapes = {}
        
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_top = landmarks[51]
        mouth_bottom = landmarks[57]
        
        mouth_width = abs(mouth_right[0] - mouth_left[0])
        mouth_height = abs(mouth_bottom[1] - mouth_top[1])
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        blendshapes["jawOpen"] = min(1.0, mouth_ratio * 2)
        
        corner_y = (mouth_left[1] + mouth_right[1]) / 2
        corner_offset = (mouth_top[1] - corner_y) / h
        
        blendshapes["mouthSmile_L"] = max(0, min(1.0, 0.5 + corner_offset * 2))
        blendshapes["mouthSmile_R"] = max(0, min(1.0, 0.5 + corner_offset * 2))
        
        left_eye_top = landmarks[36]
        left_eye_bottom = landmarks[41]
        left_eye_height = abs(left_eye_bottom[1] - left_eye_top[1])
        blendshapes["eyeBlink_L"] = max(0, min(1.0, left_eye_height / 10))
        
        right_eye_top = landmarks[42]
        right_eye_bottom = landmarks[47]
        right_eye_height = abs(right_eye_bottom[1] - right_eye_top[1])
        blendshapes["eyeBlink_R"] = max(0, min(1.0, right_eye_height / 10))
        
        left_brow = landmarks[19]
        right_brow = landmarks[24]
        
        brow_dist_left = y + h * 0.3 - left_brow[1]
        brow_dist_right = y + h * 0.3 - right_brow[1]
        
        blendshapes["browUp_L"] = max(0, min(1.0, brow_dist_left / 20))
        blendshapes["browUp_R"] = max(0, min(1.0, brow_dist_right / 20))
        blendshapes["browDown_L"] = max(0, min(1.0, -brow_dist_left / 15))
        blendshapes["browDown_R"] = max(0, min(1.0, -brow_dist_right / 15))
        
        nose_left = landmarks[31]
        nose_right = landmarks[35]
        nose_width = abs(nose_right[0] - nose_left[0])
        nose_sneer = nose_width / w * 2
        blendshapes["noseSneer_L"] = nose_sneer
        blendshapes["noseSneer_R"] = nose_sneer
        
        blendshapes["cheekPuff"] = (blendshapes["mouthSmile_L"] + blendshapes["mouthSmile_R"]) / 2
        blendshapes["cheekSquint_L"] = self._calculate_cheek_squint(landmarks, face_rect, "L")
        blendshapes["cheekSquint_R"] = self._calculate_cheek_squint(landmarks, face_rect, "R")
        
        blendshapes["eyeSquint_L"] = blendshapes["eyeBlink_L"] * 0.5
        blendshapes["eyeSquint_R"] = blendshapes["eyeBlink_R"] * 0.5
        
        eye_gaze = self._calculate_eye_gaze(landmarks, face_rect)
        blendshapes.update(eye_gaze)
        
        blendshapes["mouthClose"] = 1.0 - blendshapes["jawOpen"]
        
        mouth_open = blendshapes["jawOpen"]
        blendshapes["mouthFunnel"] = mouth_open * 0.3 if mouth_open < 0.3 else 0.3
        blendshapes["mouthPucker"] = mouth_open * 0.5
        
        blendshapes["mouthLeft"] = blendshapes["mouthSmile_L"] * 0.3
        blendshapes["mouthRight"] = blendshapes["mouthSmile_R"] * 0.3
        
        jaw_center_x = (mouth_left[0] + mouth_right[0]) / 2
        face_center_x = x + w / 2
        jaw_offset = (jaw_center_x - face_center_x) / w
        
        blendshapes["jawForward"] = 0.0
        blendshapes["jawLeft"] = max(0, min(1.0, -jaw_offset * 3))
        blendshapes["jawRight"] = max(0, min(1.0, jaw_offset * 3))
        
        blendshapes["mouthDimple_L"] = self._calculate_mouth_corner(landmarks, face_rect, 48, "L")
        blendshapes["mouthDimple_R"] = self._calculate_mouth_corner(landmarks, face_rect, 54, "R")
        
        upper_lip_y = landmarks[61][1]
        lower_lip_y = landmarks[67][1]
        blendshapes["mouthUpperUp_L"] = max(0, min(1.0, (corner_y - upper_lip_y) / 20))
        blendshapes["mouthUpperUp_R"] = blendshapes["mouthUpperUp_L"]
        
        smoothed = {}
        for name, value in blendshapes.items():
            if name in self.previous_values:
                prev = self.previous_values[name]
                smoothed[name] = prev * self.smoothing + value * (1 - self.smoothing)
            else:
                smoothed[name] = value
        
        for name in self.ARKIT_BLENDSHAPES:
            if name not in smoothed:
                smoothed[name] = self.previous_values.get(name, 0.0)
        
        self.previous_values = smoothed
        return smoothed

class HeadPoseEstimator:
    """Estimate head pose (rotation and translation) from face landmarks."""
    
    def __init__(self):
        self.previous_pose = {"rotation": [0.0, 0.0, 0.0], "translation": [0.0, 0.0, 1.5]}
        self.smoothing = 0.8
    
    def estimate(self, face_rect, frame_shape) -> Dict[str, List[float]]:
        """Estimate head pose from face bounding box."""
        if face_rect is None:
            return self.previous_pose
        
        x, y, w, h = face_rect
        frame_h, frame_w = frame_shape
        
        center_x = x + w / 2
        center_y = y + h / 2
        
        norm_x = (center_x - frame_w / 2) / frame_w * 2
        norm_y = (center_y - frame_h / 2) / frame_h * 2
        
        depth = max(0.5, min(3.0, 200 / w))
        
        yaw = norm_x * 30
        pitch = norm_y * 30
        roll = 0.0
        
        rotation = [
            pitch * self.smoothing + self.previous_pose["rotation"][0] * (1 - self.smoothing),
            yaw * self.smoothing + self.previous_pose["rotation"][1] * (1 - self.smoothing),
            roll * self.smoothing + self.previous_pose["rotation"][2] * (1 - self.smoothing),
        ]
        
        translation = [
            norm_x * self.smoothing + self.previous_pose["translation"][0] * (1 - self.smoothing),
            -norm_y * self.smoothing + self.previous_pose["translation"][1] * (1 - self.smoothing),
            depth,
        ]
        
        self.previous_pose = {"rotation": rotation, "translation": translation}
        return self.previous_pose

class MaxHeadroomTracker:
    """Main face tracking application."""
    
    WINDOW_NAME = "Max Headroom Digitizer"
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.cap = None
        self.detector = FaceDetector()
        self.blendshape_calc = BlendShapeCalculator()
        self.pose_estimator = HeadPoseEstimator()
        self.ws = None
        self.running = False
        
        self.frame_count = 0
        self.sent_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        self.current_blendshapes = {}
        self.current_landmarks = []
        self.current_pose = {}
        
        # Filter system
        self.filter_manager = None
        self._init_filters()
        
    def _init_filters(self):
        """Initialize Snapchat/WhatsApp level filter system."""
        try:
            from filters import FilterManager
            self.filter_manager = FilterManager()
            LOG.info("Filter system initialized with %d filters", 
                    len(self.filter_manager.filters))
        except Exception as e:
            LOG.warning("Filter system not available: %s", e)
    
    def init(self) -> bool:
        """Initialize camera and detector."""
        LOG.info("Loading face detector...")
        if self.detector.cascade.empty():
            LOG.error("Failed to load Haar cascade")
            return False
        LOG.info("Face detector ready")
        
        LOG.info("Opening camera %d...", self.config.camera_index)
        self.cap = cv2.VideoCapture(self.config.camera_index)
        
        if not self.cap.isOpened():
            LOG.warning("Camera not available - switching to TEST MODE")
            self.config.test_mode = True
            return True
        
        w, h = self.config.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        LOG.info("Camera ready: %dx%d", int(actual_w), int(actual_h))
        
        return True
    
    def connect_ws(self) -> bool:
        """Connect to WebSocket server."""
        if not self.config.enable_ws:
            LOG.info("WebSocket disabled")
            return True
        
        LOG.info("Connecting to WebSocket %s:%d...", self.config.ws_host, self.config.ws_port)
        try:
            self.ws = websocket.create_connection(
                f"ws://{self.config.ws_host}:{self.config.ws_port}",
                timeout=5
            )
            LOG.info("WebSocket connected!")
            return True
        except Exception as e:
            LOG.warning("WebSocket connection failed: %s. Continuing without WS.", e)
            self.config.enable_ws = False
            return False
    
    def send_ws(self, data: Dict) -> bool:
        """Send data to WebSocket server."""
        if not self.ws or not self.config.enable_ws:
            return False
        try:
            self.ws.send(json.dumps(data))
            self.sent_count += 1
            return True
        except:
            self.ws = None
            return False
    
    def process_test_frame(self) -> Tuple[Dict, List, Dict]:
        """Generate digital entity test data with oscillating values."""
        t = time.time()
        
        blendshapes = {
            "jawOpen": 0.15 + 0.1 * np.sin(t * 2.5),
            "mouthSmile_L": 0.25 + 0.15 * np.sin(t * 3.2),
            "mouthSmile_R": 0.25 + 0.15 * np.sin(t * 3.2 + 0.4),
            "eyeBlink_L": 0.0,
            "eyeBlink_R": 0.0,
            "browUp_L": 0.15 * (1 + np.sin(t * 1.8)),
            "browUp_R": 0.15 * (1 + np.sin(t * 1.8 + 0.25)),
            "browDown_L": 0.1 * max(0, np.sin(t * 1.2)),
            "browDown_R": 0.1 * max(0, np.sin(t * 1.2 + 0.3)),
            "cheekPuff": 0.15 + 0.1 * np.sin(t * 2.8),
            "cheekSquint_L": 0.1 * max(0, np.sin(t * 2)),
            "cheekSquint_R": 0.1 * max(0, np.sin(t * 2 + 0.2)),
            "noseSneer_L": 0.08 * (1 + np.sin(t * 1.5)),
            "noseSneer_R": 0.08 * (1 + np.sin(t * 1.5 + 0.1)),
            "eyeLookUp_L": 0.2 * max(0, np.sin(t * 0.8)),
            "eyeLookUp_R": 0.2 * max(0, np.sin(t * 0.8 + 0.1)),
            "eyeLookDown_L": 0.1 * max(0, -np.sin(t * 0.6)),
            "eyeLookDown_R": 0.1 * max(0, -np.sin(t * 0.6 + 0.1)),
            "mouthLeft": 0.1 * (1 + np.sin(t * 2)),
            "mouthRight": 0.1 * (1 + np.sin(t * 2 + 0.5)),
            "mouthDimple_L": 0.15 * max(0, np.sin(t * 1.5)),
            "mouthDimple_R": 0.15 * max(0, np.sin(t * 1.5 + 0.3)),
            "mouthUpperUp_L": 0.12 * max(0, np.sin(t * 1.3)),
            "mouthUpperUp_R": 0.12 * max(0, np.sin(t * 1.3 + 0.2)),
            "eyeSquint_L": 0.05 * (1 + np.sin(t * 3)),
            "eyeSquint_R": 0.05 * (1 + np.sin(t * 3 + 0.15)),
            "mouthClose": 0.85 - 0.1 * np.sin(t * 2),
            "mouthFunnel": 0.08,
            "mouthPucker": 0.12,
            "jawForward": 0.0,
            "jawLeft": 0.05 * np.sin(t * 1.7),
            "jawRight": 0.05 * np.sin(t * 1.7 + 0.3),
        }
        
        pose = {
            "rotation": [
                3 * np.sin(t * 0.6),
                8 * np.sin(t * 0.35),
                2 * np.sin(t * 0.45),
            ],
            "translation": [
                0.08 * np.sin(t * 0.55),
                0.04 * np.sin(t * 0.75),
                1.5 + 0.1 * np.sin(t * 0.4),
            ]
        }
        
        return blendshapes, [], pose
    
    def process_frame(self, frame) -> Tuple[Dict, Optional[List], Dict]:
        """Process single frame and extract face data."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = self.detector.detect(gray)
        landmarks = None
        
        if face_rect is not None:
            landmarks = self.detector.detect_landmarks(gray, face_rect)
        
        if face_rect is None:
            return self.process_test_frame()
        
        blendshapes = self.blendshape_calc.calculate(landmarks, face_rect, time.time())
        
        frame_h, frame_w = gray.shape
        pose = self.pose_estimator.estimate(face_rect, (frame_h, frame_w))
        
        self.current_blendshapes = blendshapes
        self.current_landmarks = landmarks or []
        self.current_pose = pose
        
        return blendshapes, landmarks, pose
    
    def draw_overlay(self, frame, blendshapes, pose, landmarks, face_rect):
        """Draw digital entity style tracking overlay with CRT effects."""
        h, w = frame.shape[:2]
        t = time.time()
        
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        
        for i in range(0, h, 4):
            cv2.line(frame, (0, i), (w, i), (0, 20, 0), 1)
        
        cv2.rectangle(frame, (0, 0), (w, 55), (0, 255, 0), 1)
        
        scan_y = int((t * 50) % h)
        cv2.rectangle(frame, (0, scan_y), (w, scan_y + 2), (0, 255, 70), -1)
        
        cv2.putText(frame, "MAX HEADROOM v3.0", (15, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        fps_text = f"FPS:{self.fps} | {int(t)}"
        cv2.putText(frame, fps_text, (w - 120, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 50), 1)
        
        if self.config.digital_mode and self.config.test_mode:
            cv2.rectangle(frame, (w//2 - 110, h//2 - 35), (w//2 + 110, h//2 + 35), (0, 255, 0), 2)
            
            mode_text = ">>> DIGITAL ENTITY <<<"
            cv2.putText(frame, mode_text, (w//2 - 95, h//2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            entity_text = "[LIVE DATA FEED]"
            cv2.putText(frame, entity_text, (w//2 - 75, h//2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        
        glitch_x = 0
        glitch_y = 0
        if self.config.digital_mode:
            if np.random.random() < self.config.glitch_intensity:
                glitch_x = np.random.randint(-4, 4)
                glitch_y = np.random.randint(-2, 2)
            
            if np.random.random() < self.config.glitch_intensity * 0.5:
                r, g, b = frame[:, :, 0].copy(), frame[:, :, 1].copy(), frame[:, :, 2].copy()
                shift = np.random.randint(2, 5)
                frame[:, :, 0] = np.roll(b, shift, axis=1)
                frame[:, :, 2] = np.roll(r, shift, axis=0)
        
        if face_rect is not None:
            x, y, fw, fh = face_rect
            gx, gy = x + glitch_x, y + glitch_y
            
            cv2.rectangle(frame, (gx, gy), (gx + fw, gy + fh), (0, 255, 0), 2)
            
            if landmarks:
                colors = [(0, 255, 255), (50, 255, 50), (150, 255, 0)]
                for i, (lx, ly) in enumerate(landmarks[:68]):
                    cv2.circle(frame, (lx + glitch_x, ly + glitch_y), 
                               1 + (i % 2), colors[i % 3], -1)
                
                # Advanced eye tracking + red glow effect
                if len(landmarks) >= 48:
                    self._draw_eye_effects(frame, landmarks, glitch_x, glitch_y)
        
        y_offset = 70
        display_shapes = list(self.blendshape_calc.ARKIT_BLENDSHAPES[:12])
        
        for name in display_shapes:
            if name in blendshapes:
                val = blendshapes[name]
                bar_w = int(val * 150)
                
                cv2.rectangle(frame, (15, y_offset), (15 + bar_w, y_offset + 12), (0, 255, 0), -1)
                cv2.rectangle(frame, (15, y_offset), (165, y_offset + 12), (40, 80, 40), 1)
                
                pct = int(val * 100)
                bar_pct = f"{pct:3}%"
                cv2.putText(frame, bar_pct, (170, y_offset + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 220, 50), 1)
                
                name_short = name[:10].ljust(10)
                cv2.putText(frame, name_short, (200, y_offset + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)
                
                y_offset += 16
        
        if pose:
            r = pose.get("rotation", [0, 0, 0])
            p = pose.get("translation", [0, 0, 1.5])
            
            pose_str = f"R:{r[0]:>5.1f} {r[1]:>5.1f} {r[2]:>5.1f} | T:{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}"
            cv2.putText(frame, pose_str, (15, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 50), 1)
        
        ws_color = (0, 255, 0) if self.ws else (0, 50, 0)
        cv2.putText(frame, "WS:ON" if self.ws else "WS:--", (w - 70, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, ws_color, 2)
        
        # Eye glow status indicator
        if self.config.eye_glow:
            cv2.putText(frame, "EYE:GLOW", (w - 150, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    
    def _draw_eye_effects(self, frame, landmarks, glitch_x, glitch_y):
        """Draw advanced eye tracking visualization and anime-style lens flare glow."""
        if len(landmarks) < 48:
            return
        
        h, w = frame.shape[:2]
        t = time.time()
        
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        
        def get_eye_center(indices):
            pts = [landmarks[i] for i in indices if i < len(landmarks)]
            if not pts:
                return None
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            return (cx + glitch_x, cy + glitch_y)
        
        def get_eye_openness(indices):
            if max(indices) >= len(landmarks):
                return 0.0
            top_y = min(landmarks[i][1] for i in indices)
            bottom_y = max(landmarks[i][1] for i in indices)
            return float(bottom_y - top_y)
        
        left_center = get_eye_center(left_eye_indices)
        right_center = get_eye_center(right_eye_indices)
        left_open = get_eye_openness(left_eye_indices)
        right_open = get_eye_openness(right_eye_indices)
        
        # Eye tracking HUD
        if left_center:
            lx, ly = left_center
            openness_pct = min(1.0, left_open / 20.0)
            bar_h = int(openness_pct * 30)
            cv2.rectangle(frame, (lx - 25, ly - 35), (lx - 15, ly - 35 + bar_h), (0, 255, 255), -1)
            cv2.rectangle(frame, (lx - 25, ly - 35), (lx - 15, ly - 5), (100, 100, 100), 1)
        
        if right_center:
            rx, ry = right_center
            openness_pct = min(1.0, right_open / 20.0)
            bar_h = int(openness_pct * 30)
            cv2.rectangle(frame, (rx + 15, ry - 35), (rx + 25, ry - 35 + bar_h), (0, 255, 255), -1)
            cv2.rectangle(frame, (rx + 15, ry - 35), (rx + 25, ry - 5), (100, 100, 100), 1)
        
        # Anime lens flare eye glow
        if self.config.eye_glow:
            for center in [left_center, right_center]:
                if center is None:
                    continue
                self._draw_lens_flare_eye(frame, center, t)
    
    def _draw_filter_hud(self, frame):
        """Draw filter status HUD overlay."""
        if not self.filter_manager:
            return
        
        h, w = frame.shape[:2]
        status = self.filter_manager.get_all_status()
        
        y_offset = h - 80
        x_offset = w - 160
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset - 10, y_offset - 20), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "FILTERS", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        
        y_offset += 18
        for filt in status[:5]:  # Show max 5 filters
            color = (0, 255, 0) if filt["enabled"] else (100, 100, 100)
            text = f"{'ON' if filt['enabled'] else 'OFF'} {filt['name'][:12]}"
            cv2.putText(frame, text, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y_offset += 14
    
    def _draw_lens_flare_eye(self, frame, center, t):
        """Draw an anime-style lens flare / sharingan glowing eye effect."""
        cx, cy = int(center[0]), int(center[1])
        h, w = frame.shape[:2]
        
        # Create a transparent overlay for the glow
        overlay = np.zeros_like(frame, dtype=np.float32)
        
        pulse = 0.6 + 0.4 * np.sin(t * 6)
        flicker = 0.8 + 0.2 * np.sin(t * 15)
        
        # 1. Soft radial gradient glow (large fade)
        max_radius = 80
        for r in range(max_radius, 4, -4):
            alpha = (1.0 - r / max_radius) * 0.25 * pulse
            intensity = int(255 * alpha * flicker)
            cv2.circle(overlay, (cx, cy), r, (0, 0, intensity), -1)
        
        # 2. Horizontal flare line (the classic anime eye beam)
        flare_length = 120
        flare_width = 3
        for i in range(flare_length):
            dist = abs(i - flare_length // 2)
            alpha = (1.0 - dist / (flare_length // 2)) ** 2
            intensity = int(255 * alpha * 0.8 * pulse * flicker)
            x = cx - flare_length // 2 + i
            if 0 <= x < w:
                cv2.circle(overlay, (x, cy), flare_width, (0, int(intensity * 0.3), intensity), -1)
        
        # 3. Diagonal radial spikes / starburst rays
        num_rays = 12
        for i in range(num_rays):
            angle = (2 * np.pi * i / num_rays) + t * 0.5
            ray_length = 25 + 10 * np.sin(t * 3 + i)
            end_x = int(cx + np.cos(angle) * ray_length)
            end_y = int(cy + np.sin(angle) * ray_length)
            
            # Draw ray with gradient
            steps = int(ray_length)
            for s in range(steps):
                px = int(cx + np.cos(angle) * s)
                py = int(cy + np.sin(angle) * s)
                alpha = (1.0 - s / ray_length) ** 1.5
                intensity = int(255 * alpha * 0.7 * pulse)
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(overlay, (px, py), 2, (0, int(intensity * 0.2), intensity), -1)
        
        # 4. Inner bright glow ring
        for r in [20, 14, 10]:
            alpha = (20 - r) / 20 * 0.5 * pulse
            intensity = int(255 * alpha * flicker)
            cv2.circle(overlay, (cx, cy), r, (0, int(intensity * 0.3), intensity), -1)
        
        # 5. Bright core (yellow-white center)
        core_size = 5
        core_glow = int(255 * flicker)
        cv2.circle(overlay, (cx, cy), core_size + 3, (0, core_glow // 3, core_glow), -1)
        cv2.circle(overlay, (cx, cy), core_size, (core_glow // 4, core_glow, 255), -1)
        cv2.circle(overlay, (cx, cy), 2, (core_glow // 2, 255, 255), -1)
        
        # Blend overlay onto frame using screen blend mode
        frame_float = frame.astype(np.float32)
        
        # Screen blend: 1 - (1 - a) * (1 - b)
        blended = 255 - (255 - frame_float) * (255 - overlay) / 255
        np.clip(blended, 0, 255, out=blended)
        frame[:] = blended.astype(np.uint8)
    
    def run(self):
        """Main tracking loop."""
        if not self.init():
            LOG.error("Initialization failed")
            return
        
        if not self.connect_ws():
            LOG.warning("Continuing without WebSocket")
        
        self.running = True
        LOG.info("Tracker v%s - Q:quit T:test E:eye D:android B:beauty G:bg A:AR M:morph C:color R:reset", VERSION)
        
        frame_delay = int(1000 / self.config.target_fps)
        face_rect = None
        
        while self.running:
            if self.config.test_mode:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.rectangle(frame, (100, 50), (540, 430), (0, 150, 0), 3)
                cv2.putText(frame, "NO FACE", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press T for test animation", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                face_rect = None
            else:
                ret, frame = self.cap.read()
                if not ret:
                    LOG.warning("Camera read failed")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rect = self.detector.detect(gray)
                
                if face_rect is None:
                    h, w = frame.shape[:2]
                    cv2.putText(frame, "NO FACE DETECTED", (w//2 - 100, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            blendshapes, landmarks, pose = self.process_frame(frame)
            
            self.draw_overlay(frame, blendshapes, pose, landmarks, face_rect)
            
            # Apply Snapchat/WhatsApp level filters
            if self.filter_manager:
                frame = self.filter_manager.process(
                    frame,
                    landmarks=landmarks,
                    face_rect=face_rect,
                    blendshapes=blendshapes,
                    head_pose=pose,
                    frame_id=self.frame_count
                )
            
            # Draw filter HUD
            if self.filter_manager:
                self._draw_filter_hud(frame)
            
            # Build filter status for E2E transparency
            filter_status = {}
            if self.filter_manager:
                filter_status = {
                    "active": [],
                    "params": {},
                }
                for f in self.filter_manager.filters:
                    if f.enabled:
                        filter_status["active"].append(f.name)
                        filter_status["params"][f.name] = {
                            k: v for k, v in f.params.items()
                            if isinstance(v, (int, float, str, bool, list))
                        }

            data = {
                "type": "face_data",
                "version": VERSION,
                "mode": "digital_entity" if self.config.digital_mode else "standard",
                "blendshapes": blendshapes,
                "head_pose": pose,
                "landmarks": [{"x": p[0], "y": p[1]} for p in landmarks] if landmarks else [],
                "timestamp": time.time(),
                "fps": self.fps,
                "frame_id": self.frame_count,
                "filter_status": filter_status,
            }
            
            if self.config.enable_ws:
                self.send_ws(data)
            
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
                
                if self.frame_count % 30 == 0 and self.frame_count > 0:
                    ws_status = "ON" if self.ws else "OFF"
                    LOG.info("Frame %d | FPS: %d | WS: %s | jawOpen: %.2f",
                            self.frame_count, self.fps, ws_status, blendshapes.get('jawOpen', 0))
            
            self.frame_count += 1
            
            cv2.imshow(self.WINDOW_NAME, frame)
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                LOG.info("Quit key pressed")
                break
            if key == ord('t') or key == ord('T'):
                self.config.test_mode = not self.config.test_mode
                LOG.info("Test mode toggled: %s", self.config.test_mode)
            if key == ord('e') or key == ord('E'):
                self.config.eye_glow = not self.config.eye_glow
                LOG.info("Eye glow toggled: %s", self.config.eye_glow)
            if key == ord('d') or key == ord('D'):
                if self.filter_manager:
                    self.filter_manager.toggle_filter("Max Headroom")
                    LOG.info("Android/Max Headroom filter toggled")
            
            # Filter hotkeys
            if self.filter_manager:
                if key == ord('b') or key == ord('B'):
                    self.filter_manager.toggle_filter("Skin Smoothing")
                    LOG.info("Beauty filter toggled")
                if key == ord('g') or key == ord('G'):
                    self.filter_manager.toggle_filter("Background")
                    LOG.info("Background filter toggled")
                if key == ord('a') or key == ord('A'):
                    self.filter_manager.toggle_filter("AR Overlay")
                    LOG.info("AR overlay toggled")
                if key == ord('m') or key == ord('M'):
                    self.filter_manager.toggle_filter("Face Morph")
                    LOG.info("Face morph toggled")
                if key == ord('c') or key == ord('C'):
                    self.filter_manager.toggle_filter("Color Grading")
                    LOG.info("Color grading toggled")
                if key == ord('r') or key == ord('R'):
                    self.filter_manager.reset()
                    LOG.info("All filters reset")
        
        self.stop()
    
    def stop(self):
        """Clean up resources."""
        self.running = False
        
        if self.cap:
            self.cap.release()
            LOG.info("Camera released")
        
        if self.ws:
            self.ws.close()
            LOG.info("WebSocket closed")
        
        cv2.destroyAllWindows()
        LOG.info("Stopped - processed %d frames, sent %d", self.frame_count, self.sent_count)

def main():
    parser = argparse.ArgumentParser(description="Max Headroom Digitizer Tracker v" + VERSION)
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--ws-host", default="localhost", help="WebSocket host")
    parser.add_argument("--ws-port", type=int, default=30000, help="WebSocket port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket output")
    parser.add_argument("--test", action="store_true", help="Start in test mode")
    parser.add_argument("--digital", action="store_true", default=True, help="Digital entity mode (default: on)")
    parser.add_argument("--glitch", type=float, default=None, help="Glitch intensity (0.0-1.0)")
    parser.add_argument("--eye-glow", action="store_true", help="Enable red glowing eye filter")
    parser.add_argument("--android", action="store_true", help="Enable Max Headroom android character filter")
    args = parser.parse_args()
    
    LOG.info("Max Headroom Digitizer v%s", VERSION)
    LOG.info("=" * 50)
    
    config = Config()
    config.camera_index = args.camera
    config.target_fps = args.fps
    config.resolution = (args.width, args.height)
    config.ws_host = args.ws_host
    config.ws_port = args.ws_port
    config.enable_ws = not args.no_ws
    config.test_mode = args.test
    config.digital_mode = args.digital
    if args.glitch is not None:
        config.glitch_intensity = max(0.0, min(1.0, args.glitch))
    config.eye_glow = args.eye_glow
    
    tracker = MaxHeadroomTracker(config)
    if args.android and tracker.filter_manager:
        tracker.filter_manager.enable_filter("Max Headroom")
        tracker.filter_manager.set_filter_param("Max Headroom", "intensity", 1.0)
        LOG.info("Max Headroom android filter enabled")
    tracker.run()

if __name__ == "__main__":
    main()