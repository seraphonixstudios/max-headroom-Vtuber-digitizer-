#!/usr/bin/env python3
"""
Max Headroom v3.1 - Advanced Face Tracker
MediaPipe primary detector, 3D pose estimation, Kalman smoothing
"""
import cv2
import numpy as np
import time
import json
import sys
import threading
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

# Version
VERSION = "3.1.0"

# Logging
try:
    from logging_utils import get_logger
    LOG = get_logger("Tracker")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.Tracker")

# Config
try:
    import config as cfg
except ImportError:
    cfg = None

# WebSocket
try:
    import websocket
except ImportError:
    websocket = None

# MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# 3D face model points (generic face)
FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],        # Nose tip
    [0.0, -330.0, -65.0],   # Chin
    [-225.0, 170.0, -135.0], # Left eye outer
    [225.0, 170.0, -135.0],  # Right eye outer
    [-150.0, -150.0, -125.0], # Left mouth
    [150.0, -150.0, -125.0],  # Right mouth
    [-330.0, 0.0, -100.0],    # Left temple
    [330.0, 0.0, -100.0],     # Right temple
], dtype=np.float32)

@dataclass
class KalmanFilter:
    """Simple 1D Kalman filter for smoothing."""
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    estimated_error: float = 1.0
    value: float = 0.0
    
    def update(self, measurement: float) -> float:
        """Update with new measurement."""
        # Prediction update
        self.estimated_error += self.process_noise
        
        # Measurement update
        kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_noise)
        self.value = self.value + kalman_gain * (measurement - self.value)
        self.estimated_error = (1 - kalman_gain) * self.estimated_error
        
        return self.value

class AdvancedFaceDetector:
    """Face detector with MediaPipe primary and Haar fallback."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.mp_face_detection = None
        self.mp_face_mesh = None
        self.haar_cascade = None
        self.detector_type = None
        
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the best available detector."""
        primary = self.config.get('primary', 'mediapipe')
        
        if primary == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            try:
                mp = __import__('mediapipe').solutions
                self.mp_face_detection = mp.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.config.get('face_confidence', 0.5)
                )
                self.mp_face_mesh = mp.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.config.get('face_confidence', 0.5),
                    min_tracking_confidence=self.config.get('tracking_confidence', 0.5)
                )
                self.detector_type = 'mediapipe'
                LOG.info("MediaPipe detector initialized")
                return
            except Exception as e:
                LOG.warning("MediaPipe init failed: %s", e)
        
        # Fallback to Haar
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.detector_type = 'haar'
        LOG.info("Haar cascade detector initialized")
    
    def detect(self, frame) -> Tuple[Optional[Tuple], Optional[List]]:
        """Detect face and return (rect, landmarks)."""
        if self.detector_type == 'mediapipe':
            return self._detect_mediapipe(frame)
        return self._detect_haar(frame)
    
    def _detect_mediapipe(self, frame) -> Tuple[Optional[Tuple], Optional[List]]:
        """Detect using MediaPipe Face Mesh."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None, None
        
        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert to pixel coordinates
        points = []
        for lm in landmarks:
            points.append((int(lm.x * w), int(lm.y * h)))
        
        # Calculate bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x, y = min(xs), min(ys)
        fw, fh = max(xs) - x, max(ys) - y
        padding = 20
        
        return (max(0, x-padding), max(0, y-padding), 
                min(w, fw+padding*2), min(h, fh+padding*2)), points
    
    def _detect_haar(self, frame) -> Tuple[Optional[Tuple], Optional[List]]:
        """Detect using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
        
        x, y, w, h = faces[0]
        
        # Generate approximate landmarks
        landmarks = self._generate_landmarks(x, y, w, h)
        
        return (x, y, w, h), landmarks
    
    def _generate_landmarks(self, x, y, w, h) -> List[Tuple[int, int]]:
        """Generate approximate 68 landmarks from face rect."""
        landmarks = []
        
        # Jaw (0-16)
        for i in range(17):
            lx = x + int(w * i / 16)
            ly = y + h - 5 if i < 8 or i > 8 else y + h - int(h * 0.15)
            landmarks.append((lx, ly))
        
        # Eyebrows (17-26)
        for i in range(5):
            landmarks.append((x + w//4 - 15 + i*8, y + h//4))
        for i in range(5):
            landmarks.append((x + 3*w//4 - 15 + i*8, y + h//4))
        
        # Nose (27-35)
        landmarks.append((x + w//2, y + h//3))
        for i in range(4):
            landmarks.append((x + w//2 - 10 + i*7, y + h//2))
        for i in range(4):
            landmarks.append((x + w//2 - 8 + i*5, y + h//2 + 12))
        
        # Eyes (36-47)
        for i in range(6):
            landmarks.append((x + w//4 - 12 + i*5, y + h//3 + 5))
        for i in range(6):
            landmarks.append((x + 3*w//4 - 12 + i*5, y + h//3 + 5))
        
        # Mouth (48-67)
        for i in range(12):
            mx = x + w//4 + int(w//2 * i / 11)
            my = y + 2*h//3
            landmarks.append((mx, my))
        for i in range(8):
            mx = x + w//4 + 10 + int((w//2 - 20) * i / 7)
            my = y + 2*h//3 + 8
            landmarks.append((mx, my))
        
        return landmarks
    
    def close(self):
        """Release detector resources."""
        if self.mp_face_mesh:
            self.mp_face_mesh.close()
        if self.mp_face_detection:
            self.mp_face_detection.close()

class AdvancedBlendShapeCalculator:
    """Calculate 52 ARKit-compatible blendshapes with 3D awareness."""
    
    ARKIT_BLENDSHAPES = [
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
        "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
        "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
        "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
        "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
        "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
        "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
        "noseSneerLeft", "noseSneerRight"
    ]
    
    def __init__(self, use_kalman: bool = True):
        self.use_kalman = use_kalman
        self.values = {name: 0.0 for name in self.ARKIT_BLENDSHAPES}
        self.kalman_filters = {name: KalmanFilter() for name in self.ARKIT_BLENDSHAPES} if use_kalman else {}
        self.smoothing = 0.75
        self.prev_values = {name: 0.0 for name in self.ARKIT_BLENDSHAPES}
    
    def calculate(self, landmarks: List[Tuple], frame_shape: Tuple) -> Dict[str, float]:
        """Calculate all blendshapes from landmarks."""
        if not landmarks or len(landmarks) < 68:
            return self.prev_values.copy()
        
        h, w = frame_shape[:2]
        
        # Eye calculations
        left_eye_open = self._eye_openness(landmarks, 36, 37, 38, 39, 40, 41)
        right_eye_open = self._eye_openness(landmarks, 42, 43, 44, 45, 46, 47)
        
        self.values["eyeBlinkLeft"] = 1.0 - min(1.0, left_eye_open / 12.0)
        self.values["eyeBlinkRight"] = 1.0 - min(1.0, right_eye_open / 12.0)
        self.values["eyeWideLeft"] = max(0.0, min(1.0, (left_eye_open - 10) / 8))
        self.values["eyeWideRight"] = max(0.0, min(1.0, (right_eye_open - 10) / 8))
        
        # Eyebrow calculations
        left_brow_y = (landmarks[19][1] + landmarks[21][1]) / 2
        right_brow_y = (landmarks[24][1] + landmarks[26][1]) / 2
        eye_level = (landmarks[37][1] + landmarks[44][1]) / 2
        
        brow_raise_left = max(0, (eye_level - left_brow_y - 20) / 30)
        brow_raise_right = max(0, (eye_level - right_brow_y - 20) / 30)
        
        self.values["browInnerUp"] = max(brow_raise_left, brow_raise_right)
        self.values["browOuterUpLeft"] = brow_raise_left
        self.values["browOuterUpRight"] = brow_raise_right
        
        brow_lower_left = max(0, (left_brow_y - eye_level + 10) / 20)
        brow_lower_right = max(0, (right_brow_y - eye_level + 10) / 20)
        self.values["browDownLeft"] = brow_lower_left
        self.values["browDownRight"] = brow_lower_right
        
        # Mouth calculations
        mouth_top = landmarks[51][1]
        mouth_bottom = landmarks[57][1]
        mouth_left = landmarks[48][0]
        mouth_right = landmarks[54][0]
        mouth_center_y = (mouth_top + mouth_bottom) / 2
        mouth_height = mouth_bottom - mouth_top
        mouth_width = mouth_right - mouth_left
        
        self.values["jawOpen"] = min(1.0, mouth_height / max(mouth_width * 0.8, 1))
        self.values["mouthClose"] = 1.0 - self.values["jawOpen"]
        
        # Smile detection
        left_corner_y = landmarks[48][1]
        right_corner_y = landmarks[54][1]
        smile_left = max(0, (mouth_center_y - left_corner_y) / (mouth_height + 1))
        smile_right = max(0, (mouth_center_y - right_corner_y) / (mouth_height + 1))
        self.values["mouthSmileLeft"] = min(1.0, smile_left * 1.5)
        self.values["mouthSmileRight"] = min(1.0, smile_right * 1.5)
        
        # Mouth position
        face_center_x = w / 2
        mouth_center_x = (mouth_left + mouth_right) / 2
        mouth_offset = (mouth_center_x - face_center_x) / max(mouth_width, 1)
        self.values["mouthLeft"] = max(0, -mouth_offset)
        self.values["mouthRight"] = max(0, mouth_offset)
        
        # Pucker / funnel
        self.values["mouthPucker"] = self.values["jawOpen"] * 0.5
        self.values["mouthFunnel"] = self.values["jawOpen"] * 0.3
        
        # Nose sneer
        nose_width = abs(landmarks[35][0] - landmarks[31][0])
        self.values["noseSneerLeft"] = min(1.0, nose_width / 40)
        self.values["noseSneerRight"] = self.values["noseSneerLeft"]
        
        # Cheek puff
        self.values["cheekPuff"] = (self.values["mouthSmileLeft"] + self.values["mouthSmileRight"]) / 2
        self.values["cheekSquintLeft"] = self.values["eyeBlinkLeft"] * 0.3
        self.values["cheekSquintRight"] = self.values["eyeBlinkRight"] * 0.3
        
        # Apply smoothing / Kalman
        smoothed = {}
        for name, value in self.values.items():
            if self.use_kalman and name in self.kalman_filters:
                smoothed[name] = self.kalman_filters[name].update(value)
            else:
                prev = self.prev_values.get(name, 0)
                smoothed[name] = prev * self.smoothing + value * (1 - self.smoothing)
        
        self.prev_values = smoothed
        return smoothed
    
    def _eye_openness(self, landmarks, *indices) -> float:
        """Calculate eye openness from landmark indices."""
        top = min(landmarks[i][1] for i in indices)
        bottom = max(landmarks[i][1] for i in indices)
        return float(bottom - top)

class AdvancedHeadPoseEstimator:
    """3D head pose estimation using solvePnP."""
    
    def __init__(self, smooth_rot: float = 0.8, smooth_trans: float = 0.8):
        self.smooth_rot = smooth_rot
        self.smooth_trans = smooth_trans
        self.prev_rotation = [0.0, 0.0, 0.0]
        self.prev_translation = [0.0, 0.0, 1.0]
        self.camera_matrix = None
        self.dist_coeffs = None
    
    def _get_camera_matrix(self, frame_shape):
        """Build camera matrix from frame dimensions."""
        h, w = frame_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        
        return np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def estimate(self, landmarks: List[Tuple], frame_shape: Tuple) -> Dict[str, List[float]]:
        """Estimate head pose using solvePnP."""
        if not landmarks or len(landmarks) < 6:
            return {"rotation": self.prev_rotation, "translation": self.prev_translation}
        
        # Map 2D landmarks to 3D model points
        # Use key landmarks: nose tip, chin, eyes, mouth corners
        if len(landmarks) >= 68:
            image_points = np.array([
                landmarks[30],  # Nose tip
                landmarks[8],   # Chin
                landmarks[36],  # Left eye outer
                landmarks[45],  # Right eye outer
                landmarks[48],  # Left mouth
                landmarks[54],  # Right mouth
            ], dtype=np.float32)
        else:
            # Approximate from fewer points
            image_points = np.array([
                landmarks[0], landmarks[1], landmarks[2],
                landmarks[3], landmarks[4], landmarks[5]
            ], dtype=np.float32)
        
        camera_matrix = self._get_camera_matrix(frame_shape)
        dist_coeffs = np.zeros((4, 1))
        
        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                FACE_MODEL_3D[:len(image_points)],
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                euler = self._rotation_matrix_to_euler(rotation_mat)
                
                # Smooth
                rotation = [
                    euler[0] * (1 - self.smooth_rot) + self.prev_rotation[0] * self.smooth_rot,
                    euler[1] * (1 - self.smooth_rot) + self.prev_rotation[1] * self.smooth_rot,
                    euler[2] * (1 - self.smooth_rot) + self.prev_rotation[2] * self.smooth_rot,
                ]
                
                translation = [
                    float(translation_vec[0][0] * (1 - self.smooth_trans) + self.prev_translation[0] * self.smooth_trans),
                    float(translation_vec[1][0] * (1 - self.smooth_trans) + self.prev_translation[1] * self.smooth_trans),
                    float(translation_vec[2][0] * (1 - self.smooth_trans) + self.prev_translation[2] * self.smooth_trans),
                ]
                
                self.prev_rotation = rotation
                self.prev_translation = translation
                
                return {"rotation": rotation, "translation": translation}
        
        except Exception as e:
            LOG.warning("Pose estimation failed: %s", e)
        
        return {"rotation": self.prev_rotation, "translation": self.prev_translation}
    
    def _rotation_matrix_to_euler(self, R) -> List[float]:
        """Convert rotation matrix to Euler angles (degrees)."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        if sy < 1e-6:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        
        return [float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))]

class MaxHeadroomTrackerV31:
    """Upgraded face tracker with advanced features."""
    
    WINDOW_NAME = "Max Headroom v3.1"
    
    def __init__(self, config: Dict = None):
        self.cfg = config or {}
        self.detector = None
        self.blendshape_calc = None
        self.pose_estimator = None
        
        self.cap = None
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
        
        self._init_components()
    
    def _init_components(self):
        """Initialize processing components."""
        detector_cfg = self.cfg.get('detector', {})
        self.detector = AdvancedFaceDetector(detector_cfg)
        
        blend_cfg = self.cfg.get('blendshapes', {})
        self.blendshape_calc = AdvancedBlendShapeCalculator(
            use_kalman=self.cfg.get('kalman_filter', True)
        )
        
        pose_cfg = self.cfg.get('head_pose', {})
        self.pose_estimator = AdvancedHeadPoseEstimator(
            smooth_rot=pose_cfg.get('smooth_rotation', 0.8),
            smooth_trans=pose_cfg.get('smooth_translation', 0.8)
        )
        
        LOG.info("Tracker v%s components initialized", VERSION)
    
    def init_camera(self) -> bool:
        """Initialize camera."""
        LOG.info("Initializing camera %d...", self.cfg.get('camera_index', 0))
        self.cap = cv2.VideoCapture(self.cfg.get('camera_index', 0))
        
        if not self.cap.isOpened():
            LOG.warning("Camera unavailable - test mode")
            self.cfg['test_mode'] = True
            return True
        
        res = self.cfg.get('resolution', [640, 480])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.get('target_fps', 60))
        
        LOG.info("Camera ready: %dx%d", int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return True
    
    def connect_ws(self) -> bool:
        """Connect to WebSocket with auto-reconnect."""
        if not self.cfg.get('enable_ws', True) or websocket is None:
            return True
        
        ws_host = self.cfg.get('ws_host', 'localhost')
        ws_port = self.cfg.get('ws_port', 30000)
        
        try:
            self.ws = websocket.create_connection(f"ws://{ws_host}:{ws_port}", timeout=3)
            LOG.info("WebSocket connected to %s:%d", ws_host, ws_port)
            return True
        except Exception as e:
            LOG.warning("WebSocket connection failed: %s", e)
            return False
    
    def reconnect_ws(self):
        """Attempt WebSocket reconnection."""
        if not self.cfg.get('auto_reconnect', True):
            return
        
        try:
            if self.ws:
                self.ws.close()
        except:
            pass
        
        time.sleep(self.cfg.get('reconnect_interval', 5))
        self.connect_ws()
    
    def process_frame(self, frame) -> Tuple[Dict, Optional[List], Dict]:
        """Process frame and extract face data."""
        if frame is None:
            return self._generate_test_data()
        
        face_rect, landmarks = self.detector.detect(frame)
        
        if face_rect is None:
            return self._generate_test_data()
        
        blendshapes = self.blendshape_calc.calculate(landmarks, frame.shape)
        
        pose_cfg = self.cfg.get('head_pose', {})
        if pose_cfg.get('enabled', True):
            pose = self.pose_estimator.estimate(landmarks, frame.shape)
        else:
            pose = {"rotation": [0, 0, 0], "translation": [0, 0, 1]}
        
        self.current_blendshapes = blendshapes
        self.current_landmarks = landmarks or []
        self.current_pose = pose
        
        return blendshapes, landmarks, pose
    
    def _generate_test_data(self) -> Tuple[Dict, List, Dict]:
        """Generate synthetic test data."""
        t = time.time()
        
        blendshapes = {}
        for name in self.blendshape_calc.ARKIT_BLENDSHAPES:
            # Generate varied oscillating values
            phase = hash(name) % 100 / 100.0
            freq = 0.5 + (hash(name[::-1]) % 100) / 50.0
            val = 0.5 + 0.5 * np.sin(t * freq + phase * 2 * np.pi)
            blendshapes[name] = max(0.0, min(1.0, val))
        
        pose = {
            "rotation": [
                float(5 * np.sin(t * 0.7)),
                float(8 * np.sin(t * 0.5)),
                float(2 * np.sin(t * 0.3))
            ],
            "translation": [
                float(0.1 * np.sin(t * 0.6)),
                float(0.05 * np.sin(t * 0.8)),
                float(1.5 + 0.1 * np.sin(t * 0.4))
            ]
        }
        
        return blendshapes, [], pose
    
    def run(self):
        """Main tracking loop."""
        if not self.init_camera():
            return
        
        self.connect_ws()
        self.running = True
        
        LOG.info("Tracker v%s running - Q:quit | T:test | E:eye_glow", VERSION)
        
        target_fps = self.cfg.get('target_fps', 60)
        frame_delay = max(1, int(1000 / target_fps))
        
        while self.running:
            if self.cfg.get('test_mode', False):
                frame = np.zeros((480, 640, 3), np.uint8)
                face_rect = None
            else:
                ret, frame = self.cap.read()
                if not ret:
                    LOG.warning("Camera read failed")
                    break
                face_rect = None  # Will be set by detector
            
            blendshapes, landmarks, pose = self.process_frame(frame)
            
            # Build data packet
            data = {
                "type": "face_data",
                "version": VERSION,
                "timestamp": time.time(),
                "frame_id": self.frame_count,
                "fps": self.fps,
                "blendshapes": blendshapes,
                "head_pose": pose,
                "landmarks": [{"x": p[0], "y": p[1]} for p in landmarks] if landmarks else [],
            }
            
            # Send via WebSocket
            if self.cfg.get('enable_ws', True) and self.ws:
                try:
                    self.ws.send(json.dumps(data))
                    self.sent_count += 1
                except Exception:
                    LOG.warning("WebSocket send failed, attempting reconnect")
                    self.reconnect_ws()
            
            # FPS calculation
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
            
            self.frame_count += 1
            
            # Key handling (minimal for headless/server mode)
            # cv2.waitKey is skipped in v3.1 for better performance
            
            # Frame rate limiting
            time.sleep(max(0, frame_delay / 1000.0 - 0.001))
    
    def stop(self):
        """Clean up resources."""
        self.running = False
        
        if self.cap:
            self.cap.release()
        if self.ws:
            self.ws.close()
        if self.detector:
            self.detector.close()
        
        LOG.info("Stopped - processed %d frames, sent %d", self.frame_count, self.sent_count)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Max Headroom v3.1 Tracker")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--test", action="store_true", help="Test mode")
    args = parser.parse_args()
    
    # Load config
    if cfg:
        cfg.load_config(args.config)
        tracker_cfg = cfg.get("tracker", {})
    else:
        tracker_cfg = {}
    
    if args.test:
        tracker_cfg['test_mode'] = True
    
    tracker = MaxHeadroomTrackerV31(tracker_cfg)
    
    try:
        tracker.run()
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    finally:
        tracker.stop()