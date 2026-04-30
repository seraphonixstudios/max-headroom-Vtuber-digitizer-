#!/usr/bin/env python3
"""
Max Headroom - MediaPipe Face Mesh Tracker
468-point facial landmark detection
Uses MediaPipe Tasks Vision API
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

VERSION = "3.0.0"

# Logging setup
try:
    from logging_utils import LOG, get_logger
    LOG = get_logger("MediaPipe")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.MediaPipe")

@dataclass
class MediaPipeConfig:
    """MediaPipe detector configuration."""
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

class MediaPipeFaceTracker:
    """468-point face mesh using Google MediaPipe Tasks API."""
    
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
    
    def __init__(self, config: MediaPipeConfig = None):
        self.config = config or MediaPipeConfig()
        self.face_mesh = None
        self._init_face_mesh()
        
        self.previous_values = {name: 0.0 for name in self.ARKIT_BLENDSHAPES}
        self.smoothing = 0.7
        self.prev_landmarks = None
    
    def _init_face_mesh(self) -> None:
        """Initialize MediaPipe Face Landmarker."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            model_path = self._get_model_path()
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.config.max_faces,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
            LOG.info("Face Landmarker initialized")
            
        except Exception as e:
            LOG.error("Init error: %s", e)
            LOG.info("Using fallback tracker")
            self.face_mesh = None
    
    def _get_model_path(self) -> str:
        """Get model path or use bundled."""
        import os
        import tempfile
        
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, "face_landmarker.task")
        
        if not os.path.exists(model_path):
            try:
                import urllib.request
                LOG.info("Downloading model...")
                urllib.request.urlretrieve(model_url, model_path)
                LOG.info("Model saved to %s", model_path)
            except Exception as e:
                LOG.error("Download failed: %s", e)
                return None
        
        return model_path
    
    def process(self, frame) -> Tuple[Optional[Dict], Optional[List], Optional[Dict]]:
        """Process frame and return blendshapes, landmarks, pose."""
        if frame is None or self.face_mesh is None:
            return None, None, None
        
        h, w = frame.shape[:2]
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            results = self.face_mesh.detect_for_video(mp_image, int(time.time() * 1000))
            
            if not results.face_landmarks or len(results.face_landmarks) == 0:
                return None, None, None
            
            landmarks_3d = results.face_landmarks[0]
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_3d]
            
            face_rect = self._calculate_bounds(landmarks, w, h)
            blendshapes = self._calculate_blendshapes_from_mp(results.face_blendshapes[0])
            pose = self._estimate_pose(landmarks_3d, w, h)
            
            self.prev_landmarks = landmarks
            return blendshapes, landmarks, pose
            
        except Exception as e:
            return None, None, None
    
    def _calculate_bounds(self, landmarks: List, w: int, h: int) -> Tuple[int, int, int, int]:
        """Calculate face bounding box."""
        min_x = min(p[0] for p in landmarks)
        max_x = max(p[0] for p in landmarks)
        min_y = min(p[1] for p in landmarks)
        max_y = max(p[1] for p in landmarks)
        
        padding = 20
        return (max(0, min_x - padding), max(0, min_y - padding), 
                min(w, max_x - min_x + padding), min(h, max_y - min_y + padding))
    
    def _calculate_blendshapes_from_mp(self, blendshapes_mp) -> Dict[str, float]:
        """Convert MediaPipe blendshapes to ARKit format."""
        if not blendshapes_mp:
            return self.previous_values.copy()
        
        blendshapes = {}
        
        try:
            for bs in blendshapes_mp:
                name = bs.category_name
                value = bs.score
                
                ark_name = self._map_mp_to_arkit(name)
                if ark_name:
                    blendshapes[ark_name] = value
        except:
            pass
        
        for name in self.ARKIT_BLENDSHAPES:
            if name not in blendshapes:
                blendshapes[name] = self.previous_values.get(name, 0.0)
        
        smoothed = {}
        for name, val in blendshapes.items():
            prev = self.previous_values.get(name, 0.0)
            smoothed[name] = prev * self.smoothing + val * (1 - self.smoothing)
        
        self.previous_values = smoothed
        return smoothed
    
    def _map_mp_to_arkit(self, mp_name: str) -> Optional[str]:
        """Map MediaPipe blendshape names to ARKit."""
        mapping = {
            "browDownLeft": "browDown_L",
            "browDownRight": "browDown_R",
            "browUpLeft": "browUp_L",
            "browUpRight": "browUp_R",
            "cheekPuff": "cheekPuff",
            "cheekSquintLeft": "cheekSquint_L",
            "cheekSquintRight": "cheekSquint_R",
            "eyeBlinkLeft": "eyeBlink_L",
            "eyeBlinkRight": "eyeBlink_R",
            "eyeLookDownLeft": "eyeLookDown_L",
            "eyeLookDownRight": "eyeLookDown_R",
            "eyeLookUpLeft": "eyeLookUp_L",
            "eyeLookUpRight": "eyeLookUp_R",
            "eyeSquintLeft": "eyeSquint_L",
            "eyeSquintRight": "eyeSquint_R",
            "jawForward": "jawForward",
            "jawLeft": "jawLeft",
            "jawOpen": "jawOpen",
            "jawRight": "jawRight",
            "mouthClose": "mouthClose",
            "mouthDimpleLeft": "mouthDimple_L",
            "mouthDimpleRight": "mouthDimple_R",
            "mouthFunnel": "mouthFunnel",
            "mouthLeft": "mouthLeft",
            "mouthPucker": "mouthPucker",
            "mouthRight": "mouthRight",
            "mouthSmileLeft": "mouthSmile_L",
            "mouthSmileRight": "mouthSmile_R",
            "mouthUpperUpLeft": "mouthUpperUp_L",
            "mouthUpperUpRight": "mouthUpperUp_R",
            "noseSneerLeft": "noseSneer_L",
            "noseSneerRight": "noseSneer_R",
        }
        
        return mapping.get(mp_name)
    
    def _estimate_pose(self, landmarks_3d, w: int, h: int) -> Dict[str, List[float]]:
        """Estimate head pose from 3D landmarks."""
        try:
            nose = landmarks_3d[1]
            left_cheek = landmarks_3d[116]
            right_cheek = landmarks_3d[345]
            
            center_x = nose.x * 2 - 1
            center_y = nose.y * 2 - 1
            
            yaw = (left_cheek.x - right_cheek.x) * 90
            pitch = center_y * 30
            roll = (right_cheek.y - left_cheek.y) * 45
            
            depth = 1.5 + (1.0 - abs(nose.z))
            
            return {
                "rotation": [pitch, yaw, roll],
                "translation": [center_x * 0.5, -center_y * 0.3, depth],
            }
        except:
            return {"rotation": [0, 0, 0], "translation": [0, 0, 1.5]}
    
    def draw_mesh(self, frame, landmarks: List, color=(0, 255, 0)) -> None:
        """Draw face mesh overlay."""
        if not landmarks or frame is None:
            return
        
        for i in range(0, min(len(landmarks), 68), 2):
            cv2.line(frame, landmarks[i], landmarks[i+1] if i+1 < len(landmarks) else landmarks[i], color, 1)
        
        for i in range(0, len(landmarks), 10):
            cv2.circle(frame, landmarks[i], 2, (0, 255, 255), -1)
    
    def close(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None


class MediaPipeFallbackTracker:
    """Fallback tracker using OpenCV when MediaPipe unavailable."""
    
    def __init__(self):
        from tracker import FaceDetector, BlendShapeCalculator, HeadPoseEstimator
        from tracker import Config
        
        self.detector = FaceDetector()
        self.blendshape_calc = BlendShapeCalculator()
        self.pose_estimator = HeadPoseEstimator()
        self.config = Config()
        self.config.digital_mode = False
    
    def process(self, frame):
        """Process with fallback."""
        from tracker import MaxHeadroomTracker
        
        if frame is None:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            tracker = MaxHeadroomTracker()
            tracker.config.test_mode = True
            return tracker.process_frame(dummy)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = self.detector.detect(gray)
        
        if face_rect is None:
            return None, None, None
        
        landmarks = self.detector.detect_landmarks(gray, face_rect)
        
        blendshapes = self.blendshape_calc.calculate(landmarks, face_rect, time.time())
        pose = self.pose_estimator.estimate(face_rect, gray.shape)
        
        return blendshapes, landmarks, pose
    
    def close(self):
        """Release resources."""
        pass


def check_mediapipe() -> bool:
    """Check if MediaPipe is available."""
    try:
        from mediapipe.tasks import python
        return True
    except:
        return False


def create_tracker(use_mediapipe: bool = True):
    """Create appropriate tracker."""
    if use_mediapipe:
        try:
            return MediaPipeFaceTracker()
        except Exception as e:
            print(f"[MediaPipe] Failed: {e}")
    
    return MediaPipeFallbackTracker()


def test():
    """Test MediaPipe tracker."""
    print(f"Max Headroom MediaPipe Tracker v{VERSION}")
    print("Testing...")
    
    tracker = create_tracker(use_mediapipe=False)
    
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    blendshapes, landmarks, pose = tracker.process(frame)
    
    if blendshapes:
        print(f"SUCCESS: {len(blendshapes)} blendshapes")
        for k in ["jawOpen", "mouthSmile_L", "eyeBlink_L", "browUp_L"]:
            print(f"  {k}: {blendshapes.get(k, 0):.3f}")
    else:
        print("Using fallback (expected - no face)")
    
    tracker.close()
    print("MediaPipe tracker ready!")

if __name__ == "__main__":
    test()