"""
Background Removal Filter v2 - Professional green-screen style removal.
Uses PersonSegmenter for high-quality alpha matting.
"""
import cv2
import numpy as np
from typing import Dict
from .base import Filter, FilterMode

class BackgroundRemovalFilter(Filter):
    """
    Professional background removal with modes:
    - blur: Gaussian blur background
    - color: Solid color replacement
    - replace: Image background replacement
    - remove: Full removal with alpha channel (transparent/green screen)
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("Background", mode)
        self.priority = 20
        self.params = {
            "mode": "remove",           # blur, color, replace, remove
            "blur_amount": 25,
            "replacement_color": (0, 255, 0),  # BGR green for chroma key
            "replacement_image": None,
            "edge_smooth": 12,
            "quality": "high",           # low, medium, high
        }
        self._bg_scaled = None
        self._segmenter = None
        self._mask_cache = None
        self._frame_count = 0
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled:
            return frame
        
        h, w = frame.shape[:2]
        context = context or {}
        
        # Lazy init segmenter
        if self._segmenter is None:
            from person_segmentation import PersonSegmenter
            self._segmenter = PersonSegmenter()
        
        # Get face rect from context
        face_rect = None
        if "face_rect" in context and context["face_rect"] is not None:
            face_rect = context["face_rect"]
        elif "head_pose" in context:
            # Estimate from pose if available
            pass
        
        # Compute mask every frame for temporal smoothing
        mask = self._segmenter.segment(frame, face_rect)
        
        # Edge feathering
        feather = self.params["edge_smooth"]
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
        
        bg_mode = self.params["mode"]
        
        if bg_mode == "remove":
            return self._remove_mode(frame, mask)
        elif bg_mode == "blur":
            bg = self._blur_background(frame)
        elif bg_mode == "color":
            bg = np.full_like(frame, self.params["replacement_color"])
        elif bg_mode == "replace":
            bg = self._load_background(w, h)
        else:
            bg = self._blur_background(frame)
        
        # Composite
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (frame * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
        return result
    
    def _remove_mode(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove background entirely.
        Returns green-screen style output (BGR) where bg is chroma green.
        This allows OBS or other apps to chroma key it.
        """
        h, w = frame.shape[:2]
        green = np.full_like(frame, (0, 255, 0))  # BGR green
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (frame * mask_3ch + green * (1 - mask_3ch)).astype(np.uint8)
        
        # Add subtle outline glow for better visibility
        edge = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
        result[edge > 0] = (0, 200, 255)  # Cyan edge glow
        
        return result
    
    def _blur_background(self, frame: np.ndarray) -> np.ndarray:
        blur = self.params["blur_amount"]
        if blur % 2 == 0:
            blur += 1
        return cv2.GaussianBlur(frame, (blur, blur), 0)
    
    def _load_background(self, w: int, h: int) -> np.ndarray:
        if self._bg_scaled is not None and self._bg_scaled.shape[:2] == (h, w):
            return self._bg_scaled
        
        path = self.params["replacement_image"]
        if path and cv2.os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                self._bg_scaled = cv2.resize(img, (w, h))
                return self._bg_scaled
        
        return np.full((h, w, 3), self.params["replacement_color"], dtype=np.uint8)
    
    def set_mode(self, mode: str):
        """Set background mode: blur, color, replace, remove."""
        self.params["mode"] = mode
    
    def set_background_image(self, path: str):
        self.params["replacement_image"] = path
        self._bg_scaled = None
        self.params["mode"] = "replace"
    
    def set_background_color(self, r: int, g: int, b: int):
        self.params["replacement_color"] = (b, g, r)
        self.params["mode"] = "color"
    
    def reset(self):
        super().reset()
        self._bg_scaled = None
        if self._segmenter:
            self._segmenter.reset()
