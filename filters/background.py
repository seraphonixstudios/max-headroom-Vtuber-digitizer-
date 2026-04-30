"""
Max Headroom - Background Filter
Virtual background with blur or replacement
"""
import cv2
import numpy as np
from typing import Dict
from .base import Filter, FilterMode

class BackgroundFilter(Filter):
    """
    Virtual background filter - blur or replace background.
    Uses portrait segmentation when available, falls back to color-based masking.
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("Background", mode)
        self.priority = 20
        self.requires_segmentation = True
        self.params = {
            "mode": "blur",         # "blur", "replace", "color"
            "blur_amount": 21,      # Gaussian blur kernel size
            "replacement_image": None,
            "replacement_color": (120, 80, 160),  # BGR - subtle purple
            "edge_feather": 15,     # Feather edge pixels
            "segmentation_threshold": 0.5,
        }
        self._bg_image = None
        self._bg_scaled = None
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled:
            return frame
        
        h, w = frame.shape[:2]
        
        # Get segmentation mask from context or compute simple one
        if context and "segmentation_mask" in context and context["segmentation_mask"] is not None:
            mask = context["segmentation_mask"]
            # Resize if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback: create mask from face rect
            mask = self._create_face_mask(frame, context)
        
        # Feather edges
        mask = self._feather_mask(mask, self.params["edge_feather"])
        
        # Process based on mode
        bg_mode = self.params["mode"]
        
        if bg_mode == "blur":
            background = self._blur_background(frame)
        elif bg_mode == "replace" and self.params["replacement_image"]:
            background = self._load_background(w, h)
        elif bg_mode == "color":
            background = np.full_like(frame, self.params["replacement_color"])
        else:
            background = self._blur_background(frame)
        
        # Composite foreground onto background
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (frame * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def _create_face_mask(self, frame: np.ndarray, context: Dict) -> np.ndarray:
        """Create a person mask using simple methods when segmentation unavailable."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # If we have face landmarks, create an elliptical mask
        if context and "landmarks" in context and context["landmarks"]:
            landmarks = context["landmarks"]
            if len(landmarks) >= 68:
                # Face contour points
                face_pts = np.array(landmarks[:17], dtype=np.int32)
                # Create convex hull
                hull = cv2.convexHull(face_pts)
                cv2.fillConvexPoly(mask, hull, 1.0)
                
                # Extend to include upper body (approximate)
                center_x = int(np.mean([p[0] for p in landmarks]))
                center_y = int(np.mean([p[1] for p in landmarks]))
                
                # Large ellipse for body
                body_mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(body_mask, 
                           (center_x, center_y + 50), 
                           (w//3, h//2), 
                           0, 0, 360, 1.0, -1)
                
                # Combine
                mask = np.maximum(mask, body_mask)
        else:
            # Fallback: center ellipse
            cv2.ellipse(mask, (w//2, h//2), (w//3, h//2), 0, 0, 360, 1.0, -1)
        
        # Smooth
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        return mask
    
    def _feather_mask(self, mask: np.ndarray, amount: int) -> np.ndarray:
        """Feather edges of mask."""
        if amount <= 0:
            return mask
        return cv2.GaussianBlur(mask, (amount * 2 + 1, amount * 2 + 1), 0)
    
    def _blur_background(self, frame: np.ndarray) -> np.ndarray:
        """Apply heavy Gaussian blur for background."""
        blur_amount = self.params["blur_amount"]
        # Ensure odd
        if blur_amount % 2 == 0:
            blur_amount += 1
        return cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
    
    def _load_background(self, w: int, h: int) -> np.ndarray:
        """Load and scale background image."""
        if self._bg_scaled is not None and self._bg_scaled.shape[:2] == (h, w):
            return self._bg_scaled
        
        path = self.params["replacement_image"]
        if path and cv2.os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                self._bg_scaled = cv2.resize(img, (w, h))
                return self._bg_scaled
        
        # Fallback to color
        return np.full((h, w, 3), self.params["replacement_color"], dtype=np.uint8)
    
    def set_background_image(self, path: str):
        """Set background replacement image."""
        self.params["replacement_image"] = path
        self._bg_scaled = None
    
    def set_background_color(self, r: int, g: int, b: int):
        """Set background color (RGB)."""
        self.params["replacement_color"] = (b, g, r)  # Convert to BGR
        self.params["mode"] = "color"
