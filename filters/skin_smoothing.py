"""
Max Headroom - Skin Smoothing / Beauty Filter
Snapchat/WhatsApp level skin smoothing with edge preservation
"""
import cv2
import numpy as np
from typing import Dict
from .base import Filter, FilterMode

class SkinSmoothingFilter(Filter):
    """
    Professional skin smoothing using bilateral filter + luminance masking.
    Preserves edges (eyes, mouth, hair) while smoothing skin tones.
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("Skin Smoothing", mode)
        self.priority = 10
        self.params = {
            "strength": 0.5,        # 0.0 to 1.0
            "bilateral_d": 9,       # Bilateral filter diameter
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            "skin_mask_threshold": 0.3,
            "preserve_eyes": True,
            "preserve_mouth": True,
            "preserve_eyebrows": True,
            "sharpen_edges": 0.2,
        }
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled or self.params["strength"] <= 0:
            return frame
        
        strength = self.params["strength"]
        
        # Create skin mask using HSV color space
        skin_mask = self._create_skin_mask(frame)
        
        # Apply bilateral filter for smoothing
        smoothed = cv2.bilateralFilter(
            frame,
            self.params["bilateral_d"],
            self.params["bilateral_sigma_color"],
            self.params["bilateral_sigma_space"]
        )
        
        # Additional smoothing using pyramid
        if strength > 0.3:
            smoothed = self._pyramid_smooth(smoothed, iterations=int(strength * 2))
        
        # Preserve facial features (eyes, mouth, eyebrows)
        if context and self.params.get("preserve_eyes", True):
            smoothed = self._preserve_features(smoothed, frame, context)
        
        # Blend original and smoothed based on skin mask and strength
        mask_3ch = np.stack([skin_mask] * 3, axis=-1)
        alpha = mask_3ch * strength
        result = (smoothed * alpha + frame * (1 - alpha)).astype(np.uint8)
        
        # Subtle sharpening
        if self.params.get("sharpen_edges", 0) > 0:
            result = self._subtle_sharpen(result, self.params["sharpen_edges"])
        
        return result
    
    def _create_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a mask for skin tone regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin tone range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([30, 170, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Second range for different skin tones
        lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 170, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur for soft edges
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
        
        # Normalize to 0-1
        skin_mask = skin_mask.astype(np.float32) / 255.0
        
        return skin_mask
    
    def _pyramid_smooth(self, frame: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Apply pyramid down/up for additional smoothing."""
        result = frame.copy()
        for _ in range(iterations):
            # Downsample
            down = cv2.pyrDown(result)
            # Upsample
            result = cv2.pyrUp(down, dstsize=(frame.shape[1], frame.shape[0]))
        return result
    
    def _preserve_features(self, smoothed: np.ndarray, original: np.ndarray, 
                          context: Dict) -> np.ndarray:
        """Preserve eyes, mouth, eyebrows from smoothing."""
        landmarks = context.get("landmarks", []) if context else []
        if not landmarks or len(landmarks) < 68:
            return smoothed
        
        result = smoothed.copy()
        
        # Eye regions (landmarks 36-47)
        for eye_indices in [list(range(36, 42)), list(range(42, 48))]:
            pts = np.array([landmarks[i] for i in eye_indices if i < len(landmarks)])
            if len(pts) > 0:
                x, y, w, h = cv2.boundingRect(pts)
                padding = 5
                y1, y2 = max(0, y-padding), min(original.shape[0], y+h+padding)
                x1, x2 = max(0, x-padding), min(original.shape[1], x+w+padding)
                result[y1:y2, x1:x2] = original[y1:y2, x1:x2]
        
        # Mouth region (landmarks 48-67)
        mouth_pts = np.array([landmarks[i] for i in range(48, 68) if i < len(landmarks)])
        if len(mouth_pts) > 0:
            x, y, w, h = cv2.boundingRect(mouth_pts)
            padding = 3
            y1, y2 = max(0, y-padding), min(original.shape[0], y+h+padding)
            x1, x2 = max(0, x-padding), min(original.shape[1], x+w+padding)
            result[y1:y2, x1:x2] = original[y1:y2, x1:x2]
        
        # Eyebrows (landmarks 17-26)
        brow_pts = np.array([landmarks[i] for i in range(17, 27) if i < len(landmarks)])
        if len(brow_pts) > 0:
            x, y, w, h = cv2.boundingRect(brow_pts)
            padding = 3
            y1, y2 = max(0, y-padding), min(original.shape[0], y+h+padding)
            x1, x2 = max(0, x-padding), min(original.shape[1], x+w+padding)
            result[y1:y2, x1:x2] = original[y1:y2, x1:x2]
        
        return result
    
    def _subtle_sharpen(self, frame: np.ndarray, amount: float) -> np.ndarray:
        """Apply subtle unsharp mask for clarity."""
        blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        return sharpened
