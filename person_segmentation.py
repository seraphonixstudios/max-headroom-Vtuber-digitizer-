"""
Person Segmentation - Background removal with alpha matting.
Uses skin detection + morphological refinement + guided filter edge smoothing.
No ML models required.
"""
import cv2
import numpy as np
from typing import Optional, Tuple

class PersonSegmenter:
    """
    Segments person from background using:
    1. Skin color detection in HSV
    2. Face rectangle expansion for body estimate
    3. Morphological closing to fill gaps
    4. Guided filter for edge-aware alpha matting
    """
    
    def __init__(self):
        self._prev_mask = None
        self._temporal_alpha = 0.3
    
    def segment(self, frame: np.ndarray,
                face_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Returns float32 alpha mask (0=background, 1=person).
        """
        h, w = frame.shape[:2]
        
        # Start with skin detection
        skin_mask = self._detect_skin(frame)
        
        # Expand using face rect to include body
        if face_rect is not None:
            body_mask = self._estimate_body_mask(frame.shape[:2], face_rect)
            mask = np.maximum(skin_mask, body_mask)
        else:
            mask = skin_mask
        
        # Morphological cleanup
        mask = self._morphological_cleanup(mask)
        
        # Edge-aware refinement with guided filter
        mask = self._refine_edges(frame, mask)
        
        # Temporal smoothing
        if self._prev_mask is not None:
            mask = self._prev_mask * self._temporal_alpha + mask * (1 - self._temporal_alpha)
        self._prev_mask = mask.copy()
        
        return np.clip(mask, 0, 1).astype(np.float32)
    
    def remove_background(self, frame: np.ndarray,
                          face_rect: Optional[Tuple[int, int, int, int]] = None,
                          background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove background and return BGRA frame.
        If background is provided, composite onto it.
        """
        mask = self.segment(frame, face_rect)
        h, w = frame.shape[:2]
        
        # Create BGRA output
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = (mask * 255).astype(np.uint8)
        
        if background is not None:
            # Resize background if needed
            if background.shape[:2] != (h, w):
                background = cv2.resize(background, (w, h))
            # Composite
            mask_3ch = np.stack([mask] * 3, axis=-1)
            composite = (frame * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
            return composite
        
        return bgra
    
    def _detect_skin(self, frame: np.ndarray) -> np.ndarray:
        """Detect skin pixels using HSV color range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # HSV ranges for skin (tuned for various skin tones)
        lower1 = np.array([0, 20, 70], dtype=np.uint8)
        upper1 = np.array([20, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 20, 70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        return skin_mask.astype(np.float32) / 255.0
    
    def _estimate_body_mask(self, shape: Tuple[int, ...],
                           face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Estimate body region below face."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        fx, fy, fw, fh = face_rect
        
        # Face ellipse
        cx, cy = fx + fw // 2, fy + fh // 2
        cv2.ellipse(mask, (cx, cy), (fw // 2 + 10, fh // 2 + 10), 0, 0, 360, 0.8, -1)
        
        # Body estimate (shoulders to bottom)
        body_top = fy + fh
        body_width = int(fw * 2.5)
        body_height = h - body_top
        
        if body_height > 0:
            # Create trapezoid for body
            pts = np.array([
                [cx - fw // 2, body_top],
                [cx + fw // 2, body_top],
                [cx + body_width // 2, h - 20],
                [cx - body_width // 2, h - 20]
            ], np.int32)
            cv2.fillPoly(mask, [pts], 0.5)
        
        return mask
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Close holes and remove noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill any remaining small holes
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Keep only the largest contour (person)
            largest = max(contours, key=cv2.contourArea)
            mask.fill(0)
            cv2.drawContours(mask, [largest], -1, 1.0, -1)
        
        return mask
    
    def _refine_edges(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use guided filter for edge-aware alpha refinement."""
        try:
            from filters.graphics_engine import GuidedFilter
            refined = GuidedFilter.apply(frame, (mask * 255).astype(np.uint8), radius=8)
            return refined.astype(np.float32) / 255.0
        except Exception:
            # Fallback: simple Gaussian blur
            return cv2.GaussianBlur(mask, (21, 21), 0)
    
    def reset(self):
        """Reset temporal state."""
        self._prev_mask = None
