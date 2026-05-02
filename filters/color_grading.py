"""
Max Headroom - Color Grading Filter
LUT-based color filters for mood/atmosphere
"""
import cv2
import numpy as np
from typing import Dict
from .base import Filter, FilterMode

class ColorGradingFilter(Filter):
    """
    Color grading with preset LUTs and real-time adjustments.
    Supports warmth, cool, cyberpunk, vintage, noir, and custom curves.
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("Color Grading", mode)
        self.priority = 5
        self.params = {
            "preset": "none",       # none, warm, cool, cyberpunk, vintage, noir, matrix
            "contrast": 1.0,        # 0.5 to 2.0
            "saturation": 1.0,      # 0.0 to 2.0
            "brightness": 0.0,      # -50 to 50
            "vignette": 0.0,        # 0.0 to 1.0
            "tint": [0, 0, 0],      # RGB tint added
        }
        self._lut_cache = {}
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled:
            return frame
        
        preset = self.params["preset"]
        
        if preset == "none" and self.params["contrast"] == 1.0 and self.params["saturation"] == 1.0:
            return frame
        
        result = frame.copy()
        
        # Apply preset LUT
        if preset != "none":
            result = self._apply_preset(result, preset)
        
        # Apply adjustments
        if self.params["contrast"] != 1.0 or self.params["brightness"] != 0:
            result = self._adjust_contrast_brightness(result)
        
        if self.params["saturation"] != 1.0:
            result = self._adjust_saturation(result)
        
        if self.params["tint"] != [0, 0, 0]:
            result = self._apply_tint(result)
        
        if self.params["vignette"] > 0:
            result = self._apply_vignette(result)
        
        return result
    
    def _apply_preset(self, frame: np.ndarray, preset: str) -> np.ndarray:
        """Apply a color preset."""
        if preset == "warm":
            # Increase warmth (red/yellow)
            lut = self._get_cached_lut(preset, lambda: self._create_warm_lut())
        elif preset == "cool":
            # Increase blues
            lut = self._get_cached_lut(preset, lambda: self._create_cool_lut())
        elif preset == "cyberpunk":
            # High contrast, neon magenta/cyan
            lut = self._get_cached_lut(preset, lambda: self._create_cyberpunk_lut())
        elif preset == "vintage":
            # Sepia-like, faded
            lut = self._get_cached_lut(preset, lambda: self._create_vintage_lut())
        elif preset == "noir":
            # High contrast B&W
            lut = self._get_cached_lut(preset, lambda: self._create_noir_lut())
        elif preset == "matrix":
            # Green tint
            lut = self._get_cached_lut(preset, lambda: self._create_matrix_lut())
        else:
            return frame
        
        return cv2.LUT(frame, lut)
    
    def _get_cached_lut(self, name: str, factory) -> np.ndarray:
        """Get cached LUT or create new one."""
        if name not in self._lut_cache:
            self._lut_cache[name] = factory()
        return self._lut_cache[name]
    
    def _create_warm_lut(self) -> np.ndarray:
        """Create warm color LUT."""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            lut[i, 0, 2] = min(255, int(i * 1.1))  # More red
            lut[i, 0, 1] = min(255, int(i * 1.05))  # Slightly more green
            lut[i, 0, 0] = i  # Blue unchanged
        return lut
    
    def _create_cool_lut(self) -> np.ndarray:
        """Create cool color LUT."""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            lut[i, 0, 2] = i
            lut[i, 0, 1] = i
            lut[i, 0, 0] = min(255, int(i * 1.15))  # More blue
        return lut
    
    def _create_cyberpunk_lut(self) -> np.ndarray:
        """Create cyberpunk neon LUT."""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # Boost shadows to blue, highlights to magenta
            if i < 128:
                lut[i, 0, 0] = min(255, int(i * 1.3))  # Blue
                lut[i, 0, 2] = int(i * 0.5)
            else:
                lut[i, 0, 2] = min(255, int(i * 1.2))  # Red
                lut[i, 0, 0] = min(255, int(i * 1.1))  # Blue
            lut[i, 0, 1] = int(i * 0.7)  # Less green
        return lut
    
    def _create_vintage_lut(self) -> np.ndarray:
        """Create vintage/sepia LUT."""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # Sepia formula
            r = min(255, int(i * 1.0))
            g = min(255, int(i * 0.85))
            b = min(255, int(i * 0.65))
            lut[i, 0] = [b, g, r]
        return lut
    
    def _create_noir_lut(self) -> np.ndarray:
        """Create film noir LUT."""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # High contrast grayscale
            val = int(255 * ((i / 255.0) ** 1.5))
            lut[i, 0] = [val, val, val]
        return lut
    
    def _create_matrix_lut(self) -> np.ndarray:
        """Create matrix green LUT."""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            val = min(255, int(i * 1.2))
            lut[i, 0] = [0, val, 0]  # Green only
        return lut
    
    def _adjust_contrast_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Adjust contrast and brightness in Lab space for perceptual correctness."""
        alpha = self.params["contrast"]
        beta = self.params["brightness"]
        
        # For high contrast adjustments, use Lab L channel
        if abs(alpha - 1.0) > 0.3 or abs(beta) > 20:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab)
            l = cv2.convertScaleAbs(l, alpha=alpha, beta=beta)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    def _adjust_saturation(self, frame: np.ndarray) -> np.ndarray:
        """Adjust saturation in HSV space."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.params["saturation"], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _apply_tint(self, frame: np.ndarray) -> np.ndarray:
        """Apply RGB tint with gamma correction."""
        tint = np.array(self.params["tint"], dtype=np.float32)
        # Apply in linear light for perceptual correctness
        linear = np.power(frame.astype(np.float32) / 255.0, 2.2)
        tinted = linear + (tint / 255.0)
        return np.clip(np.power(tinted, 1.0 / 2.2) * 255, 0, 255).astype(np.uint8)
    
    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Apply vignette effect with proper gamma-correct darkening."""
        h, w = frame.shape[:2]
        strength = self.params["vignette"]
        
        # Create radial gradient
        X_resultant_kernel = cv2.getGaussianKernel(w, w/2)
        Y_resultant_kernel = cv2.getGaussianKernel(h, h/2)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        
        # Invert for vignette (darker edges)
        mask = 1 - (1 - mask) * strength
        mask = np.dstack([mask] * 3)
        
        # Apply in linear light
        linear = np.power(frame.astype(np.float32) / 255.0, 2.2)
        result = linear * mask
        return np.clip(np.power(result, 1.0 / 2.2) * 255, 0, 255).astype(np.uint8)
    
    def set_preset(self, preset: str):
        """Set color preset."""
        valid = ["none", "warm", "cool", "cyberpunk", "vintage", "noir", "matrix"]
        if preset in valid:
            self.params["preset"] = preset
