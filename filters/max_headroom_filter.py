"""
Max Headroom - Android/Digital Entity Character Filter
Transforms the user into a Max Headroom styled android/digital character.

Effects:
- Heavy cyan/monochrome color grading with high contrast
- Pronounced CRT scanlines
- Temporal stutter / frame drop simulation
- Chromatic aberration (RGB channel splitting)
- Edge enhancement for sharp machine-like appearance
- Blocky pixelation for digital artifacting
- Geometric neon grid background overlay
- Glitch blocks (random rectangle corruption)
- Scrolling data text overlay (hex codes, status)
- Heavy vignette for broadcast intrusion feel
"""
import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from .base import Filter, FilterMode

class MaxHeadroomFilter(Filter):
    """
    Max Headroom styled android/digital entity transformation filter.
    
    Applies a complete visual overhaul to create the classic 1980s
    cyberpunk broadcast intrusion aesthetic - heavy scanlines,
    cyan monochrome, chromatic glitch, stuttering motion, and
    data overlays.
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("Max Headroom", mode)
        self.priority = 2  # Run early in pipeline
        self.enabled = False
        
        # Parameters
        self.params = {
            # Master intensity (0.0 - 1.0)
            "intensity": 1.0,
            
            # Color grading
            "monochrome": True,           # Convert to grayscale + cyan tint
            "cyan_boost": 1.8,            # How much to boost cyan channel
            "contrast": 1.6,              # High contrast
            "brightness": -10,            # Slightly dark
            
            # Scanlines
            "scanlines": True,
            "scanline_thickness": 2,      # Pixel thickness
            "scanline_spacing": 3,        # Every N pixels
            "scanline_alpha": 0.35,       # Darkness of lines
            
            # Chromatic aberration
            "chromatic": True,
            "chromatic_shift": 4,         # Pixel shift amount
            "chromatic_probability": 0.3, # Chance per frame
            
            # Edge enhancement
            "sharpen": True,
            "sharpen_amount": 1.2,        # Unsharp mask strength
            
            # Pixelation
            "pixelate": True,
            "pixelate_scale": 0.25,       # Downscale factor
            
            # Temporal stutter
            "stutter": True,
            "stutter_probability": 0.08,  # Chance to repeat previous frame
            "stutter_frames": 2,          # How many frames to repeat
            
            # Glitch blocks
            "glitch_blocks": True,
            "glitch_block_probability": 0.15,
            "glitch_block_count": 3,
            
            # Geometric grid
            "grid": True,
            "grid_spacing": 40,
            "grid_color": [0, 255, 255],  # Cyan in BGR
            "grid_alpha": 0.25,
            
            # Data text overlay
            "data_overlay": True,
            "data_text": [
                "SIGNAL: NOMINAL",
                "BROADCAST: LIVE",
                "ENTITY: MAX",
                "MODE: DIGITAL",
                "ORIGIN: CYBERSPACE",
                "STATUS: ONLINE",
                "FORMAT: NTSC",
                "NOISE: 0.04%",
                "SYNC: LOCKED",
            ],
            
            # Vignette
            "vignette": True,
            "vignette_strength": 0.6,
        }
        
        # Temporal state
        self._prev_frame = None
        self._stutter_counter = 0
        self._last_glitch_time = 0
        self._scanline_mask = None
        self._vignette_mask = None
        self._grid_overlay = None
        self._data_scroll_offset = 0
        self._last_data_update = 0
        self._glitch_regions = []
        self._frame_id = 0
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled or frame is None or frame.size == 0:
            return frame
        
        self._frame_id += 1
        intensity = self.params["intensity"]
        if intensity <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Check temporal stutter first - if active, return previous fully-processed frame
        if self.params["stutter"] and intensity > 0.3:
            stuttered = self._apply_stutter(frame)
            if stuttered is not frame:
                # Stutter is active, return the previous processed frame directly
                return stuttered
        
        result = frame.copy()
        
        # 1. Pixelation (downscale + upscale for blocky look)
        if self.params["pixelate"] and intensity > 0.2:
            result = self._apply_pixelation(result)
        
        # 2. Color grading: monochrome + cyan boost + high contrast
        if self.params["monochrome"]:
            result = self._apply_monochrome(result, intensity)
        
        # 3. Edge enhancement (sharpen for machine-like edges)
        if self.params["sharpen"]:
            result = self._apply_sharpen(result, intensity)
        
        # 4. Chromatic aberration (RGB channel shift)
        if self.params["chromatic"] and intensity > 0.3:
            result = self._apply_chromatic(result, w, h, intensity)
        
        # 5. Glitch blocks (random rectangle corruption)
        if self.params["glitch_blocks"] and intensity > 0.4:
            result = self._apply_glitch_blocks(result, w, h, intensity)
        
        # 6. Heavy scanlines
        if self.params["scanlines"]:
            result = self._apply_scanlines(result, h, w, intensity)
        
        # 7. Geometric neon grid overlay
        if self.params["grid"] and intensity > 0.2:
            result = self._apply_grid(result, w, h, intensity)
        
        # 8. Data text overlay
        if self.params["data_overlay"] and intensity > 0.3:
            result = self._apply_data_overlay(result, w, h, intensity)
        
        # 9. Heavy vignette
        if self.params["vignette"]:
            result = self._apply_vignette(result, h, w, intensity)
        
        # Store for stutter
        self._prev_frame = result.copy()
        
        return result
    
    def _apply_stutter(self, frame: np.ndarray) -> np.ndarray:
        """Apply temporal stutter by occasionally repeating previous frame.
        
        Returns:
            Previous processed frame if stuttering, otherwise the input frame.
        """
        # If we're in an active stutter sequence, return previous frame
        if self._stutter_counter > 0:
            self._stutter_counter -= 1
            if self._prev_frame is not None and self._prev_frame.shape == frame.shape:
                return self._prev_frame.copy()
        
        # Check if we should start a new stutter sequence
        prob = self.params["stutter_probability"]
        if np.random.random() < prob:
            self._stutter_counter = self.params["stutter_frames"]
        
        # Return input frame to signal "no stutter"
        return frame
    
    def _apply_pixelation(self, frame: np.ndarray) -> np.ndarray:
        """Apply blocky pixelation by downscaling and upscaling."""
        scale = max(0.05, min(0.5, self.params["pixelate_scale"]))
        h, w = frame.shape[:2]
        
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def _apply_monochrome(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Convert to high-contrast grayscale with cyan tint."""
        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # High contrast
        contrast = 1.0 + (self.params["contrast"] - 1.0) * intensity
        brightness = self.params["brightness"] * intensity
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
        
        # Create cyan-tinted BGR from grayscale
        cyan_boost = 1.0 + (self.params["cyan_boost"] - 1.0) * intensity
        
        result = np.zeros_like(frame)
        result[:, :, 0] = np.clip(gray * cyan_boost, 0, 255).astype(np.uint8)  # Blue (cyan component)
        result[:, :, 1] = np.clip(gray * 0.9, 0, 255).astype(np.uint8)         # Green (slight)
        result[:, :, 2] = np.clip(gray * 0.3, 0, 255).astype(np.uint8)         # Red (minimal)
        
        # Blend with original based on intensity
        if intensity < 1.0:
            result = cv2.addWeighted(frame, 1.0 - intensity, result, intensity, 0)
        
        return result
    
    def _apply_sharpen(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply unsharp mask for sharp machine-like edges."""
        amount = 1.0 + (self.params["sharpen_amount"] - 1.0) * intensity
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
        
        # Unsharp mask: original + (original - blurred) * amount
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _apply_chromatic(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
        """Apply chromatic aberration by shifting RGB channels."""
        prob = self.params["chromatic_probability"]
        if np.random.random() > prob:
            return frame
        
        shift = int(self.params["chromatic_shift"] * intensity)
        if shift < 1:
            return frame
        
        b, g, r = cv2.split(frame)
        
        # Shift red channel right
        r_shifted = np.zeros_like(r)
        r_shifted[:, shift:] = r[:, :-shift]
        
        # Shift blue channel left
        b_shifted = np.zeros_like(b)
        b_shifted[:, :-shift] = b[:, shift:]
        
        # Green stays center
        return cv2.merge([b_shifted, g, r_shifted])
    
    def _apply_glitch_blocks(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
        """Apply random glitch block corruptions."""
        prob = self.params["glitch_block_probability"] * intensity
        if np.random.random() > prob:
            return frame
        
        result = frame.copy()
        count = int(self.params["glitch_block_count"] * intensity)
        
        for _ in range(count):
            bw = np.random.randint(20, 80)
            bh = np.random.randint(5, 20)
            bx = np.random.randint(0, max(1, w - bw))
            by = np.random.randint(0, max(1, h - bh))
            
            glitch_type = np.random.randint(0, 3)
            if glitch_type == 0:
                # Invert colors
                result[by:by+bh, bx:bx+bw] = 255 - result[by:by+bh, bx:bx+bw]
            elif glitch_type == 1:
                # Horizontal shift
                shift_x = np.random.randint(-20, 20)
                block = result[by:by+bh, bx:bx+bw].copy()
                if shift_x > 0:
                    result[by:by+bh, bx+shift_x:bx+bw] = block[:, :-shift_x]
                elif shift_x < 0:
                    result[by:by+bh, bx:bx+bw+shift_x] = block[:, -shift_x:]
            else:
                # Solid cyan block
                result[by:by+bh, bx:bx+bw] = [255, 255, 0]  # Cyan in BGR
        
        return result
    
    def _apply_scanlines(self, frame: np.ndarray, h: int, w: int, intensity: float) -> np.ndarray:
        """Apply heavy CRT scanlines."""
        if self._scanline_mask is None or self._scanline_mask.shape[:2] != (h, w):
            self._scanline_mask = np.ones((h, w, 3), dtype=np.float32)
            thickness = self.params["scanline_thickness"]
            spacing = self.params["scanline_spacing"]
            
            for y in range(0, h, spacing):
                end_y = min(y + thickness, h)
                self._scanline_mask[y:end_y, :] = 1.0 - self.params["scanline_alpha"]
        
        alpha = self.params["scanline_alpha"] * intensity
        mask = 1.0 - (self._scanline_mask - (1.0 - alpha)) * (alpha / self.params["scanline_alpha"])
        mask = np.clip(mask, 0.3, 1.0)
        
        return (frame.astype(np.float32) * mask).astype(np.uint8)
    
    def _apply_grid(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
        """Apply geometric neon grid overlay."""
        alpha = self.params["grid_alpha"] * intensity
        if alpha < 0.05:
            return frame
        
        spacing = self.params["grid_spacing"]
        color = np.array(self.params["grid_color"], dtype=np.uint8)
        
        overlay = np.zeros_like(frame)
        
        # Vertical lines
        for x in range(0, w, spacing):
            cv2.line(overlay, (x, 0), (x, h), color.tolist(), 1)
        
        # Horizontal lines
        for y in range(0, h, spacing):
            cv2.line(overlay, (0, y), (w, y), color.tolist(), 1)
        
        # Perspective effect - lines converge toward center
        cx, cy = w // 2, h // 2
        for i in range(1, 6):
            offset = i * spacing
            # Draw perspective rectangle frames
            pts = np.array([
                [cx - offset, cy - offset // 2],
                [cx + offset, cy - offset // 2],
                [cx + offset * 2, cy + offset],
                [cx - offset * 2, cy + offset]
            ], np.int32)
            cv2.polylines(overlay, [pts], True, color.tolist(), 1)
        
        return cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
    
    def _apply_data_overlay(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
        """Apply scrolling data text overlay with hex codes and status."""
        t = time.time()
        
        # Update scroll offset every 0.5 seconds
        if t - self._last_data_update > 0.5:
            self._data_scroll_offset = (self._data_scroll_offset + 1) % len(self.params["data_text"])
            self._last_data_update = t
        
        alpha = 0.6 * intensity
        overlay = frame.copy()
        
        # Background bar for text
        bar_height = 20
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        
        frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
        
        # Top bar: scrolling status
        texts = self.params["data_text"]
        idx = self._data_scroll_offset
        display_text = f">>> {texts[idx]} <<<"
        
        cv2.putText(frame, display_text, (10, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Top right: hex timestamp
        hex_time = f"0x{int(t * 1000) % 0xFFFF:04X}"
        cv2.putText(frame, hex_time, (w - 70, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        
        # Bottom bar: frame info
        frame_hex = f"FRAME_ID: 0x{self._frame_id % 0xFFFF:04X}"
        cv2.putText(frame, frame_hex, (10, h - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        # Bottom right: signal strength bars
        bar_count = 5
        bar_w = 4
        bar_gap = 2
        start_x = w - 60
        for i in range(bar_count):
            bar_h = 5 + i * 3
            bx = start_x + i * (bar_w + bar_gap)
            by = h - bar_height + (bar_height - bar_h) // 2
            color = (0, 255, 255) if i < 4 else (0, 100, 100)
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), color, -1)
        
        # Random hex dump on the side
        if intensity > 0.6:
            hex_lines = [
                "7F 3A 9E 2B",
                "C4 11 88 FF",
                "00 7F FE 01",
                "AA 55 AA 55",
            ]
            for i, line in enumerate(hex_lines):
                y_pos = 60 + i * 14
                if y_pos < h - 40:
                    cv2.putText(frame, line, (w - 90, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 150, 200), 1)
        
        return frame
    
    def _apply_vignette(self, frame: np.ndarray, h: int, w: int, intensity: float) -> np.ndarray:
        """Apply heavy vignette for broadcast intrusion feel."""
        if self._vignette_mask is None or self._vignette_mask.shape[:2] != (h, w):
            # Create radial gradient mask
            X = cv2.getGaussianKernel(w, w * 0.6)
            Y = cv2.getGaussianKernel(h, h * 0.6)
            kernel = Y * X.T
            self._vignette_mask = kernel / kernel.max()
        
        strength = self.params["vignette_strength"] * intensity
        mask = 1.0 - (1.0 - self._vignette_mask) * strength
        mask = np.dstack([mask] * 3)
        
        return (frame.astype(np.float32) * mask).astype(np.uint8)
    
    def set_intensity(self, intensity: float):
        """Set overall filter intensity (0.0 - 1.0)."""
        self.params["intensity"] = max(0.0, min(1.0, intensity))
    
    def cycle_intensity(self):
        """Cycle through preset intensity levels."""
        levels = [0.0, 0.3, 0.6, 1.0]
        current = self.params["intensity"]
        idx = (levels.index(min(levels, key=lambda x: abs(x - current))) + 1) % len(levels)
        self.params["intensity"] = levels[idx]
        return levels[idx]
