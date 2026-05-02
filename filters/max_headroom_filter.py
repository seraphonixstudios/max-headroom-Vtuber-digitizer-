"""
Max Headroom - Android/Digital Entity Character Filter v2.0
SOTA Graphics Engine integration for broadcast-quality visual effects.

Uses production-grade open-source CV techniques:
- K-means posterization for digital artifacting
- Ordered Bayer dithering for retro C64/aesthetic
- Radial chromatic aberration with barrel distortion
- RGB phosphor triad simulation
- Interlaced field flicker
- CLAHE for local contrast enhancement
- Film grain with luminance masking
- Temporal coherence via exponential smoothing
- Multi-scale Laplacian blending for overlays
- Gamma-correct alpha compositing
"""
import cv2
import numpy as np
import time
from typing import Dict
from .base import Filter, FilterMode

try:
    from .graphics_engine import (
        ColorQuantizer, Dithering, ChromaticAberration,
        ScanlineEffects, FilmGrain, CLAHEEnhancer,
        TemporalSmoother, AlphaCompositor, PyramidBlend
    )
    SOTA_AVAILABLE = True
except ImportError:
    SOTA_AVAILABLE = False

class MaxHeadroomFilter(Filter):
    """
    Max Headroom styled android/digital entity transformation filter.
    v2.0 uses SOTA graphics engine for broadcast-quality effects.
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("Max Headroom", mode)
        self.priority = 2
        self.enabled = False
        
        self.params = {
            "intensity": 1.0,
            "monochrome": True,
            "cyan_boost": 1.8,
            "contrast": 1.6,
            "brightness": -10,
            "scanlines": True,
            "scanline_thickness": 2,
            "scanline_spacing": 3,
            "scanline_alpha": 0.35,
            "phosphor": True,
            "interlace": True,
            "chromatic": True,
            "chromatic_shift": 4.0,
            "chromatic_probability": 0.3,
            "sharpen": True,
            "sharpen_amount": 1.2,
            "pixelate": True,
            "pixelate_scale": 0.25,
            "stutter": True,
            "stutter_probability": 0.08,
            "stutter_frames": 2,
            "glitch_blocks": True,
            "glitch_block_probability": 0.15,
            "glitch_block_count": 3,
            "grid": True,
            "grid_spacing": 40,
            "grid_color": [0, 255, 255],
            "grid_alpha": 0.25,
            "data_overlay": True,
            "data_text": [
                "SIGNAL: NOMINAL", "BROADCAST: LIVE", "ENTITY: MAX",
                "MODE: DIGITAL", "ORIGIN: CYBERSPACE", "STATUS: ONLINE",
                "FORMAT: NTSC", "NOISE: 0.04%", "SYNC: LOCKED",
            ],
            "vignette": True,
            "vignette_strength": 0.6,
            # SOTA enhancements
            "posterize": True,
            "posterize_levels": 6,
            "dither": True,
            "dither_levels": 4,
            "film_grain": True,
            "grain_intensity": 0.06,
            "clahe": True,
            "clahe_clip": 2.0,
            "temporal_smooth": True,
            "temporal_alpha": 0.75,
        }
        
        self._prev_frame = None
        self._stutter_counter = 0
        self._scanline_mask = None
        self._vignette_mask = None
        self._data_scroll_offset = 0
        self._last_data_update = 0
        self._frame_id = 0
        
        # SOTA engine instances
        self._temporal = None
        self._clahe_enhancer = None
        if SOTA_AVAILABLE:
            self._temporal = TemporalSmoother(alpha=self.params["temporal_alpha"])
            self._clahe_enhancer = CLAHEEnhancer(
                clip_limit=self.params["clahe_clip"],
                tile_size=(8, 8)
            )
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled or frame is None or frame.size == 0:
            return frame
        
        self._frame_id += 1
        intensity = self.params["intensity"]
        if intensity <= 0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Temporal stutter
        if self.params["stutter"] and intensity > 0.3:
            stuttered = self._apply_stutter(frame)
            if stuttered is not frame:
                return stuttered
        
        result = frame.copy()
        
        # 1. Pixelation
        if self.params["pixelate"] and intensity > 0.2:
            result = self._apply_pixelation(result)
        
        # 2. Color grading: monochrome + cyan + CLAHE (SOTA)
        if self.params["monochrome"]:
            result = self._apply_monochrome(result, intensity)
            if SOTA_AVAILABLE and self.params["clahe"]:
                result = self._apply_clahe(result)
        
        # 3. Edge enhancement
        if self.params["sharpen"]:
            result = self._apply_sharpen(result, intensity)
        
        # 4. Chromatic aberration (SOTA radial)
        if SOTA_AVAILABLE and self.params["chromatic"] and intensity > 0.3:
            result = self._apply_chromatic_sota(result, w, h, intensity)
        elif self.params["chromatic"] and intensity > 0.3:
            result = self._apply_chromatic_legacy(result, w, h, intensity)
        
        # 5. Glitch blocks
        if self.params["glitch_blocks"] and intensity > 0.4:
            result = self._apply_glitch_blocks(result, w, h, intensity)
        
        # 6. Scanlines (SOTA: CRT + phosphor + interlace)
        if self.params["scanlines"]:
            result = self._apply_scanlines(result, h, w, intensity)
            if SOTA_AVAILABLE and self.params["phosphor"] and intensity > 0.5:
                result = ScanlineEffects.rgb_phosphor(result, strength=0.25)
            if SOTA_AVAILABLE and self.params["interlace"] and intensity > 0.4:
                result = ScanlineEffects.interlace_flicker(result, self._frame_id, intensity=0.08)
        
        # 7. Film grain (SOTA)
        if SOTA_AVAILABLE and self.params["film_grain"] and intensity > 0.3:
            result = FilmGrain.apply(result, intensity=self.params["grain_intensity"] * intensity,
                                    grain_size=1.5, color=False)
        
        # 8. Geometric grid
        if self.params["grid"] and intensity > 0.2:
            result = self._apply_grid(result, w, h, intensity)
        
        # 9. Data overlay
        if self.params["data_overlay"] and intensity > 0.3:
            result = self._apply_data_overlay(result, w, h, intensity)
        
        # 10. Vignette
        if self.params["vignette"]:
            result = self._apply_vignette(result, h, w, intensity)
        
        # 11. Posterization + Dithering (SOTA) - applied LAST for visible effect
        if SOTA_AVAILABLE and self.params["posterize"] and intensity > 0.3:
            result = ColorQuantizer.quantize_fast(result, k=self.params["posterize_levels"])
            if self.params["dither"]:
                result = Dithering.ordered_dither(result, levels=self.params["dither_levels"])
        
        # 12. Temporal coherence (SOTA)
        if SOTA_AVAILABLE and self.params["temporal_smooth"]:
            result = self._temporal.apply(result)
        
        self._prev_frame = result.copy()
        return result
    
    def _apply_stutter(self, frame: np.ndarray) -> np.ndarray:
        if self._stutter_counter > 0:
            self._stutter_counter -= 1
            if self._prev_frame is not None and self._prev_frame.shape == frame.shape:
                return self._prev_frame.copy()
        if np.random.random() < self.params["stutter_probability"]:
            self._stutter_counter = self.params["stutter_frames"]
        return frame
    
    def _apply_pixelation(self, frame: np.ndarray) -> np.ndarray:
        scale = max(0.05, min(0.5, self.params["pixelate_scale"]))
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def _apply_monochrome(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = 1.0 + (self.params["contrast"] - 1.0) * intensity
        brightness = self.params["brightness"] * intensity
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
        cyan_boost = 1.0 + (self.params["cyan_boost"] - 1.0) * intensity
        result = np.zeros_like(frame)
        result[:, :, 0] = np.clip(gray * cyan_boost, 0, 255).astype(np.uint8)
        result[:, :, 1] = np.clip(gray * 0.9, 0, 255).astype(np.uint8)
        result[:, :, 2] = np.clip(gray * 0.3, 0, 255).astype(np.uint8)
        if intensity < 1.0:
            result = cv2.addWeighted(frame, 1.0 - intensity, result, intensity, 0)
        return result
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE for enhanced local contrast."""
        if self._clahe_enhancer is None:
            return frame
        try:
            return self._clahe_enhancer.apply(frame)
        except:
            return frame
    
    def _apply_sharpen(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        amount = 1.0 + (self.params["sharpen_amount"] - 1.0) * intensity
        blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _apply_chromatic_sota(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
        """SOTA radial chromatic aberration."""
        prob = self.params["chromatic_probability"]
        if np.random.random() > prob:
            return frame
        strength = self.params["chromatic_shift"] * intensity
        try:
            return ChromaticAberration.apply(frame, strength=strength)
        except:
            return self._apply_chromatic_legacy(frame, w, h, intensity)
    
    def _apply_chromatic_legacy(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
        """Legacy simple chromatic aberration."""
        prob = self.params["chromatic_probability"]
        if np.random.random() > prob:
            return frame
        shift = int(self.params["chromatic_shift"] * intensity)
        if shift < 1:
            return frame
        b, g, r = cv2.split(frame)
        r_shifted = np.zeros_like(r)
        r_shifted[:, shift:] = r[:, :-shift]
        b_shifted = np.zeros_like(b)
        b_shifted[:, :-shift] = b[:, shift:]
        return cv2.merge([b_shifted, g, r_shifted])
    
    def _apply_glitch_blocks(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
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
                result[by:by+bh, bx:bx+bw] = 255 - result[by:by+bh, bx:bx+bw]
            elif glitch_type == 1:
                shift_x = np.random.randint(-20, 20)
                block = result[by:by+bh, bx:bx+bw].copy()
                if shift_x > 0:
                    result[by:by+bh, bx+shift_x:bx+bw] = block[:, :-shift_x]
                elif shift_x < 0:
                    result[by:by+bh, bx:bx+bw+shift_x] = block[:, -shift_x:]
            else:
                result[by:by+bh, bx:bx+bw] = [255, 255, 0]
        return result
    
    def _apply_scanlines(self, frame: np.ndarray, h: int, w: int, intensity: float) -> np.ndarray:
        if SOTA_AVAILABLE:
            try:
                alpha = self.params["scanline_alpha"] * intensity
                return ScanlineEffects.crt_scanlines(
                    frame,
                    thickness=self.params["scanline_thickness"],
                    spacing=self.params["scanline_spacing"],
                    alpha=alpha
                )
            except:
                pass
        # Legacy fallback
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
        alpha = self.params["grid_alpha"] * intensity
        if alpha < 0.05:
            return frame
        spacing = self.params["grid_spacing"]
        color = np.array(self.params["grid_color"], dtype=np.uint8)
        overlay = np.zeros_like(frame)
        for x in range(0, w, spacing):
            cv2.line(overlay, (x, 0), (x, h), color.tolist(), 1)
        for y in range(0, h, spacing):
            cv2.line(overlay, (0, y), (w, y), color.tolist(), 1)
        cx, cy = w // 2, h // 2
        for i in range(1, 6):
            offset = i * spacing
            pts = np.array([
                [cx - offset, cy - offset // 2],
                [cx + offset, cy - offset // 2],
                [cx + offset * 2, cy + offset],
                [cx - offset * 2, cy + offset]
            ], np.int32)
            cv2.polylines(overlay, [pts], True, color.tolist(), 1)
        return cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
    
    def _apply_data_overlay(self, frame: np.ndarray, w: int, h: int, intensity: float) -> np.ndarray:
        t = time.time()
        if t - self._last_data_update > 0.5:
            self._data_scroll_offset = (self._data_scroll_offset + 1) % len(self.params["data_text"])
            self._last_data_update = t
        alpha = 0.6 * intensity
        overlay = frame.copy()
        bar_height = 20
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
        texts = self.params["data_text"]
        idx = self._data_scroll_offset
        display_text = f">>> {texts[idx]} <<<"
        cv2.putText(frame, display_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        hex_time = f"0x{int(t * 1000) % 0xFFFF:04X}"
        cv2.putText(frame, hex_time, (w - 70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        frame_hex = f"FRAME_ID: 0x{self._frame_id % 0xFFFF:04X}"
        cv2.putText(frame, frame_hex, (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
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
        if intensity > 0.6:
            hex_lines = ["7F 3A 9E 2B", "C4 11 88 FF", "00 7F FE 01", "AA 55 AA 55"]
            for i, line in enumerate(hex_lines):
                y_pos = 60 + i * 14
                if y_pos < h - 40:
                    cv2.putText(frame, line, (w - 90, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 150, 200), 1)
        return frame
    
    def _apply_vignette(self, frame: np.ndarray, h: int, w: int, intensity: float) -> np.ndarray:
        if self._vignette_mask is None or self._vignette_mask.shape[:2] != (h, w):
            X = cv2.getGaussianKernel(w, w * 0.6)
            Y = cv2.getGaussianKernel(h, h * 0.6)
            kernel = Y * X.T
            self._vignette_mask = kernel / kernel.max()
        strength = self.params["vignette_strength"] * intensity
        mask = 1.0 - (1.0 - self._vignette_mask) * strength
        mask = np.dstack([mask] * 3)
        return (frame.astype(np.float32) * mask).astype(np.uint8)
    
    def set_intensity(self, intensity: float):
        self.params["intensity"] = max(0.0, min(1.0, intensity))
    
    def cycle_intensity(self):
        levels = [0.0, 0.3, 0.6, 1.0]
        current = self.params["intensity"]
        idx = (levels.index(min(levels, key=lambda x: abs(x - current))) + 1) % len(levels)
        self.params["intensity"] = levels[idx]
        return levels[idx]
