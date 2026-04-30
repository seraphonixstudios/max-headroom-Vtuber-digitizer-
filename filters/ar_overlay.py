"""
Max Headroom - AR Overlay Filter
Stickers and effects anchored to facial landmarks
"""
import cv2
import numpy as np
import math
from typing import Dict, List, Tuple
from .base import Filter, FilterMode

class AROverlayFilter(Filter):
    """
    Augmented Reality overlay system.
    Places stickers/effects on facial landmarks with proper anchoring.
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("AR Overlay", mode)
        self.priority = 30
        self.requires_landmarks = True
        self.params = {
            "stickers": [],  # List of active stickers
            "scale": 1.0,
            "alpha": 0.9,
        }
        self._sticker_cache = {}
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled or not context:
            return frame
        
        landmarks = context.get("landmarks", [])
        if not landmarks or len(landmarks) < 68:
            return frame
        
        result = frame.copy()
        stickers = self.params.get("stickers", [])
        
        for sticker in stickers:
            if not sticker.get("enabled", True):
                continue
            
            sticker_type = sticker.get("type", "none")
            anchor = sticker.get("anchor", "forehead")
            
            if sticker_type == "glasses":
                result = self._draw_glasses(result, landmarks, sticker)
            elif sticker_type == "hat":
                result = self._draw_hat(result, landmarks, sticker)
            elif sticker_type == "mustache":
                result = self._draw_mustache(result, landmarks, sticker)
            elif sticker_type == "blush":
                result = self._draw_blush(result, landmarks, sticker)
            elif sticker_type == "crown":
                result = self._draw_crown(result, landmarks, sticker)
            elif sticker_type == "tears":
                result = self._draw_tears(result, landmarks, sticker)
        
        return result
    
    def add_sticker(self, sticker_type: str, anchor: str = "auto", **kwargs):
        """Add a sticker to the active list."""
        sticker = {
            "type": sticker_type,
            "anchor": anchor,
            "enabled": True,
            **kwargs
        }
        self.params["stickers"].append(sticker)
    
    def clear_stickers(self):
        """Remove all stickers."""
        self.params["stickers"] = []
    
    def _draw_glasses(self, frame: np.ndarray, landmarks: List, sticker: Dict) -> np.ndarray:
        """Draw AR glasses on eyes."""
        # Left eye center (36-41), Right eye center (42-47)
        left_eye_pts = [landmarks[i] for i in range(36, 42)]
        right_eye_pts = [landmarks[i] for i in range(42, 48)]
        
        if not left_eye_pts or not right_eye_pts:
            return frame
        
        left_center = tuple(np.mean(left_eye_pts, axis=0).astype(int))
        right_center = tuple(np.mean(right_eye_pts, axis=0).astype(int))
        
        eye_distance = int(np.linalg.norm(np.array(right_center) - np.array(left_center)))
        
        color = sticker.get("color", (0, 255, 255))  # Cyan default
        thickness = sticker.get("thickness", 2)
        
        # Draw frames
        radius = int(eye_distance * 0.35)
        cv2.circle(frame, left_center, radius, color, thickness)
        cv2.circle(frame, right_center, radius, color, thickness)
        
        # Bridge
        cv2.line(frame, 
                (left_center[0] + radius, left_center[1]),
                (right_center[0] - radius, right_center[1]),
                color, thickness)
        
        # Temple arms
        cv2.line(frame, (left_center[0] - radius, left_center[1]),
                (left_center[0] - radius - int(eye_distance*0.3), left_center[1] - 5), color, thickness)
        cv2.line(frame, (right_center[0] + radius, right_center[1]),
                (right_center[0] + radius + int(eye_distance*0.3), right_center[1] - 5), color, thickness)
        
        # Lenses (semi-transparent fill)
        overlay = frame.copy()
        cv2.circle(overlay, left_center, radius - 2, (color[0]//4, color[1]//4, color[2]//4), -1)
        cv2.circle(overlay, right_center, radius - 2, (color[0]//4, color[1]//4, color[2]//4), -1)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def _draw_hat(self, frame: np.ndarray, landmarks: List, sticker: Dict) -> np.ndarray:
        """Draw a hat above the forehead."""
        # Forehead area - use eyebrows as reference
        brow_y = min(landmarks[i][1] for i in range(17, 27))
        center_x = int(np.mean([landmarks[i][0] for i in range(17, 27)]))
        
        color = sticker.get("color", (139, 69, 19))  # Brown
        
        # Hat dimensions
        hat_width = int(abs(landmarks[0][0] - landmarks[16][0]) * 1.2)
        hat_height = int(hat_width * 0.6)
        
        top_y = max(0, brow_y - hat_height - 10)
        
        # Draw hat body (rounded rectangle)
        pt1 = (max(0, center_x - hat_width//2), top_y)
        pt2 = (min(frame.shape[1], center_x + hat_width//2), brow_y - 5)
        cv2.rectangle(frame, pt1, pt2, color, -1)
        cv2.rectangle(frame, pt1, pt2, (color[0]//2, color[1]//2, color[2]//2), 2)
        
        # Hat brim
        brim_y = brow_y - 5
        cv2.ellipse(frame, (center_x, brim_y), (hat_width//2 + 5, 10), 0, 0, 360, color, -1)
        cv2.ellipse(frame, (center_x, brim_y), (hat_width//2 + 5, 10), 0, 0, 360, (0,0,0), 2)
        
        return frame
    
    def _draw_mustache(self, frame: np.ndarray, landmarks: List, sticker: Dict) -> np.ndarray:
        """Draw a mustache above the upper lip."""
        # Use mouth landmarks 48-54 (upper lip)
        lip_pts = [landmarks[i] for i in range(48, 55)]
        if not lip_pts:
            return frame
        
        center_x = int(np.mean([p[0] for p in lip_pts]))
        center_y = int(np.mean([p[1] for p in lip_pts])) - 5
        
        width = int(abs(landmarks[48][0] - landmarks[54][0]) * 1.1)
        height = int(width * 0.3)
        
        color = sticker.get("color", (50, 50, 50))  # Dark gray
        
        # Draw handlebar mustache using bezier curves
        pts = np.array([
            [center_x - width//2, center_y],
            [center_x - width//4, center_y - height//2],
            [center_x, center_y],
            [center_x + width//4, center_y - height//2],
            [center_x + width//2, center_y],
            [center_x + width//3, center_y + height//3],
            [center_x, center_y + height//4],
            [center_x - width//3, center_y + height//3],
        ], np.int32)
        
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (0, 0, 0), 1)
        
        return frame
    
    def _draw_blush(self, frame: np.ndarray, landmarks: List, sticker: Dict) -> np.ndarray:
        """Draw blush on cheeks."""
        color = sticker.get("color", (200, 150, 255))  # Pinkish BGR
        alpha = sticker.get("alpha", 0.2)
        
        # Left cheek (landmark 29 is nose tip, 2 is jaw left)
        left_cheek = (
            (landmarks[2][0] + landmarks[29][0]) // 2,
            (landmarks[2][1] + landmarks[29][1]) // 2
        )
        
        # Right cheek
        right_cheek = (
            (landmarks[14][0] + landmarks[29][0]) // 2,
            (landmarks[14][1] + landmarks[29][1]) // 2
        )
        
        overlay = frame.copy()
        
        for cheek in [left_cheek, right_cheek]:
            cv2.circle(overlay, cheek, 20, color, -1)
        
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame
    
    def _draw_crown(self, frame: np.ndarray, landmarks: List, sticker: Dict) -> np.ndarray:
        """Draw a crown on the head."""
        brow_y = min(landmarks[i][1] for i in range(17, 27))
        center_x = int(np.mean([landmarks[i][0] for i in range(17, 27)]))
        
        color = sticker.get("color", (0, 215, 255))  # Gold BGR
        
        # Crown points
        crown_width = int(abs(landmarks[0][0] - landmarks[16][0]) * 0.9)
        crown_height = 30
        top_y = max(0, brow_y - crown_height - 20)
        
        # 5 points
        pts = np.array([
            [center_x - crown_width//2, brow_y - 10],
            [center_x - crown_width//4, top_y],
            [center_x, brow_y - 10],
            [center_x + crown_width//4, top_y],
            [center_x + crown_width//2, brow_y - 10],
        ], np.int32)
        
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (0, 100, 200), 2)
        
        # Jewels
        for pt in pts[:-1]:
            cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)  # Red jewels
        
        return frame
    
    def _draw_tears(self, frame: np.ndarray, landmarks: List, sticker: Dict) -> np.ndarray:
        """Draw anime-style tears under eyes."""
        color = sticker.get("color", (255, 255, 0))  # Cyan/yellow BGR
        
        # Left eye bottom (landmark 41)
        left_tear = landmarks[41]
        # Right eye bottom (landmark 47)
        right_tear = landmarks[47]
        
        for eye_pt in [left_tear, right_tear]:
            x, y = eye_pt
            # Draw tear drop
            pts = np.array([
                [x, y + 5],
                [x - 5, y + 20],
                [x, y + 35],
                [x + 5, y + 20],
            ], np.int32)
            cv2.fillPoly(frame, [pts], color)
            cv2.polylines(frame, [pts], True, (200, 200, 0), 1)
            
            # Sparkle
            cv2.circle(frame, (x, y + 30), 2, (255, 255, 255), -1)
        
        return frame
