"""
Max Headroom - Face Morph Filter
Subtle face morphing: slimming, eye enlarging, jaw shaping
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
from .base import Filter, FilterMode

class FaceMorphFilter(Filter):
    """
    Subtle face morphing using mesh deformation.
    Implements face slimming, eye enlarging, and jaw shaping.
    """
    
    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("Face Morph", mode)
        self.priority = 15
        self.requires_landmarks = True
        self.params = {
            "slimming": 0.0,        # -1.0 to 1.0, 0 = off
            "eye_enlarge": 0.0,     # 0.0 to 1.0
            "jaw_shaping": 0.0,     # -1.0 to 1.0
            "chin_shaping": 0.0,    # -1.0 to 1.0
        }
    
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled or not context:
            return frame
        
        landmarks = context.get("landmarks", [])
        if not landmarks or len(landmarks) < 68:
            return frame
        
        # Check if any morphing is active
        if all(abs(self.params.get(k, 0)) < 0.01 for k in ["slimming", "eye_enlarge", "jaw_shaping", "chin_shaping"]):
            return frame
        
        # Create mesh from landmarks
        src_points = np.array(landmarks, dtype=np.float32)
        dst_points = src_points.copy()
        
        h, w = frame.shape[:2]
        
        # Apply morphs
        if abs(self.params.get("slimming", 0)) > 0.01:
            dst_points = self._apply_slimming(dst_points)
        
        if self.params.get("eye_enlarge", 0) > 0.01:
            dst_points = self._apply_eye_enlarge(dst_points)
        
        if abs(self.params.get("jaw_shaping", 0)) > 0.01:
            dst_points = self._apply_jaw_shaping(dst_points)
        
        if abs(self.params.get("chin_shaping", 0)) > 0.01:
            dst_points = self._apply_chin_shaping(dst_points)
        
        # Add boundary points for stable mesh
        boundary = np.array([
            [0, 0], [w//2, 0], [w-1, 0],
            [w-1, h//2], [w-1, h-1], [w//2, h-1], [0, h-1], [0, h//2]
        ], dtype=np.float32)
        
        src_points = np.vstack([src_points, boundary])
        dst_points = np.vstack([dst_points, boundary])
        
        # Compute Delaunay triangulation
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for pt in dst_points:
            subdiv.insert((float(pt[0]), float(pt[1])))
        
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32).reshape(-1, 3, 2)
        
        # Warp image
        result = self._warp_image(frame, src_points, dst_points, triangles)
        
        return result
    
    def _apply_slimming(self, points: np.ndarray) -> np.ndarray:
        """Slim the face by moving jaw and cheek inward."""
        strength = self.params["slimming"]
        center_x = (points[0][0] + points[16][0]) / 2
        
        # Jaw points: 0-16
        jaw_indices = list(range(0, 17))
        for idx in jaw_indices:
            dist_from_center = points[idx][0] - center_x
            points[idx][0] -= dist_from_center * strength * 0.15
        
        # Cheek points: 1-15 (upper jaw)
        for idx in range(1, 16):
            dist_from_center = points[idx][0] - center_x
            points[idx][0] -= dist_from_center * strength * 0.1
        
        return points
    
    def _apply_eye_enlarge(self, points: np.ndarray) -> np.ndarray:
        """Enlarge eyes by moving eye corners outward."""
        strength = self.params["eye_enlarge"]
        
        # Left eye center
        left_center = np.mean(points[36:42], axis=0)
        for idx in range(36, 42):
            direction = points[idx] - left_center
            points[idx] = left_center + direction * (1 + strength * 0.15)
        
        # Right eye center
        right_center = np.mean(points[42:48], axis=0)
        for idx in range(42, 48):
            direction = points[idx] - right_center
            points[idx] = right_center + direction * (1 + strength * 0.15)
        
        return points
    
    def _apply_jaw_shaping(self, points: np.ndarray) -> np.ndarray:
        """Shape jaw line (up/down)."""
        strength = self.params["jaw_shaping"]
        
        # Lower jaw points
        for idx in range(4, 13):
            points[idx][1] += strength * 8
        
        return points
    
    def _apply_chin_shaping(self, points: np.ndarray) -> np.ndarray:
        """Shape chin point (up/down)."""
        strength = self.params["chin_shaping"]
        
        # Chin point is landmark 8
        points[8][1] += strength * 12
        
        # Smooth neighbors
        points[7][1] += strength * 6
        points[9][1] += strength * 6
        
        return points
    
    def _warp_image(self, img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, 
                   triangles: np.ndarray) -> np.ndarray:
        """Warp image using affine transformations on triangles."""
        h, w = img.shape[:2]
        result = np.zeros_like(img)
        
        for tri in triangles:
            # Get triangle vertices
            tri_dst = np.float32(tri)
            
            # Find corresponding source triangle
            src_tri = []
            for pt in tri_dst:
                distances = np.linalg.norm(dst_pts - pt, axis=1)
                idx = np.argmin(distances)
                src_tri.append(src_pts[idx])
            src_tri = np.float32(src_tri)
            
            # Validate triangles
            if src_tri.shape != (3, 2) or tri_dst.shape != (3, 2):
                continue
            
            # Check for collinear points
            area = abs((src_tri[1][0] - src_tri[0][0]) * (src_tri[2][1] - src_tri[0][1]) - 
                      (src_tri[2][0] - src_tri[0][0]) * (src_tri[1][1] - src_tri[0][1]))
            if area < 1.0:
                continue
            
            # Get bounding rectangles
            r_dst = cv2.boundingRect(tri_dst)
            r_src = cv2.boundingRect(src_tri)
            
            if r_src[2] <= 0 or r_src[3] <= 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
                continue
            
            # Offset points to bounding rect
            tri_dst_cropped = tri_dst - [r_dst[0], r_dst[1]]
            tri_src_cropped = src_tri - [r_src[0], r_src[1]]
            
            # Affine transform
            try:
                warp_mat = cv2.getAffineTransform(tri_src_cropped, tri_dst_cropped)
            except cv2.error:
                continue
            
            # Crop source
            src_cropped = img[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]
            
            if src_cropped.size == 0:
                continue
            
            # Warp
            dst_cropped = cv2.warpAffine(src_cropped, warp_mat, (r_dst[2], r_dst[3]),
                                        None, flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
            
            # Create mask for triangle
            mask = np.zeros((r_dst[3], r_dst[2]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tri_dst_cropped), 255)
            
            # Blend
            roi = result[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]]
            masked_dst = cv2.bitwise_and(dst_cropped, dst_cropped, mask=mask)
            masked_roi = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            result[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] = cv2.add(masked_roi, masked_dst)
        
        return result
