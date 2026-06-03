"""
Motion Capture Visualization Filter.
Renders tracking overlay: wireframe face mesh, tracking points, skeleton, head pose axes.
"""
import cv2
import numpy as np
import math
from typing import Dict, List, Tuple
from .base import Filter, FilterMode

FACEMESH_TRIANGLES = [
    (0, 1, 2), (1, 3, 4), (3, 5, 6), (5, 7, 8),
    (7, 9, 10), (9, 11, 12), (11, 13, 14), (13, 15, 16),
    (15, 17, 18), (17, 19, 20), (14, 21, 22), (21, 23, 24),
    (23, 25, 26), (25, 27, 28), (27, 29, 30), (29, 31, 32),
    (31, 33, 34), (33, 35, 36), (35, 37, 38), (37, 39, 40),
    (39, 41, 42), (36, 43, 44), (43, 45, 46), (45, 47, 48),
]

FACEMESH_CONTOURS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
    (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (20, 21),
    (22, 23), (23, 24), (24, 25), (25, 26),
    (27, 28), (28, 29), (29, 30), (30, 31),
    (32, 33), (33, 34), (34, 35), (35, 36),
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42),
    (43, 44), (44, 45), (45, 46), (46, 47),
]

FACEMESH_TESSELATION = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    (12, 13), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (20, 21),
    (22, 23), (23, 24), (24, 25), (25, 26),
    (27, 28), (28, 29), (29, 30), (30, 31),
    (32, 33), (33, 34), (34, 35), (35, 36),
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42),
    (43, 44), (44, 45), (45, 46), (46, 47),
    (0, 17), (2, 22), (4, 27), (6, 32), (8, 36), (10, 43),
    (12, 16), (14, 21), (16, 26), (21, 31), (26, 36), (31, 41),
]


class MoCapVizFilter(Filter):
    """
    Motion capture visualization overlay.
    Renders face wireframe, tracking points, head pose axes,
    and optional skeleton overlay for a professional mocap look.
    """

    def __init__(self, mode: FilterMode = FilterMode.OFF):
        super().__init__("MoCap Viz", mode)
        self.priority = 35
        self.requires_landmarks = True
        self.params = {
            "wireframe": True,
            "wireframe_color": (0, 200, 255),
            "wireframe_alpha": 0.6,
            "tracking_points": True,
            "point_color": (0, 255, 255),
            "point_size": 3,
            "pose_axes": True,
            "pose_axis_length": 40,
            "skeleton": False,
            "skeleton_color": (0, 100, 255),
            "labels": False,
            "label_color": (255, 255, 255),
            "intensity": 1.0,
            "style": "tech",  # tech, neon, dark, minimal
        }
        self._styles = {
            "tech": {"wf": (0, 200, 255), "pt": (0, 255, 255), "sk": (0, 100, 255)},
            "neon": {"wf": (255, 0, 255), "pt": (0, 255, 128), "sk": (128, 0, 255)},
            "dark": {"wf": (60, 60, 80), "pt": (100, 100, 120), "sk": (40, 40, 60)},
            "minimal": {"wf": (100, 200, 255), "pt": (200, 255, 255), "sk": (50, 150, 200)},
        }

    def set_style(self, style: str):
        if style in self._styles:
            self.params["style"] = style
            s = self._styles[style]
            self.params["wireframe_color"] = s["wf"]
            self.params["point_color"] = s["pt"]
            self.params["skeleton_color"] = s["sk"]

    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        if not self.enabled or not context:
            return frame
        landmarks = context.get("landmarks", [])
        if not landmarks or len(landmarks) < 68:
            return frame

        intensity = self.params.get("intensity", 1.0)
        if intensity <= 0:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]

        if self.params.get("wireframe", True) and intensity > 0.2:
            result = self._draw_wireframe(result, landmarks, intensity)

        if self.params.get("tracking_points", True) and intensity > 0.3:
            result = self._draw_tracking_points(result, landmarks, intensity)

        if self.params.get("pose_axes", True) and intensity > 0.3:
            pose = context.get("head_pose", {})
            if pose:
                result = self._draw_pose_axes(result, landmarks, pose, intensity)

        if self.params.get("skeleton", False) and intensity > 0.4:
            result = self._draw_skeleton(result, landmarks, intensity)

        if self.params.get("labels", False) and intensity > 0.5:
            result = self._draw_labels(result, landmarks, intensity)

        return result

    def _draw_wireframe(self, frame: np.ndarray, landmarks: List,
                        intensity: float) -> np.ndarray:
        color = self.params.get("wireframe_color", (0, 200, 255))
        alpha = self.params.get("wireframe_alpha", 0.6) * intensity
        pts = np.array(landmarks[:68], dtype=np.int32)

        overlay = frame.copy()

        # Draw tesselation edges
        for i, j in FACEMESH_TESSELATION:
            if i < len(pts) and j < len(pts):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[j]), color, 1, cv2.LINE_AA)

        # Draw contours with thicker lines
        contour_color = (color[0] // 2, color[1] // 2, color[2] // 2)
        for i, j in FACEMESH_CONTOURS:
            if i < len(pts) and j < len(pts):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[j]), contour_color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame

    def _draw_tracking_points(self, frame: np.ndarray, landmarks: List,
                              intensity: int) -> np.ndarray:
        color = self.params.get("point_color", (0, 255, 255))
        base_size = self.params.get("point_size", 3)
        size = max(1, int(base_size * intensity))

        for i, pt in enumerate(landmarks[:68]):
            x, y = int(pt[0]), int(pt[1])
            if i in (36, 39, 42, 45, 30, 33, 48, 51, 54, 57, 8, 62, 66):
                cv2.circle(frame, (x, y), size + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)

        return frame

    def _draw_pose_axes(self, frame: np.ndarray, landmarks: List,
                        pose: Dict, intensity: int) -> np.ndarray:
        axis_len = int(self.params.get("pose_axis_length", 40) * intensity)
        if len(landmarks) < 30:
            return frame

        nose = (int(landmarks[30][0]), int(landmarks[30][1]))

        rot = pose.get("rotation", [0, 0, 0])
        roll, pitch, yaw = [math.radians(r) for r in rot]

        cos_r, sin_r = math.cos(roll), math.sin(roll)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        axes = {
            "X": (axis_len, 0, 0),
            "Y": (0, -axis_len, 0),
            "Z": (0, 0, axis_len),
        }
        colors = {"X": (0, 0, 255), "Y": (0, 255, 0), "Z": (255, 0, 0)}

        for axis, (dx, dy, dz) in axes.items():
            x_rot = dx * cos_r * cos_y + dy * (sin_p * sin_r * cos_y - cos_p * sin_y) + dz * (cos_p * sin_r * cos_y + sin_p * sin_y)
            y_rot = dx * cos_r * sin_y + dy * (sin_p * sin_r * sin_y + cos_p * cos_y) + dz * (cos_p * sin_r * sin_y - sin_p * cos_y)
            end_pt = (int(nose[0] + x_rot), int(nose[1] + y_rot))
            cv2.arrowedLine(frame, nose, end_pt, colors[axis], 2, cv2.LINE_AA, tipLength=0.3)

        return frame

    def _draw_skeleton(self, frame: np.ndarray, landmarks: List,
                       intensity: int) -> np.ndarray:
        color = self.params.get("skeleton_color", (0, 100, 255))
        if len(landmarks) < 68:
            return frame
        pts = np.array(landmarks[:68], dtype=np.int32)

        skeleton_edges = [
            (1, 15), (15, 14), (14, 13), (13, 3),
            (2, 12), (12, 11), (11, 10), (10, 4),
            (3, 4), (3, 5), (4, 6),
            (5, 7), (6, 8), (7, 9), (8, 10),
            (27, 28), (28, 29), (29, 30),
            (31, 32), (32, 33), (33, 34), (34, 35),
            (36, 37), (37, 38), (38, 39), (39, 40), (40, 41),
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56),
            (56, 57), (57, 58), (58, 59),
            (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),
        ]

        thickness = max(1, int(2 * intensity))
        for i, j in skeleton_edges:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, thickness, cv2.LINE_AA)

        return frame

    def _draw_labels(self, frame: np.ndarray, landmarks: List,
                     intensity: int) -> np.ndarray:
        color = self.params.get("label_color", (255, 255, 255))
        key_points = {
            30: "NOSE", 33: "NOSE_BR", 8: "CHIN",
            36: "L_EYE", 45: "R_EYE",
            48: "L_MOUTH", 54: "R_MOUTH",
            17: "L_BROW", 26: "R_BROW",
        }

        for idx, label in key_points.items():
            if idx < len(landmarks):
                pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
                cv2.putText(frame, label, (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        return frame
