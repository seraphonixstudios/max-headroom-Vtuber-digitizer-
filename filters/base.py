"""
Max Headroom - Base Filter Class
All filters inherit from this
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class FilterMode(Enum):
    """Filter operation modes."""
    OFF = 0
    PREVIEW = 1      # Show in UI only
    STREAM = 2       # Apply to stream output
    BOTH = 3         # Preview + Stream

class Filter(ABC):
    """Abstract base class for all video filters."""
    
    def __init__(self, name: str, mode: FilterMode = FilterMode.OFF):
        self.name = name
        self.mode = mode
        self.enabled = False
        self.params: Dict[str, Any] = {}
        self.priority = 50  # Lower = earlier in pipeline
        self.requires_landmarks = False
        self.requires_segmentation = False
    
    @abstractmethod
    def process(self, frame: np.ndarray, context: Dict = None) -> np.ndarray:
        """
        Process a frame and return modified frame.
        
        Args:
            frame: BGR image (H, W, 3)
            context: Dictionary with 'landmarks', 'face_rect', 'segmentation_mask', etc.
        
        Returns:
            Modified frame
        """
        pass
    
    def set_param(self, key: str, value: Any):
        """Update a filter parameter."""
        self.params[key] = value
    
    def get_param(self, key: str, default=None):
        """Get a filter parameter."""
        return self.params.get(key, default)
    
    def toggle(self):
        """Toggle filter on/off."""
        self.enabled = not self.enabled
        return self.enabled
    
    def enable(self):
        """Enable filter."""
        self.enabled = True
    
    def disable(self):
        """Disable filter."""
        self.enabled = False

class FilterContext:
    """Context passed between filters in pipeline."""
    
    def __init__(self):
        self.landmarks: Optional[List[Tuple[int, int]]] = None
        self.face_rect: Optional[Tuple[int, int, int, int]] = None
        self.segmentation_mask: Optional[np.ndarray] = None
        self.head_pose: Optional[Dict] = None
        self.blendshapes: Optional[Dict[str, float]] = None
        self.frame_id: int = 0
        self.timestamp: float = 0.0
        self.metadata: Dict[str, Any] = {}
    
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and len(self.landmarks) > 0
    
    def has_segmentation(self) -> bool:
        return self.segmentation_mask is not None