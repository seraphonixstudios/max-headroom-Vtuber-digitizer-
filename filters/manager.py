"""
Max Headroom - Filter Manager
Orchestrates the filter pipeline
"""
import time
import numpy as np
from typing import Dict, List, Optional

from .base import Filter, FilterContext
from .skin_smoothing import SkinSmoothingFilter
from .background_removal import BackgroundRemovalFilter
from .ar_overlay import AROverlayFilter
from .face_morph import FaceMorphFilter
from .color_grading import ColorGradingFilter
from .max_headroom_filter import MaxHeadroomFilter

try:
    from logging_utils import get_logger
    LOG = get_logger("Filters")
except Exception:
    import logging
    LOG = logging.getLogger("MaxHeadroom.Filters")

class FilterManager:
    """
    Manages all video filters and applies them in priority order.
    Provides hot-toggle, parameter control, and performance monitoring.
    """
    
    def __init__(self):
        self.filters: List[Filter] = []
        self.context = FilterContext()
        self.enabled = True
        
        # Performance tracking
        self.frame_times = []
        self.last_report = 0
        
        self._init_default_filters()
    
    def _init_default_filters(self):
        """Initialize default filter set."""
        # Priority order: lower = earlier
        self.filters.append(MaxHeadroomFilter())        # 2
        self.filters.append(ColorGradingFilter())       # 5
        self.filters.append(SkinSmoothingFilter())      # 10
        self.filters.append(FaceMorphFilter())          # 15
        self.filters.append(BackgroundRemovalFilter())  # 20
        self.filters.append(AROverlayFilter())          # 30
        
        LOG.info("Initialized %d filters", len(self.filters))
    
    def process(self, frame: np.ndarray, **context_data) -> np.ndarray:
        """
        Process frame through all enabled filters.
        
        Args:
            frame: Input BGR frame
            **context_data: landmarks, face_rect, blendshapes, head_pose, etc.
        
        Returns:
            Processed frame
        """
        if not self.enabled or not frame.size:
            return frame
        
        start_time = time.time()
        
        # Update context
        self.context.landmarks = context_data.get("landmarks")
        self.context.face_rect = context_data.get("face_rect")
        self.context.blendshapes = context_data.get("blendshapes")
        self.context.head_pose = context_data.get("head_pose")
        self.context.frame_id = context_data.get("frame_id", 0)
        self.context.timestamp = time.time()
        
        result = frame
        active_count = 0
        
        # Sort by priority
        sorted_filters = sorted(self.filters, key=lambda f: f.priority)
        
        for filt in sorted_filters:
            if not filt.enabled:
                continue
            
            # Check requirements
            if filt.requires_landmarks and not self.context.has_landmarks():
                continue
            
            try:
                result = filt.process(result, {
                    "landmarks": self.context.landmarks,
                    "face_rect": self.context.face_rect,
                    "blendshapes": self.context.blendshapes,
                    "head_pose": self.context.head_pose,
                    "segmentation_mask": self.context.segmentation_mask,
                })
                active_count += 1
            except Exception as e:
                LOG.warning("Filter %s failed: %s", filt.name, e)
        
        # Performance tracking
        elapsed = (time.time() - start_time) * 1000
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        if time.time() - self.last_report > 10:
            avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
            LOG.info("Filter pipeline: %.2fms avg, %d active", avg_time, active_count)
            self.last_report = time.time()
        
        return result
    
    def get_filter(self, name: str) -> Optional[Filter]:
        """Get filter by name."""
        for filt in self.filters:
            if filt.name.lower() == name.lower():
                return filt
        return None
    
    def toggle_filter(self, name: str) -> bool:
        """Toggle filter by name. Returns new state."""
        filt = self.get_filter(name)
        if filt:
            return filt.toggle()
        LOG.warning("Filter not found: %s", name)
        return False
    
    def enable_filter(self, name: str):
        """Enable a filter."""
        filt = self.get_filter(name)
        if filt:
            filt.enable()
            LOG.info("Enabled filter: %s", name)
    
    def disable_filter(self, name: str):
        """Disable a filter."""
        filt = self.get_filter(name)
        if filt:
            filt.disable()
            LOG.info("Disabled filter: %s", name)
    
    def set_filter_param(self, name: str, key: str, value):
        """Set a filter parameter."""
        filt = self.get_filter(name)
        if filt:
            filt.set_param(key, value)
            LOG.debug("Set %s.%s = %s", name, key, value)
    
    def get_filter_params(self, name: str) -> Optional[Dict]:
        """Get all parameters for a filter."""
        filt = self.get_filter(name)
        if filt:
            return {
                "name": filt.name,
                "enabled": filt.enabled,
                "priority": filt.priority,
                "params": filt.params
            }
        return None
    
    def get_all_status(self) -> List[Dict]:
        """Get status of all filters."""
        return [
            {
                "name": f.name,
                "enabled": f.enabled,
                "priority": f.priority,
            }
            for f in sorted(self.filters, key=lambda x: x.priority)
        ]
    
    def reset(self):
        """Reset all filters to defaults."""
        for filt in self.filters:
            filt.disable()
        LOG.info("All filters reset")
    
    def get_performance_stats(self) -> Dict:
        """Get filter pipeline performance stats."""
        if not self.frame_times:
            return {"avg_ms": 0, "max_ms": 0, "min_ms": 0}
        
        return {
            "avg_ms": round(sum(self.frame_times) / len(self.frame_times), 2),
            "max_ms": round(max(self.frame_times), 2),
            "min_ms": round(min(self.frame_times), 2),
            "frames": len(self.frame_times),
        }