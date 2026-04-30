"""
Max Headroom - Filter System Package
SOTA filter pipeline for real-time face effects
"""
from .base import Filter, FilterMode
from .manager import FilterManager
from .skin_smoothing import SkinSmoothingFilter
from .background import BackgroundFilter
from .ar_overlay import AROverlayFilter
from .face_morph import FaceMorphFilter
from .color_grading import ColorGradingFilter
from .max_headroom_filter import MaxHeadroomFilter

__all__ = [
    'Filter', 'FilterMode', 'FilterManager',
    'SkinSmoothingFilter', 'BackgroundFilter',
    'AROverlayFilter', 'FaceMorphFilter', 'ColorGradingFilter',
    'MaxHeadroomFilter'
]