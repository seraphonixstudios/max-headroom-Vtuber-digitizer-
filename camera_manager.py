#!/usr/bin/env python3
"""
Camera Manager - Professional camera handling with discovery and hot-swap.
"""
import cv2
import time
import threading
import concurrent.futures
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass

@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int
    fps: float
    available: bool

class CameraManager:
    """
    Professional camera manager with discovery, non-blocking init,
    and hot-swappable camera indices.
    """
    
    def __init__(self, preferred_index: int = 0, timeout: float = 3.0):
        self.preferred_index = preferred_index
        self.timeout = timeout
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_info: Optional[CameraInfo] = None
        self._lock = threading.Lock()
        self._running = False
        self._on_frame: Optional[Callable] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._latest_frame = None
    
    @staticmethod
    def discover(max_index: int = 5) -> List[CameraInfo]:
        """Discover all available cameras."""
        cameras = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if __import__('sys').platform == 'win32' else cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cameras.append(CameraInfo(
                        index=i, name=f"Camera {i}",
                        width=w, height=h, fps=fps, available=True
                    ))
                cap.release()
        return cameras
    
    def open(self, index: Optional[int] = None, width: int = 640, height: int = 480) -> bool:
        """Open camera with timeout. Returns True on success."""
        index = index if index is not None else self.preferred_index
        
        # Close existing
        self.close()
        
        def _try_open():
            backend = cv2.CAP_DSHOW if __import__('sys').platform == 'win32' else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ret, frame = cap.read()
                if ret:
                    return cap
                cap.release()
            return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_try_open)
            try:
                cap = future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                cap = None
        
        if cap is not None:
            with self._lock:
                self.cap = cap
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.current_info = CameraInfo(
                    index=index, name=f"Camera {index}",
                    width=w, height=h, fps=fps, available=True
                )
            return True
        return False
    
    def read(self) -> Optional:
        """Read a frame. Thread-safe."""
        with self._lock:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame
        return None
    
    def close(self):
        """Release camera."""
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.current_info = None
    
    def is_opened(self) -> bool:
        with self._lock:
            return self.cap is not None and self.cap.isOpened()
    
    def start_capture(self, callback: Callable):
        """Start background capture thread."""
        self._on_frame = callback
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
    
    def stop_capture(self):
        """Stop background capture."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
    
    def _capture_loop(self):
        while self._running:
            frame = self.read()
            if frame is not None and self._on_frame:
                self._on_frame(frame)
            else:
                time.sleep(0.005)
    
    def __del__(self):
        self.close()
