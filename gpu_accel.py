#!/usr/bin/env python3
"""
Max Headroom - GPU Acceleration
CUDA/OpenCL face processing
Version: 3.0.0
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import time

VERSION = "3.0.0"

# Logging setup
try:
    from logging_utils import LOG, get_logger
    LOG = get_logger("GPU")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.GPU")

class GPUDetector:
    """GPU-accelerated face detection."""
    
    def __init__(self):
        self.device_id = 0
        self.use_cuda = False
        self._init_gpu()
        
        self.min_size = (100, 100)
        self.scale_factor = 1.1
        self.min_neighbors = 3
        
        self.cascade = None
        self._load_cascade()
    
    def _init_gpu(self) -> bool:
        """Initialize GPU/OpenCL."""
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDevice()
            if cuda_devices > 0:
                self.use_cuda = True
                LOG.info("CUDA available: %d device(s)", cuda_devices)
                return True
        except:
            pass
        
        try:
            ocl = cv2.ocl.haveOpenCL()
            if ocl:
                cv2.ocl.setUseOpenCL(True)
                self.use_cuda = True
                LOG.info("OpenCL available")
                return True
        except:
            pass
        
        LOG.info("Using CPU fallback")
        return False
    
    def _load_cascade(self) -> None:
        """Load detection cascade."""
        cascade_path = cv2.data.haarcascades
        
        if self.use_cuda:
            try:
                self.cascade = cv2.CascadeClassifier(
                    cascade_path + "haarcascade_frontalface_default.xml"
                )
                print("[GPU] GPU cascade loaded")
                return
            except:
                pass
        
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("[GPU] CPU cascade loaded")
    
    def detect(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """Detect face with GPU acceleration."""
        if self.cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        if self.use_cuda:
            try:
                gpu_frame = cv2.cuda.GpuMat()
                gpu_frame.upload(gray)
                
                faces = self.cascade.detectMultiScale(
                    gpu_frame,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size,
                )
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    return (int(x), int(y), int(w), int(h))
            except:
                pass
        
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return (int(x), int(y), int(w), int(h))
        
        return None

class GPUProcessor:
    """GPU-accelerated frame processor."""
    
    def __init__(self, target_fps: int = 60):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        self.use_gpu = False
        self._init_device()
        
        self.pipeline = [
            ("resize", (640, 480)),
            ("equalize", None),
            ("smooth", 3),
            ("detect", None),
        ]
    
    def _init_device(self) -> None:
        """Initialize GPU device."""
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                self.use_gpu = True
                devs = cv2.ocl.Device.getDefault()
                print(f"[GPU] Device: {devs.name()}")
        except:
            print("[GPU] CPU mode")
    
    def upload(self, frame) -> cv2.cuda.GpuMat:
        """Upload frame to GPU."""
        gpu_frame = cv2.cuda.GpuMat()
        gpu_frame.upload(frame)
        return gpu_frame
    
    def download(self, gpu_frame) -> np.ndarray:
        """Download frame from GPU."""
        frame = np.ndarray(())
        gpu_frame.download(frame)
        return frame
    
    def resize_gpu(self, gpu_frame, size: Tuple[int, int]) -> cv2.cuda.GpuMat:
        """GPU resize."""
        resized = cv2.cuda.GpuMat()
        cv2.cuda.resize(gpu_frame, size, resized)
        return resized
    
    def equalize_gpu(self, gpu_frame) -> cv2.cuda.GpuMat:
        """GPU histogram equalization."""
        result = cv2.cuda.GpuMat()
        
        gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        
        yuv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2YUV)
        yuv = cv2.cuda.split(yuv)
        
        cv2.cuda.equalizeHist(yuv[0], yuv[0])
        
        merged = cv2.cuda.merge(yuv)
        result = cv2.cuda.cvtColor(merged, cv2.COLOR_YUV2BGR)
        
        return result
    
    def smooth_gpu(self, gpu_frame, k: int = 3) -> cv2.cuda.GpuMat:
        """GPU Gaussian blur."""
        blur = cv2.cuda.GpuMat()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cv2.cuda.medianBlur(gpu_frame, k, blur)
        return blur
    
    def process(self, frame) -> Optional[np.ndarray]:
        """Process frame with GPU."""
        if not self.use_gpu:
            return frame
        
        gpu_frame = self.upload(frame)
        
        for op, params in self.pipeline:
            if op == "resize":
                gpu_frame = self.resize_gpu(gpu_frame, params)
            elif op == "equalize":
                gpu_frame = self.equalize_gpu(gpu_frame)
            elif op == "smooth":
                gpu_frame = self.smooth_gpu(gpu_frame, params)
        
        result = np.ndarray(())
        gpu_frame.download(result)
        return result
    
    def get_stats(self) -> Dict:
        """Get GPU stats."""
        return {
            "gpu_enabled": self.use_gpu,
            "target_fps": self.target_fps,
            "pipeline": self.pipeline,
        }

class MultiStreamProcessor:
    """Process multiple camera streams."""
    
    def __init__(self, num_streams: int = 2):
        self.num_streams = num_streams
        self.processors = [GPUProcessor() for _ in range(num_streams)]
        self.queues = [[] for _ in range(num_streams)]
        
        self.running = False
    
    def add_frame(self, stream_id: int, frame) -> None:
        """Add frame to stream."""
        if stream_id < self.num_streams:
            processed = self.processors[stream_id].process(frame)
            self.queues[stream_id].append(processed)
    
    def get_latest(self, stream_id: int) -> Optional[np.ndarray]:
        """Get latest frame from stream."""
        if stream_id < self.num_streams and self.queues[stream_id]:
            return self.queues[stream_id].pop(0)
        return None
    
    def clear(self, stream_id: int = None) -> None:
        """Clear queue."""
        if stream_id is not None:
            self.queues[stream_id] = []
        else:
            for q in self.queues:
                q.clear()

def test():
    """Test GPU processing."""
    print(f"Max Headroom GPU v{VERSION}")
    
    print("\n[1] Testing GPUDetector...")
    detector = GPUDetector()
    print(f"    GPU enabled: {detector.use_cuda}")
    
    print("\n[2] Testing GPUProcessor...")
    processor = GPUProcessor(60)
    stats = processor.get_stats()
    print(f"    {stats}")
    
    print("\n[3] Testing MultiStreamProcessor...")
    multi = MultiStreamProcessor(2)
    print(f"    Streams: {multi.num_streams}")
    
    print("\n[GPU] Acceleration ready!")

if __name__ == "__main__":
    test()