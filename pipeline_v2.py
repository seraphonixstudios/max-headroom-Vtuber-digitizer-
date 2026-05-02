#!/usr/bin/env python3
"""
Pipeline v2 - Professional FrameBus Architecture
Producer-consumer pipeline with clean separation of concerns.
"""
import time
import queue
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque

@dataclass
class FramePacket:
    """A frame traveling through the pipeline with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    face_rect: Optional[tuple] = None
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_pose: Dict[str, Any] = field(default_factory=dict)
    filters_active: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    dropped: bool = False

class FrameBus:
    """
    Central message bus for frame packets.
    Producers push frames; consumers pull them.
    """
    
    def __init__(self, maxsize: int = 2):
        self._q = queue.Queue(maxsize=maxsize)
        self._listeners: List[Callable] = []
        self._lock = threading.Lock()
    
    def publish(self, packet: FramePacket) -> bool:
        """Publish a packet. Drops oldest if full (keep-latest)."""
        try:
            self._q.put_nowait(packet)
            return True
        except queue.Full:
            # Drop oldest, insert newest (keep-latest semantics)
            try:
                self._q.get_nowait()
                self._q.put_nowait(packet)
                return True
            except (queue.Empty, queue.Full):
                return False
    
    def subscribe(self, callback: Callable[[FramePacket], None]):
        """Subscribe to receive every published packet."""
        with self._lock:
            self._listeners.append(callback)
    
    def unsubscribe(self, callback: Callable):
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)
    
    def get(self, timeout: float = 0.1) -> Optional[FramePacket]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def notify_listeners(self, packet: FramePacket):
        with self._lock:
            for cb in self._listeners:
                try:
                    cb(packet)
                except Exception:
                    pass

class PipelineStage:
    """Base class for a pipeline processing stage."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._input_bus: Optional[FrameBus] = None
        self._output_bus: Optional[FrameBus] = None
        self._stats = {"processed": 0, "dropped": 0, "avg_ms": 0.0}
        self._stat_lock = threading.Lock()
    
    def connect(self, input_bus: FrameBus, output_bus: FrameBus):
        self._input_bus = input_bus
        self._output_bus = output_bus
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"Stage-{self.name}", daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _run(self):
        while self._running:
            if self._input_bus is None:
                time.sleep(0.01)
                continue
            packet = self._input_bus.get(timeout=0.05)
            if packet is None:
                continue
            if not self.enabled:
                self._forward(packet)
                continue
            t0 = time.time()
            try:
                result = self.process(packet)
                if result is not None:
                    result.processing_time_ms += (time.time() - t0) * 1000
                    self._update_stats(result.processing_time_ms)
                    self._forward(result)
            except Exception as e:
                print(f"[Pipeline:{self.name}] Error: {e}")
                self._forward(packet)
    
    def _forward(self, packet: FramePacket):
        if self._output_bus:
            self._output_bus.publish(packet)
            self._output_bus.notify_listeners(packet)
    
    def process(self, packet: FramePacket) -> Optional[FramePacket]:
        """Override this in subclasses."""
        return packet
    
    def _update_stats(self, ms: float):
        with self._stat_lock:
            self._stats["processed"] += 1
            n = self._stats["processed"]
            self._stats["avg_ms"] = (self._stats["avg_ms"] * (n - 1) + ms) / n
    
    def get_stats(self) -> Dict:
        with self._stat_lock:
            return dict(self._stats)

class PipelineCoordinatorV2:
    """
    Orchestrates stages: Capture -> Detect -> Filter -> Output
    """
    
    def __init__(self):
        self.capture_bus = FrameBus(maxsize=2)
        self.detect_bus = FrameBus(maxsize=2)
        self.filter_bus = FrameBus(maxsize=2)
        self.display_bus = FrameBus(maxsize=2)
        
        self.stages: List[PipelineStage] = []
        self._running = False
    
    def add_stage(self, stage: PipelineStage, input_bus: FrameBus, output_bus: FrameBus):
        stage.connect(input_bus, output_bus)
        self.stages.append(stage)
    
    def start(self):
        self._running = True
        for stage in self.stages:
            stage.start()
    
    def stop(self):
        self._running = False
        for stage in self.stages:
            stage.stop()
    
    def get_all_stats(self) -> Dict[str, Dict]:
        return {s.name: s.get_stats() for s in self.stages}
    
    def inject_frame(self, frame: np.ndarray, frame_id: int):
        """Inject a raw frame into the pipeline (from camera or test)."""
        packet = FramePacket(
            frame=frame,
            timestamp=time.time(),
            frame_id=frame_id
        )
        self.capture_bus.publish(packet)
