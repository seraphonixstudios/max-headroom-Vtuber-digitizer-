#!/usr/bin/env python3
"""
Max Headroom - Recording & Playback System
Save and replay tracking data for debugging
"""
import json
import time
import os
import gzip
import pickle
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

VERSION = "3.0.0"

# Logging setup
try:
    from logging_utils import LOG, get_logger
    LOG = get_logger("Recorder")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.Recorder")

@dataclass
class FrameRecord:
    """Single frame of tracking data."""
    timestamp: float
    frame_id: int
    blendshapes: Dict[str, float]
    head_pose: Dict[str, List[float]]
    landmarks: List[Dict[str, float]]
    fps: int

class RecordingSession:
    """Recording session with playback."""
    
    def __init__(self, name: str = None):
        self.name = name or f"rec_{int(time.time())}"
        self.frames: List[FrameRecord] = []
        self.start_time = 0.0
        self.duration = 0.0
        self.fps = 0
        self.frame_count = 0
    
    def add_frame(self, data: Dict) -> None:
        """Add frame to recording."""
        record = FrameRecord(
            timestamp=data.get("timestamp", time.time()),
            frame_id=data.get("frame_id", self.frame_count),
            blendshapes=data.get("blendshapes", {}),
            head_pose=data.get("head_pose", {}),
            landmarks=data.get("landmarks", []),
            fps=data.get("fps", 0),
        )
        self.frames.append(record)
        self.frame_count += 1
        
        if self.start_time == 0:
            self.start_time = record.timestamp
    
    def save(self, path: str = None) -> str:
        """Save recording to file."""
        path = path or f"{self.name}.mhr"
        
        self.duration = self.frames[-1].timestamp - self.start_time if self.frames else 0
        
        data = {
            "version": VERSION,
            "name": self.name,
            "start_time": self.start_time,
            "duration": self.duration,
            "frame_count": len(self.frames),
            "frames": [
                {
                    "t": f.timestamp,
                    "id": f.frame_id,
                    "bs": f.blendshapes,
                    "hp": f.head_pose,
                    "lm": f.landmarks,
                    "fps": f.fps,
                }
                for f in self.frames
            ]
        }
        
        with open(path, "w") as f:
            json.dump(data, f)
        
        return path
    
    @classmethod
    def load(cls, path: str) -> "RecordingSession":
        """Load recording from file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        session = cls(data.get("name", "loaded"))
        session.start_time = data.get("start_time", 0)
        session.duration = data.get("duration", 0)
        
        for frame_data in data.get("frames", []):
            record = FrameRecord(
                timestamp=frame_data["t"],
                frame_id=frame_data["id"],
                blendshapes=frame_data["bs"],
                head_pose=frame_data["hp"],
                landmarks=frame_data["lm"],
                fps=frame_data["fps"],
            )
            session.frames.append(record)
        
        session.frame_count = len(session.frames)
        return session

class Recorder:
    """Recording manager."""
    
    def __init__(self, max_frames: int = 10000):
        self.max_frames = max_frames
        self.current: Optional[RecordingSession] = None
        self.is_recording = False
        self.is_playing = False
        self.playback_index = 0
    
    def start(self, name: str = None) -> None:
        """Start recording."""
        self.current = RecordingSession(name)
        self.is_recording = True
        LOG.info("Started recording: %s", self.current.name)
    
    def stop(self) -> Optional[RecordingSession]:
        """Stop recording."""
        self.is_recording = False
        
        if self.current:
            LOG.info("Stopped recording: %d frames", self.current.frame_count)
        
        return self.current
    
    def add(self, data: Dict) -> None:
        """Add frame if recording."""
        if self.is_recording and self.current:
            self.current.add_frame(data)
            
            if len(self.current.frames) >= self.max_frames:
                self.stop()
    
    def save(self, path: str = None) -> Optional[str]:
        """Save current recording."""
        if self.current:
            return self.current.save(path)
        return None
    
    def load(self, path: str) -> bool:
        """Load recording for playback."""
        try:
            self.current = RecordingSession.load(path)
            self.playback_index = 0
            print(f"[Record] Loaded: {self.current.name} ({self.current.frame_count} frames)")
            return True
        except Exception as e:
            print(f"[Record] Load error: {e}")
            return False
    
    def play(self) -> Optional[Dict]:
        """Get next playback frame."""
        if not self.current or self.playback_index >= len(self.current.frames):
            return None
        
        frame = self.current.frames[self.playback_index]
        self.playback_index += 1
        
        return {
            "type": "face_data",
            "version": VERSION,
            "blendshapes": frame.blendshapes,
            "head_pose": frame.head_pose,
            "landmarks": frame.landmarks,
            "timestamp": frame.timestamp,
            "fps": frame.fps,
            "frame_id": frame.frame_id,
            "playback": True,
        }
    
    def seek(self, index: int) -> None:
        """Seek to frame."""
        if self.current:
            self.playback_index = max(0, min(index, len(self.current.frames) - 1))
    
    def rewind(self) -> None:
        """Rewind to start."""
        self.playback_index = 0
    
    def get_status(self) -> Dict:
        """Get recorder status."""
        return {
            "recording": self.is_recording,
            "playing": self.is_playing,
            "total_frames": len(self.current.frames) if self.current else 0,
            "current_frame": self.playback_index,
            "name": self.current.name if self.current else None,
        }

class LiveRecorder:
    """Live recording from WebSocket stream."""
    
    def __init__(self, recorder: Recorder = None):
        self.recorder = recorder or Recorder()
        self.buffer = deque(maxlen=60)
        self.connected = False
    
    def connect(self, host="localhost", port=30000) -> bool:
        """Connect to WebSocket server."""
        import websocket
        
        try:
            self.ws = websocket.create_connection(
                f"ws://{host}:{port}",
                timeout=1,
            )
            self.connected = True
            print(f"[Live] Connected to {host}:{port}")
            return True
        except:
            self.connected = False
            return False
    
    def record(self, duration: float = 60.0) -> Optional[RecordingSession]:
        """Record live stream for duration."""
        if not self.connected:
            return None
        
        self.recorder.start()
        
        start = time.time()
        
        while time.time() - start < duration and self.recorder.is_recording:
            try:
                msg = self.ws.recv()
                data = json.loads(msg)
                self.recorder.add(data)
            except:
                break
        
        return self.recorder.stop()
    
    def close(self) -> None:
        """Close connection."""
        if self.connected and hasattr(self, "ws"):
            self.ws.close()
        
        self.connected = False

recorder = Recorder()

def start_recording(name: str = None) -> None:
    """Start recording."""
    recorder.start(name)

def stop_recording() -> Optional[RecordingSession]:
    """Stop recording."""
    return recorder.stop()

def save_recording(path: str = None) -> Optional[str]:
    """Save recording."""
    return recorder.save(path)

def load_recording(path: str) -> bool:
    """Load recording."""
    return recorder.load(path)

def playback() -> Optional[Dict]:
    """Get playback frame."""
    return recorder.play()

def get_status() -> Dict:
    """Get status."""
    return recorder.get_status()

def test():
    """Test recorder."""
    print(f"Max Headroom Recording v{VERSION}")
    
    start_recording("test_rec")
    
    for i in range(10):
        data = {
            "blendshapes": {"jawOpen": i * 0.1, "mouthSmile_L": 0.5},
            "head_pose": {"rotation": [0, 0, 0], "translation": [0, 0, 1.5]},
            "landmarks": [],
            "timestamp": time.time(),
            "frame_id": i,
            "fps": 30,
        }
        recorder.add(data)
    
    session = stop_recording()
    print(f"Recorded {session.frame_count} frames")
    
    path = save_recording("test.mhr")
    print(f"Saved to {path}")

if __name__ == "__main__":
    test()