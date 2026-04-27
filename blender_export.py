#!/usr/bin/env python3
"""
Max Headroom - Blender Live Export
Direct bone/input export to running Blender instance
Version: 3.0.0
"""
import json
import time
import socket
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

VERSION = "3.0.0"

ARKIT_TO_BLENDER = {
    "jawOpen": "jaw_master",
    "jawLeft": "jaw_left",
    "jawRight": "jaw_right",
    "jawForward": "jaw_fwd",
    "mouthClose": "mouth_close",
    "mouthFunnel": "mouth_funnel",
    "mouthPucker": "mouth_pucker",
    "mouthSmile_L": "mouth_smile_l",
    "mouthSmile_R": "mouth_smile_r",
    "mouthLeft": "mouth_left",
    "mouthRight": "mouth_right",
    "mouthDimple_L": "mouth_corner_l",
    "mouthDimple_R": "mouth_corner_r",
    "mouthUpperUp_L": "mouth_upper_l",
    "mouthUpperUp_R": "mouth_upper_r",
    "eyeBlink_L": "eye_blink_l",
    "eyeBlink_R": "eye_blink_r",
    "eyeSquint_L": "eye_squint_l",
    "eyeSquint_R": "eye_squint_r",
    "eyeLookUp_L": "eye_up_l",
    "eyeLookDown_L": "eye_down_l",
    "eyeLookUp_R": "eye_up_r",
    "eyeLookDown_R": "eye_down_r",
    "browUp_L": "brow_up_l",
    "browUp_R": "brow_up_r",
    "browDown_L": "brow_down_l",
    "browDown_R": "brow_down_r",
    "cheekPuff": "cheek_puff",
    "cheekSquint_L": "cheek_l",
    "cheekSquint_R": "cheek_r",
    "noseSneer_L": "nose_sneer_l",
    "noseSneer_R": "nose_sneer_r",
}

@dataclass
class BlenderTarget:
    """Blender target shape/bone."""
    name: str
    value: float
    bone_mode: bool = False

class BlenderExporter:
    """Export blendshapes to Blender."""
    
    def __init__(self, host="localhost", port=30001, use_tcp=False):
        self.host = host
        self.port = port
        self.use_tcp = use_tcp
        self.socket = None
        self.connected = False
        self.running = False
        self.thread = None
        
        self.mapping = ARKIT_TO_BLENDER.copy()
        self.smoothing = 0.3
        
        self.prev_values = {}
        self.frame_count = 0
    
    def connect(self) -> bool:
        """Connect to Blender."""
        try:
            if self.use_tcp:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
            else:
                import websocket
                self.socket = websocket.create_connection(
                    f"ws://{self.host}:{self.port}",
                    timeout=1,
                )
            
            self.connected = True
            print(f"[Blender] Connected to {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"[Blender] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Blender."""
        if self.socket:
            self.socket.close()
        self.connected = False
        self.socket = None
    
    def export(self, blendshapes: Dict[str, float], pose: Dict = None) -> bool:
        """Export blendshapes to Blender."""
        if not self.connected:
            return False
        
        exports = self._map_blendshapes(blendshapes)
        
        if pose:
            self._add_pose(exports, pose)
        
        payload = {
            "type": "blendshapes",
            "version": VERSION,
            "frame": self.frame_count,
            "targets": exports,
        }
        
        try:
            msg = json.dumps(payload)
            
            if self.use_tcp:
                self.socket.sendall((msg + "\n").encode())
            else:
                self.socket.send(msg)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"[Blender] Export error: {e}")
            self.connected = False
            return False
    
    def _map_blendshapes(self, blendshapes: Dict) -> Dict[str, float]:
        """Map ARKit names to Blender shape names."""
        mapped = {}
        
        for ark, value in blendshapes.items():
            blender_name = self.mapping.get(ark, ark)
            
            prev = self.prev_values.get(blender_name, 0.0)
            smoothed = prev * self.smoothing + value * (1 - self.smoothing)
            
            mapped[blender_name] = smoothed
            self.prev_values[blender_name] = smoothed
        
        return mapped
    
    def _add_pose(self, exports: Dict, pose: Dict) -> None:
        """Add head pose to exports."""
        rot = pose.get("rotation", [0, 0, 0])
        trans = pose.get("translation", [0, 0, 1.5])
        
        exports["head_rot_x"] = rot[0]
        exports["head_rot_y"] = rot[1]
        exports["head_rot_z"] = rot[2]
        exports["head_pos_x"] = trans[0]
        exports["head_pos_y"] = trans[1]
        exports["head_pos_z"] = trans[2]
    
    def start_live(self, get_data_fn=None) -> None:
        """Start live export loop."""
        self.running = True
        
        def loop():
            while self.running:
                if get_data_fn:
                    data = get_data_fn()
                    if data:
                        self.export(
                            data.get("blendshapes", {}),
                            data.get("head_pose", {}),
                        )
                time.sleep(1 / 60)
        
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
    
    def stop_live(self) -> None:
        """Stop live export."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

class BlenderServer:
    """TCP server for Blender to connect to."""
    
    def __init__(self, port=30001):
        self.port = port
        self.socket = None
        self.running = False
        self.clients = []
    
    def start(self) -> bool:
        """Start TCP server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("0.0.0.0", self.port))
            self.socket.listen(1)
            self.socket.settimeout(1)
            
            self.running = True
            print(f"[Blender Server] Listening on port {self.port}")
            return True
            
        except Exception as e:
            print(f"[Blender Server] Error: {e}")
            return False
    
    def accept_connection(self) -> bool:
        """Accept incoming connection."""
        try:
            conn, addr = self.socket.accept()
            self.clients.append(conn)
            print(f"[Blender Server] Connected: {addr}")
            return True
        except:
            return False
    
    def send(self, data: str) -> None:
        """Send data to all clients."""
        for client in self.clients[:]:
            try:
                client.sendall((data + "\n").encode())
            except:
                self.clients.remove(client)
    
    def stop(self) -> None:
        """Stop server."""
        self.running = False
        for client in self.clients:
            client.close()
        if self.socket:
            self.socket.close()

def export_blendshapes(blendshapes: Dict, pose: Dict = None) -> bool:
    """Quick export function."""
    exporter = BlenderExporter()
    if exporter.connect():
        result = exporter.export(blendshapes, pose)
        exporter.disconnect()
        return result
    return False

def test():
    """Test Blender export."""
    print(f"Max Headroom Blender Export v{VERSION}")
    
    exporter = BlenderExporter()
    
    test_shapes = {
        "jawOpen": 0.3,
        "mouthSmile_L": 0.5,
        "mouthSmile_R": 0.4,
        "eyeBlink_L": 0.0,
        "eyeBlink_R": 0.0,
        "browUp_L": 0.2,
        "browUp_R": 0.2,
        "cheekPuff": 0.3,
    }
    
    test_pose = {
        "rotation": [5.0, 10.0, 0.0],
        "translation": [0.1, 0.0, 1.5],
    }
    
    if exporter.connect():
        print("Connected!")
        for i in range(3):
            exporter.export(test_shapes, test_pose)
            print(f"Frame {i} sent")
            time.sleep(0.1)
        
        exporter.disconnect()
        print("Export complete")
    else:
        print("No Blender connection (expected if Blender not running)")

if __name__ == "__main__":
    test()