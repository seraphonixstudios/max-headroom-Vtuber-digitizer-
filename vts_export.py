#!/usr/bin/env python3
"""
Max Headroom - VTuber Studio Export
Export to VTube Studio JSON format
Version: 3.0.0
"""
import json
import time
import struct
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

VERSION = "3.0.0"

# Logging setup
try:
    from logging_utils import LOG, get_logger
    LOG = get_logger("VTS")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.VTS")

ARKIT_TO_VTS = {
    "eyeBlink_L": "EyeL",
    "eyeBlink_R": "EyeR",
    "eyeSquint_L": "EyeSquintL",
    "eyeSquint_R": "EyeSquintR",
    "eyeLookUp_L": "EyeLookUpL",
    "eyeLookDown_L": "EyeLookDownL",
    "eyeLookUp_R": "EyeLookUpR",
    "eyeLookDown_R": "EyeLookDownR",
    "browUp_L": "BrowLUp",
    "browUp_R": "BrowRUp",
    "browDown_L": "BrowLDown",
    "browDown_R": "BrowRDown",
    "browOuter_L": "BrowLOuter",
    "browOuter_R": "BrowROuter",
    "jawOpen": "JawOpen",
    "jawForward": "JawForward",
    "jawLeft": "JawLeft",
    "jawRight": "JawRight",
    "mouthSmile_L": "MouthSmileL",
    "mouthSmile_R": "MouthSmileR",
    "mouthDimple_L": "MouthDimpleL",
    "mouthDimple_R": "MouthDimpleR",
    "mouthLeft": "MouthLeft",
    "mouthRight": "MouthRight",
    "mouthPucker": "MouthPucker",
    "mouthFunnel": "MouthFunnel",
    "mouthUpperUp_L": "MouthUpperUpL",
    "mouthUpperUp_R": "MouthUpperUpR",
    "cheekPuff": "CheekPuffL",
    "cheekSquint_L": "CheekSquintL",
    "cheekSquint_R": "CheekSquintR",
    "noseSneer_L": "NoseSneerL",
    "noseSneer_R": "NoseSneerR",
    "tongueOut": "TongueOut",
    "tongueUp": "TongueUp",
    "tongueDown": "TongueDown",
    "tongueLeft": "TongueLeft",
    "tongueRight": "TongueRight",
    "mouthClose": "MouthClose",
}

@dataclass
class VTSRequest:
    """VTube Studio API request."""
    api_name: str
    api_version: str = "1.0"
    request_id: str = ""

class VTSExporter:
    """Export to VTube Studio."""
    
    def __init__(self, port=9001):
        self.port = port
        self.connected = False
        self.socket = None
        
        self.mapping = ARKIT_TO_VTS.copy()
        self.frame_count = 0
    
    def connect(self) -> bool:
        """Connect to VTube Studio."""
        import socket as sock
        
        try:
            self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            self.socket.settimeout(1)
            self.socket.connect(("127.0.0.1", self.port))
            
            self.connected = True
            print(f"[VTS] Connected to port {self.port}")
            return True
            
        except Exception as e:
            print(f"[VTS] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from VTube Studio."""
        if self.socket:
            self.socket.close()
        self.connected = False
    
    def set_blendshape(self, name: str, value: float) -> bool:
        """Set single blendshape."""
        if not self.connected:
            return False
        
        if name in self.mapping:
            vts_name = self.mapping[name]
        else:
            vts_name = name
        
        request = {
            "apiName": "SetLive2DParameter",
            "apiVersion": "1.0",
            "requestId": f"mhr_{self.frame_count}",
            "parameters": [
                {
                    "id": vts_name,
                    "value": max(0.0, min(1.0, value)),
                }
            ]
        }
        
        return self._send_request(request)
    
    def set_blendshapes(self, blendshapes: Dict[str, float], filter_status: Dict = None) -> bool:
        """Set multiple blendshapes with optional filter metadata."""
        if not self.connected:
            return False
        
        params = []
        
        for ark_name, value in blendshapes.items():
            vts_name = self.mapping.get(ark_name, ark_name)
            params.append({
                "id": vts_name,
                "value": max(0.0, min(1.0, value)),
            })
        
        # Add filter metadata as custom parameters if provided
        if filter_status:
            active = filter_status.get("active", [])
            if "Max Headroom" in active:
                params.append({"id": "AndroidMode", "value": 1.0})
            else:
                params.append({"id": "AndroidMode", "value": 0.0})
        
        request = {
            "apiName": "SetLive2DParameter",
            "apiVersion": "1.0",
            "requestId": f"mhr_{self.frame_count}",
            "parameters": params,
        }
        
        result = self._send_request(request)
        self.frame_count += 1
        return result
    
    def _send_request(self, request: Dict) -> bool:
        """Send request to VTS."""
        import socket as sock
        
        if not self.socket:
            return False
        
        try:
            data = json.dumps(request).encode("utf-8")
            length = struct.pack("I", len(data))
            
            self.socket.sendall(length + data)
            
            response_length_bytes = self.socket.recv(4)
            
            if response_length_bytes:
                response_length = struct.unpack("I", response_length_bytes)[0]
                
                if response_length > 0:
                    response = self.socket.recv(response_length)
                    return True
            
            return False
            
        except Exception as e:
            print(f"[VTS] Error: {e}")
            self.connected = False
            return False
    
    def get_parameters(self) -> List[Dict]:
        """Get available parameters from VTS."""
        if not self.connected:
            return []
        
        request = {
            "apiName": "GetCurrentParameterList",
            "apiVersion": "1.0",
            "requestId": "mhr_list",
        }
        
        if self._send_request(request):
            return []
        
        return []

class VTSPipe:
    """Named pipe for VTS communication."""
    
    def __init__(self, pipe_name="VTubeStudioIn"):
        self.pipe_name = pipe_name
        self.handle = None
    
    def open(self) -> bool:
        """Open named pipe."""
        try:
            import win32pipe
            import win32file
            
            self.handle = win32pipe.CreateNamedPipe(
                f"\\\\.\\pipe\\{self.pipe_name}",
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
                1,
                4096,
                4096,
                0,
                None,
            )
            
            return True
            
        except ImportError:
            print("[VTS] win32pipe not available")
            return False
        except Exception as e:
            print(f"[VTS] Pipe error: {e}")
            return False
    
    def write(self, data: Dict) -> bool:
        """Write to pipe."""
        if not self.handle:
            return False
        
        try:
            import win32file
            import win32pipe
            
            msg = json.dumps(data).encode("utf-8")
            win32file.WriteFile(self.handle, msg)
            
            return True
            
        except Exception as e:
            print(f"[VTS] Write error: {e}")
            return False
    
    def close(self) -> None:
        """Close pipe."""
        if self.handle:
            try:
                import win32file
                win32file.CloseHandle(self.handle)
            except:
                pass
        self.handle = None

def export_to_vts(blendshapes: Dict) -> bool:
    """Quick export to VTS."""
    exporter = VTSExporter()
    if exporter.connect():
        result = exporter.set_blendshapes(blendshapes)
        exporter.disconnect()
        return result
    return False

def test():
    """Test VTS export."""
    print(f"Max Headroom VTS Export v{VERSION}")
    
    exporter = VTSExporter()
    
    test_shapes = {
        "eyeBlink_L": 0.0,
        "eyeBlink_R": 0.0,
        "jawOpen": 0.3,
        "mouthSmile_L": 0.5,
        "mouthSmile_R": 0.4,
        "browUp_L": 0.2,
        "browUp_R": 0.2,
    }
    
    if exporter.connect():
        print("Connected to VTS!")
        
        for i in range(3):
            exporter.set_blendshapes(test_shapes)
            print(f"Frame {i}")
        
        exporter.disconnect()
        print("Export complete")
    else:
        print("VTS not running (expected)")

if __name__ == "__main__":
    test()