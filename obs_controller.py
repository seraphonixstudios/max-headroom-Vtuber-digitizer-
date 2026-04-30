#!/usr/bin/env python3
"""
Max Headroom - OBS WebSocket Controller
OBS scene switching and filter control
"""
import json
import time
import threading
from typing import Dict, List, Optional, Any

VERSION = "3.0.0"

# Logging setup
try:
    from logging_utils import LOG, get_logger
    LOG = get_logger("OBS")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.OBS")

class OBSController:
    """OBS WebSocket control for Max Headroom."""
    
    def __init__(self, host="localhost", port=4455, password=""):
        self.host = host
        self.port = port
        self.password = password
        self.ws = None
        self.connected = False
        self.running = False
        self.thread = None
        
        self.current_scene = "Live"
        self.scenes: List[str] = []
        
        self.config = {
            "auto_switch": True,
            "scene_on_face": True,
            "filter_on_motion": True,
        }
    
    def connect(self) -> bool:
        """Connect to OBS WebSocket."""
        try:
            import websocket
            
            url = f"ws://{self.host}:{self.port}"
            self.ws = websocket.create_connection(
                url,
                timeout=5,
            )
            
            auth = self._build_auth()
            self.ws.send(json.dumps(auth))
            
            resp = json.loads(self.ws.recv())
            
            if resp.get("status") == "ok":
                self.connected = True
                LOG.info("Connected to %s:%d", self.host, self.port)
                self._get_scenes()
                return True
            
        except Exception as e:
            LOG.error("Connection failed: %s", e)
        
        self.connected = False
        return False
    
    def _build_auth(self) -> Dict:
        """Build authentication message."""
        return {
            "op": 1,
            "d": {
                "rpcVersion": 1,
                "eventSubscriptions": [1, 2],
                "authentication": None,
            }
        }
    
    def _get_scenes(self) -> None:
        """Get available scenes."""
        if not self.connected:
            return
        
        msg = {"op": 6, "d": {"requestType": "GetSceneList"}}
        self.ws.send(json.dumps(msg))
        
        try:
            resp = json.loads(self.ws.recv())
            data = resp.get("d", {}).get("responseData", {})
            self.scenes = data.get("scenes", ["Live"])
        except:
            self.scenes = ["Live"]
    
    def switch_scene(self, scene_name: str) -> bool:
        """Switch to scene."""
        if not self.connected:
            return False
        
        msg = {
            "op": 6,
            "d": {
                "requestType": "SetCurrentScene",
                "requestData": {"sceneName": scene_name},
            }
        }
        
        try:
            self.ws.send(json.dumps(msg))
            resp = json.loads(self.ws.recv())
            
            if resp.get("status") == "ok":
                self.current_scene = scene_name
                print(f"[OBS] Scene: {scene_name}")
                return True
        except:
            pass
        
        return False
    
    def set_filter(self, source: str, filter: str, enabled: bool) -> bool:
        """Toggle filter on source."""
        if not self.connected:
            return False
        
        msg = {
            "op": 6,
            "d": {
                "requestType": "SetSourceFilterEnabled",
                "requestData": {
                    "sourceName": source,
                    "filterName": filter,
                    "filterEnabled": enabled,
                }
            }
        }
        
        try:
            self.ws.send(json.dumps(msg))
            return True
        except:
            return False
    
    def get_source(self, source: str) -> Optional[Dict]:
        """Get source status."""
        if not self.connected:
            return None
        
        msg = {
            "op": 6,
            "d": {
                "requestType": "GetSourceActive",
                "requestData": {"sourceName": source},
            }
        }
        
        try:
            self.ws.send(json.dumps(msg))
            resp = json.loads(self.ws.recv())
            return resp.get("d", {}).get("responseData", {})
        except:
            return None
    
    def start_auto_switch(self, tracker_data: Dict) -> None:
        """Auto-switch based on tracking data."""
        if not self.config.get("auto_switch"):
            return
        
        blends = tracker_data.get("blendshapes", {})
        
        if blends.get("jawOpen", 0) > 0.5:
            self.set_filter("Face", "Glow", True)
        else:
            self.set_filter("Face", "Glow", False)
        
        if blends.get("mouthSmile_L", 0) > 0.5 or blends.get("mouthSmile_R", 0) > 0.5:
            if self.current_scene != "Excited":
                self.switch_scene("Excited")
        elif self.current_scene != "Live":
            self.switch_scene("Live")
    
    def close(self) -> None:
        """Disconnect from OBS."""
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        self.connected = False

class OBSManager:
    """Manages OBS connections for multiple targets."""
    
    def __init__(self):
        self.controllers: Dict[str, OBSController] = {}
        self.default = None
    
    def add(self, name: str, host="localhost", port=4455, password="") -> OBSController:
        """Add OBS controller."""
        ctrl = OBSController(host, port, password)
        self.controllers[name] = ctrl
        
        if not self.default:
            self.default = name
        
        return ctrl
    
    def connect(self, name: str = None) -> bool:
        """Connect to OBS."""
        name = name or self.default
        
        if name in self.controllers:
            return self.controllers[name].connect()
        
        return False
    
    def switch_scene(self, scene: str, name: str = None) -> bool:
        """Switch scene on default OBS."""
        name = name or self.default
        
        if name in self.controllers:
            return self.controllers[name].switch_scene(scene)
        
        return False
    
    def get_controller(self, name: str = None) -> Optional[OBSController]:
        """Get controller."""
        name = name or self.default
        return self.controllers.get(name)

obs_manager = OBSManager()
obs_manager.add("default")

def connect(host="localhost", port=4455, password="") -> bool:
    """Connect to OBS."""
    return obs_manager.connect("default")

def switch_scene(scene: str) -> bool:
    """Switch OBS scene."""
    return obs_manager.switch_scene(scene)

def set_filter(source: str, filter: str, enabled: bool) -> bool:
    """Toggle filter."""
    ctrl = obs_manager.get_controller()
    if ctrl:
        return ctrl.set_filter(source, filter, enabled)
    return False

def is_connected() -> bool:
    """Check connection."""
    ctrl = obs_manager.get_controller()
    return ctrl.connected if ctrl else False

def get_scenes() -> List[str]:
    """Get scenes."""
    ctrl = obs_manager.get_controller()
    return ctrl.scenes if ctrl else []

def test():
    """Test OBS connection."""
    print(f"Max Headroom OBS Controller v{VERSION}")
    
    if connect():
        print("Connected to OBS!")
        print(f"Scenes: {get_scenes()}")
        switch_scene("Live")
        print("Scene switched")
    else:
        print("OBS not connected (expected if OBS not running)")

if __name__ == "__main__":
    test()