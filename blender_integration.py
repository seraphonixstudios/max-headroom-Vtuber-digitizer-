"""
Max Headroom Digitizer - Blender Integration
Real-time face tracking receiver with CRT/Glitch shader for Max Headroom aesthetic
"""
import bpy
import asyncio
import threading
import time
import json
import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

VERSION = "2.0.0"

@dataclass
class FaceTrackingData:
    blendshapes: Dict[str, float]
    head_pose: Dict[str, List[float]]
    landmarks: List[Dict[str, float]]
    timestamp: float

class MaxHeadroomBlender:
    """Blender integration for Max Headroom Digitizer."""
    
    ARKIT_BLENDSHAPES = [
        "browDown_L", "browDown_R", "browUp_L", "browUp_R",
        "cheekPuff", "cheekSquint_L", "cheekSquint_R",
        "eyeBlink_L", "eyeBlink_R",
        "eyeLookDown_L", "eyeLookDown_R",
        "eyeLookUp_L", "eyeLookUp_R",
        "eyeSquint_L", "eyeSquint_R",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimple_L", "mouthDimple_R",
        "mouthFunnel", "mouthLeft", "mouthPucker",
        "mouthRight", "mouthSmile_L", "mouthSmile_R",
        "mouthUpperUp_L", "mouthUpperUp_R",
        "noseSneer_L", "noseSneer_R",
    ]
    
    def __init__(self):
        self.running = False
        self.ws = None
        self.data: Optional[FaceTrackingData] = None
        self.smoothed_data: Optional[FaceTrackingData] = None
        self.smoothing = 0.7
        
        self.head_mesh: Optional[bpy.types.Object] = None
        self.camera: Optional[bpy.types.Object] = None
        
        self._ws_thread = None
        self._update_thread = None
        
    def setup_scene(self):
        """Setup scene with head mesh and camera."""
        print("[MH] Setting up scene...")
        
        self._create_head_mesh()
        self._setup_camera()
        self._setup_lighting()
        self._create_crt_material()
        
        print("[MH] Scene setup complete")
    
    def _create_head_mesh(self):
        """Create procedural head mesh with shape keys."""
        mesh_name = "Head_Mesh"
        
        if mesh_name in bpy.data.objects:
            self.head_mesh = bpy.data.objects[mesh_name]
            return
        
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, segments=32, ring_count=16)
        self.head_mesh = bpy.data.objects.active
        self.head_mesh.name = mesh_name
        
        mesh = self.head_mesh.data
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=2)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        basis = self.head_mesh.shape_key_add(name="Basis")
        
        for shape_name in self.ARKIT_BLENDSHAPAPES:
            self.head_mesh.shape_key_add(name=shape_name)
        
        print(f"[MH] Created head mesh with {len(mesh.shape_keys.key_blocks)} shape keys")
    
    def _setup_camera(self):
        """Setup camera looking at head mesh."""
        camera_name = "Head_Cam"
        
        if camera_name in bpy.data.objects:
            self.camera = bpy.data.objects[camera_name]
            return
        
        bpy.ops.object.camera_add()
        self.camera = bpy.data.objects.active
        self.camera.name = camera_name
        self.camera.location = (0, 0, 1.5)
        self.camera.rotation_euler = (1.5708, 0, 0)
        
        bpy.context.scene.camera = self.camera
        
        print("[MH] Camera setup complete")
    
    def _setup_lighting(self):
        """Setup three-point lighting."""
        lights = [
            ("Key_Light", (2, 2, 3), 200),
            ("Fill_Light", (-2, 1, 2), 100),
            ("Rim_Light", (0, -2, 2), 150),
        ]
        
        for name, loc, energy in lights:
            if name in bpy.data.objects:
                continue
            
            bpy.ops.object.light_add(type='SUN', location=loc)
            light = bpy.data.objects.active
            light.name = name
            light.data.energy = energy
        
        print("[MH] Lighting setup complete")
    
    def _create_crt_material(self):
        """Create Max Headroom CRT glitch material."""
        mat_name = "MaxHeadroom_CRT"
        
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (0, 0)
        
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (400, 0)
        principled.inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1)
        principled.inputs['Roughness'].default_value = 0.6
        
        emission = nodes.new('ShaderNodeEmission')
        emission.location = (400, 200)
        emission.inputs['Color'].default_value = (0.0, 1.0, 0.0, 1)
        emission.inputs['Strength'].default_value = 0.3
        
        mix_glitch = nodes.new('ShaderNodeMixShader')
        mix_glitch.location = (200, 0)
        mix_glitch.inputs['Fac'].default_value = 0.15
        
        mix_rgb = nodes.new('ShaderNodeMixShader')
        mix_rgb.location = (200, 200)
        mix_rgb.inputs['Fac'].default_value = 0.1
        
        noise = nodes.new('ShaderNodeTexNoise')
        noise.location = (600, 0)
        noise.inputs['Scale'].default_value = 50.0
        
        links.new(principled.outputs['BSDF'], mix_glitch.inputs[1])
        links.new(emission.outputs['Emission'], mix_glitch.inputs[2])
        links.new(mix_glitch.outputs['Shader'], output.inputs['Surface'])
        links.new(noise.outputs['Fac'], mix_rgb.inputs[0])
        links.new(emission.outputs['Emission'], mix_rgb.inputs[1])
        
        if self.head_mesh:
            self.head_mesh.data.materials.append(mat)
        
        print("[MH] CRT material created")
    
    def _update_blendshapes(self, blendshapes: Dict[str, float]):
        """Update head mesh shape keys from blendshapes."""
        if not self.head_mesh or not self.head_mesh.data.shape_keys:
            return
        
        key_blocks = self.head_mesh.data.shape_keys.key_blocks
        
        for name, value in blendshapes.items():
            if name in key_blocks:
                key_blocks[name].value = value
    
    def _update_head_pose(self, pose: Dict[str, List[float]]):
        """Update head mesh position and rotation."""
        if not self.head_mesh:
            return
        
        translation = pose.get("translation", [0, 0, 1.5])
        rotation = pose.get("rotation", [0, 0, 0])
        
        self.head_mesh.location = tuple(translation)
        
        rx = math.radians(rotation[0])
        ry = math.radians(rotation[1])
        rz = math.radians(rotation[2])
        self.head_mesh.rotation_euler = (rx, ry, rz)
    
    def _update_from_data(self, data: FaceTrackingData):
        """Update Blender from tracking data with smoothing."""
        if self.smoothed_data is None:
            self.smoothed_data = data
            return
        
        s = self.smoothing
        t = 1 - s
        
        smoothed_blends = {}
        for name, value in data.blendshapes.items():
            prev = self.smoothed_data.blendshapes.get(name, 0)
            smoothed_blends[name] = prev * s + value * t
        
        smoothed_pose = {
            "translation": [
                self.smoothed_data.head_pose.get("translation", [0, 0, 1.5])[0] * s + data.head_pose.get("translation", [1.5, 0, 0])[0] * t,
                self.smoothed_data.head_pose.get("translation", [0, 0, 1.5])[1] * s + data.head_pose.get("translation", [1.5, 0, 0])[1] * t,
                self.smoothed_data.head_pose.get("translation", [0, 0, 1.5])[2] * s + data.head_pose.get("translation", [1.5, 0, 0])[2] * t,
            ],
            "rotation": [
                self.smoothed_data.head_pose.get("rotation", [0, 0, 0])[0] * s + data.head_pose.get("rotation", [0, 0, 0])[0] * t,
                self.smoothed_data.head_pose.get("rotation", [0, 0, 0])[1] * s + data.head_pose.get("rotation", [0, 0, 0])[1] * t,
                self.smoothed_data.head_pose.get("rotation", [0, 0, 0])[2] * s + data.head_pose.get("rotation", [0, 0, 0])[2] * t,
            ]
        }
        
        self.smoothed_data = FaceTrackingData(
            blendshapes=smoothed_blends,
            head_pose=smoothed_pose,
            landmarks=data.landmarks,
            timestamp=data.timestamp,
        )
        
        self._update_blendshapes(smoothed_blends)
        self._update_head_pose(smoothed_pose)
    
    def _ws_loop(self, host: str, port: int):
        """WebSocket receive loop."""
        import websockets
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def receive():
            import websockets
            
            async def handler(ws, path):
                print("[WS] Tracker connected")
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "face_data":
                            self.data = FaceTrackingData(
                                blendshapes=data.get("blendshapes", {}),
                                head_pose=data.get("head_pose", {}),
                                landmarks=data.get("landmarks", []),
                                timestamp=data.get("timestamp", 0),
                            )
                    except:
                        pass
            
            async with websockets.serve(handler, host, port, ping_interval=None):
                print(f"[WS] Listening on {host}:{port}")
                await asyncio.Future()
        
        try:
            loop.run_until_complete(receive())
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            loop.close()
    
    def _update_loop(self):
        """Main update loop running in Blender context."""
        while self.running:
            if self.data is not None:
                self._update_from_data(self.data)
            time.sleep(1/60)
    
    def start(self, host: str = "localhost", port: int = 30000):
        """Start the Blender integration."""
        if self.running:
            print("[MH] Already running")
            return
        
        self.running = True
        
        self.setup_scene()
        
        print("[MH] Starting WebSocket receiver...")
        self._ws_thread = threading.Thread(target=self._ws_loop, args=(host, port), daemon=True)
        self._ws_thread.start()
        
        print("[MH] Max Headroom Blender integration started")
        print("[MH] Commands: mh.get_data(), mh.stop(), mh.stats()")
    
    def stop(self):
        """Stop the integration."""
        print("[MH] Stopping...")
        self.running = False
        if self._ws_thread:
            self._ws_thread.join(timeout=2)
        print("[MH] Stopped")
    
    def get_data(self) -> Optional[Dict]:
        """Get current tracking data."""
        if self.data:
            return {
                "blendshapes": self.data.blendshapes,
                "head_pose": self.data.head_pose,
                "timestamp": self.data.timestamp,
            }
        return None
    
    def stats(self):
        """Print statistics."""
        if self.data:
            print(f"Timestamp: {self.data.timestamp}")
            print(f"Blendshapes: {list(self.data.blendshapes.keys())}")
            print(f"Head pose: {self.data.head_pose}")
        else:
            print("No data received yet")

mh = MaxHeadroomBlender()

def start(host: str = "localhost", port: int = 30000):
    """Start Max Headroom Blender integration."""
    mh.start(host, port)

def stop():
    """Stop Max Headroom Blender integration."""
    mh.stop()

def get_data():
    """Get current tracking data."""
    return mh.get_data()

def stats():
    """Show statistics."""
    mh.stats()

print(f"Max Headroom Blender v{VERSION} loaded")
print("Commands: start(), stop(), get_data(), stats()")