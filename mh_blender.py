"""
Max Headroom Digitizer - Complete Blender Integration
Full face mesh with ARKit blendshapes, body rig, and holographic effects
Run in Blender Python console
Version: 3.0.0
"""
import bpy
import math
import json
import time
import threading
import asyncio
from typing import Dict, Any, List, Optional

VERSION = "3.0.0"

MH_CONFIG = {
    "ws_host": "localhost",
    "ws_port": 30000,
    "smoothing": 0.8,
    "auto_setup": True,
}

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

class MHBlender:
    """Max Headroom Blender Integration"""
    
    def __init__(self, config: Dict = None):
        self.config = config or MH_CONFIG
        self.running = False
        self.ws_thread = None
        self.current_data = None
        self.smoothed_data = None
        self.head_mesh = None
        self.body_armature = None
    
    def setup_scene(self):
        """Setup complete scene"""
        print("[MH] Setting up Max Headroom scene...")
        
        self.create_head_mesh()
        self.create_body_rig()
        self.setup_camera()
        self.setup_lighting()
        self.create_materials()
        self.setup_post_processing()
        
        print("[MH] Scene setup complete!")
    
    def create_head_mesh(self):
        """Create face mesh with blendshapes"""
        print("[MH] Creating head mesh...")
        
        mesh_name = "MH_Head"
        
        if mesh_name not in bpy.data.objects:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, segments=32, ring_count=16)
            head = bpy.data.objects.active
            head.name = mesh_name
            
            mesh = head.data
            
            bpy.ops.object.mode_set(mode='EDIT')
            for _ in range(3):
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.subdivide(number_cuts=1)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            head.shape_key_add(name="Basis")
            
            for shape_name in ARKIT_BLENDSHAPES:
                head.shape_key_add(name=shape_name)
            
            print(f"[MH] Created head with {len(mesh.shape_keys.key_blocks)} shape keys")
        
        self.head_mesh = bpy.data.objects[mesh_name]
    
    def create_body_rig(self):
        """Create body rig"""
        print("[MH] Creating body rig...")
        
        rig_name = "MH_Body"
        
        if rig_name not in bpy.data.objects:
            bpy.ops.object.armature_add()
            armature = bpy.data.objects.active
            armature.name = rig_name
            
            bones = armature.data.edit_bones
            bones.clear()
            
            root = bones.new("Root")
            root.head = (0, 0, 0)
            root.tail = (0, 0, 0.3)
            
            spine = bones.new("Spine")
            spine.head = (0, 0, 0.3)
            spine.tail = (0, 0, 1.0)
            spine.parent = root
            
            neck = bones.new("Neck")
            neck.head = (0, 0, 1.0)
            neck.tail = (0, 0, 1.2)
            neck.parent = spine
            
            head_bone = bones.new("Head")
            head_bone.head = (0, 0, 1.2)
            head_bone.head = (0, 0, 1.3)
            head_bone.parent = neck
            
            bpy.ops.object.mode_set(mode='OBJECT')
        
        self.body_armature = bpy.data.objects[rig_name]
        print("[MH] Body rig created")
    
    def setup_camera(self):
        """Setup camera"""
        print("[MH] Setting up camera...")
        
        cam_name = "MH_Camera"
        
        if cam_name not in bpy.data.objects:
            bpy.ops.object.camera_add()
            camera = bpy.data.objects.active
            camera.name = cam_name
            camera.location = (0, -1.5, 1.2)
            camera.rotation_euler = (1.2, 0, 0)
            bpy.context.scene.camera = camera
        
        print("[MH] Camera ready")
    
    def setup_lighting(self):
        """Setup three-point lighting"""
        print("[MH] Setting up lighting...")
        
        lights = [
            ("MH_Key", (-1, -1, 2), 200),
            ("MH_Fill", (1, -1, 1.5), 100),
            ("MH_Rim", (0, 1, 1.5), 150),
        ]
        
        for name, loc, energy in lights:
            if name not in bpy.data.objects:
                bpy.ops.object.light_add(type='SUN', location=loc)
                light = bpy.data.objects.active
                light.name = name
                light.data.energy = energy
        
        print("[MH] Lighting ready")
    
    def create_materials(self):
        """Create holographic materials"""
        print("[MH] Creating materials...")
        
        if "MH_Head_MAT" not in bpy.data.materials:
            mat = bpy.data.materials.new(name="MH_Head_MAT")
            mat.use_nodes = True
            
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            
            output = nodes.new('ShaderNodeOutputMaterial')
            output.location = (800, 0)
            
            principled = nodes.new('ShaderNodeBsdfPrincipled')
            principled.location = (400, 0)
            principled.inputs['Base Color'].default_value = (0.02, 0.08, 0.02, 1)
            principled.inputs['Emission Color'].default_value = (0.0, 1.0, 0.0, 1)
            principled.inputs['Emission Strength'].default_value = 0.6
            principled.inputs['Roughness'].default_value = 0.3
            principled.inputs['Metallic'].default_value = 0.2
            
            mix = nodes.new('ShaderNodeMixShader')
            mix.location = (600, 0)
            mix.inputs['Fac'].default_value = 0.15
            
            links.new(principled.outputs['BSDF'], mix.inputs[1])
            links.new(principled.outputs['Emission'], mix.inputs[2])
            links.new(mix.outputs['Shader'], output.inputs['Surface'])
            
            if self.head_mesh:
                self.head_mesh.data.materials.append(mat)
        
        print("[MH] Materials ready")
    
    def setup_post_processing(self):
        """Setup post-processing"""
        print("[MH] Setting up post-processing...")
        
        bpy.context.scene.eevee.use_bloom = True
        bpy.context.scene.eevee.bloom_threshold = 0.7
        bpy.context.scene.eevee.bloom_intensity = 0.4
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        
        print("[MH] Post-processing ready")
    
    def update_blendshapes(self, blendshapes: Dict[str, float]):
        """Update face mesh blendshapes"""
        if not self.head_mesh or not self.head_mesh.data.shape_keys:
            return
        
        key_blocks = self.head_mesh.data.shape_keys.key_blocks
        
        for name, value in blendshapes.items():
            if name in key_blocks:
                key_blocks[name].value = max(0.0, min(1.0, value))
    
    def update_body_pose(self, pose: Dict[str, List[float]]):
        """Update body rig pose"""
        if not self.body_armature:
            return
        
        rotation = pose.get("rotation", [0, 0, 0])
        
        pose_bones = self.body_armature.pose.bones
        
        if "Neck" in pose_bones:
            pose_bones["Neck"].rotation_euler = (
                math.radians(rotation[0] * 0.5),
                math.radians(rotation[1] * 0.3),
                math.radians(rotation[2] * 0.2)
            )
        
        if "Spine" in pose_bones:
            pose_bones["Spine"].rotation_euler = (
                math.radians(rotation[0] * 0.2),
                0,
                math.radians(rotation[2] * 0.3)
            )
    
    def update_position(self, translation: List[float]):
        """Update head position"""
        if not self.head_mesh:
            return
        
        self.head_mesh.location = (
            translation[0],
            translation[2] - 1.0,
            translation[1]
        )
    
    def apply_smoothing(self, blendshapes: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing"""
        if self.smoothed_data is None:
            self.smoothed_data = blendshapes.copy()
            return blendshapes
        
        s = self.config.get("smoothing", 0.8)
        smoothed = {}
        
        for name, value in blendshapes.items():
            prev = self.smoothed_data.get(name, 0)
            smoothed[name] = prev * s + value * (1 - s)
        
        self.smoothed_data = smoothed
        return smoothed
    
    def ws_loop(self):
        """WebSocket receive loop"""
        import websockets
        import asyncio
        
        async def receive():
            import websockets
            
            async def handler(ws, path):
                print("[WS] Tracker connected")
                
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "face_data":
                            self.current_data = data
                            
                            if data.get("blendshapes"):
                                smoothed = self.apply_smoothing(data["blendshapes"])
                                self.update_blendshapes(smoothed)
                            
                            if data.get("head_pose"):
                                self.update_body_pose(data["head_pose"])
                                trans = data["head_pose"].get("translation", [0, 0, 1.0])
                                self.update_position(trans)
                            
                            if self.head_mesh:
                                self.head_mesh.update_tag()
                                
                    except Exception as e:
                        print(f"[WS] Error: {e}")
            
            async with websockets.serve(handler, self.config["ws_host"], self.config["ws_port"]):
                await asyncio.Future()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(receive())
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            loop.close()
    
    def start(self):
        """Start integration"""
        if self.running:
            print("[MH] Already running!")
            return
        
        if self.config.get("auto_setup", True):
            self.setup_scene()
        
        self.running = True
        self.ws_thread = threading.Thread(target=self.ws_loop, daemon=True)
        self.ws_thread.start()
        
        print(f"[MH] Started - listening on {self.config['ws_host']}:{self.config['ws_port']}")
        print("[MH] Commands: mh.get_data(), mh.stop(), mh.stats()")
    
    def stop(self):
        """Stop integration"""
        print("[MH] Stopping...")
        self.running = False
    
    def get_data(self):
        """Get current tracking data"""
        return self.current_data
    
    def stats(self):
        """Print statistics"""
        if self.current_data:
            blends = self.current_data.get("blendshapes", {})
            print(f"Blendshapes: {len(blends)} keys")
            print(f"Keys: {list(blends.keys())[:5]}...")
        else:
            print("No data received yet")

mh = MHBlender()

def start(host="localhost", port=30000, auto_setup=True):
    """Start Max Headroom"""
    mh.config["ws_host"] = host
    mh.config["ws_port"] = port
    mh.config["auto_setup"] = auto_setup
    mh.start()

def stop():
    """Stop Max Headroom"""
    mh.stop()

def get_data():
    """Get current data"""
    return mh.get_data()

def stats():
    """Show stats"""
    mh.stats()

print(f"Max Headroom Blender v{VERSION} loaded")
print("Commands: start(), stop(), get_data(), stats()")