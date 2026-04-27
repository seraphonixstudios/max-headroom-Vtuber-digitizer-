"""
Max Headroom Digitizer - Blender Integration
Complete VRM-ready face mesh with ARKit blendshapes and body tracking
"""
import bpy
import math
import json
import time
import threading
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

VERSION = "3.0.0"

@dataclass
class MHConfig:
    ws_host = "localhost"
    ws_port = 30000
    smoothing = 0.8
    auto_setup = True

class MHBlender:
    """Max Headroom Blender integration."""
    
    # ARKit 52 blendshapes
    ARKIT_BLENDSHAPES = [
        "browDown_L", "browDown_R", "browUp_L", "browUp_R",
        "cheekPuff", "cheekSquint_L", "cheekSquint_R",
        "eyeBlink_L", "eyeBlink_R", "eyeDilation", "eyeConstrict",
        "eyeLookDown_L", "eyeLookDown_R", "eyeLookUp_L", "eyeLookUp_R",
        "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L", "eyeLookOut_R",
        "eyeSquint_L", "eyeSquint_R", "eyeWide_L", "eyeWide_R",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimple_L", "mouthDimple_R",
        "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R",
        "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper",
        "mouthSmile_L", "mouthSmile_R", "mouthStretch",
        "mouthUpperUp_L", "mouthUpperUp_R",
        "noseDilation", "noseSneer_L", "noseSneer_R",
        "tongueOut",
    ]
    
    def __init__(self, config: MHConfig = None):
        self.config = config or MHConfig()
        
        self.running = False
        self.ws = None
        self.ws_thread = None
        
        self.current_data = None
        self.smoothed_data = None
        
        self.head_mesh = None
        self.body_armature = None
        
        self._handlers = []
    
    def setup_scene(self):
        """Setup complete scene."""
        print("[MH] Setting up Max Headroom scene...")
        
        self._create_face_mesh()
        self._create_body_rig()
        self._setup_camera()
        self._setup_lighting()
        self._create_materials()
        self._create_post_processing()
        
        print("[MH] Scene setup complete!")
    
    def _create_face_mesh(self):
        """Create full face mesh with 52 blendshapes."""
        print("[MH] Creating face mesh...")
        
        mesh_name = "MH_Head"
        
        # Create or get head mesh
        if mesh_name not in bpy.data.objects:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, segments=32, ring_count=16)
            head = bpy.data.objects.active
            head.name = mesh_name
            
            mesh = head.data
            bpy.ops.object.mode_set(mode='EDIT')
            
            # Subdivide for better definition
            for _ in range(3):
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.subdivide(number_cuts=1)
            
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Create basis shape key
            basis = head.shape_key_add(name="Basis")
            basis.interpolation = 'KEY_LINEAR'
            
            # Create ARKit blendshape keys
            for shape_name in self.ARKIT_BLENDSHAPES:
                head.shape_key_add(name=shape_name)
            
            print(f"[MH] Created face with {len(mesh.shape_keys.key_blocks)} shape keys")
        
        self.head_mesh = bpy.data.objects[mesh_name]
    
    def _create_body_rig(self):
        """Create procedural body rig."""
        print("[MH] Creating body rig...")
        
        rig_name = "MH_Body"
        
        if rig_name not in bpy.data.objects:
            # Create armature
            bpy.ops.object.armature_add()
            armature = bpy.data.objects.active
            armature.name = rig_name
            
            # Add bones
            armature.data.edit_bones.clear()
            
            # Root
            root = armature.data.edit_bones.new("Root")
            root.head = (0, 0, 0)
            root.tail = (0, 0, 0.2)
            
            # Spine
            spine = armature.data.edit_bones.new("Spine")
            spine.head = (0, 0, 0.2)
            spine.tail = (0, 0, 0.8)
            spine.parent = root
            
            # Neck
            neck = armature.data.edit_bones.new("Neck")
            neck.head = (0, 0, 0.8)
            neck.tail = (0, 0, 1.0)
            neck.parent = spine
            
            # Head
            head = armature.data.edit_bones.new("Head")
            head.head = (0, 0, 1.0)
            head.tail = (0, 0, 1.1)
            head.parent = neck
            
            # Shoulders
            for side in [-1, 1]:
                shoulder = armature.data.edit_bones.new(f"Shoulder_{'L' if side < 0 else 'R'}")
                shoulder.head = (side * 0.1, 0, 0.85)
                shoulder.tail = (side * 0.25, 0, 0.85)
                shoulder.parent = spine
            
            # Arms
            for side in [-1, 1]:
                for i, bone_name in enumerate(["UpperArm", "LowerArm", "Hand"]):
                    name = f"{bone_name}_{'L' if side < 0 else 'R'}"
                    bone = armature.data.edit_bones.new(name)
                    bone.parent = armature.data.edit_bones[f"Shoulder_{'L' if side < 0 else 'R'}"]
                    
                    if i == 0:
                        bone.head = (side * 0.25, 0, 0.85)
                        bone.tail = (side * 0.4, 0, 0.7)
                    elif i == 1:
                        bone.head = (side * 0.4, 0, 0.7)
                        bone.tail = (side * 0.5, 0, 0.5)
                    else:
                        bone.head = (side * 0.5, 0, 0.5)
                        bone.tail = (side * 0.55, 0, 0.45)
            
            # Convert to pose mode
            bpy.ops.object.mode_set(mode='POSE')
            bpy.ops.object.mode_set(mode='OBJECT')
        
        self.body_armature = bpy.data.objects[rig_name]
        print("[MH] Body rig created")
    
    def _setup_camera(self):
        """Setup camera."""
        print("[MH] Setting up camera...")
        
        cam_name = "MH_Camera"
        
        if cam_name not in bpy.data.objects:
            bpy.ops.object.camera_add()
            camera = bpy.data.objects.active
            camera.name = cam_name
            
            camera.location = (0, -1.5, 1.2)
            camera.rotation_euler = (1.2, 0, 0)
            
            # Set as active camera
            bpy.context.scene.camera = camera
        
        print("[MH] Camera ready")
    
    def _setup_lighting(self):
        """Setup three-point lighting."""
        print("[MH] Setting up lighting...")
        
        lights = [
            ("MH_Key", (-1, -1, 2), 200, (-0.5, -0.5, 0)),
            ("MH_Fill", (1, -1, 1.5), 100, (0.5, -0.5, 0)),
            ("MH_Rim", (0, 1, 1.5), 150, (0, 0.5, 0)),
        ]
        
        for name, loc, energy, euler in lights:
            if name not in bpy.data.objects:
                bpy.ops.object.light_add(type='SUN', location=loc)
                light = bpy.data.objects.active
                light.name = name
                light.data.energy = energy
        
        print("[MH] Lighting ready")
    
    def _create_materials(self):
        """Create Max Headroom holographic materials."""
        print("[MH] Creating materials...")
        
        # Head material
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
            principled.inputs['Base Color'].default_value = (0.02, 0.06, 0.02, 1)
            principled.inputs['Emission Color'].default_value = (0.0, 1.0, 0.0, 1)
            principled.inputs['Emission Strength'].default_value = 0.5
            principled.inputs['Roughness'].default_value = 0.4
            principled.inputs['Metallic'].default_value = 0.1
            
            # Add scanline effect
            mix = nodes.new('ShaderNodeMixShader')
            mix.location = (600, 0)
            mix.inputs['Fac'].default_value = 0.1
            
            links.new(principled.outputs['BSDF'], mix.inputs[1])
            links.new(principled.outputs['Emission'], mix.inputs[2])
            links.new(mix.outputs['Shader'], output.inputs['Surface'])
            
            if self.head_mesh:
                self.head_mesh.data.materials.append(mat)
        
        # Body material
        if "MH_Body_MAT" not in bpy.data.materials:
            mat = bpy.data.materials.new(name="MH_Body_MAT")
            mat.use_nodes = True
            
            nodes = mat.node_tree.nodes
            principled = nodes.get('Principled BSDF')
            if principled:
                principled.inputs['Base Color'].default_value = (0.02, 0.04, 0.02, 1)
                principled.inputs['Roughness'].default_value = 0.6
        
        print("[MH] Materials ready")
    
    def _create_post_processing(self):
        """Create post-processing setup."""
        print("[MH] Setting up post-processing...")
        
        # Enable bloom for holographic effect
        bpy.context.scene.eevee.use_bloom = True
        bpy.context.scene.eevee.bloom_threshold = 0.8
        bpy.context.scene.eevee.bloom_intensity = 0.3
        
        # Enable viewport AO
        bpy.context.scene.eevee.use_gtao = True
        
        # Set render engine to EEVEE
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        
        print("[MH] Post-processing ready")
    
    def _update_blendshapes(self, blendshapes: Dict[str, float]):
        """Update face mesh blendshapes."""
        if not self.head_mesh or not self.head_mesh.data.shape_keys:
            return
        
        key_blocks = self.head_mesh.data.shape_keys.key_blocks
        
        for name, value in blendshapes.items():
            if name in key_blocks:
                key_blocks[name].value = max(0.0, min(1.0, value))
    
    def _update_body_pose(self, pose: Dict[str, List[float]]):
        """Update body rig pose."""
        if not self.body_armature:
            return
        
        rotation = pose.get("rotation", [0, 0, 0])
        translation = pose.get("translation", [0, 0, 1.0])
        
        # Rotate spine based on head pose
        pose_bones = self.body_armature.pose.bones
        
        if "Spine" in pose_bones:
            pose_bones["Spine"].rotation_euler = (
                math.radians(rotation[0] * 0.3),
                0,
                math.radians(rotation[2] * 0.3)
            )
        
        if "Neck" in pose_bones:
            pose_bones["Neck"].rotation_euler = (
                math.radians(rotation[0] * 0.5),
                math.radians(rotation[1] * 0.3),
                math.radians(rotation[2] * 0.2)
            )
        
        if "Head" in pose_bones:
            pose_bones["Head"].rotation_euler = (
                math.radians(rotation[0]),
                math.radians(rotation[1]),
                math.radians(rotation[2])
            )
    
    def _update_position(self, translation: List[float]):
        """Update mesh position."""
        if not self.head_mesh:
            return
        
        self.head_mesh.location = (
            translation[0],
            translation[2] - 1.0,
            translation[1]
        )
    
    def _apply_smoothing(self, blendshapes: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing."""
        if self.smoothed_data is None:
            self.smoothed_data = blendshapes.copy()
            return blendshapes
        
        s = self.config.smoothing
        smoothed = {}
        
        for name, value in blendshapes.items():
            prev = self.smoothed_data.get(name, 0)
            smoothed[name] = prev * s + value * (1 - s)
        
        self.smoothed_data = smoothed
        return smoothed
    
    def _web_socket_loop(self):
        """WebSocket receive loop."""
        import websockets
        import asyncio
        
        async def receive():
            import websockets
            
            async def handler(ws, path):
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "face_data":
                            self.current_data = data
                            
                            # Apply smoothing
                            if data.get("blendshapes"):
                                smoothed = self._apply_smoothing(data["blendshapes"])
                                data["blendshapes"] = smoothed
                            
                            self._update_from_data(data)
                    except:
                        pass
            
            async with websockets.serve(handler, self.config.ws_host, self.config.ws_port):
                await asyncio.Future()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(receive())
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            loop.close()
    
    def _update_from_data(self, data: Dict):
        """Update scene from tracking data."""
        blendshapes = data.get("blendshapes", {})
        head_pose = data.get("head_pose", {})
        
        if blendshapes and self.head_mesh:
            self._update_blendshapes(blendshapes)
        
        if head_pose and self.body_armature:
            self._update_body_pose(head_pose)
            
            translation = head_pose.get("translation", [0, 0, 1.0])
            self._update_position(translation)
        
        # Force update
        if self.head_mesh:
            self.head_mesh.update_tag()
    
    def start(self):
        """Start integration."""
        if self.running:
            print("[MH] Already running!")
            return
        
        if self.config.auto_setup:
            self.setup_scene()
        
        self.running = True
        self._ws_thread = threading.Thread(target=self._web_socket_loop, daemon=True)
        self._ws_thread.start()
        
        print("[MH] Max Headroom integration started")
        print(f"[MH] Listening on {self.config.ws_host}:{self.config.ws_port}")
        print("[MH] Commands: mh.get_data(), mh.stop(), mh.stats()")
    
    def stop(self):
        """Stop integration."""
        print("[MH] Stopping...")
        self.running = False
    
    def get_data(self):
        """Get current tracking data."""
        return self.current_data
    
    def stats(self):
        """Print statistics."""
        if self.current_data:
            blends = self.current_data.get("blendshapes", {})
            print(f"Blendshapes: {len(blends)} | Keys: {list(blends.keys())[:5]}...")
        else:
            print("No data received yet")

# Global instance
mh = MHBlender()

# Module functions
def start():
    """Start Max Headroom."""
    mh.start()

def stop():
    """Stop Max Headroom."""
    mh.stop()

def get_data():
    """Get current data."""
    return mh.get_data()

def stats():
    """Show stats."""
    mh.stats()

print(f"Max Headroom Blender v{VERSION} loaded")
print("Commands: start(), stop(), get_data(), stats(), mh.setup_scene()")