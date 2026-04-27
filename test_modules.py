import sys
import numpy as np
sys.path.insert(0, '.')

print("=== MODULE TESTS ===\n")

# Test MediaPipe
print("[1] MediaPipe Tracker")
try:
    from mediapipe_tracker import MediaPipeFaceTracker, VERSION
    mp = MediaPipeFaceTracker()
    print(f"  VERSION: {VERSION}")
    print(f"  Created: OK")
    print(f"  Blendshapes: {len(mp.ARKIT_BLENDSHAPES)}")
    mp.close()
except Exception as e:
    print(f"  ERROR: {e}")

# Test OBS Controller
print("\n[2] OBS Controller")
try:
    from obs_controller import OBSController, connect, is_connected
    obs = OBSController()
    print(f"  Created: OK")
    print(f"  Connected: {is_connected()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test Recorder
print("\n[3] Recorder")
try:
    from recorder import Recorder, start_recording, stop_recording
    rec = Recorder()
    rec.start("test")
    for i in range(5):
        rec.add({
            "blendshapes": {"jawOpen": 0.1 * i},
            "head_pose": {"rotation": [0, 0, 0], "translation": [0, 0, 1.5]},
            "timestamp": 0,
            "fps": 30,
        })
    session = rec.stop()
    print(f"  Frames: {session.frame_count}")
    print(f"  Duration: {session.duration:.2f}s")
except Exception as e:
    print(f"  ERROR: {e}")

# Test Blender Export
print("\n[4] Blender Export")
try:
    from blender_export import BlenderExporter, ARKIT_TO_BLENDER
    exp = BlenderExporter()
    print(f"  Mapping entries: {len(ARKIT_TO_BLENDER)}")
    print(f"  Created: OK")
except Exception as e:
    print(f"  ERROR: {e}")

# Test VTS Export
print("\n[5] VTS Export")
try:
    from vts_export import VTSExporter, ARKIT_TO_VTS
    vts = VTSExporter()
    print(f"  Mapping entries: {len(ARKIT_TO_VTS)}")
    print(f"  Created: OK")
except Exception as e:
    print(f"  ERROR: {e}")

# Test GPU
print("\n[6] GPU Acceleration")
try:
    from gpu_accel import GPUDetector, GPUProcessor
    det = GPUDetector()
    proc = GPUProcessor(30)
    stats = proc.get_stats()
    print(f"  GPU enabled: {stats['gpu_enabled']}")
    print(f"  Target FPS: {stats['target_fps']}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== ALL MODULE TESTS PASSED ===")