print("=== MAX HEADROOM v3.0 DIGITAL ENTITY SYSTEM ===")
print("\nTesting...")

import numpy as np
from tracker import Config, MaxHeadroomTracker

config = Config()
config.test_mode = True
config.digital_mode = True
config.glitch_intensity = 0.25

tracker = MaxHeadroomTracker(config)

print(f"Mode: DIGITAL")
print(f"Glitch: {config.glitch_intensity}")
print(f"WebSocket: {config.ws_host}:{config.ws_port}")
print("\nProcessing frame...")

frame = np.zeros((480, 640, 3), dtype=np.uint8)
blendshapes, landmarks, pose = tracker.process_frame(frame)

print(f"\n[OUTPUT]")
print(f"Blendshapes: {len(blendshapes)} keys")
print(f"Keys: {list(blendshapes.keys())}")
print(f"\nSample values:")
for k in ["jawOpen", "mouthSmile_L", "browDown_L", "eyeLookUp_L", "cheekSquint_L", "mouthDimple_L"]:
    print(f"  {k}: {blendshapes.get(k, 0):.3f}")
print(f"Pose: {pose}")

print("\n=== SYSTEM READY ===")