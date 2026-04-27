import sys
import numpy as np
sys.path.insert(0, '.')

print("=== TRACKER COMPONENT TESTS ===\n")

# Test 1: Config
print("[1] Config test")
from tracker import Config
c = Config()
print(f"  ws_host: {c.ws_host}")
print(f"  digital_mode: {c.digital_mode}")
print(f"  glitch_intensity: {c.glitch_intensity}")

# Test 2: BlendShapeCalculator
print("\n[2] BlendShapeCalculator test")
from tracker import BlendShapeCalculator
calc = BlendShapeCalculator()
test_landmarks = [(320 + i*2, 280 + i) for i in range(68)]
result = calc.calculate(test_landmarks, (200, 150, 240, 260), 0)
print(f"  Keys: {len(result)}")
print(f"  jawOpen: {result.get('jawOpen', 0):.2f}")
print(f"  mouthSmile_L: {result.get('mouthSmile_L', 0):.2f}")

# Test 3: FaceDetector
print("\n[3] FaceDetector test")
from tracker import FaceDetector
det = FaceDetector()
print(f"  Cascade loaded: {not det.cascade.empty()}")

# Test 4: MaxHeadroomTracker
print("\n[4] MaxHeadroomTracker init")
from tracker import MaxHeadroomTracker
t = MaxHeadroomTracker()
print(f"  Created: OK")

# Test 5: Test mode
print("\n[5] Test mode frame generation")
t.config.test_mode = True
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
bs, lm, pose = t.process_frame(dummy_frame)
print(f"  Blendshapes: {len(bs)} keys")
print(f"  Pose: {pose is not None}")

print("\n=== ALL TRACKER TESTS PASSED ===")