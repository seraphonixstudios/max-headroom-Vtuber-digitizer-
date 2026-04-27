import sys
import numpy as np
sys.path.insert(0, '.')

print('[Testing MediaPipe module]')
from mediapipe_tracker import MediaPipeFaceTracker, create_tracker

tracker = create_tracker(use_mediapipe=False)

frame = np.zeros((480, 640, 3), dtype=np.uint8)

blendshapes, landmarks, pose = tracker.process(frame)

print(f'Result: blendshapes={blendshapes is not None}, landmarks={landmarks is not None}, pose={pose is not None}')

if blendshapes:
    print(f'Blendshapes: {len(blendshapes)}')
    j = blendshapes.get('jawOpen', 0)
    print(f'jawOpen: {j:.3f}')

tracker.close()
print('MediaPipe test complete')