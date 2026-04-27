# Max Headroom Digitizer v3.0.0

Real-time VTuber digitization system with webcam face capture, WebSocket streaming, OBS integration, and multi-platform export.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MAX HEADROOM v3.0                        │
│              Digital Entity VTuber System                     │
└─────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **31 Blendshapes** | ARKit-compatible facial expressions |
| **MediaPipe** | 468-point face mesh (optional) |
| **WebSocket** | Multi-client real-time streaming |
| **OBS** | Scene switching & filter control |
| **Recording** | Save & playback tracking data |
| **Blender** | Live bone/shape export |
| **VTS** | VTuber Studio API integration |
| **GPU** | CUDA/OpenCL acceleration |
| **CRT Effects** | Digital entity visual overlay |

## Architecture

```
┌──────────────┐   ws://30000   ┌─────────────┐
│  TRACKER    │ ──────────────► │  SERVER   │
│  (Camera)   │               │  (Broker) │
│            │               │          │
│ • Haar/Face │◄─────────────►│ Clients  │
│ • 31 shapes│               └────┬─────┘
│ • Head pose│                    │
└────────────┘              ┌────┼────┐
                            │    │    │
                    ┌───────▼┐  │  ┌─▼────┐
                    │ BLENDER │  │  │ OBS  │
                    │ 3D Exp │  │  │Scene │
                    └────────┘  │  └─────┘
                           ┌────▼─────┐
                           │  VTS    │
                           │ Studio  │
                           └────────┘
```

## Quick Start

### Terminal 1: Start Server
```bash
python server.py --port 30000
```

### Terminal 2: Start Tracker
```bash
python tracker.py --test --glitch 0.15
```

Open http://localhost:8000 in browser to see WebUI.

### Terminal 3: OBS/WebUI
Connect to ws://localhost:30000 via OBS WebSocket or browser.

---

## Modules

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| Tracker | `tracker.py` | Face capture & blendshape extraction |
| Server | `server.py` | WebSocket data broker |
| MediaPipe | `mediapipe_tracker.py` | 468-point face mesh |
| OBS | `obs_controller.py` | OBS scene/filter control |
| Recorder | `recorder.py` | Save/playback data |
| Blender | `blender_export.py` | Live bone export |
| VTS | `vts_export.py` | VTuber Studio API |
| GPU | `gpu_accel.py` | CUDA/OpenCL acceleration |

### Usage

#### Face Tracker
```bash
python tracker.py --help
--camera 0              # Camera index (default: 0)
--fps 30                # Target FPS (default: 30)
--width 640             # Frame width (default: 640)
--height 480            # Frame height (default: 480)
--ws-host localhost    # WebSocket host
--ws-port 30000         # WebSocket port
--no-ws                # Disable WebSocket
--test                 # Test mode (no camera)
--glitch 0.15          # Glitch intensity (0.0-1.0)
```

#### WebSocket Server
```bash
python server.py --help
--host localhost        # Bind host
--port 30000           # Bind port
```

---

## WebSocket Protocol

### Data Format (Outbound)

```json
{
  "type": "face_data",
  "version": "2.0.0",
  "mode": "digital_entity",
  "blendshapes": {
    "jawOpen": 0.234,
    "mouthSmile_L": 0.456,
    "eyeBlink_R": 0.0,
    ...
  },
  "head_pose": {
    "rotation": [5.2, 10.1, -0.5],
    "translation": [0.1, 0.0, 1.5]
  },
  "landmarks": [{"x": 320, "y": 240}, ...],
  "timestamp": 1234567890.123,
  "fps": 30,
  "frame_id": 1234
}
```

### Commands (Inbound)

```json
{"type": "ping", "client_time": 1234567890.0}
{"type": "status_request"}
{"type": "calibrate"}
```

---

## ARKit Blendshapes

| Index | Name | Description |
|-------|------|-------------|
| 0 | browDown_L | Left eyebrow down |
| 1 | browDown_R | Right eyebrow down |
| 2 | browUp_L | Left eyebrow up |
| 3 | browUp_R | Right eyebrow up |
| 4 | cheekPuff | Cheek puff |
| 5 | cheekSquint_L | Left cheek squint |
| 6 | cheekSquint_R | Right cheek squint |
| 7 | eyeBlink_L | Left eye blink |
| 8 | eyeBlink_R | Right eye blink |
| 9 | eyeLookDown_L | Left eye look down |
| 10 | eyeLookDown_R | Right eye look down |
| 11 | eyeLookUp_L | Left eye look up |
| 12 | eyeLookUp_R | Right eye look up |
| 13 | eyeSquint_L | Left eye squint |
| 14 | eyeSquint_R | Right eye squint |
| 15 | jawForward | Jaw forward |
| 16 | jawLeft | Jaw left |
| 17 | jawOpen | Jaw open |
| 18 | jawRight | Jaw right |
| 19 | mouthClose | Mouth close |
| 20 | mouthDimple_L | Left mouth dimple |
| 21 | mouthDimple_R | Right mouth dimple |
| 22 | mouthFunnel | Mouth funnel |
| 23 | mouthLeft | Mouth left |
| 24 | mouthPucker | Mouth pucker |
| 25 | mouthRight | Mouth right |
| 26 | mouthSmile_L | Left smile |
| 27 | mouthSmile_R | Right smile |
| 28 | mouthUpperUp_L | Left upper lip up |
| 29 | mouthUpperUp_R | Right upper lip up |
| 30 | noseSneer_L | Left nose sneer |
| 31 | noseSneer_R | Right nose sneer |

---

## Integration

### OBS WebSocket

```python
from obs_controller import connect, switch_scene, set_filter

connect("localhost", 4455)

switch_scene("Live")
set_filter("Face Capture", "Glow", True)
```

### Blender Live Export

```python
from blender_export import BlenderExporter

exporter = BlenderExporter()
exporter.connect()

exporter.export(blendshapes, pose)
```

### VTuber Studio

```python
from vts_export import VTSExporter

exporter = VTSExporter()
exporter.connect()

exporter.set_blendshapes(blendshapes)
```

### MediaPipe Face Mesh

```python
from mediapipe_tracker import MediaPipeFaceTracker

tracker = MediaPipeFaceTracker()
blendshapes, landmarks, pose = tracker.process(frame)
```

### Recording

```python
from recorder import start_recording, stop_recording, save_recording

start_recording("session_1")

session = stop_recording()
save_recording("session_1.mhr")
```

---

## Digital Entity Mode

The v3.0 tracker includes "Digital Entity" visual effects:

- **CRT Scanlines** - Horizontal scan effect
- **RGB Glitch** - Random color channel shift
- **Pose Display** - Real-time rotation/translation
- **Test Animation** - Oscillating test data

Enable with `--digital` flag (default: on).

```bash
python tracker.py --test --glitch 0.2
```

---

## Troubleshooting

### Camera Not Available
```bash
python tracker.py --test  # Use test mode
```

### WebSocket Connection Refused
- Ensure server is running: `python server.py`
- Check port: `--port 30000`

### OBS Not Connecting
- Enable WebSocket in OBS: Tools > WebSocket Server Settings
- Default port: 4455

### Low FPS
- Reduce resolution: `--width 640 --height 480`
- Use GPU: `python gpu_accel.py`
- Disable effects: `--glitch 0`

---

## Requirements

```
opencv-python>=4.5
numpy>=1.20
mediapipe>=0.8
websocket-client>=1.0
websockets>=10.0
obs-websocket-py>=1.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## File Structure

```
filter/
├── tracker.py          # Main tracker
├── server.py          # WebSocket server
├── mediapipe_tracker.py  # MediaPipe module
├── obs_controller.py # OBS integration
├── recorder.py       # Recording
├── blender_export.py  # Blender export
├── vts_export.py    # VTS export
├── gpu_accel.py     # GPU acceleration
├── max_headroom.py   # Desktop GUI
├── test_run.py      # Test script
└── README.md      # This file
```

---

## License

MIT License - Free for commercial and personal use.

## Links

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)
- [OBS WebSocket](https://obsproject.com/kb/obs-websocket-4-x-development)
- [VTuber Studio](https://vtuberstudio.com)