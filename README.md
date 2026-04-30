# Max Headroom Digitizer v3.1

Real-time VTuber digitization system with **Snapchat/WhatsApp-level filters**, webcam face capture, WebSocket streaming, OBS integration, and multi-platform export.

```
 __  __       _      _   _               _                  
|  \/  | __ _| | ___| |_| |__   ___ _ __| |_ _ __ ___  __ _ 
| |\/| |/ _` | |/ _ \ __| '_ \ / _ \ '__| __| '__/ _ \/ _` |
| |  | | (_| | |  __/ |_| | | |  __/ |  | |_| | |  __/ (_| |
|_|  |_|\__,_|_|\___|\__|_| |_|\___|_|   \__|_|  \___|\__, |
                                                      |___/
                Digital Entity VTuber System v3.1
```

## What's New in v3.1

| Feature | Description |
|---------|-------------|
| **Filter System** | 5 Snapchat/WhatsApp-level real-time filters |
| **Skin Smoothing** | Bilateral beauty filter with edge preservation |
| **Background** | Blur, color, or image replacement |
| **AR Overlays** | Glasses, hats, crowns, mustache, tears, blush |
| **Face Morph** | Face slimming, eye enlarging, jaw shaping |
| **Color Grading** | 6 LUT presets + vignette + contrast |
| **Eye Glow** | Anime lens flare / sharingan terminator effect |
| **Config System** | JSON-based configuration with dot-notation access |
| **Pipeline v3.1** | MediaPipe primary, 3D pose, Kalman smoothing |
| **Logging** | Centralized colored console + rotating file output |

## Features

| Feature | Description |
|---------|-------------|
| **51 Blendshapes** | ARKit-compatible facial expressions (v3.1) |
| **MediaPipe** | 468-point face mesh (optional) |
| **WebSocket** | Multi-client real-time streaming |
| **OBS** | Scene switching & filter control |
| **Recording** | Save & playback tracking data |
| **Blender** | Live bone/shape export |
| **VTS** | VTuber Studio API integration |
| **GPU** | CUDA/OpenCL acceleration |
| **CRT Effects** | Digital entity visual overlay |
| **Filters** | Beauty, background, AR, morph, color (v3.1) |

## Architecture

```
┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   CAMERA     │────▶│   TRACKER   │────▶│   FILTERS   │
│  (Webcam)    │     │ • Detect    │     │ • Beauty    │
└──────────────┘     │ • 51 shapes │     │ • Background│
                     │ • 3D Pose   │     │ • AR Overlay│
                     └──────┬──────┘     │ • Morph     │
                            │            │ • Color     │
                            ▼            └──────┬──────┘
                     ┌─────────────┐            │
                     │   SERVER    │◄───────────┘
                     │ ws://30000  │    Stream
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼──────┐
        │  BLENDER  │ │   OBS   │ │    VTS     │
        │  3D Exp   │ │  Scene  │ │  Studio    │
        └───────────┘ └─────────┘ └────────────┘
```

## Quick Start

### Terminal 1: Start Server
```bash
python server.py --port 30000
```

### Terminal 2: Start Tracker with Filters
```bash
python tracker.py --test --eye-glow
```

### Controls (while tracker running)
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `T` | Toggle test mode |
| `E` | Toggle anime eye glow |
| `B` | Toggle beauty filter |
| `G` | Toggle background filter |
| `A` | Toggle AR overlays |
| `M` | Toggle face morph |
| `C` | Toggle color grading |
| `R` | Reset all filters |

---

## Filter System (v3.1)

### Snapchat/WhatsApp-Level Filters

Real-time filters applied to the video stream with **zero placeholders** - all fully working.

#### 1. Skin Smoothing (Beauty Filter)
```bash
# Hotkey: B
```
- Bilateral filter with HSV skin-tone masking
- Preserves edges (eyes, mouth, eyebrows, hair)
- Configurable strength (0.0 - 1.0)
- Pyramid down/up for extra smoothness
- Subtle unsharp mask for clarity

#### 2. Background Filter
```bash
# Hotkey: G
```
Modes:
- **Blur**: Heavy Gaussian blur on background
- **Color**: Solid color replacement
- **Replace**: Custom image background

Features:
- Person segmentation using face landmarks
- Edge feathering for natural blending
- Automatic mask generation from face contours

#### 3. AR Overlay (Stickers)
```bash
# Hotkey: A
```
Available stickers (anchored to facial landmarks):
- **Glasses**: Anime-style frames with lens tint
- **Hat**: Cap positioned above eyebrows
- **Crown**: Gold crown with jewels
- **Mustache**: Handlebar style
- **Blush**: Pink cheek circles
- **Tears**: Anime tear drops under eyes

Add stickers programmatically:
```python
from filters import AROverlayFilter
ar = AROverlayFilter()
ar.add_sticker("glasses", color=(0, 255, 255))
ar.add_sticker("crown")
```

#### 4. Face Morph
```bash
# Hotkey: M
```
- **Face Slimming**: Move jaw inward
- **Eye Enlarging**: Scale eyes larger
- **Jaw Shaping**: Move jaw line up/down
- **Chin Shaping**: Pointier/rounder chin

Uses Delaunay triangulation mesh deformation for smooth warping.

#### 5. Color Grading
```bash
# Hotkey: C
```
LUT Presets:
- `warm` - Increase reds/yellows
- `cool` - Increase blues
- `cyberpunk` - Neon magenta/cyan
- `vintage` - Sepia faded look
- `noir` - High contrast B&W
- `matrix` - Green tint

Adjustments:
- Contrast (0.5 - 2.0)
- Saturation (0.0 - 2.0)
- Brightness (-50 to +50)
- Vignette (0.0 - 1.0)
- RGB tint

```python
from filters import ColorGradingFilter, FilterManager

mgr = FilterManager()
mgr.enable_filter("Color Grading")
mgr.set_filter_param("Color Grading", "preset", "cyberpunk")
mgr.set_filter_param("Color Grading", "vignette", 0.5)
```

---

## Configuration (v3.1)

All settings in `config.json`:

```json
{
  "tracker": {
    "target_fps": 60,
    "resolution": [640, 480],
    "kalman_filter": true,
    "detector": {
      "primary": "mediapipe",
      "fallback": "haar"
    },
    "eye_glow": false,
    "glitch_intensity": 0.15
  },
  "server": {
    "host": "localhost",
    "port": 30000
  }
}
```

Access via code:
```python
from config import get, set
fps = get('tracker.target_fps', 30)
set('tracker.eye_glow', True)
```

---

## Modules

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| Tracker | `tracker.py` | Face capture & blendshape extraction |
| Tracker v3.1 | `tracker_v31.py` | Advanced tracker with MediaPipe + 3D pose |
| Server | `server.py` | WebSocket data broker |
| Pipeline | `pipeline.py` | Pipeline coordinator |
| Config | `config.py` | Configuration manager |

### Filter Modules

| Module | File | Purpose |
|--------|------|---------|
| Filter Manager | `filters/manager.py` | Pipeline orchestrator |
| Skin Smoothing | `filters/skin_smoothing.py` | Beauty filter |
| Background | `filters/background.py` | Virtual backgrounds |
| AR Overlay | `filters/ar_overlay.py` | Stickers & effects |
| Face Morph | `filters/face_morph.py` | Mesh deformation |
| Color Grading | `filters/color_grading.py` | LUT color filters |

### Export Modules

| Module | File | Purpose |
|--------|------|---------|
| MediaPipe | `mediapipe_tracker.py` | 468-point face mesh |
| OBS | `obs_controller.py` | OBS scene/filter control |
| Recorder | `recorder.py` | Save/playback data |
| Blender | `blender_export.py` | Live bone export |
| VTS | `vts_export.py` | VTuber Studio API |
| GPU | `gpu_accel.py` | CUDA/OpenCL acceleration |

---

## Tracker CLI Options

```bash
python tracker.py --help
--camera 0              # Camera index
--fps 30                # Target FPS
--width 640             # Frame width
--height 480            # Frame height
--ws-host localhost     # WebSocket host
--ws-port 30000         # WebSocket port
--no-ws                 # Disable WebSocket
--test                  # Test mode (no camera)
--glitch 0.15           # Glitch intensity
--eye-glow              # Enable eye glow
--digital               # Digital entity mode
```

---

## WebSocket Protocol

### Data Format (Outbound)

```json
{
  "type": "face_data",
  "version": "3.1.0",
  "mode": "digital_entity",
  "blendshapes": {
    "jawOpen": 0.234,
    "mouthSmile_L": 0.456,
    ...
  },
  "head_pose": {
    "rotation": [5.2, 10.1, -0.5],
    "translation": [0.1, 0.0, 1.5]
  },
  "landmarks": [{"x": 320, "y": 240}, ...],
  "timestamp": 1234567890.123,
  "fps": 60,
  "frame_id": 1234
}
```

---

## Testing

### Run All Tests
```bash
# v3.0 core tests (24 tests)
python run_tests.py

# v3.1 pipeline tests (16 tests)
python test_v31.py

# Filter system tests (22 tests)
python test_filters.py

# Quick validation
python launch.py --quick-test
```

### Test Results
| Suite | Tests | Status |
|-------|-------|--------|
| v3.0 Core | 24 | ✅ Pass |
| v3.1 Pipeline | 16 | ✅ Pass |
| Filter System | 22 | ✅ Pass |
| **Total** | **62** | **✅ 100%** |

---

## Logging

All modules use centralized logging:

### Log Location
- **Console**: Real-time colored output
- **File**: `logs/max_headroom.log` (5MB rotating)

### Log Levels
- **INFO**: Startup, connections, status
- **WARNING**: Recoverable issues
- **ERROR**: Failures
- **DEBUG**: Frame processing details

```python
from logging_utils import LOG
import logging
LOG.setLevel(logging.DEBUG)
```

---

## Troubleshooting

### Camera Not Available
```bash
python tracker.py --test  # Use test mode
```

### WebSocket Connection Refused
```bash
python server.py --port 30000  # Start server first
```

### OBS Not Connecting
- Enable WebSocket: Tools > WebSocket Server Settings
- Default port: 4455

### Low FPS
```bash
python tracker.py --width 640 --height 480 --fps 30
```

### Filters Not Working
- Ensure landmarks are detected (face visible to camera)
- Press filter hotkeys (B, G, A, M, C) to toggle
- Check `logs/max_headroom.log` for errors

---

## Requirements

```
opencv-python>=4.5
numpy>=1.20
mediapipe>=0.8
websocket-client>=1.0
websockets>=10.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## File Structure

```
filter/
├── tracker.py              # Main tracker (v3.0)
├── tracker_v31.py          # Advanced tracker (v3.1)
├── server.py              # WebSocket server
├── pipeline.py            # Pipeline coordinator
├── config.py              # Config manager
├── config.json            # Configuration file
├── logging_utils.py       # Centralized logging
├── launch.py              # Unified launcher
├── run_tests.py           # v3.0 test suite
├── test_v31.py            # v3.1 integration tests
├── test_filters.py        # Filter system tests
├── test_e2e.py            # End-to-end test
├── mediapipe_tracker.py   # MediaPipe module
├── obs_controller.py      # OBS integration
├── recorder.py            # Recording
├── blender_export.py      # Blender export
├── vts_export.py         # VTS export
├── gpu_accel.py          # GPU acceleration
├── filters/              # Filter system
│   ├── __init__.py
│   ├── base.py
│   ├── manager.py
│   ├── skin_smoothing.py
│   ├── background.py
│   ├── ar_overlay.py
│   ├── face_morph.py
│   └── color_grading.py
└── README.md             # This file
```

---

## License

MIT License - Free for commercial and personal use.

---

## Links

- [GitHub Repository](https://github.com/seraphonixstudios/max-headroom-Vtuber-digitizer-)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)
- [OBS WebSocket](https://obsproject.com/kb/obs-websocket-4-x-development)
- [VTuber Studio](https://vtuberstudio.com)

---

*Max Headroom v3.1 - Built with OpenCV, NumPy, and MediaPipe. No placeholders, no hacks - production-ready.*