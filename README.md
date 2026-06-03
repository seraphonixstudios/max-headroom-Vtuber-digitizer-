# Max Headroom Digitizer v3.4

Real-time VTuber digitization system with **cinematic rendered graphics, motion-capture overlays, and Snapchat/WhatsApp-level filters** — webcam face capture, WebSocket streaming, OBS integration, and multi-platform export.

```
 __  __       _      _   _               _                  
|  \/  | __ _| | ___| |_| |__   ___ _ __| |_ _ __ ___  __ _ 
| |\/| |/ _` | |/ _ \ __| '_ \ / _ \ '__| __| '__/ _ \/ _` |
| |  | | (_| | |  __/ |_| | | |  __/ |  | |_| | |  __/ (_| |
|_|  |_|\__,_|_|\___|\__|_| |_|\___|_|   \__|_|  \___|\__, |
                                                       |___/
       Digital Entity VTuber System v3.4 — Render Quality
```

## What's New in v3.4

| Feature | Description |
|---------|-------------|
| **Cel-Shading** | 3 toon rendering modes: Canny outlines, Sobel edges, comic-book halftone |
| **Bloom/Glow** | Threshold-based bloom + neon edge glow with configurable tint |
| **Stylized Edges** | Ink-style DoG edges and colored neon outlines |
| **MoCap Viz Filter** | Face wireframe mesh, tracking points, head pose axes, skeleton, labels |
| **AR Wireframe Sticker** | Face mesh wireframe overlay as an AR sticker |
| **AR Tracking Dots** | Glowing tracking point markers as an AR sticker |
| **7-Filter Pipeline** | New MoCap Viz filter (priority 35) fully integrated |
| **Per-Filter Intensity Sliders** | Individual intensity sliders (0-100%) for each filter |
| **Filter Reorder UI** | ▲/▼ buttons to change filter pipeline order |
| **Color Presets Tab** | One-click color presets: None, Warm, Cool, Cyberpunk, Vintage, Noir, Matrix |
| **Tabbed Sidebar** | Five-tab Notebook: Camera, Filters, Color, Network, Mixer |
| **Waveform Panel** | Animated Perlin-noise waveform replacing old System panel |
| **Dev Console** | Live camera index, resolution, frame time, pipeline status, errors |
| **204 Tests** | Full coverage for all filters, graphics engine, MoCap, and integration |

## What's New in v3.3

| Feature | Description |
|---------|-------------|
| **CameraManager** | Professional camera handling with discovery, hot-swap, 3s timeout |
| **FrameBus Architecture** | Producer-consumer pipeline with clean stage separation |
| **OBS-Style UI** | Dark broadcast studio interface with docked panels |
| **Scene Presets** | One-click filter combos: Default, Android, Beauty, Color |
| **Source Toggles** | Eye-icon filter list with color-coded abbreviations |
| **Camera Diagnostics** | TEST CAM button scans all cameras and reports status |
| **Rate-Limited Logging** | Prevents console spam with 0.1s throttle |
| **Graceful Degradation** | Face detector failure auto-switches to test mode |

## What's New in v3.2

| Feature | Description |
|---------|-------------|
| **LED Status Indicators** | Pulsing CAM / TRACK / NET status lights with color coding |
| **Real Blendshape Bars** | Live horizontal bar chart showing top 12 active blendshapes |
| **Head Pose Display** | Real-time rotation & translation values panel |
| **Filter Button Feedback** | Active filters highlight with cyan glow; inactive dim |
| **Keyboard Shortcuts** | D/B/C/G/A/M/R/Q hotkeys bound in GUI |
| **Camera Auto-Recovery** | Automatic camera reconnection with 3s retry backoff |
| **WS Auto-Reconnect** | Exponential backoff WebSocket reconnection (up to 5 attempts) |
| **Thread-Safe GUI** | Frame queue with lock; skips updates if GUI falls behind |
| **Performance HUD** | Live FPS, packet count, and frame processing time display |
| **Shortcut Legend** | Visual hotkey reference in bottom status bar |

## Features

| Feature | Description |
|---------|-------------|
| **51 Blendshapes** | ARKit-compatible facial expressions |
| **7 Real-Time Filters** | Full pipeline with priority ordering |
| **Cel-Shading** | Canny/Sobel edge outlines + k-means color quantization |
| **Bloom/Glow** | Real-time threshold bloom with configurable tint |
| **MoCap Overlays** | Wireframe mesh, tracking points, pose axes, skeleton, labels |
| **AR Stickers** | Glasses, hats, crowns + wireframe mesh + tracking dots |
| **MediaPipe** | 468-point face mesh (optional) |
| **WebSocket** | Multi-client real-time streaming |
| **OBS** | Scene switching & filter control |
| **Recording** | Save & playback tracking data |
| **Blender** | Live bone/shape export |
| **VTS** | VTuber Studio API integration |
| **GPU** | CUDA/OpenCL acceleration |
| **CRT Effects** | Scanlines, phosphor triads, interlace flicker |
| **Android Mode** | Max Headroom styled character filter |

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
| `D` | Toggle Max Headroom android filter |
| `B` | Toggle beauty filter |
| `G` | Toggle background filter |
| `A` | Toggle AR overlays |
| `M` | Toggle face morph |
| `C` | Toggle color grading |
| `R` | Reset all filters |

---

## Themed Desktop GUI

Launch the fully themed desktop interface:
```bash
python max_headroom.py --gui
# or
python launch.py --tracker --gui
```

### Visual Themes

The GUI fuses four visual styles into one cohesive interface:

| Theme | Elements | Colors |
|-------|----------|--------|
| **Matrix** | Falling code rain, terminal log, green data streams | `#00FF41` |
| **Max Headroom** | CRT scanlines, glitch text, cyan accents | `#00FFFF` |
| **Atlantean** | Sacred geometry, crystal hexagons, rotating patterns | `#00E5FF` |
| **Sci-Fi HUD** | Targeting reticle, data rings, waveform visualizer | `#BF00FF` |

### Interface Layout (v3.4)

```
┌──────────────────────────────────────────────────────────────────────┐
│  [GLITCH TITLE]                              [MATRIX RAIN ANIMATION] │
├───────────────────────────────┬──────────────────────────────────────┤
│                               │  ┌─ TABBED SIDEBAR ───────────────┐ │
│    [VIDEO PREVIEW]            │  │ ▸ CAMERA   FILTERS  COLOR      │ │
│    + CRT scanline (drawn)     │  │   NETWORK  MIXER               │ │
│    + HUD reticle (drawn)      │  ├────────────────────────────────┤ │
│                               │  │ ⬤ MH   Max Headroom   ▲ ▼  │ │
│                               │  │ ⬤ SKIN Skin Smoothing ▲ ▼  │ │
│                               │  │ ⬤ CLR  Color Grading  ▲ ▼  │ │
│                               │  │ ⬤ MORPH Face Morph    ▲ ▼  │ │
│                               │  │ ⬤ BG   Background     ▲ ▼  │ │
│                               │  │ ⬤ AR   AR Overlay     ▲ ▼  │ │
│                               │  │ ⬤ MOCAP MoCap Viz     ▲ ▼  │ │
│                               │  │   [████████████░░░] 50%     │ │
│                               │  │ Global: [████|████]  Glitch │ │
│                               │  └────────────────────────────────┘ │
├───────────┬───────────┬───────┴────────────────────────────────────┤
│ HEAD POSE │ WAVEFORM  │  CONSOLE           │ DEV                  │
│ ROT 0 0 0 │ [== ██==] │  System online     │ Cam:0 Size:640x480   │
│ POS 0 0 0 │ [==██ ██] │  Filters ready     │ Pipe:active 19ms    │
├───────────┴───────────┴────────────────────┴──────────────────────┤
│ CAMERA: STANDBY              FPS: 0    PKT: 0                     │
└────────────────────────────────────────────────────────────────────┘
```

### Animated Elements (Real tkinter Canvas Animation)

- **Matrix Rain** — Japanese characters + hex falling in columns, 25 FPS
- **Sacred Geometry** — Rotating flower of life, Metatron's cube, crystal hexagon (in Mixer tab)
- **CRT Scanlines** — Drawn directly on video canvas for proper compositing
- **HUD Reticle** — Animated targeting reticle, data ring, pulse dot drawn on video
- **Waveform** — Perlin-noise style bars in bottom panel (20 FPS)
- **Hex Display** — Live-updating memory dump in Mixer tab
- **Glitch Label** — Title occasionally corrupts characters and shifts color
- **Blendshape Bars** — Real-time bar chart of top 12 blendshapes (in Mixer tab)

### GUI Filter Controls (v3.4)

The **Filters** tab in the Notebook sidebar provides per-filter controls:
- **Eye Toggle** (⬤) — Enable/disable individual filters
- **▲ ▼** — Reorder filter priority in the pipeline
- **Intensity Slider** (0-100%) — Per-filter intensity with live percentage readout
- **Global Quality** — Low/Medium/High quality preset for all filters
- **Glitch Intensity** — Global glitch effect amount

Additional tabs:
- **Camera** — Device dropdown, SCAN/REFRESH, resolution/FPS info
- **Color** — Instant color preset swatches (None, Warm, Cool, Cyberpunk, Vintage, Noir, Matrix)
- **Network** — WebSocket host/port, CONNECT, Simulation Mode
- **Mixer** — Blendshape bars + Sacred Geometry + Hex dump VU meters

Hotkeys: D (Android), B (Beauty), C (Color), G (Background), A (AR), M (Morph), R (Reset), Q (Quit)

---

## Filter System (v3.4)

### 7 Professional Real-Time Filters

All filters applied in priority order through the pipeline. Each has per-filter intensity control and individual enable/disable.

#### 0. MoCap Viz (NEW in v3.4)
```bash
# Toggle: MoCap Viz filter in Filters tab
```
Motion capture visualization overlay — transforms your video feed into a professional mocap studio view.

**Features:**
- **Face Wireframe** — Delaunay-like tesselation mesh with contour highlights
- **Tracking Points** — Glowing dots on key landmarks (eyes, nose, mouth, chin)
- **Head Pose Axes** — RGB XYZ rotation arrows from nose, using full rotation matrix
- **Skeleton** — Face skeleton edges (jaw, brow, eye, nose, mouth connectivity)
- **Landmark Labels** — NOSE, L_EYE, R_EYE, CHIN, etc.
- **4 Style Presets** — `tech` (cyan), `neon` (magenta), `dark` (subdued), `minimal` (light)
- **Per-Feature Toggle** — Each visualization layer independently toggleable

```python
from filters import FilterManager
mgr = FilterManager()
mgr.enable_filter("MoCap Viz")
mgr.set_filter_param("MoCap Viz", "wireframe", True)
mgr.set_filter_param("MoCap Viz", "tracking_points", True)
mgr.set_filter_param("MoCap Viz", "pose_axes", True)
mgr.set_filter_param("MoCap Viz", "style", "neon")
```

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
- **Wireframe** (v3.4): Face mesh wireframe with tesselation + contour lines
- **Tracking Dots** (v3.4): Glowing tracking points with halo rings

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

#### 6. Max Headroom Android Filter
```bash
# Hotkey: D
# CLI: python tracker.py --android
```
Transforms you into a **Max Headroom styled android/digital character** with the classic 1980s cyberpunk broadcast intrusion aesthetic.

**Effects:**
- **Cyan Monochrome** — High-contrast grayscale with cyan/blue tint
- **Heavy CRT Scanlines** — Thick, pronounced horizontal scan lines
- **Temporal Stutter** — Occasional frame repeats for bad-signal look
- **Chromatic Aberration** — RGB channel splitting for lens distortion
- **Edge Enhancement** — Sharp unsharp mask for machine-like edges
- **Pixelation** — Blocky digital artifacting
- **Geometric Neon Grid** — Perspective cyan grid overlay
- **Glitch Blocks** — Random rectangle corruption/inversion
- **Data Overlay** — Scrolling status text, hex codes, signal bars
- **Heavy Vignette** — Dark edges for broadcast intrusion feel

**v2.0 SOTA Enhancements:**
- **Posterization** — K-means color quantization (6-level digital palette)
- **Ordered Dithering** — Bayer 4×4 matrix for retro C64 aesthetic
- **Film Grain** — Luminance-only Gaussian noise with configurable size
- **CLAHE** — Contrast Limited Adaptive Histogram Equalization on L channel
- **Radial Chromatic Aberration** — Physically-based lens distortion with barrel effect
- **RGB Phosphor Triads** — Vertical RGB stripe simulation
- **Interlace Flicker** — Alternating field dimming
- **Temporal Smoothing** — Exponential moving average for frame coherence

**v3.4 Digital Graphics Modes:**
- **Cel-Shading** — Canny or Sobel edge outlines + k-means flat color quantization (configurable k=2-16)
- **Bloom/Glow** — Threshold-based glow on bright areas with optional tint (e.g. cyan bloom)
- **Comic Style** — Comic-book aesthetic: quantized colors + halftone dots + bold outlines
- **Neon Edges** — Colored edge outlines in any RGB color (e.g. magenta neon)
- **Ink Edges** — Difference-of-Gaussians dark ink outlines
- **Presentation Mode** — One-switch look: `crt`, `cel`, `bloom`, `comic`, `neon`, or `clean`

```python
# Enable cel-shading
mgr.set_filter_param("Max Headroom", "cel_shading", True)
mgr.set_filter_param("Max Headroom", "cel_shading_k", 8)
mgr.set_filter_param("Max Headroom", "cel_edge_style", "canny")

# Enable bloom
mgr.set_filter_param("Max Headroom", "bloom", True)
mgr.set_filter_param("Max Headroom", "bloom_threshold", 0.7)
mgr.set_filter_param("Max Headroom", "bloom_intensity", 0.4)

# Comic style
mgr.set_filter_param("Max Headroom", "comic_style", True)
mgr.set_filter_param("Max Headroom", "comic_dots", 0.15)
```

---

## SOTA Graphics Engine

Production-grade open-source computer vision techniques used across all filters.

### Core Technologies

| Technique | Purpose | Source |
|-----------|---------|--------|
| **Laplacian Pyramid Blending** | Seamless multi-scale compositing | Burt & Adelson (1983) |
| **Gamma-Correct Alpha** | Perceptually correct transparency | sRGB standard |
| **CLAHE** | Local contrast enhancement without noise | OpenCV |
| **K-Means Quantization** | Posterization / color reduction | OpenCV |
| **Guided Filter** | Edge-preserving smoothing (faster than bilateral) | He et al. (2010) |
| **Ordered Bayer Dither** | Retro digital halftone | Classic algorithm |
| **Floyd-Steinberg** | Error diffusion dithering | Classic algorithm |
| **Radial CA** | Lens distortion simulation | OpenCV remap |
| **Temporal EMA** | Frame-to-frame coherence | Exponential smoothing |
| **GPU CUDA/OpenCL** | Hardware acceleration | OpenCV GPU module |
| **Cel-Shading** (v3.4) | Canny/Sobel edge outlines + k-means flat color | OpenCV |
| **Bloom/Glow** (v3.4) | Threshold + Gaussian blur glow composite | OpenCV |
| **Ink Edges** (v3.4) | Difference-of-Gaussians dark outlines | Classic algorithm |
| **Colored Edges** (v3.4) | Neon/stylized colored edge outlines | OpenCV |
| **Comic Style** (v3.4) | Halftone dots + quantization + outlines | OpenCV |

### Performance Benchmarks (100×100 frame)

| Operation | Time |
|-----------|------|
| Alpha compositing | ~1.0ms |
| Pyramid blend | ~0.5ms |
| CLAHE enhancement | ~0.2ms |
| K-means quantization | ~0.2ms |
| Ordered dither | ~0.2ms |
| Guided filter | ~0.7ms |
| Film grain | ~0.3ms |
| Chromatic aberration | ~0.2ms |
| CRT scanlines | ~0.1ms |

### Usage

```python
from filters.graphics_engine import (
    AlphaCompositor, PyramidBlend, CLAHEEnhancer,
    ColorQuantizer, Dithering, GuidedFilter,
    FilmGrain, ChromaticAberration, ScanlineEffects,
    CelShading, BloomEffect, StylizedEdges,
)

# Gamma-correct blending
result = AlphaCompositor.composite(background, foreground, alpha_mask)

# Cel-shading (v3.4)
result = CelShading.apply(frame, quantize_levels=6)
result = CelShading.apply_color_quantized(frame, k=8, edge_style="sobel")
result = CelShading.comic_style(frame, k=6, dot_density=0.15)

# Bloom / glow (v3.4)
result = BloomEffect.apply(frame, threshold=0.7, intensity=0.4)
result = BloomEffect.apply(frame, color=(0, 200, 255))  # Tinted cyan
result = BloomEffect.glow_edges(frame, glow_color=(255, 0, 255))  # Neon magenta

# Stylized edges (v3.4)
result = StylizedEdges.ink_edges(frame, strength=1.5)
result = StylizedEdges.colored_edges(frame, edge_color=(255, 100, 255))

# Multi-scale seamless blend
result = PyramidBlend.blend(image_a, image_b, mask, levels=4)

# CLAHE local contrast
clahe = CLAHEEnhancer(clip_limit=2.0)
result = clahe.apply(frame)

# Posterization
result = ColorQuantizer.quantize_fast(frame, k=8)

# Retro dithering
result = Dithering.ordered_dither(frame, levels=4)

# Edge-preserving smoothing
result = GuidedFilter.apply(frame, frame, radius=8)

# Film grain
result = FilmGrain.apply(frame, intensity=0.1, color=False)

# Chromatic aberration
result = ChromaticAberration.apply(frame, strength=3.0)

# CRT effects
result = ScanlineEffects.crt_scanlines(frame)
result = ScanlineEffects.rgb_phosphor(frame, strength=0.3)
```

---

## Configuration (v3.1)

mgr = FilterManager()
mgr.enable_filter("Max Headroom")
mgr.set_filter_param("Max Headroom", "intensity", 1.0)  # Full effect
mgr.set_filter_param("Max Headroom", "intensity", 0.5)  # Subtle

# Toggle individual effects
mgr.set_filter_param("Max Headroom", "scanlines", True)
mgr.set_filter_param("Max Headroom", "glitch_blocks", False)
mgr.set_filter_param("Max Headroom", "data_overlay", True)
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
| AR Overlay | `filters/ar_overlay.py` | Stickers & effects (wireframe + tracking dots in v3.4) |
| Face Morph | `filters/face_morph.py` | Mesh deformation |
| Color Grading | `filters/color_grading.py` | LUT color filters |
| Max Headroom | `filters/max_headroom_filter.py` | Android character transformation + v3.4 digital graphics |
| MoCap Viz | `filters/mocap_viz.py` | Motion capture visualization overlay (v3.4) |

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
--android               # Enable Max Headroom android filter
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

### Run All Tests (204 total)
```bash
# v3.0 core tests (24 tests)
python run_tests.py

# v3.1 pipeline tests (16 tests)
python test_v31.py

# SOTA graphics engine tests (54 tests)
python test_graphics.py

# Filter system tests (69 tests) — covers all 7 filters + MoCap
python test_filters.py

# End-to-end integration tests (41 tests)
python test_e2e.py

# Run all 5 test suites at once
python launch.py --all-tests

# Quick validation
python launch.py --quick-test
```

### Test Results
| Suite | Tests | Status |
|-------|-------|--------|
| v3.0 Core | 24 | ✅ Pass |
| v3.1 Pipeline | 16 | ✅ Pass |
| SOTA Graphics Engine | 54 | ✅ Pass |
| Filter System | 69 | ✅ Pass |
| End-to-End | 41 | ✅ Pass |
| **Total** | **204** | **✅ 100%** |

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
- Press filter hotkeys (D, B, G, A, M, C) to toggle
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
├── gui_themes.py         # Sci-Fi / Matrix / Atlantean GUI components
├── max_headroom.py       # Desktop application with themed GUI
├── camera_manager.py     # Professional camera handling
├── pipeline_v2.py        # FrameBus producer-consumer architecture
├── test_graphics.py      # SOTA graphics engine tests
├── filters/              # Filter system
│   ├── __init__.py
│   ├── base.py
│   ├── manager.py
│   ├── skin_smoothing.py
│   ├── background.py
│   ├── ar_overlay.py
│   ├── face_morph.py
│   ├── color_grading.py
│   ├── max_headroom_filter.py  # Android character filter
│   ├── mocap_viz.py            # MoCap visualization overlay (v3.4)
│   └── graphics_engine.py      # SOTA graphics engine
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

*Max Headroom v3.4 - Built with OpenCV, NumPy, and MediaPipe. No placeholders, no hacks - production-ready.*