# Sensory System

A comprehensive, real-time monitoring system that combines computer vision, facial recognition, activity detection, and app usage tracking to build a contextual understanding of user behavior.

## Overview

Sensory System is a modular sensor hub that runs multiple parallel detection threads, each contributing to a unified context model. It tracks user presence, gaze direction, active applications, recognized faces, and detected objects—all while providing natural, context-aware feedback.

### Core Capabilities

- **Activity Detection**: Monitors user presence and gaze to determine if they're active, idle, distracted, or away
- **Face Recognition**: Identifies known individuals via DeepFace with persistent memory and cooldown logic
- **Object Detection**: Uses YOLOv8 to detect objects and people in real-time
- **App Usage Tracking**: Monitors which application is currently in focus (macOS, Windows, Linux)
- **Proactive Feedback**: Delivers context-aware reminders (break suggestions, arrivals, weather alerts)

## Architecture

### Components

#### Core Modules

| Module | Purpose |
|--------|---------|
| `camera_manager.py` | Handles camera initialization, frame capture, and error recovery |
| `context_fusion.py` | Thread-safe key-value store with change callbacks and version tracking |
| `idle_detection_combined.py` | Monitors gaze and presence; determines activity state |
| `obj_face_recog.py` | Detects and identifies people via YOLO + DeepFace |
| `app_usage_monitor.py` | Tracks foreground application (macOS, Windows, Linux) |
| `proactive_outputs.py` | Generates context-aware reminders and logs changes to CSV |
| `main.py` | Bootstrap and supervision logic; starts all sensors |

#### Key Design Patterns

**Thread-Safe Context Store**: All sensor threads read/write to a shared `CONTEXT` object with automatic change notifications and thread-safe locking.

**Supervisor Pattern**: Each sensor runs in a dedicated thread with automatic restart on crash, heartbeat monitoring, and hang detection.

**Batch Updates**: Related context changes are grouped to reduce notification callbacks.

## Installation

### Prerequisites

- Python 3.10+
- macOS, Windows, or Linux (with platform-specific dependencies)

### Dependencies

```bash
pip install opencv-python mediapipe deepface pillow imagehash ultralytics
```

### Platform-Specific Setup

**macOS:**
```bash
pip install pyobjc-framework-Quartz
```

**Windows:**
```bash
pip install pywin32
```

**Linux:**
```bash
# Install wmctrl for window detection
sudo apt-get install wmctrl
```

### Models & Data

The first run will auto-download required models (YOLOv8, MediaPipe, DeepFace embeddings). A `knownfaces/` directory will be created to store recognized face crops.

## Usage

### Quick Start

```python
from main import start_sensor_hub
import signal
import sys

hub = start_sensor_hub(debug=False, console_interval=2.0)
signal.signal(signal.SIGINT, lambda *_: hub.stop() or sys.exit(0))

try:
    while not hub.any_dead():
        import time
        time.sleep(1)
finally:
    hub.stop()
```

### API

#### Starting the Hub

```python
hub = start_sensor_hub(
    camera_index=0,           # Camera device index
    debug=False,              # Enable debug output & visualizations
    console_interval=2.0      # Print context every N seconds (None to disable)
)
```

#### Accessing Context

```python
from context_fusion import CONTEXT

# Get current value
activity = CONTEXT.get("activity_state")

# Get full snapshot
version, snapshot = CONTEXT.snapshot()
print(snapshot)

# Register callback for changes
def on_change(key, old_value, new_value):
    print(f"{key}: {old_value} → {new_value}")

CONTEXT.register_callback(on_change)
```

#### Stopping

```python
hub.stop()  # Gracefully shutdown all sensors
```

## Context Variables

The `CONTEXT` store maintains these keys:

### Activity & Presence

| Key | Values | Description |
|-----|--------|-------------|
| `activity_state` | `"active"`, `"idle"`, `"distracted"`, `"away"`, `"unknown"` | Overall user activity level |
| `user_present` | `bool` | Face detected in camera frame |
| `gaze_centered` | `bool` | User looking toward screen |
| `last_active_time` | `float` (timestamp) | Last time user was actively engaged |

### Recognition

| Key | Values | Description |
|-----|--------|-------------|
| `faces_recognized` | `List[str]` | Currently visible recognized faces (excludes "Unknown") |
| `objects_seen` | `List[str]` | YOLO-detected object classes in frame |

### Application Tracking

| Key | Values | Description |
|-----|--------|-------------|
| `active_app` | `str` | Name of foreground application |
| `active_app_start_time` | `float` (timestamp) | When user switched to current app |
| `active_app_duration` | `float` (seconds) | Time spent in current app |

### System Health

| Key | Values | Description |
|-----|--------|-------------|
| `{sensor}_status` | `"running"`, `"crashed"`, `"restarting"`, etc. | Per-sensor status |
| `{sensor}_heartbeat` | `float` (timestamp) | Last heartbeat from sensor |
| `{sensor}_restart_count` | `int` | Number of restart attempts |

## Configuration

### Idle Detection (`idle_detection_combined.py`)

```python
CFG = {
    "IDLE_TIMEOUT": 60,        # Seconds before marking as idle
    "AWAY_TIMEOUT": 300,       # Seconds before marking as away
    "LOOP_INTERVAL": 0.2,      # Frame processing interval
    "GAZE_TOLERANCE": 0.12,    # % of frame width for centered gaze
}
```

### Face Recognition (`obj_face_recog.py`)

```python
CONFIG = {
    "yolo_weights": "yolov8n.pt",
    "face_model_name": "Facenet512",
    "face_distance_metric": "cosine",
    "face_every_n_frames": 1,
    "face_save_cooldown": 20,        # Seconds between saves
    "min_face_size": 50,             # Minimum face crop size (px)
    "min_face_std": 8,               # Blur filter (std threshold)
    "dup_hash_thresh": 5,            # Perceptual hash distance
}
```

### Proactive Outputs (`proactive_outputs.py`)

```python
responder = NaturalResponder(
    break_delay=300,        # Idle > 5 min → suggest break
    break_cooldown=1800,    # 30 min silence after break prompt
    app_cooldown=3600,      # 1 hour silence after app prompt
    log_file="context_log.csv",
    speak_fn=my_tts_function,
)
```

## Output & Logging

### Console Output

When `console_interval` is set, ARGUS prints the full context snapshot periodically:

```
=== Context v142 14:23:45 ===
activity_state: active
user_present: True
gaze_centered: True
active_app: PyCharm
faces_recognized: ['alice', 'bob']
objects_seen: ['person', 'keyboard']
idle_detection_status: running
...
```

### CSV Logging

All context changes are logged to `context_log.csv` with timestamps:

```csv
timestamp,key,value
1699564425.123,activity_state,"active"
1699564426.456,active_app,"Chrome"
1699564427.789,faces_recognized,"[\"alice\"]"
```

### Face Crops

Recognized faces are saved to `knownfaces/<name>/` as 224×224 JPEG files. Duplicates are filtered via perceptual hashing.

## Troubleshooting

### Camera Not Opening

- Check camera permissions (especially on macOS/Linux)
- Verify camera index with `ls /dev/video*` (Linux) or `System Preferences` (macOS)
- Restart camera service or unplug/replug camera

### Face Recognition Always Returns "Unknown"

- Ensure `knownfaces/` directory contains sample images organized by name:
  ```
  knownfaces/
  ├── alice/
  │   ├── alice_001.jpg
  │   └── alice_002.jpg
  └── bob/
      └── bob_001.jpg
  ```
- Verify faces in images are clearly visible and >= 50×50 pixels
- Adjust `min_face_std` if faces are being filtered as too blurry

### High CPU Usage

- Lower `face_every_n_frames` (process fewer frames)
- Reduce `resize_for_face` (downscale before processing)
- Increase `LOOP_INTERVAL` in idle detection config
- Disable debug mode (`debug=False`)

### Hang Detector Killing Sensors

- A sensor hasn't written a heartbeat in 6+ seconds
- Check logs for exceptions or deadlocks
- Verify camera is working (try manual frame capture)
- Increase `_HANG_TIMEOUT` in `SensorHub` if running on slow hardware

## Development

### Adding Custom Context Callbacks

```python
def my_callback(key, old_value, new_value):
    if key == "faces_recognized" and new_value:
        print(f"New faces: {new_value}")

CONTEXT.register_callback(my_callback)
```

### Extending with New Sensors

1. Create a loop function accepting `stop_event: threading.Event`
2. Call `CONTEXT.update(key, value)` to publish state
3. Update heartbeat: `CONTEXT.update("{sensor}_heartbeat", time.time())`
4. Return cleanly on `stop_event.is_set()`
5. Call `hub._spawn(name, target_func, *args, **kwargs)` to register

