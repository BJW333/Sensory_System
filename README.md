# Sensory System

A comprehensive, real-time monitoring system that combines computer vision, facial recognition, activity detection, and app usage tracking to build a contextual understanding of user behavior.

## Overview

Sensory System is a modular sensor hub that runs multiple parallel detection threads, each contributing to a unified context model. It tracks user presence, gaze direction, active applications, recognized faces, and detected objects—all while providing natural, context-aware feedback.

### Core Capabilities

- **Activity Detection**: Monitors user presence and gaze to determine if they're active, idle, distracted, or away using OpenCV Haar Cascades
- **Face Recognition**: Identifies known individuals via DeepFace with persistent memory, temporal voting, and cooldown logic
- **Object Detection**: Uses YOLOv8 to detect objects and people in real-time
- **App Usage Tracking**: Monitors which application is currently in focus (macOS, Windows, Linux)
- **Proactive Feedback**: Delivers context-aware reminders (break suggestions, arrivals, weather alerts)
- **Privacy Mode**: Optional anonymization of sensitive data in logs

## Project Structure

```
project_root/
├── src/
│   ├── camera_manager.py
│   ├── context_fusion.py
│   ├── idle_detection_combined.py
│   ├── obj_face_recog.py
│   ├── app_usage_monitor.py
│   └── proactive_outputs.py
├── data/
│   ├── knownfaces/          # Face recognition database
│   │   ├── person1/
│   │   └── person2/
│   └── debug_output/        # Debug frames (when enabled)
├── main.py                  # Entry point
└── context_log.csv          # Activity log
```

## Architecture

### Components

#### Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | Bootstrap and supervision logic with self-restarting threads |
| `src/camera_manager.py` | Handles camera initialization, frame capture with platform-specific backends |
| `src/context_fusion.py` | Thread-safe key-value store with change callbacks and version tracking |
| `src/idle_detection_combined.py` | Monitors gaze and presence using Haar Cascades; determines activity state |
| `src/obj_face_recog.py` | Detects and identifies people via YOLO + DeepFace with temporal voting |
| `src/app_usage_monitor.py` | Tracks foreground application across platforms |
| `src/proactive_outputs.py` | Generates context-aware reminders and logs changes to CSV |

#### Key Design Patterns

- **Thread-Safe Context Store**: All sensor threads read/write to a shared `CONTEXT` object with atomic operations
- **Supervisor Pattern**: Each sensor runs with automatic restart on crash, heartbeat monitoring (6s timeout), and hang detection
- **Temporal Voting**: Face recognition uses vote aggregation over 1.5s window for stability
- **Batch Updates**: Related context changes are grouped to reduce notification overhead

## Installation

### Prerequisites

- Python 3.8+
- Webcam/camera device
- macOS, Windows, or Linux

### Core Dependencies

```bash
pip install opencv-python numpy pillow imagehash ultralytics deepface tensorflow
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
# Ensure v4l2 support for cameras
sudo apt-get install v4l-utils
```

### Face Recognition Models

The system will auto-download required models on first run:
- YOLOv8n weights (~6MB)
- DeepFace Facenet512 model (~95MB)
- OpenCV Haar Cascade classifiers (included with opencv-python)

### Setting Up Known Faces

Create the directory structure and add sample images:
data/data_training_tools use trainingfaceimagecapture.py to take photos for the knownfaces folder
```bash
mkdir -p data/knownfaces/alice
mkdir -p data/knownfaces/bob
#Add clear face photos (224x224 px minimum) to each folder
```

## Usage

### Quick Start

```python
from main import start_sensor_hub
import signal
import sys
import time

# Start the sensor hub
hub = start_sensor_hub(
    camera_index=0,           # Default camera
    debug=False,              # Set True for visualization windows
    console_interval=2.0      # Print context every 2 seconds
)

# Setup graceful shutdown
signal.signal(signal.SIGINT, lambda *_: hub.stop() or sys.exit(0))
signal.signal(signal.SIGTERM, lambda *_: hub.stop() or sys.exit(0))

try:
    while not hub.any_dead():
        time.sleep(1)
finally:
    hub.stop()
```

### Direct Module Usage

```python
# Using context directly
from src.context_fusion import CONTEXT

# Get current values
activity = CONTEXT.get("activity_state", "unknown")
faces = CONTEXT.get("faces_recognized", [])
app = CONTEXT.get("active_app")

# Register for changes
def on_change(key, old_value, new_value):
    if key == "activity_state":
        print(f"Activity changed: {old_value} → {new_value}")

CONTEXT.register_callback(on_change)

# Batch updates
with CONTEXT.batch():
    CONTEXT.update("key1", value1)
    CONTEXT.update("key2", value2)
```

## Context Variables

The `CONTEXT` store maintains these real-time values:

### Activity & Presence

| Key | Type | Description |
|-----|------|-------------|
| `activity_state` | `str` | `"active"`, `"idle"`, `"distracted"`, `"away"`, `"unknown"`, `"error"` |
| `user_present` | `bool` | Face detected in camera frame |
| `gaze_centered` | `bool` | User looking toward screen (based on eye position) |
| `last_active_time` | `float` | Unix timestamp of last active state |

### Recognition

| Key | Type | Description |
|-----|------|-------------|
| `faces_recognized` | `List[str]` | Currently visible recognized faces (names only) |
| `objects_seen` | `List[str]` | YOLO-detected object classes |

### Application Tracking

| Key | Type | Description |
|-----|------|-------------|
| `active_app` | `str/None` | Name of foreground application |
| `active_app_start_time` | `float` | When user switched to current app |
| `active_app_duration` | `float` | Seconds spent in current app |

### System Health

| Key | Type | Description |
|-----|------|-------------|
| `{sensor}_status` | `str` | `"starting"`, `"running"`, `"crashed"`, `"restarting"`, `"stopped"`, `"finished"` |
| `{sensor}_last_beat` | `float` | Last heartbeat timestamp |
| `{sensor}_restart_count` | `int` | Number of automatic restarts |
| `sensorhub_shutdown` | `bool` | Global shutdown flag |

## Configuration

### Idle Detection (`src/idle_detection_combined.py`)

```python
CFG = {
    "IDLE_TIMEOUT": 60,              # Seconds before marking idle
    "AWAY_TIMEOUT": 300,             # Seconds before marking away
    "LOOP_INTERVAL": 0.1,            # Processing interval (10 FPS)
    "GAZE_TOLERANCE": 0.15,          # Eye position tolerance (±15% of face width)
    "MIN_FACE_WIDTH": 50,            # Minimum face size in pixels
    "FACE_SCALE": 1.05,              # Haar cascade scale factor
    "FACE_MIN_NEIGHBORS": 5,         # Detection confidence
    "PUPIL_DETECTION": False,        # Advanced gaze estimation
    "HEAD_POSE_FALLBACK": True,      # Use head pose when eyes not visible
}
```

### Face Recognition (`src/obj_face_recog.py`)

```python
CONFIG = {
    "yolo_weights": "yolov8n.pt",
    "obj_conf_thresh": 0.55,
    "face_model_name": "Facenet512",      # DeepFace model
    "face_distance_metric": "cosine",
    "face_distance_threshold": 0.40,      # Recognition threshold
    "min_votes_for_confirm": 1,           # Temporal voting requirement
    "vote_history_window": 1.5,           # Seconds to aggregate votes
    "face_save_cooldown": 20,             # Seconds between saving same person
    "face_folder_max": 30,                # Max images per person
    "min_face_size": 50,                  # Minimum face crop size
    "min_face_std": 8,                    # Blur detection threshold
    "dup_hash_thresh": 5,                 # Perceptual hash difference
    "face_every_n_frames": 1,             # Process every Nth frame
    "save_debug_images": False,           # Save annotated frames
}
```

### App Monitor (`src/app_usage_monitor.py`)

```python
CHECK_INTERVAL = 5.0  # Seconds between application polls
```

### Proactive Outputs (`src/proactive_outputs.py`)

```python
responder = NaturalResponder(
    break_delay=300,              # 5 min idle → suggest break
    break_cooldown=1800,          # 30 min between break prompts
    app_cooldown=3600,            # 1 hour between app prompts
    face_prompt_cooldown=600,     # 10 min between same face greetings
    left_announcement_delay=3.0,  # Wait before "left" announcement
    log_file="context_log.csv",
    max_log_size=50_000_000,      # 50MB before rotation
    privacy_mode=False,           # Set True to anonymize logs
    speak_fn=my_tts_function,
)
```

## Output & Logging

### Console Output

With `console_interval` enabled:

```
=== Context v142 14:23:45 ===
activity_state: active
user_present: True
gaze_centered: True
active_app: Chrome
active_app_duration: 234.5
faces_recognized: ['alice', 'bob']
objects_seen: ['person', 'keyboard', 'mouse']
idle_detection_status: running
obj_person_recog_status: running
app_usage_monitor_status: running
```

### CSV Logging

All context changes logged to `context_log.csv`:

```csv
timestamp,key,value
1699564425.123,activity_state,"active"
1699564426.456,active_app,"Chrome"
1699564427.789,faces_recognized,"[\"alice\"]"
```

Log rotation occurs at 50MB, keeping last 5 files.

### Face Database

Recognized faces saved to `data/knownfaces/{name}/`:
- 224×224 JPEG format
- Automatic deduplication via perceptual hashing
- Maximum 30 images per person (oldest deleted)
- 20-second cooldown between saves

### Debug Output

When `save_debug_images: True`, annotated frames saved to `data/debug_output/`:
- Shows person bounding boxes (blue)
- Face detections with IoU scores
- Recognition results with confidence

## Troubleshooting

### Camera Issues

**Camera not opening:**
- Check permissions: `System Preferences > Security & Privacy > Camera` (macOS)
- Verify device: `ls /dev/video*` (Linux) or try different `camera_index` values
- Test with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**Poor performance:**
- Reduce `resize_for_face` to 0.5 for faster processing
- Increase `face_every_n_frames` to process fewer frames
- Disable `PUPIL_DETECTION` in idle detection config

### Face Recognition

**Always returns "Unknown":**
- Verify `data/knownfaces/` structure with person folders
- Check face images are clear, well-lit, ≥224×224 pixels
- Lower `face_distance_threshold` (e.g., 0.45) for looser matching
- Enable `save_debug_images` to inspect detections
- Check DeepFace cache: delete `.pkl` files in `data/knownfaces/`

**Inconsistent recognition:**
- Increase `min_votes_for_confirm` to 2 for stability
- Adjust `vote_history_window` for longer temporal averaging
- Ensure consistent lighting conditions

### High Resource Usage

**CPU optimization:**
- Set `face_every_n_frames: 3` to process every 3rd frame
- Increase `LOOP_INTERVAL` to 0.2 in idle detection
- Set `resize_for_face: 0.5` for smaller processing size
- Disable debug mode and visualization windows

**Memory issues:**
- Reduce `face_folder_max` to limit stored face images
- Lower `max_log_size` for more frequent rotation
- Clear `data/debug_output/` regularly if debug enabled

### Sensor Crashes

**Heartbeat timeout:**
- Check logs for exceptions in sensor threads
- Increase `_HANG_TIMEOUT` in `SensorHub` (default 6s)
- Verify camera isn't being accessed by another application

**Restart loops:**
- Check `{sensor}_traceback` in context for error details
- Increase `_RESTART_DELAY` for slower retry (default 3s)
- Monitor `{sensor}_restart_count` for problematic sensors

## Platform Notes

### macOS
- Requires camera permissions granted to Terminal/IDE
- AVFoundation backend used by default
- Quartz framework required for app monitoring

### Windows
- DirectShow (DSHOW) backend for camera
- Win32 API for window detection
- May require admin rights for some features

### Linux
- V4L2 backend for camera access
- wmctrl required for window manager interaction
- X11 environment assumed (Wayland partially supported)

## Development

### Adding Custom Sensors

1. Create sensor function with heartbeat updates:

```python
def my_sensor_loop(stop_event=None):
    while stop_event is None or not stop_event.is_set():
        # Update heartbeat
        CONTEXT.update("my_sensor_last_beat", time.time())
        
        # Do work
        data = collect_data()
        CONTEXT.update("my_data", data)
        
        time.sleep(0.1)
```

2. Register in `main.py`:

```python
self._spawn("my_sensor", my_sensor_loop)
```

### Custom Callbacks

```python
def handle_arrivals(key, old_value, new_value):
    if key == "faces_recognized":
        new_faces = set(new_value) - set(old_value or [])
        for face in new_faces:
            notify_arrival(face)

CONTEXT.register_callback(handle_arrivals)
```

## License & Contributing

MIT License 

## Known Issues

- Face recognition may struggle with masks or extreme angles
- App detection on Linux requires X11 (limited Wayland support)
- High CPU usage on older hardware (consider reducing frame rates)
- Some USB cameras may disconnect/reconnect causing temporary failures

## Support

For issues or questions, please check the logs first:
- Sensor status: Check `{sensor}_status` in context
- Error details: Look for `{sensor}_traceback` entries
- Debug output: Enable `debug=True` and check console
- CSV logs: Review `context_log.csv` for state history