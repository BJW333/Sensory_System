# obj_face_recog.py - FIXED VERSION with extensive debugging
from __future__ import annotations

import cv2
import numpy as np
import os
import threading
import time
import io
import logging
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from deepface import DeepFace
from PIL import Image
import imagehash
from ultralytics import YOLO

from context_fusion import CONTEXT

###############################################################################
# CONFIGURATION
###############################################################################

CONFIG: Dict[str, int | float | bool | str] = {
    "yolo_weights": "yolov8n.pt",
    "obj_conf_thresh": 0.55,
    "face_model_name": "Facenet512",
    "face_distance_metric": "cosine",
    "face_every_n_frames": 1,
    "face_save_cooldown": 20,
    "face_folder_max": 40,
    "dup_hash_thresh": 5,
    "min_face_std": 8,
    "min_face_size": 50,
    "sleep_between_frames": 0.005,
    "resize_for_face": 0.75,
    "face_label_timeout": 6,
    # 🆕 NEW DEBUG OPTIONS
    "save_debug_images": True,  # Save what camera sees
    "debug_image_interval": 10,  # Save every N frames
}

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = SCRIPT_DIR / "knownfaces"
KNOWN_FACES_DIR.mkdir(exist_ok=True)

# 🆕 NEW: Debug output directory
DEBUG_DIR = SCRIPT_DIR / "debug_output"
DEBUG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format="[ARGUS %(levelname)s] %(message)s",
    level=logging.INFO,
)

# Detector backend selection
_AVAILABLE_DETECTORS = ["mediapipe", "opencv"]
FACE_DETECTOR_BACKEND = None
for det in _AVAILABLE_DETECTORS:
    try:
        logging.info(f"Testing DeepFace detector backend: {det}")
        _ = DeepFace.extract_faces(
            img_path=np.zeros((250, 250, 3), dtype=np.uint8),
            detector_backend=det,
            enforce_detection=False,
        )
        FACE_DETECTOR_BACKEND = det
        logging.info(f"✅ Using DeepFace detector backend: {FACE_DETECTOR_BACKEND}")
        break
    except Exception as e:
        logging.warning(f"Detector backend '{det}' failed: {e}")

if FACE_DETECTOR_BACKEND is None:
    raise RuntimeError("No supported face-detector backends available")

# Load YOLO
YOLO_MODEL = YOLO(CONFIG["yolo_weights"])
logging.info("✅ YOLO model loaded")

# Warm up DeepFace
_known_imgs = [p for p in KNOWN_FACES_DIR.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
if _known_imgs:
    logging.info(f"Warming up DeepFace with {len(_known_imgs)} known faces...")
    DeepFace.find(
        img_path=str(_known_imgs[0]),
        db_path=str(KNOWN_FACES_DIR),
        model_name=CONFIG["face_model_name"],
        detector_backend=FACE_DETECTOR_BACKEND,
        distance_metric=CONFIG["face_distance_metric"],
        enforce_detection=False,
        silent=True,
    )
    logging.info("✅ DeepFace warmed up")
else:
    logging.warning(f"⚠️ No images in {KNOWN_FACES_DIR} - all faces will be 'Unknown'")

# Global state
last_save_time: Dict[str, float] = {}
state_lock = threading.Lock()

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))

def cull_old_faces(folder: Path, max_images: int) -> None:
    files = sorted(
        (f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".png")),
        key=lambda p: p.stat().st_mtime,
    )
    for f in files[:-max_images]:
        f.unlink(missing_ok=True)

def is_duplicate_face(cropped: np.ndarray, folder: Path, *, hash_thresh: int, check_last: int = 20) -> bool:
    new_hash = imagehash.phash(Image.fromarray(cropped))
    recent = sorted(
        (f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:check_last]
    for f in recent:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img_resized = cv2.resize(img, cropped.shape[:2][::-1])
        if abs(new_hash - imagehash.phash(Image.fromarray(img_resized))) < hash_thresh:
            return True
    return False

def save_cropped_face(img_or_frame, name: str, bbox=None, already_cropped=False):
    if already_cropped:
        cropped = cv2.resize(img_or_frame, (224, 224))
    else:
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        h, w = img_or_frame.shape[:2]
        x1, y1 = _clamp(x1, 0, w - 1), _clamp(y1, 0, h - 1)
        x2, y2 = _clamp(x2, x1 + 1, w), _clamp(y2, y1 + 1, h)
        cropped = img_or_frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return
        cropped = cv2.resize(cropped, (224, 224))

    if cropped.shape[0] < CONFIG["min_face_size"] or cropped.shape[1] < CONFIG["min_face_size"]:
        logging.debug(f"❌ Face too small: {cropped.shape[:2]}")
        return

    face_std = np.std(cropped)
    if face_std < CONFIG["min_face_std"]:
        logging.debug(f"❌ Face too blurry: std={face_std:.1f}")
        return

    folder = KNOWN_FACES_DIR / name
    folder.mkdir(exist_ok=True)

    now = time.time()
    with state_lock:
        if now - last_save_time.get(name, 0) < CONFIG["face_save_cooldown"]:
            logging.debug("❌ Cooldown active")
            return
        last_save_time[name] = now

    if is_duplicate_face(cropped, folder, hash_thresh=CONFIG["dup_hash_thresh"]):
        logging.debug("❌ Duplicate face")
        return

    cull_old_faces(folder, CONFIG["face_folder_max"])
    filename = folder / f"{name}_{int(now*1000)}.jpg"
    cv2.imwrite(str(filename), cropped)
    logging.info(f"💾 Saved: {filename}")

###############################################################################
# DETECTION FUNCTIONS
###############################################################################

def detect_objects(image: np.ndarray, conf_thresh: float) -> Dict[str, List]:
    results = YOLO_MODEL(image, verbose=False)[0]
    labels, boxes, scores = [], [], []
    for b in results.boxes:
        score = float(b.conf[0])
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        boxes.append((y1, x1, y2, x2))
        labels.append(results.names[int(b.cls[0])])
        scores.append(score)
    return {"labels": labels, "boxes": boxes, "scores": scores}

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)

# 🆕 NEW: Save debug visualization
def save_debug_frame(frame: np.ndarray, frame_count: int, faces_data: List[dict], person_boxes: List, recognized_names: List[str]):
    """Save annotated frame showing what the system sees"""
    debug_frame = frame.copy()
    
    # Draw person boxes in blue
    for (py1, px1, py2, px2) in person_boxes:
        cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(debug_frame, "PERSON", (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw face boxes with recognition results
    for i, face_data in enumerate(faces_data):
        x1, y1, x2, y2 = face_data['bbox']
        iou_val = face_data.get('max_iou', 0)
        name = recognized_names[i] if i < len(recognized_names) else "Unknown"
        
        # Color based on IoU threshold
        color = (0, 255, 0) if iou_val >= 0.40 else (0, 0, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{name} (IoU:{iou_val:.2f})"
        cv2.putText(debug_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add info text
    info_text = [
        f"Frame: {frame_count}",
        f"Persons: {len(person_boxes)}",
        f"Faces: {len(faces_data)}",
        f"Recognized: {len([n for n in recognized_names if n != 'Unknown'])}",
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(debug_frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Save
    output_path = DEBUG_DIR / f"frame_{frame_count:06d}.jpg"
    cv2.imwrite(str(output_path), debug_frame)
    logging.info(f"🖼️ Saved debug frame: {output_path}")

###############################################################################
# MAIN LOOP
###############################################################################

def obj_person_recog_loop(
    camera_manager,
    *,
    show_debug: bool = False,
    stop_event: Optional[threading.Event] = None,
):
    """Main recognition loop with enhanced debugging"""
    
    if show_debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    last_objects: List[str] = []
    last_faces: List[str] = []
    last_known_faces: Dict[str, Tuple[int, int, int, int, float]] = {}
    frame_count = 0
    
    logging.info("🚀 Starting object/person recognition loop")
    logging.info(f"📁 Known faces directory: {KNOWN_FACES_DIR}")
    logging.info(f"📁 Debug output directory: {DEBUG_DIR}")
    
    try:
        while stop_event is None or not stop_event.is_set():
            start_ts = time.time()
            CONTEXT.update("obj_person_recog_last_beat", start_ts)
            
            frame = camera_manager.get_frame()
            if frame is None:
                logging.warning("⚠️ No frame from camera")
                CONTEXT.update("objects_seen", [])
                CONTEXT.update("faces_recognized", [])
                time.sleep(max(0, CONFIG["sleep_between_frames"]))
                continue
            
            # Log frame info periodically
            if frame_count % 300 == 0:
                logging.info(f"📷 Camera frame: {frame.shape}, dtype: {frame.dtype}")
            
            # Resize for processing
            if CONFIG["resize_for_face"] < 1.0:
                frame_small = cv2.resize(
                    frame, None, 
                    fx=CONFIG["resize_for_face"], 
                    fy=CONFIG["resize_for_face"], 
                    interpolation=cv2.INTER_AREA
                )
            else:
                frame_small = frame
            
            # Object detection
            objs = detect_objects(frame_small, CONFIG["obj_conf_thresh"])
            
            # Scale back to full resolution
            scale = (1.0 / CONFIG["resize_for_face"]) if CONFIG["resize_for_face"] < 1.0 else 1.0
            #person_boxes = [
            #    (int(by*scale), int(bx*scale), int(by2*scale), int(bx2*scale))
            #    for (by, bx, by2, bx2), lbl in zip(objs["boxes"], objs["labels"])
            #    if lbl == "person"
            #]
            # Expand person boxes by 20% in all directions
            person_boxes = []
            for (by, bx, by2, bx2), lbl in zip(objs["boxes"], objs["labels"]):
                if lbl == "person":
                    # Scale to full resolution
                    by, bx, by2, bx2 = int(by*scale), int(bx*scale), int(by2*scale), int(bx2*scale)
                    
                    # Expand box by 20%
                    height, width = by2 - by, bx2 - bx
                    by = max(0, by - int(height * 0.2))
                    bx = max(0, bx - int(width * 0.2))
                    by2 = by2 + int(height * 0.2)
                    bx2 = bx2 + int(width * 0.2)
                    
                    person_boxes.append((by, bx, by2, bx2))

            # Face recognition
            recognized_names: List[str] = []
            faces_debug_data = []  # 🆕 For debug visualization
            
            if frame_count % CONFIG["face_every_n_frames"] == 0:
                logging.debug(f"🔍 Frame {frame_count}: Detecting faces...")
                
                # 🔧 FIX: Convert BGR to RGB for DeepFace
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                
                try:
                    faces = DeepFace.extract_faces(
                        img_path=frame_rgb,  # ✅ Now using RGB
                        detector_backend=FACE_DETECTOR_BACKEND,
                        enforce_detection=False,
                    )
                    logging.debug(f"   Found {len(faces)} face(s)")
                except Exception as e:
                    logging.error(f"❌ Face extraction failed: {e}")
                    faces = []
                
                for idx, face in enumerate(faces):
                    r = face["facial_area"]
                    x1, y1, x2, y2 = r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]
                    
                    # Scale back to full resolution
                    if frame_small is not frame:
                        x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
                    
                    # ✅ NEW: Check if face center is inside any person box (better for multiple people)
                    face_center_x = (x1 + x2) // 2
                    face_center_y = (y1 + y2) // 2
                    
                    inside_person = False
                    closest_person_box = None
                    min_distance = float('inf')
                    
                    for person_box in person_boxes:
                        py1, px1, py2, px2 = person_box
                        
                        # Check if face center is inside this person box
                        if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                            inside_person = True
                            closest_person_box = person_box
                            break
                        
                        # Calculate distance to this person box (for logging)
                        center_dist = abs(face_center_x - (px1 + px2) // 2) + abs(face_center_y - (py1 + py2) // 2)
                        if center_dist < min_distance:
                            min_distance = center_dist
                            closest_person_box = person_box
                    
                    # For debug: still calculate IoU for display
                    if closest_person_box:
                        max_iou = _iou((x1, y1, x2, y2), closest_person_box)
                    else:
                        max_iou = 0
                    
                    faces_debug_data.append({
                        'bbox': (x1, y1, x2, y2),
                        'max_iou': max_iou,
                        'face_img': face["face"]
                    })
                    
                    logging.debug(f"   Face {idx}: bbox=({x1},{y1},{x2},{y2}), max_IoU={max_iou:.3f}")
                    
                    # ✅ NEW: Reject if face center not in any person box
                    if not inside_person:
                        logging.debug(f"   ❌ Face {idx} rejected: face center not inside any person box")
                        name = "Unknown"
                        recognized_names.append(name)
                        continue
                    
                    logging.debug(f"   ✅ Face {idx} passed IoU check")
                    
                    # Try to recognize
                    img = face["face"].astype(np.uint8)
                    THRESH = 0.5
                    
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = tmp.name
                        # Save as BGR (cv2.imwrite expects BGR)
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(tmp_path, img_bgr)
                        
                        try:
                            with io.StringIO() as buf, redirect_stdout(buf):
                                result = DeepFace.find(
                                    img_path=tmp_path,
                                    db_path=str(KNOWN_FACES_DIR),
                                    model_name=CONFIG["face_model_name"],
                                    detector_backend=FACE_DETECTOR_BACKEND,
                                    distance_metric=CONFIG["face_distance_metric"],
                                    threshold=THRESH,
                                    enforce_detection=False,
                                    silent=True,
                                )
                            
                            if result and not result[0].empty:
                                distance = float(result[0].iloc[0]["distance"])
                                if distance <= THRESH:
                                    name = Path(result[0].iloc[0]["identity"]).parent.name
                                    #logging.info(f"   ✅ Recognized: {name} (distance: {distance:.3f})")
                                    save_cropped_face(img_bgr, name, already_cropped=True)
                                    last_known_faces[name] = (x1, y1, x2, y2, time.time())
                                else:
                                    logging.debug(f"   ❌ Distance {distance:.3f} > threshold {THRESH}")
                                    name = "Unknown"
                            else:
                                logging.debug(f"   ❌ No matches in database")
                                name = "Unknown"
                        
                        except Exception as e:
                            logging.error(f"   ❌ Recognition error: {e}")
                            name = "Unknown"
                        
                        finally:
                            os.unlink(tmp_path)
                    
                    recognized_names.append(name)
            
            # 🆕 Save debug visualization
            if CONFIG["save_debug_images"] and frame_count % CONFIG["debug_image_interval"] == 0:
                save_debug_frame(frame, frame_count, faces_debug_data, person_boxes, recognized_names)
            
            # Update context
            objects_now = sorted(set(objs["labels"]))
            faces_now = sorted({n for n in recognized_names if n != "Unknown"})
            
            with state_lock:
                if objects_now != last_objects or faces_now != last_faces:
                    with CONTEXT.batch():
                        CONTEXT.update("objects_seen", objects_now)
                        CONTEXT.update("faces_recognized", faces_now)
                    last_objects = objects_now
                    last_faces = faces_now
            
            # Show debug window
            if show_debug and frame_count % CONFIG["face_every_n_frames"] == 0:
                debug_frame = frame.copy()
                for (py1, px1, py2, px2) in person_boxes:
                    cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                for face_data, name in zip(faces_debug_data, recognized_names):
                    x1, y1, x2, y2 = face_data['bbox']
                    color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imshow("ARGUS Recognition", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            frame_count += 1
            
            # Clean stale memory
            if frame_count % 60 == 0:
                now = time.time()
                last_known_faces = {
                    name: (x1, y1, x2, y2, ts)
                    for name, (x1, y1, x2, y2, ts) in last_known_faces.items()
                    if now - ts < CONFIG["face_label_timeout"]
                }
            
            # Sleep
            elapsed = time.time() - start_ts
            time.sleep(max(0, CONFIG["sleep_between_frames"] - elapsed))
    
    except KeyboardInterrupt:
        logging.info("⏹️ Interrupted by user")
    except Exception as e:
        logging.exception(f"💥 Loop crashed: {e}")
    finally:
        cv2.destroyAllWindows()
        logging.info("⏹️ Recognition loop stopped")

###############################################################################
# THREAD WRAPPER
###############################################################################

def start_obj_person_recog_thread(
    camera_manager, 
    *, 
    show_debug: bool = False
) -> Tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    t = threading.Thread(
        target=obj_person_recog_loop,
        args=(camera_manager,),
        kwargs={"show_debug": show_debug, "stop_event": stop_event},
        daemon=True,
    )
    t.start()
    return t, stop_event