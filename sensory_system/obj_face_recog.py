# obj_person_recog_fixed_v2.py – ARGUS object & face recognition loop (improved)
# ---------------------------------------------------------------------------
# Key upgrades vs. the user-supplied draft (2025-06-17):
#   • Robust face-detector selection – prefers RetinaFace → Mediapipe → HaarCascade
#   • Converts camera BGR frames → RGB before DeepFace calls (color-space bug)
#   • Uses DeepFace.find() warm-up on a *real* face instead of a zero-array (fixed
#     buggy "dummy" representation build that sometimes produced empty vectors)
#   • Optional dynamic frame resize for faster DeepFace extraction on high-res cams
#   • Added visual face-box overlay in debug mode to verify detection visually
#   • Reduced CONFIG["face_every_n_frames"] default to 3 for snappier response
#   • Hardened bbox cropping (clamp within frame and ignore out-of-bounds)
#   • Extra logging for detector backend, warm-up status, and per-frame timings
# ---------------------------------------------------------------------------


#WHAT TO ADD NEXT
# ---------------------------------------------------------------------------
#   • Add a "face recognition cooldown" to prevent rapid re-saving of the same face
#Face disappears? If a person box overlaps the last known face location, 
#we assume it’s the same person — greatly reducing flicker and misidentification.


from __future__ import annotations

import cv2
import numpy as np
import os
import threading
import time
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from deepface import DeepFace
from PIL import Image
import imagehash
from ultralytics import YOLO

from context_fusion import CONTEXT  # project-local singleton

###############################################################################
# CONFIGURATION
###############################################################################

CONFIG: Dict[str, int | float | bool | str] = {
    # YOLO (object detection)
    "yolo_weights": "yolov8n.pt",            # path or model name
    "obj_conf_thresh": 0.55,     #orgninal value 0.35     # a bit lower to catch more objs

    # Face processing
    "face_model_name": "Facenet512",
    "face_distance_metric": "cosine",
    "face_every_n_frames": 1,                 #faster recognition cadence
    "face_save_cooldown": 20,                 #secs between saves of same person
    "face_folder_max": 40,                    #keep last N imgs per person
    "dup_hash_thresh": 5,                     #perceptual-hash distance
    "min_face_std": 8,      #orginal value 12      # blur filter (np.std threshold)
    # "min_face_size": 60,                      # px
    "min_face_size": 50,    #px

    # Performance tuning
    "sleep_between_frames": 0.005,            # lighter nap; will auto-adapt
    "resize_for_face": 0.75,                  # <1.0 downsamples frame before
    
    "face_label_timeout": 6, # secs to keep a face label after last detection
}


#INITIALISATION

logging.basicConfig(
    format="[ARGUS %(levelname)s] %(message)s",
    level=logging.INFO,
)

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = SCRIPT_DIR / "knownfaces"
KNOWN_FACES_DIR.mkdir(exist_ok=True)

# Choose the best available DeepFace detector backend (accuracy ⇢ speed)
_AVAILABLE_DETECTORS = ["mediapipe", "opencv"]

FACE_DETECTOR_BACKEND = None
for det in _AVAILABLE_DETECTORS:
    try:
        logging.info(f"Testing DeepFace detector backend: {det}")
        _ = DeepFace.extract_faces(
            img_path=np.zeros((250, 250, 3), dtype=np.uint8),  # dummy black image
            detector_backend=det,
            enforce_detection=False,
        )
        FACE_DETECTOR_BACKEND = det
        logging.info(f"Using DeepFace detector backend: {FACE_DETECTOR_BACKEND}")
        break
    except Exception as e:
        logging.warning(f"Detector backend '{det}' failed: {e}")

if FACE_DETECTOR_BACKEND is None:
    raise RuntimeError("No supported face-detector backends are available for DeepFace.")

# Load YOLOv8 once
YOLO_MODEL = YOLO(CONFIG["yolo_weights"])

# Build DeepFace reps *once* using a real image (important for cosine distances)
_known_imgs = [p for p in KNOWN_FACES_DIR.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
if _known_imgs:
    logging.info("Warming-up DeepFace representations (%d images) …", len(_known_imgs))
    DeepFace.find(
        img_path=str(_known_imgs[0]),  # warm-up with first image
        db_path=str(KNOWN_FACES_DIR),
        model_name=CONFIG["face_model_name"],
        detector_backend=FACE_DETECTOR_BACKEND,
        distance_metric=CONFIG["face_distance_metric"],
        enforce_detection=False,
        silent=True,
    )
else:
    logging.warning("No images in %s – face recognition will always return Unknown", KNOWN_FACES_DIR)

# Global state guarded by a lock
last_save_time: Dict[str, float] = {}
state_lock = threading.Lock()


#UTILITY FUNCTIONS

def _clamp(val: int, lo: int, hi: int) -> int:  # small helper
    return max(lo, min(val, hi))

def cull_old_faces(folder: Path, max_images: int) -> None:
    """Keep only the most recent *max_images* in *folder*."""
    files = sorted(
        (f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".png")),
        key=lambda p: p.stat().st_mtime,
    )
    for f in files[:-max_images]:
        f.unlink(missing_ok=True)
        logging.debug("Culled old face image: %s", f)

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
            logging.debug("Duplicate face %s skipped (phash diff < %d)", f.name, hash_thresh)
            return True
    return False

#def save_cropped_face(frame: np.ndarray, bbox: Tuple[int, int, int, int], name: str) -> None:
def save_cropped_face(img_or_frame, name: str, bbox=None, already_cropped=False):
    """
    Save a 224×224 face crop into knownfaces/<name>/…jpg and emit DEBUG
    messages whenever a crop is rejected.
    """
    # ── 1. produce a (224,224,3) crop ─────────────────────────────────────
    if already_cropped:
        cropped = cv2.resize(img_or_frame, (224, 224))
    else:
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        h, w           = img_or_frame.shape[:2]
        x1, y1 = _clamp(x1, 0, w - 1), _clamp(y1, 0, h - 1)
        x2, y2 = _clamp(x2, x1 + 1, w), _clamp(y2, y1 + 1, h)
        cropped = img_or_frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return
        cropped = cv2.resize(cropped, (224, 224))

    # ── 2. quality / duplicate / cooldown gates ──────────────────────────
    if cropped.shape[0] < CONFIG["min_face_size"] or \
       cropped.shape[1] < CONFIG["min_face_size"]:
        logging.debug("skip save: face too small (%s×%s)", *cropped.shape[:2])
        return

    face_std = np.std(cropped)
    if face_std < CONFIG["min_face_std"]:
        logging.debug("skip save: face too blurry/low-contrast (std %.1f)", face_std)
        return

    folder = KNOWN_FACES_DIR / name
    folder.mkdir(exist_ok=True)

    now = time.time()
    with state_lock:
        if now - last_save_time.get(name, 0) < CONFIG["face_save_cooldown"]:
            logging.debug("skip save: cool-down")
            return
        last_save_time[name] = now

    if is_duplicate_face(cropped, folder, hash_thresh=CONFIG["dup_hash_thresh"]):
        logging.debug("skip save: duplicate phash")
        return

    cull_old_faces(folder, CONFIG["face_folder_max"])

    # ── 3. write file ─────────────────────────────────────────────────────
    filename = folder / f"{name}_{int(now*1000)}.jpg"
    cv2.imwrite(str(filename), cropped)
    logging.info("Saved new face image: %s", filename)

###############################################################################
# CORE DETECTION HELPERS
###############################################################################

def detect_objects(image: np.ndarray, conf_thresh: float) -> Dict[str, List]:
    """YOLOv8 inference wrapper – returns dict of labels / boxes / scores"""
    results = YOLO_MODEL(image, verbose=False)[0]
    labels, boxes, scores = [], [], []
    for b in results.boxes:
        score = float(b.conf[0])
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        boxes.append((y1, x1, y2, x2))  # (ymin,xmin,ymax,xmax)
        labels.append(results.names[int(b.cls[0])])
        scores.append(score)
    return {"labels": labels, "boxes": boxes, "scores": scores}

###############################################################################
# MAIN LOOP
###############################################################################
def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """intersection-over-union of two (x1,y1,x2,y2) boxes"""
    xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)

def obj_person_recog_loop(
    camera_manager,
    *,
    show_debug: bool = False,
    stop_event: Optional[threading.Event] = None,
):
    """Continuously pull frames, detect objects & faces, and update CONTEXT."""

    if show_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    last_objects: List[str] = []
    last_faces: List[str] = []
    last_known_faces: Dict[str, Tuple[int, int, int, int, float]] = {}  # name → (x1, y1, x2, y2, timestamp)
    frame_count = 0

    try:
        while stop_event is None or not stop_event.is_set():
            start_ts = time.time()
            CONTEXT.update("obj_person_recog_last_beat", start_ts)
            frame = camera_manager.get_frame()
            if frame is None:
                CONTEXT.update("objects_seen", [])
                CONTEXT.update("faces_recognized", [])
                #if no frame wait and try again smoothly sleeps
                elapsed = time.time() - start_ts
                sleep_time = max(0, CONFIG["sleep_between_frames"] - elapsed)
                time.sleep(sleep_time)
                continue

            # Optional resize to speed up detect / extract
            if CONFIG["resize_for_face"] < 1.0:
                frame_small = cv2.resize(
                    frame, None, fx=CONFIG["resize_for_face"], fy=CONFIG["resize_for_face"], interpolation=cv2.INTER_AREA
                )
            else:
                frame_small = frame

            # ── Object Detection ────────────────────────────────────────────
            objs = detect_objects(frame_small, CONFIG["obj_conf_thresh"])
            if show_debug:
                for (y1, x1, y2, x2), label, score in zip(objs["boxes"], objs["labels"], objs["scores"]):
                    # scale boxes back to full-res if we downsized
                    if frame_small is not frame:
                        scale = 1 / CONFIG["resize_for_face"]
                        y1, x1, y2, x2 = [int(v * scale) for v in (y1, x1, y2, x2)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label}:{int(score*100)}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 1)

            # ── Face Detection & Recognition (every N frames) ───────────────
            recognized_names: List[str] = []
            if frame_count % CONFIG["face_every_n_frames"] == 0:
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                faces = DeepFace.extract_faces(
                    img_path=frame_rgb,
                    detector_backend=FACE_DETECTOR_BACKEND,
                    enforce_detection=False,
                )
                # collect YOLO person boxes in full-res coords
                person_boxes = [
                    (int(bx*scale), int(by*scale), int(bx2*scale), int(by2*scale))
                    for (by, bx, by2, bx2), lbl in zip(objs["boxes"], objs["labels"])
                    if lbl == "person"
                    for scale in [(1/CONFIG["resize_for_face"]) if frame_small is not frame else (1,)]
                ]

                for face in faces:
                    img = face["face"].astype(np.uint8)
                    r = face["facial_area"]
                    # Scale bbox back to full-res if needed
                    x1, y1, x2, y2 = r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]
                    if frame_small is not frame:
                        scale = 1 / CONFIG["resize_for_face"]
                        x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
                    # Require ≥40 % IoU with at least one YOLO person box
                    if not any(_iou((x1,y1,x2,y2), pb) >= 0.40 for pb in person_boxes):
                        if show_debug:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 1)
                            
                        #recognized_names.append("Unknown")
                        # Default to Unknown
                        name = "Unknown"

                        # Try to infer identity based on recent position
                        for prev_name, (lx1, ly1, lx2, ly2, ts) in last_known_faces.items():
                            if time.time() - ts < CONFIG["face_save_cooldown"] and _iou((x1, y1, x2, y2), (lx1, ly1, lx2, ly2)) >= 0.40:
                                name = prev_name
                                break

                        recognized_names.append(name)
                        continue

                    
                    #img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    #this is meant to show me and save a image of what the model is trying to recognize
                    #it also shows me what the model is seeing
                    #timestamp = int(time.time() * 1000)
                    #attempt_path = SCRIPT_DIR / f"recognition_attempt_{timestamp}.jpg"
                    #cv2.imwrite(str(attempt_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))  # RGB → BGR for cv2.imwrite
                    #logging.info(f"Saved recognition attempt image: {attempt_path}")
                    
                    #print("[DEBUG] img shape:", img.shape, "dtype:", img.dtype, "min/max:", img.min(), img.max())
                    
                    THRESH = 0.38   # Facenet512+cosine sweet-spot (0.35–0.40)
                    with io.StringIO() as buf, redirect_stdout(buf):  # silence DeepFace logs
                        result = DeepFace.find(
                            img_path=img,
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
                            recognized_names.append(name)
                            save_cropped_face(face["face"], name, already_cropped=True)
                            # Update last known face position
                            last_known_faces[name] = (x1, y1, x2, y2, time.time()) # save last known face position this is for if we lose the face in recog 
                        else:
                            recognized_names.append("Unknown")
                    else:
                        recognized_names.append("Unknown")

                    if show_debug:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, recognized_names[-1], (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

            if not recognized_names and person_boxes:
                now = time.time()
                for (px1, py1, px2, py2) in person_boxes:
                    best_match = None
                    best_iou   = 0.0
                    for prev_name, (lx1, ly1, lx2, ly2, ts) in last_known_faces.items():
                        if now - ts > CONFIG["face_label_timeout"]:
                            continue
                        iou = _iou((px1, py1, px2, py2), (lx1, ly1, lx2, ly2))
                        if iou > 0.40 and iou > best_iou:
                            best_match, best_iou = prev_name, iou
                    if best_match:
                        recognized_names.append(best_match)
                
                recognized_names = list(dict.fromkeys(recognized_names))
            
            # ── CONTEXT updates (only on change) ────────────────────────────
            objects_now = sorted(set(objs["labels"]))
            faces_now = sorted({n for n in recognized_names if n != "Unknown"})

            #this below is the objects updates
            if objects_now != last_objects:
                with state_lock:
                    CONTEXT.update("objects_seen", objects_now)
                    last_objects = objects_now
                    
            #this below is the faces updates
            #maybe put it so that if blake_weiss is recognized
            #how should it treat a person it just recognized when the face becomes 
            #unrecognizable for a few frames but a person object is still present.
            #it should return blaek_weiss as the recognized person still maybe
            
            #basiclly what im saying is that
            #if it just saw Blake’s face here and 
            #now there’s a person being detected still in that spot
            #then assume it’s still Blake
            if faces_now != last_faces:
                with state_lock:
                    CONTEXT.update("faces_recognized", faces_now)
                    last_faces = faces_now

            # ── Debug display & timing ──────────────────────────────────────
            if show_debug:
                cv2.imshow("ARGUS Obj/Face", frame)
                logging.debug("Frame %d – objs:%s faces:%s (%.1f ms)",
                              frame_count, objects_now, faces_now, (time.time()-start_ts)*1000)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            
            # Clean stale face memory
            if frame_count % 60 == 0:
                now = time.time()
                last_known_faces = {
                    name: (x1, y1, x2, y2, ts)
                    for name, (x1, y1, x2, y2, ts) in last_known_faces.items()
                    if now - ts < CONFIG["face_label_timeout"]
                }
                
            # ── Sleep / adapt frame rate ───────────────────────────────────
            # If we processed a frame too fast, sleep to maintain CONFIG["sleep_between_frames"]
            # This allows the loop to adapt to camera FPS and processing speed.
            # If the frame processing took too long, we skip the sleep to catch up.
            # This way we can handle variable FPS cameras without stuttering.
            # Note: CONFIG["sleep_between_frames"] is a target, not a strict limit.
            elapsed = time.time() - start_ts
            time.sleep(max(0, CONFIG["sleep_between_frames"] - elapsed))
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logging.exception("Obj/Person Recognition loop crashed: %s", e)
    finally:
        cv2.destroyAllWindows()

###############################################################################
# THREAD CONVENIENCE WRAPPER
###############################################################################

def start_obj_person_recog_thread(camera_manager, *, show_debug: bool = False) -> threading.Thread:
    """Fire-and-forget helper that starts the recognition loop in a daemon thread."""
    stop_event = threading.Event()
    t = threading.Thread(
        target=obj_person_recog_loop,
        args=(camera_manager,),
        kwargs={"show_debug": show_debug, "stop_event": stop_event},
        daemon=True,
    )
    t.start()
    return t  # caller can .join() or stop_event.set() to terminate
