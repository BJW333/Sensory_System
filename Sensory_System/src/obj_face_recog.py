from __future__ import annotations
import warnings
#Suppress specific DeepFace warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mtcnn.mtcnn")

import cv2
import numpy as np
import os
import threading
import time
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from deepface import DeepFace
from PIL import Image
import imagehash
from ultralytics import YOLO
from src.context_fusion import CONTEXT
from collections import Counter
import traceback


#CONFIGURATION
CONFIG: Dict[str, int | float | bool | str] = {
    "yolo_weights": "yolov8n.pt",
    "obj_conf_thresh": 0.55,
    "face_model_name": "Facenet512", #changed from "VGG-Face" to "Facenet512"
    "face_distance_metric": "cosine",  #changed from "euclidean_l2" to "cosine"
    "face_save_cooldown": 20, #seconds between saving faces of same person
    "face_folder_max": 30,  #max images to keep per person in the folders
    "dup_hash_thresh": 5, 
    "min_face_std": 8,
    "min_face_size": 50,
    "sleep_between_frames": 0.005,
    "resize_for_face": 0.75,
    "face_label_timeout": 6, #seconds to keep showing recognized name
    #debug options
    "save_debug_images": False,  #Save what camera sees Set to true or false to save images
    "debug_image_interval": 10,  #Save every N frames
    #Tuned recognition parameters
    "face_distance_threshold": 0.40,  #Tighter threshold for better accuracy
    "min_votes_for_confirm": 1,       #Instant recognition on first detection or set to 2 for more certainty becasue two frames will be checked
    "vote_history_window": 1.5,       #Keep original window
    "face_every_n_frames": 1,
}

#Directories
PROJECT_ROOT = Path(__file__).parent.parent 
KNOWN_FACES_DIR = PROJECT_ROOT / "data" / "knownfaces"
KNOWN_FACES_DIR.mkdir(exist_ok=True)

#Debug output directory
DEBUG_DIR = PROJECT_ROOT / "data" / "debug_output"
DEBUG_DIR.mkdir(exist_ok=True)

#Detector backend selection
#Try detectors in order of reliability
_AVAILABLE_DETECTORS = ["retinaface", "mtcnn", "mediapipe", "opencv"]
FACE_DETECTOR_BACKEND = None
for det in _AVAILABLE_DETECTORS:
    try:
        logging.debug(f"Testing DeepFace detector backend: {det}")
        test_img = np.ones((250, 250, 3), dtype=np.uint8) * 128  #Gray image
        _ = DeepFace.extract_faces(
            img_path=test_img,
            detector_backend=det,
            enforce_detection=False,
            align=True, #Enable face alignment
        )
        FACE_DETECTOR_BACKEND = det
        logging.debug(f"Using DeepFace detector backend: {FACE_DETECTOR_BACKEND}")
        break
    except Exception as e:
        logging.warning(f"Detector backend '{det}' failed: {e}")

if FACE_DETECTOR_BACKEND is None:
    logging.warning("Using fallback opencv detector - expect issues!")
    FACE_DETECTOR_BACKEND = "opencv"
    
#Load YOLO
YOLO_MODEL = YOLO(CONFIG["yolo_weights"])
logging.debug("YOLO model loaded")

#DEEPFACE WARMUP
def clean_deepface_cache():
    """Remove stale DeepFace index files to force clean rebuild"""
    pkl_files = list(KNOWN_FACES_DIR.glob("*.pkl"))
    for pkl in pkl_files:
        try:
            pkl.unlink()
            logging.debug(f"Cleaned cache: {pkl.name}")
        except Exception as e:
            logging.warning(f"Could not delete {pkl.name}: {e}")

#Call BEFORE the warmup to ensure fresh indexing
clean_deepface_cache()

#Database diagnostic
logging.debug("=== DATABASE DIAGNOSTIC ===")
subdirs = [d for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()]
logging.debug(f"Found {len(subdirs)} person directories:")
for subdir in subdirs:
    imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
    logging.debug(f"  {subdir.name}: {len(imgs)} images")
logging.debug("========================")

#Build database
_known_imgs = [p for p in KNOWN_FACES_DIR.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
if _known_imgs:
    logging.debug(f"Building database with {len(_known_imgs)} known faces...")
    
    try:
        #Must specify detector_backend for consistency
        result = DeepFace.find(
            img_path=str(_known_imgs[0]),
            db_path=str(KNOWN_FACES_DIR),
            model_name=CONFIG["face_model_name"],
            distance_metric=CONFIG["face_distance_metric"],
            enforce_detection=False,
            detector_backend=FACE_DETECTOR_BACKEND, 
            align=True,  #Enable alignment
            silent=True, #Suppress verbose output
            threshold=100.0,
            refresh_database=True
        )
        
        logging.debug(f"Database built. Result shape: {result[0].shape if result and len(result) > 0 else 'empty'}")
        
        #Verify the index
        pkl_files = list(KNOWN_FACES_DIR.glob("*.pkl"))
        if pkl_files:
            import pickle
            with open(pkl_files[0], 'rb') as f:
                db_data = pickle.load(f)
            logging.debug(f"Database contains {len(db_data) if isinstance(db_data, dict) else 'unknown'} entries")
            
        if result and len(result) > 0 and not result[0].empty:
            logging.debug(f"Self-test SUCCESS: {len(result[0])} matches")
            for idx, row in result[0].head(3).iterrows():
                name = Path(row['identity']).parent.name
                dist = row['distance']
                logging.debug(f"   {name}: distance={dist:.4f}")
        else:
            logging.error("Database built but no self-matches found!")
            
    except Exception as e:
        logging.error(f"Database build failed: {e}")
        traceback.print_exc()
        raise
else:
    logging.warning(f"No images in {KNOWN_FACES_DIR}")
    
#Global state
last_save_time: Dict[str, float] = {}
state_lock = threading.Lock()

#UTILITY FUNCTIONS
def _clamp(val: int, lo: int, hi: int) -> int:
    """
    Clamp val between lo and hi.
    Args:
        val: Value to clamp
        lo: Minimum value
        hi: Maximum value
    Returns:
        int: Clamped value
    """
    return max(lo, min(val, hi))

def cull_old_faces(folder: Path, max_images: int) -> None:
    """
    Remove oldest images in folder to keep total under max_images.
    Args:
        folder: Folder path
        max_images: Maximum number of images to keep
    Returns:
        None
    """
    files = sorted(
        (f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".png")),
        key=lambda p: p.stat().st_mtime,
    )
    for f in files[:-max_images]:
        f.unlink(missing_ok=True)

def is_duplicate_face(cropped: np.ndarray, folder: Path, *, hash_thresh: int, check_last: int = 20) -> bool:
    """
    Check if the cropped face is a duplicate of recent images in the folder.
    Args:
        cropped: Cropped face image (numpy array)
        folder: Folder to check against
        hash_thresh: Hamming distance threshold for considering a duplicate
        check_last: Number of recent images to check
    Returns:
        bool: True if duplicate, False otherwise
    """
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
    """
    Save cropped face image to known faces directory.
    Args:
        img_or_frame: Original image or frame (numpy array)
        name: Name of the person
        bbox: Bounding box (x1, y1, x2, y2) if not already cropped
        already_cropped: If True, img_or_frame is already the cropped face
    Returns:
        None
    """
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
        logging.debug(f"Face too small: {cropped.shape[:2]}")
        return

    face_std = np.std(cropped)
    if face_std < CONFIG["min_face_std"]:
        logging.debug(f"Face too blurry: std={face_std:.1f}")
        return

    folder = KNOWN_FACES_DIR / name
    folder.mkdir(exist_ok=True)

    now = time.time()
    with state_lock:
        if now - last_save_time.get(name, 0) < CONFIG["face_save_cooldown"]:
            logging.debug("Cooldown active")
            return
        last_save_time[name] = now

    if is_duplicate_face(cropped, folder, hash_thresh=CONFIG["dup_hash_thresh"]):
        logging.debug("Duplicate face")
        return

    cull_old_faces(folder, CONFIG["face_folder_max"])
    filename = folder / f"{name}_{int(now*1000)}.jpg"
    cv2.imwrite(str(filename), cropped)
    logging.debug(f"Saved: {filename}")

#DETECTION FUNCTIONS
def detect_objects(image: np.ndarray, conf_thresh: float) -> Dict[str, List]:
    """
    Detect objects in an image using YOLOv8 model.
    Args:
        image (np.ndarray): input image
        conf_thresh (float): confidence threshold
    Returns:
        Dict with keys: "labels" (List[str]), "boxes" (List[Tuple[y1,x1,y2,x2]]), "scores" (List[float])
    """
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
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Args:
        a: Bounding box A (x1, y1, x2, y2)
        b: Bounding box B (x1, y1, x2, y2)
    Returns:
        IoU value (float)
    """
    xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)

def save_debug_frame(frame: np.ndarray, frame_count: int, faces_data: List[dict], person_boxes: List, recognized_names: List[str]):
    """
    Save annotated frame showing what the system sees
    Args:
        frame: Original camera frame
        frame_count: Current frame number
        faces_data: List of face detection data dicts
        person_boxes: List of person bounding boxes
        recognized_names: List of recognized names corresponding to faces_data
    Returns:
        None
    """
    debug_frame = frame.copy()
    
    #Draw person boxes in blue
    for (py1, px1, py2, px2) in person_boxes:
        cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(debug_frame, "PERSON", (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    #Draw face boxes with recognition results
    for i, face_data in enumerate(faces_data):
        x1, y1, x2, y2 = face_data['bbox']
        iou_val = face_data.get('max_iou', 0)
        name = recognized_names[i] if i < len(recognized_names) else "Unknown"
        
        #Color based on IoU threshold
        color = (0, 255, 0) if iou_val >= 0.40 else (0, 0, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{name} (IoU:{iou_val:.2f})"
        cv2.putText(debug_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    #Add info text
    info_text = [
        f"Frame: {frame_count}",
        f"Persons: {len(person_boxes)}",
        f"Faces: {len(faces_data)}",
        f"Recognized: {len([n for n in recognized_names if n != 'Unknown'])}",
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(debug_frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    #Save
    output_path = DEBUG_DIR / f"frame_{frame_count:06d}.jpg"
    cv2.imwrite(str(output_path), debug_frame)
    logging.debug(f"Saved debug frame: {output_path}")

def prepare_face_image(face_img):
    """
    Convert face image to proper format for recognition
    Purpose: Ensure face image is in uint8 BGR format resized to 224x224
    Args:
        face_img: Input face image (numpy array)
    Returns:
        Processed face image (numpy array)
    """
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(np.uint8)
    
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    elif face_img.shape[2] == 4:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
    elif face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    
    return cv2.resize(face_img, (224, 224))

#MAIN LOOP
def obj_person_recog_loop(camera_manager, *, show_debug: bool = False, stop_event: Optional[threading.Event] = None,):
    """
    Main loop for object and person recognition.
    Args:
        camera_manager: Instance of CameraManager to get frames from.
        show_debug (bool): If True, enable debug logging.
        stop_event (threading.Event | None): Optional event to signal loop termination. 
    Returns:
        None
    """
    if show_debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    last_objects: List[str] = []
    last_faces: List[str] = []
    last_known_faces: Dict[str, Tuple[int, int, int, int, float]] = {} #name -> (x1,y1,x2,y2,timestamp)
    frame_count = 0
    
    recognized_history: Dict[int, List[Tuple[str, float]]] = {} #person_idx -> list of (name, timestamp)
    face_position_history: Dict[int, List[Tuple[int, int, int, int, float]]] = {} #person_idx -> list of (x1,y1,x2,y2,timestamp)

    logging.debug("Starting object/person recognition loop")
    logging.debug(f"Known faces directory: {KNOWN_FACES_DIR}")
    logging.debug(f"Debug output directory: {DEBUG_DIR}")
    
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
            
            #Log frame info periodically
            if frame_count % 300 == 0:
                logging.debug(f"Camera frame: {frame.shape}, dtype: {frame.dtype}")
            
            #Resize for processing
            if CONFIG["resize_for_face"] < 1.0:
                frame_small = cv2.resize(
                    frame, None, 
                    fx=CONFIG["resize_for_face"], 
                    fy=CONFIG["resize_for_face"], 
                    interpolation=cv2.INTER_AREA
                )
            else:
                frame_small = frame
            
            #Object detection
            objs = detect_objects(frame_small, CONFIG["obj_conf_thresh"])
            
            #Scale back to full resolution
            scale = (1.0 / CONFIG["resize_for_face"]) if CONFIG["resize_for_face"] < 1.0 else 1.0
            
            #Expand person boxes by 20% in all directions
            person_boxes = []
            for (by, bx, by2, bx2), lbl in zip(objs["boxes"], objs["labels"]):
                if lbl == "person":
                    #Scale to full resolution
                    by, bx, by2, bx2 = int(by*scale), int(bx*scale), int(by2*scale), int(bx2*scale)
                    
                    #Expand box by 20%
                    height, width = by2 - by, bx2 - bx
                    by = max(0, by - int(height * 0.2))
                    bx = max(0, bx - int(width * 0.2))
                    by2 = by2 + int(height * 0.2)
                    bx2 = bx2 + int(width * 0.2)
                    
                    person_boxes.append((by, bx, by2, bx2))

            #Face recognition - STORE previous names for reuse
            if not hasattr(obj_person_recog_loop, '_prev_recognized_names'):
                obj_person_recog_loop._prev_recognized_names = []
            
            recognized_names: List[str] = []
            faces_debug_data = []  #for debug visualization
            
            if frame_count % CONFIG["face_every_n_frames"] == 0:
                logging.debug(f"🔍 Frame {frame_count}: Detecting faces...")
                
                #convert BGR to RGB for DeepFace
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    faces = DeepFace.extract_faces(
                        img_path=frame_rgb,  #Now using RGB
                        detector_backend=FACE_DETECTOR_BACKEND,
                        enforce_detection=False,
                        align=True,  #enable alignment for better recognition
                    )
                    logging.debug(f"   Found {len(faces)} face(s)")
                except Exception as e:
                    logging.error(f"Face extraction failed: {e}")
                    faces = []
                
                #match each face to its individual person box
                # ============ STEP 1: Map each face to its individual person box ============
                faces_by_person = {i: [] for i in range(len(person_boxes))}
                
                for idx, face in enumerate(faces):
                    r = face["facial_area"]
                    x1, y1, x2, y2 = r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]
                    logging.debug(f"  Face {idx}: x={r['x']} y={r['y']} w={r['w']} h={r['h']}")
                    
                    #Calculate face center
                    face_center_x = (x1 + x2) // 2
                    face_center_y = (y1 + y2) // 2
                    
                    #Find which person this face belongs to
                    assigned_person = None
                    assigned_person_box = None
                    max_iou = 0
                    
                    for person_idx, (py1, px1, py2, px2) in enumerate(person_boxes):
                        iou = _iou((x1, y1, x2, y2), (px1, py1, px2, py2))
                        
                        #Check if face center is inside this person box
                        if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                            assigned_person = person_idx
                            assigned_person_box = (py1, px1, py2, px2)
                            max_iou = iou
                            break
                        
                    logging.debug(f"  Face {idx}: assigned={assigned_person}, iou={max_iou:.3f}, center_inside={assigned_person is not None}")

                    # ========== Smooth face position across frames ==========
                    if assigned_person is not None and assigned_person_box is not None:
                        if assigned_person not in face_position_history:
                            face_position_history[assigned_person] = []
                        
                        #Add current detection
                        face_position_history[assigned_person].append((x1, y1, x2, y2, time.time()))
                        
                        #Keep only recent detections (last 0.5 seconds)
                        now = time.time()
                        face_position_history[assigned_person] = [
                            (bx1, by1, bx2, by2, t) 
                            for bx1, by1, bx2, by2, t in face_position_history[assigned_person]
                            if now - t < 0.5
                        ]
                        
                        #Use smoothed/averaged position if we have history
                        if len(face_position_history[assigned_person]) >= 2:
                            #Average the positions
                            avg_x1 = int(np.mean([bx1 for bx1, _, _, _, _ in face_position_history[assigned_person]]))
                            avg_y1 = int(np.mean([by1 for _, by1, _, _, _ in face_position_history[assigned_person]]))
                            avg_x2 = int(np.mean([bx2 for _, _, bx2, _, _ in face_position_history[assigned_person]]))
                            avg_y2 = int(np.mean([by2 for _, _, _, by2, _ in face_position_history[assigned_person]]))
                            
                            #Use smoothed position
                            x1, y1, x2, y2 = avg_x1, avg_y1, avg_x2, avg_y2
                            
                            py1, px1, py2, px2 = assigned_person_box
                            max_iou = _iou((x1, y1, x2, y2), (px1, py1, px2, py2))
                            logging.debug(f"Face {idx} (smoothed) → Person {assigned_person} (IoU={max_iou:.3f})")
                        else:
                            logging.debug(f"Face {idx} → Person {assigned_person} (IoU={max_iou:.3f})")
                    
                    #HARD REQUIREMENT: face must be in a person box with reasonable overlap
                    if assigned_person is None or max_iou < 0.05: #used to be 0.10 for IoU threshold
                        print(f"Face {idx} rejected: no matching person box (IoU={max_iou:.2f})")
                        logging.debug(f"Face {idx} rejected: not clearly in person box (IoU={max_iou:.2f})")
                        continue
                    
                    logging.debug(f"Face {idx} → Person {assigned_person} (IoU={max_iou:.3f})")
                    
                    faces_by_person[assigned_person].append({
                        'face_img': face["face"],
                        'bbox': (x1, y1, x2, y2),
                        'iou': max_iou,
                    })
                    logging.debug(f"  Face {idx} assigned to Person {assigned_person}")
                    
                # ============ STEP 2: Recognize faces within each person region ============
                for person_idx in range(len(person_boxes)):
                    name = "Unknown"  #ALWAYS initialize default
                    
                    if not faces_by_person[person_idx]:
                        logging.debug(f"Person {person_idx}: no face detected")
                        recognized_names.append(name)
                        continue
                    
                    logging.debug(f"[RECOG] Person {person_idx}: {len(faces_by_person[person_idx])} faces to recognize")
                    logging.debug(f"[MAPPING] {len(person_boxes)} person boxes, {len(faces)} faces to map")

                    #Take the best face for this person (highest IoU)
                    best_face_data = max(faces_by_person[person_idx], key=lambda f: f['iou'])
                    x1, y1, x2, y2 = best_face_data['bbox']
                    iou_val = best_face_data['iou']
                    
                    faces_debug_data.append({'bbox': (x1, y1, x2, y2), 'max_iou': iou_val})
                    
                    try:
                        #Extract and convert face properly
                        #uses best_face_data 
                        #uses x1, y1, x2, y2 = best_face_data['bbox']
                        face_img = best_face_data['face_img']

                        #Prepare face image for recognition using function
                        face_img = prepare_face_image(face_img)
                        
                        if face_img.size == 0:
                            logging.warning(f"Person {person_idx}: empty face region")
                            recognized_names.append(name)
                            continue
                        
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            tmp_path = tmp.name
                            success = cv2.imwrite(tmp_path, face_img)
                            
                            if not success or os.path.getsize(tmp_path) < 100:
                                logging.error(f"Person {person_idx}: Failed to save face region")
                                recognized_names.append(name)
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                                continue
                            
                            try:
                                logging.debug(f"[QUERY] Person {person_idx}: Querying database...")
                                
                                result = DeepFace.find(
                                    img_path=tmp_path,
                                    db_path=str(KNOWN_FACES_DIR),
                                    model_name=CONFIG["face_model_name"],
                                    distance_metric=CONFIG["face_distance_metric"],
                                    threshold=100.0,
                                    enforce_detection=False,
                                    detector_backend=FACE_DETECTOR_BACKEND,  
                                    align=True, #Enable alignment
                                    silent=True, #Suppress verbose output
                                    refresh_database=False,
                                )
                                
                                #print what DeepFace returned
                                logging.debug(f"  [DEBUG] DeepFace.find() returned: {type(result)}, len={len(result) if result else 0}")
                                if result and len(result) > 0:
                                    logging.debug(f"  [DEBUG] result[0] shape: {result[0].shape}")
                                    logging.debug(f"  [DEBUG] result[0] empty? {result[0].empty}")
                                    if not result[0].empty:
                                        logging.debug(f"  [DEBUG] First 3 rows of result[0]:")
                                        for idx, row in result[0].head(3).iterrows():
                                            logging.debug(f"    {Path(row['identity']).parent.name}: distance={row['distance']:.4f}")
                                else:
                                    logging.debug(f"  [DEBUG] result is None or empty!")
                                    
                                    
                                THRESH = CONFIG["face_distance_threshold"]
                                best_distance = float('inf')

                                if result and len(result) > 0 and not result[0].empty:
                                    best_row = result[0].iloc[0]
                                    best_distance = float(best_row["distance"])
                                    matched_name = Path(best_row["identity"]).parent.name
                                    
                                    logging.debug(f"  [MATCH] {matched_name} @ distance {best_distance:.4f} (threshold: {THRESH})")
                                    
                                    if best_distance <= THRESH:
                                        # ========== TEMPORAL VOTING ==========
                                        if person_idx not in recognized_history:
                                            recognized_history[person_idx] = []
                                        
                                        recognized_history[person_idx].append((matched_name, time.time()))
                                        
                                        now = time.time()
                                        recognized_history[person_idx] = [
                                            (n, t) for n, t in recognized_history[person_idx]
                                            if now - t < CONFIG["vote_history_window"]
                                        ][-15:]
                                        
                                        vote_count = len(recognized_history[person_idx])
                                        min_votes = CONFIG["min_votes_for_confirm"]
                                        
                                        logging.debug(f"  [VOTES] Person {person_idx}: {vote_count}/{min_votes} for {matched_name}")
                                        
                                        if vote_count >= min_votes:
                                            votes = Counter([n for n, _ in recognized_history[person_idx]])
                                            most_common, vote_agreement = votes.most_common(1)[0]
                                            
                                            if vote_agreement >= min_votes:
                                                name = most_common
                                                logging.debug(
                                                    f"Person {person_idx}: CONFIRMED {name} "
                                                    f"({vote_agreement}/{vote_count} votes, dist={best_distance:.3f})"
                                                )
                                                save_cropped_face(frame, name, bbox=(x1, y1, x2, y2), already_cropped=False)
                                                last_known_faces[name] = (x1, y1, x2, y2, time.time())
                                            else:
                                                logging.debug(f"  [VOTES] Person {person_idx}: Split votes: {dict(votes)} - waiting")
                                        else:
                                            logging.debug(f"  [VOTES] Person {person_idx}: {matched_name} ({vote_count}/{min_votes})")
                                    else:
                                        logging.debug(
                                            f"Person {person_idx}: distance {best_distance:.3f} > threshold {THRESH}"
                                        )
                                else:
                                    logging.debug(f"  [NO-MATCH] Person {person_idx}: no database matches")
                            
                            except Exception as e:
                                logging.error(f"Person {person_idx} recognition error: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            finally:
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                    
                    except Exception as e:
                        logging.error(f"Person {person_idx}: Failed to extract face region: {e}")
                    
                    #Always append name at the end
                    recognized_names.append(name)
                    logging.debug(f"  [RESULT] Person {person_idx}: {name}\n")
                    
                #Clean up stale data
                now = time.time()
                
                #Clean old face positions
                for person_idx in list(face_position_history.keys()):
                    if person_idx >= len(person_boxes):
                        del face_position_history[person_idx]
                    else:
                        face_position_history[person_idx] = [
                            (bx1, by1, bx2, by2, t) 
                            for bx1, by1, bx2, by2, t in face_position_history[person_idx]
                            if now - t < 0.5
                        ]
                        if not face_position_history[person_idx]:
                            del face_position_history[person_idx]
                            
                for old_person_idx in list(recognized_history.keys()):
                    if old_person_idx >= len(person_boxes):
                        del recognized_history[old_person_idx]
                        continue

                    elif not faces_by_person[old_person_idx]:
                        recognized_history[old_person_idx] = [
                            (n, t) for n, t in recognized_history[old_person_idx]
                            if time.time() - t < 0.3
                        ]
                        if not recognized_history[old_person_idx]:
                            del recognized_history[old_person_idx]
                
                #Store for reuse on skip frames
                obj_person_recog_loop._prev_recognized_names = recognized_names
            
            else:
                #Reuse previous names on frames where we skip detection
                if obj_person_recog_loop._prev_recognized_names:
                    recognized_names = obj_person_recog_loop._prev_recognized_names
                else:
                    recognized_names = ["Unknown"] * len(person_boxes)
                
                logging.debug(f"[SKIP] Frame {frame_count}: Using cached names = {recognized_names}")
            
            #Save debug visualization
            if CONFIG["save_debug_images"] and frame_count % CONFIG["debug_image_interval"] == 0:
                save_debug_frame(frame, frame_count, faces_debug_data, person_boxes, recognized_names)
                        
            #Update context
            objects_now = sorted(set(objs["labels"]))
            faces_now = sorted({n for n in recognized_names if n != "Unknown"})
            
            logging.debug(f"  Updating CONTEXT.faces_recognized = {faces_now}")
            logging.debug(f"  Updating CONTEXT.objects_seen = {objects_now}")
            #Debug the context update (add this right before CONTEXT.update):
            logging.debug(f"\n[CONTEXT-DEBUG] Frame {frame_count}:")
            logging.debug(f"  recognized_names: {recognized_names}")
            logging.debug(f"  faces_now raw: {[n for n in recognized_names if n != 'Unknown']}")
            logging.debug(f"  faces_now sorted: {faces_now}")

            with state_lock:
                if objects_now != last_objects or faces_now != last_faces:
                    with CONTEXT.batch():
                        CONTEXT.update("objects_seen", objects_now)
                        CONTEXT.update("faces_recognized", faces_now)
                    last_objects = objects_now
                    last_faces = faces_now
                    logging.debug(f"  CONTEXT updated!")
                else:
                    logging.debug(f"  (no change)")
            
            #Show debug window
            if show_debug and frame_count % CONFIG["face_every_n_frames"] == 0:
                debug_frame = frame.copy()
                for (py1, px1, py2, px2) in person_boxes:
                    cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                for face_data, name in zip(faces_debug_data, recognized_names):
                    x1, y1, x2, y2 = face_data['bbox']
                    color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imshow("Recognition", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            frame_count += 1
            
            #Clean stale memory
            if frame_count % 60 == 0:
                now = time.time()
                last_known_faces = {
                    name: (x1, y1, x2, y2, ts)
                    for name, (x1, y1, x2, y2, ts) in last_known_faces.items()
                    if now - ts < CONFIG["face_label_timeout"]
                }
            
            #Sleep
            elapsed = time.time() - start_ts
            time.sleep(max(0, CONFIG["sleep_between_frames"] - elapsed))
    
    except KeyboardInterrupt:
        logging.debug("Interrupted by user")
    except Exception as e:
        logging.exception(f"Loop crashed: {e}")
    finally:
        cv2.destroyAllWindows()
        logging.debug("️Recognition loop stopped")

#THREAD WRAPPER
def start_obj_person_recog_thread(camera_manager, *, show_debug: bool = False) -> Tuple[threading.Thread, threading.Event]:
    """
    Start the object and person recognition loop in a separate thread.
    Args:
        camera_manager: Instance of CameraManager to get frames from.
        show_debug (bool): If True, enable debug logging.
    Returns:
        Tuple of (threading.Thread, threading.Event) for the recognition loop.
    """
    stop_event = threading.Event()
    t = threading.Thread(
        target=obj_person_recog_loop,
        args=(camera_manager,),
        kwargs={"show_debug": show_debug, "stop_event": stop_event},
        daemon=True,
    )
    t.start()
    return t, stop_event