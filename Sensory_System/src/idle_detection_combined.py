"""
Idle detection using OpenCV Haar Cascades (no MediaPipe dependency).
Detects user presence and gaze direction to track activity state.
"""
from __future__ import annotations
import threading
import time
import logging
from typing import Tuple, Optional
import cv2
import numpy as np
from src.context_fusion import CONTEXT

logger = logging.getLogger(__name__)

#CONFIGURATION
CFG = {
    "IDLE_TIMEOUT": 60,              #seconds: gaze moved away = idle
    "AWAY_TIMEOUT": 300,             #seconds: user not present = away
    "LOOP_INTERVAL": 0.1,            #seconds between frames (10 FPS)
    "GAZE_TOLERANCE": 0.15,          #normalized: eyes must be within +or-15% of face center
    "MIN_FACE_WIDTH": 50,            #Allow smaller faces
    "FACE_SCALE": 1.05,              #More thorough search
    "FACE_MIN_NEIGHBORS": 5,         #Lower confidence threshold
    "EYE_SCALE": 1.01,               #Finer eye detection (was 1.02)
    "EYE_MIN_NEIGHBORS": 3,          #More sensitive (was 4)
    "PUPIL_DETECTION": False,        #Enable advanced pupil based gaze estimation
    "HEAD_POSE_FALLBACK": True,      #Use head pose when eyes not visible
}

#IDLE DETECTION STATE MACHINE
class _IdleDetector:
    """
    Manages idle state transitions with hysteresis to prevent flickering.
    """
    def __init__(self, min_frames_to_switch: int = 3):
        """
        Initialize the idle detector.
        Args:
            min_frames_to_switch: Number of consecutive consistent readings 
                                        required before changing state
        Returns:
            None
        """
        self._state_confidence = 0
        self._min_frames_to_switch = min_frames_to_switch
        self._last_desired_state: Optional[str] = None  #INITIALIZE
        self._current_state = "active"  #INITIALIZE to "active"
        
    def update_state(
        self, 
        present: bool, 
        centered: bool, 
        now: float, 
        last_active_time: float
    ) -> str:
        """
        Update state with hysteresis to prevent rapid state switching.
        Args:
            present: User face detected
            centered: User gaze is centered on screen
            now: Current timestamp
            last_active_time: When user was last active
        Returns:
            Current state after hysteresis filtering
        """
        desired_state = self._compute_desired_state(present, centered, now, last_active_time)
        
        #Only switch if we've had consistent readings
        if desired_state == self._last_desired_state:
            self._state_confidence += 1
        else:
            self._state_confidence = 1
            self._last_desired_state = desired_state
        
        #State changed means update current state
        if self._state_confidence >= self._min_frames_to_switch:
            self._current_state = desired_state  #UPDATE current state
        
        return self._current_state
    
    def _compute_desired_state(
        self, 
        present: bool, 
        centered: bool, 
        now: float, 
        last_active_time: float
    ) -> str:
        """
        Compute what state SHOULD be (before hysteresis).
        Args:
            present: User face detected
            centered: User gaze is centered on screen
            now: Current timestamp
            last_active_time: When user was last active
        Returns:
            Desired state based on current conditions
        """
        if present:
            if centered:
                return "active"
            else:
                return (
                    "idle"
                    if now - last_active_time > CFG["IDLE_TIMEOUT"]
                    else "distracted"
                )
        else:
            away_duration = now - last_active_time
            if away_duration > CFG["AWAY_TIMEOUT"]:
                return "away"
            elif away_duration > CFG["IDLE_TIMEOUT"]:
                return "idle"
            else:
                return "distracted"
            
#CASCADE INITIALIZATION
def _load_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """
    Load OpenCV Haar Cascade classifiers for face and eye detection.
    Returns:
        (face_cascade, eye_cascade)
    Raises:
        RuntimeError: If cascades cannot be loaded
    """
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load face cascade: {face_cascade_path}")
    if eye_cascade.empty():
        raise RuntimeError(f"Failed to load eye cascade: {eye_cascade_path}")
    
    logger.info("Haar cascades loaded successfully")
    return face_cascade, eye_cascade


#CORE DETECTION FUNCTION
def _detect_face_and_eyes(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
) -> Tuple[bool, bool, Optional[Tuple[int, int, int, int]], Optional[list]]:
    """
    Detect face and eyes, estimate gaze direction with improved accuracy.
    Args:
        frame: BGR image from camera
        face_cascade: Preloaded face cascade classifier
        eye_cascade: Preloaded eye cascade classifier
    Returns:
        (user_present, gaze_centered, face_bbox, eye_points)
    """
    user_present = False
    gaze_centered = False
    face_bbox = None
    eye_points = None
    
    #Preprocess frame
    #Convert to grayscale and equalize histogram
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    
    #Detect faces
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=CFG["FACE_SCALE"],
        minNeighbors=CFG["FACE_MIN_NEIGHBORS"],
        minSize=(CFG["MIN_FACE_WIDTH"], CFG["MIN_FACE_WIDTH"]),
    )
    
    #No faces detected
    if len(faces) == 0:
        return user_present, gaze_centered, face_bbox, eye_points
    
    user_present = True
    
    #Select the largest face
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    
    x_face, y_face, w_face, h_face = faces[0]
    face_bbox = (x_face, y_face, w_face, h_face)
    
    #Define eye search regions
    eye_region_y = y_face
    eye_region_h = int(h_face * 0.6)
    
    #Search for eyes with bounds checking: Ensure all are within image bounds
    roi_y_start = max(0, eye_region_y) 
    roi_y_end = min(gray_eq.shape[0], eye_region_y + eye_region_h) 
    roi_x_start = max(0, x_face)
    roi_x_end = min(gray_eq.shape[1], x_face + w_face)
    
    roi_gray = gray_eq[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    #If no ROI fallback
    if roi_gray.size == 0:
        return _estimate_gaze_from_head_pose(frame, x_face, y_face, w_face, h_face)
    
    #Safe min/max sizes
    min_eye_width = max(5, int(w_face * 0.12))
    min_eye_height = max(5, int(w_face * 0.08))
    max_eye_size = max(min_eye_width + 1, int(w_face * 0.3))
    
    #Detect eyes
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=CFG["EYE_SCALE"],
        minNeighbors=CFG["EYE_MIN_NEIGHBORS"],
        minSize=(min_eye_width, min_eye_height),
        maxSize=(max_eye_size, max_eye_size),
    )
    
    #Filter unlikely eye candidates
    valid_eyes = []
    for (ex, ey, ew, eh) in eyes:
        if ey + eh/2 < roi_gray.shape[0] * 0.7:  #Use roi_gray.shape instead of eye_region_h
            eye_center_x = ex + ew/2
            if w_face > 0:
                #Calculate relative x position within face
                relative_x = eye_center_x / w_face
                #Accept eyes in left and right thirds of face
                if 0.15 < relative_x < 0.45 or 0.55 < relative_x < 0.85:
                    valid_eyes.append((ex, ey, ew, eh))
                
    #Need at least two valid eyes for gaze estimation
    if len(valid_eyes) < 2:
        #CHECK THE CONFIG FLAG
        if CFG["HEAD_POSE_FALLBACK"]:
            return _estimate_gaze_from_head_pose(frame, x_face, y_face, w_face, h_face)
        else:
            #Return with simple centered check
            return user_present, False, face_bbox, None

    #Sort eyes by x coordinate and take two closest to center
    valid_eyes = sorted(valid_eyes, key=lambda e: e[0])[:2]

    #Convert eye coordinates back to frame space (accounting for ROI offset)
    eye_points = [
        (roi_x_start + ex + ew // 2, roi_y_start + ey + eh // 2)
        for ex, ey, ew, eh in valid_eyes
    ]

    #Check if pupil detection is enabled
    if CFG["PUPIL_DETECTION"]:
        #Use advanced pupil based gaze estimation
        gaze_centered = _estimate_gaze_direction(
            gray_eq,       
            valid_eyes, 
            roi_x_start,
            roi_y_start,
            w_face
        )
    else:
        #Simple gaze based on eye center positions only
        face_center_x = x_face + w_face / 2
        eye_centers_x = [(roi_x_start + ex + ew/2) for ex, _, ew, _ in valid_eyes]
        avg_eye_x = np.mean(eye_centers_x)
        distance_from_center = abs(avg_eye_x - face_center_x)
        gaze_centered = distance_from_center < (w_face * CFG["GAZE_TOLERANCE"])
        logger.debug(f"Simple gaze (no pupil): dist={distance_from_center:.0f}, centered={gaze_centered}")

    return user_present, gaze_centered, face_bbox, eye_points

def _estimate_gaze_from_head_pose(
    frame: np.ndarray, 
    x: int, 
    y: int, 
    w: int, 
    h: int
) -> Tuple[bool, bool, Tuple[int, int, int, int], None]:
    """
    Estimate gaze based on head pose when eyes aren't clearly visible.
    Uses face symmetry and position to estimate if user is looking at screen.
    Args:
        frame: BGR image from camera
        x, y, w, h: Face bounding box
    Returns:
        (user_present, gaze_centered, face_bbox, None)
    """
    frame_center_x = frame.shape[1] / 2
    face_center_x = x + w / 2
    
    #Check if face is reasonably centered
    distance_from_center = abs(face_center_x - frame_center_x)
    centered_tolerance = frame.shape[1] * 0.2  #20% tolerance
    
    #Check face size (closer faces = more likely engaged)
    face_area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
    engaged_size = face_area_ratio > 0.04  #Face takes up >4% of frame
    
    #Check vertical position (too high/low = not looking at screen)
    face_center_y = y + h / 2
    frame_center_y = frame.shape[0] / 2
    vertical_distance = abs(face_center_y - frame_center_y)
    vertical_tolerance = frame.shape[0] * 0.25
    
    gaze_centered = (
        distance_from_center < centered_tolerance and
        vertical_distance < vertical_tolerance and
        engaged_size
    )
    
    logger.debug(
        f"Head pose fallback: centered={gaze_centered}, "
        f"h_dist={distance_from_center:.0f}, v_dist={vertical_distance:.0f}, "
        f"size={face_area_ratio:.3f}"
    )
    
    return True, gaze_centered, (x, y, w, h), None

def _estimate_gaze_direction(
    gray: np.ndarray,
    eyes: list,
    face_x: int,
    eye_region_y: int,
    face_w: int,
) -> bool:
    """
    Estimate gaze direction using pupil position within eyes.
    Args:
        gray: Grayscale image
        eyes: List of detected eye bounding boxes
        face_x: X coordinate of face bounding box
        eye_region_y: Y coordinate of eye search region
        face_w: Width of face bounding box
    Returns:
        gaze_centered: True if gaze is centered on screen
    """
    #Need at least two eyes
    if len(eyes) < 2:
        return False
    
    left_eye, right_eye = eyes[0], eyes[1]
    pupil_positions = []
    
    for (ex, ey, ew, eh) in [left_eye, right_eye]:
        #Extract eye region with bounds checking
        y_start = max(0, eye_region_y + ey)
        y_end = min(gray.shape[0], eye_region_y + ey + eh)
        x_start = max(0, face_x + ex)
        x_end = min(gray.shape[1], face_x + ex + ew)
        
        eye_roi = gray[y_start:y_end, x_start:x_end]
        
        if eye_roi.size == 0 or eye_roi.shape[0] < 5 or eye_roi.shape[1] < 5:
            continue
        
        #Apply Gaussian blur (kernel size must be odd and > 0)
        kernel_size = min(5, min(eye_roi.shape[0], eye_roi.shape[1]))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size >= 3:
            eye_roi_blur = cv2.GaussianBlur(eye_roi, (kernel_size, kernel_size), 0)
        else:
            eye_roi_blur = eye_roi
        
        #Find darkest region
        min_loc = cv2.minMaxLoc(eye_roi_blur)[2]
        
        #Calculate relative position with division by zero check
        if min_loc and ew > 0:
            pupil_x = ex + min_loc[0]
            pupil_relative_x = pupil_x / ew
            pupil_positions.append(pupil_relative_x)
    
    #Estimate gaze based on pupil positions
    if len(pupil_positions) >= 2:
        avg_pupil_position = np.mean(pupil_positions)
        gaze_centered = 0.35 < avg_pupil_position < 0.65
        
        logger.debug(
            f"Pupil analysis: positions={pupil_positions}, "
            f"avg={avg_pupil_position:.2f}, centered={gaze_centered}"
        )
    else:
        #Fallback to simpler method
        face_center_x = face_x + face_w / 2
        eye_centers_x = [(face_x + ex + ew/2) for ex, _, ew, _ in eyes]
        avg_eye_x = np.mean(eye_centers_x) if eye_centers_x else face_center_x
        
        distance_from_center = abs(avg_eye_x - face_center_x)
        tolerance_pixels = face_w * CFG["GAZE_TOLERANCE"]
        
        gaze_centered = distance_from_center < tolerance_pixels
        
        logger.debug(f"Simple eye center: dist={distance_from_center:.0f}, centered={gaze_centered}")
    
    return gaze_centered

#MAIN LOOP
def idle_detection_loop(
    camera_manager,
    *,
    show_debug: bool = False,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    Main idle detection loop.
    Args:
        camera_manager: CameraManager instance
        show_debug: If True, display debug visualization
        stop_event: Threading event to signal shutdown
    Returns:
        None
    """
    #Load cascades once
    try:
        face_cascade, eye_cascade = _load_cascades()
    except RuntimeError as e:
        logger.error("Cannot start idle detection: %s", e)
        CONTEXT.update("activity_state", "error")
        return
    
    #Validate config
    _validate_config()
    
    last_active_time = time.time()
    detector = _IdleDetector(min_frames_to_switch=3)
    state = "active"
    
    try:
        while stop_event is None or not stop_event.is_set():
            loop_start = time.time()
            CONTEXT.update("idle_detection_last_beat", loop_start)
            
            #Get frame from camera
            frame = camera_manager.get_frame(wait=False)
            if frame is None:
                with CONTEXT.batch():
                    CONTEXT.update("activity_state", "unknown")
                    CONTEXT.update("user_present", False)
                time.sleep(CFG["LOOP_INTERVAL"])
                continue
            
            #Detect face and eyes
            now = time.time()
            present, centered, face_bbox, eye_points = _detect_face_and_eyes(
                frame, face_cascade, eye_cascade
            )
            
            #State machine with hysteresis
            if present and centered:
                last_active_time = now
            
            state = detector.update_state(present, centered, now, last_active_time)
            
            
            #Update context 
            with CONTEXT.batch():
                CONTEXT.update("activity_state", state)
                CONTEXT.update("last_active_time", last_active_time)
                CONTEXT.update("user_present", present)
                CONTEXT.update("gaze_centered", centered)
            
            #Debug visualization
            if show_debug:
                _debug_display(
                    frame, state, present, centered, face_bbox, eye_points, now, last_active_time
                )
            
            #Adaptive sleep
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, CFG["LOOP_INTERVAL"] - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Idle detection interrupted by user")
    except Exception as e:
        logger.exception("Idle detection loop crashed: %s", e)
        CONTEXT.update("activity_state", "error")
    finally:
        cv2.destroyAllWindows()
        logger.info("Idle detection loop stopped")

#DEBUG HELPERS
def _debug_display(
    frame: np.ndarray,
    state: str,
    present: bool,
    centered: bool,
    face_bbox: Optional[Tuple[int, int, int, int]],
    eye_points: Optional[list],
    now: float,
    last_active_time: float,
) -> None:
    """
    Display debug visualization window.
    Args:
        frame: BGR image from camera
        state: Current activity state
        present: User face detected
        centered: User gaze is centered on screen
        face_bbox: Detected face bounding box
        eye_points: Detected eye center points
        now: Current timestamp
        last_active_time: When user was last active
    Returns:
        None
    """
    dbg_frame = frame.copy()
    h, w = dbg_frame.shape[:2]
    
    #Draw face box if detected
    if face_bbox:
        x, y, fw, fh = face_bbox
        color = (0, 255, 0) if centered else (0, 0, 255)
        cv2.rectangle(dbg_frame, (x, y), (x + fw, y + fh), color, 2)
        
        #Draw eye points
        if eye_points:
            for ex, ey in eye_points:
                cv2.circle(dbg_frame, (ex, ey), 3, (255, 255, 0), -1)
        
        #Draw face center line (for reference)
        cv2.line(dbg_frame, (x + fw // 2, y), (x + fw // 2, y + fh), (255, 0, 0), 1)
    
    #Status text
    idle_duration = int(now - last_active_time)
    text = f"{state.upper()} | present={present} | gaze={centered} | idle={idle_duration}s"
    color = (0, 255, 0) if state == "active" else (0, 165, 255) if state == "distracted" else (0, 0, 255)
    cv2.putText(
        dbg_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    
    #FPS counter
    cv2.putText(
        dbg_frame,
        f"Press 'q' to quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    
    cv2.imshow("Idle Detection Debug", dbg_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise KeyboardInterrupt()

def _validate_config() -> None:
    """
    Validate CONFIG values at startup.
    Raises:
        AssertionError: If any config value is invalid.
    """
    #Validation checks
    assert CFG["IDLE_TIMEOUT"] > 0, "IDLE_TIMEOUT must be positive"
    assert CFG["AWAY_TIMEOUT"] > CFG["IDLE_TIMEOUT"], "AWAY_TIMEOUT must be > IDLE_TIMEOUT"
    assert 0 < CFG["GAZE_TOLERANCE"] < 0.5, "GAZE_TOLERANCE should be 0-50%"
    assert CFG["MIN_FACE_WIDTH"] > 10, "MIN_FACE_WIDTH must be > 10px"
    assert CFG["LOOP_INTERVAL"] > 0, "LOOP_INTERVAL must be positive"
    
    #Log configuration state
    logger.info("=== Idle Detection Configuration ===")
    logger.info(f"Pupil detection: {'ENABLED' if CFG['PUPIL_DETECTION'] else 'DISABLED'}")
    logger.info(f"Head pose fallback: {'ENABLED' if CFG['HEAD_POSE_FALLBACK'] else 'DISABLED'}")
    logger.info(f"Idle timeout: {CFG['IDLE_TIMEOUT']}s")
    logger.info(f"Away timeout: {CFG['AWAY_TIMEOUT']}s")
    logger.debug("Config validation passed")
    
#THREADING WRAPPER
def start_idle_detection_thread(camera_manager, *, show_debug: bool = False,) -> Tuple[threading.Thread, threading.Event]:
    """
    Start idle detection in a background thread.
    Args:
        camera_manager: CameraManager instance
        show_debug: Enable debug visualization  
    Returns:
        (thread, stop_event) - call stop_event.set() to stop the thread
    """
    stop_event = threading.Event()
    t = threading.Thread(
        target=idle_detection_loop,
        args=(camera_manager,),
        kwargs={"show_debug": show_debug, "stop_event": stop_event},
        name="IdleDetection",
        daemon=True,
    )
    t.start()
    logger.info("Idle detection thread started")
    return t, stop_event