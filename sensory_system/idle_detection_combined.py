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
from context_fusion import CONTEXT

# ============================================================================
# CONFIG
# ============================================================================

CFG = {
    "IDLE_TIMEOUT": 60,              # seconds: gaze moved away → idle
    "AWAY_TIMEOUT": 300,             # seconds: user not present → away
    "LOOP_INTERVAL": 0.1,            # seconds between frames (≈10 FPS)
    "GAZE_TOLERANCE": 0.15,          # normalized: eyes must be within ±15% of face center
    "MIN_FACE_WIDTH": 50,       # ← Allow smaller faces
    "FACE_SCALE": 1.05,        # ← More thorough search
    "FACE_MIN_NEIGHBORS": 5,   # ← Lower confidence threshold
    "EYE_SCALE": 1.02,       
    "EYE_MIN_NEIGHBORS": 4,
}

logger = logging.getLogger(__name__)
# ============================================================================
# CORRECTED _IdleDetector CLASS - Replace the one in idle_detection_combined.py
# ============================================================================

class _IdleDetector:
    """Manages idle state transitions with hysteresis to prevent flickering."""
    
    def __init__(self, min_frames_to_switch: int = 3):
        """
        Initialize the idle detector.
        
        Args:
            min_frames_to_switch: Number of consecutive consistent readings 
                                 required before changing state
        """
        self._state_confidence = 0
        self._min_frames_to_switch = min_frames_to_switch
        self._last_desired_state: Optional[str] = None  # ✅ INITIALIZE
        self._current_state = "active"  # ✅ INITIALIZE to "active"
    
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
        
        # Only switch if we've had consistent readings
        if desired_state == self._last_desired_state:
            self._state_confidence += 1
        else:
            self._state_confidence = 1
            self._last_desired_state = desired_state
        
        # State changed - update current state
        if self._state_confidence >= self._min_frames_to_switch:
            self._current_state = desired_state  # ✅ UPDATE current state
        
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
            
# ============================================================================
# CASCADE INITIALIZATION
# ============================================================================

def _load_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """Load OpenCV Haar Cascade classifiers for face and eye detection.
    
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


# ============================================================================
# CORE DETECTION
# ============================================================================

def _detect_face_and_eyes(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
) -> Tuple[bool, bool, Optional[Tuple[int, int, int, int]], Optional[list]]:
    """Detect face and eyes, estimate gaze direction.
    
    Args:
        frame: BGR image from camera
        face_cascade: Haar cascade for face detection
        eye_cascade: Haar cascade for eye detection
        
    Returns:
        (user_present, gaze_centered, face_bbox, eye_points)
        - user_present: bool
        - gaze_centered: bool (only meaningful if user_present)
        - face_bbox: (x, y, w, h) or None
        - eye_points: [(x, y), ...] or None
    """
    user_present = False
    gaze_centered = False
    face_bbox = None
    eye_points = None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)  # improve contrast for cascade
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=CFG["FACE_SCALE"],
        minNeighbors=CFG["FACE_MIN_NEIGHBORS"],
        minSize=(CFG["MIN_FACE_WIDTH"], CFG["MIN_FACE_WIDTH"]),
    )
    
    if len(faces) == 0:
        return user_present, gaze_centered, face_bbox, eye_points
    
    user_present = True
    x_face, y_face, w_face, h_face = faces[0]  # largest/first face
    face_bbox = (x_face, y_face, w_face, h_face)
    
    # Search for eyes within face ROI
    roi_gray = gray_eq[y_face : y_face + h_face, x_face : x_face + w_face]
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=CFG["EYE_SCALE"],
        minNeighbors=CFG["EYE_MIN_NEIGHBORS"],
        minSize=(10, 10),
    )
    
    if len(eyes) < 2:
        # Fallback: estimate gaze from face center (better than blind assumption)
        # If face is mostly centered in frame, assume user is looking at screen
        frame_center_x = frame.shape[1] / 2
        face_center_x = x_face + w_face / 2
        
        distance_from_frame_center = abs(face_center_x - frame_center_x)
        tolerance_pixels = frame.shape[1] * 0.25  # Allow 25% tolerance
        
        gaze_centered = distance_from_frame_center < tolerance_pixels
        logger.debug("Using face position as gaze proxy; centered=%s", gaze_centered)
        return user_present, gaze_centered, face_bbox, eye_points
    
    # Convert eye coordinates back to frame space
    eye_points = [
        (x_face + ex + ew // 2, y_face + ey + eh // 2)
        for ex, ey, ew, eh in eyes[:2]  # take first 2 eyes
    ]
    
    # Estimate gaze: are eyes centered in face?
    face_center_x = x_face + w_face / 2
    eye_centers_x = [pt[0] for pt in eye_points]
    avg_eye_x = np.mean(eye_centers_x)
    
    # Normalize distance: how far from center as % of face width?
    distance_from_center = abs(avg_eye_x - face_center_x)
    tolerance_pixels = w_face * CFG["GAZE_TOLERANCE"]
    
    gaze_centered = distance_from_center < tolerance_pixels
    
    return user_present, gaze_centered, face_bbox, eye_points


# ============================================================================
# MAIN LOOP
# ============================================================================

def idle_detection_loop(
    camera_manager,
    *,
    show_debug: bool = False,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Main idle detection loop.
    
    Args:
        camera_manager: CameraManager instance
        show_debug: If True, display debug visualization
        stop_event: Threading event to signal shutdown
    """
    # Load cascades once
    try:
        face_cascade, eye_cascade = _load_cascades()
    except RuntimeError as e:
        logger.error("Cannot start idle detection: %s", e)
        CONTEXT.update("activity_state", "error")
        return
    
    # Validate config
    _validate_config()
    
    last_active_time = time.time()
    detector = _IdleDetector(min_frames_to_switch=3)
    state = "active"
    
    try:
        while stop_event is None or not stop_event.is_set():
            loop_start = time.time()
            CONTEXT.update("idle_detection_last_beat", loop_start)
            
            # Get frame from camera
            frame = camera_manager.get_frame(wait=False)
            if frame is None:
                with CONTEXT.batch():
                    CONTEXT.update("activity_state", "unknown")
                    CONTEXT.update("user_present", False)
                time.sleep(CFG["LOOP_INTERVAL"])
                continue
            
            # Detect face and eyes
            now = time.time()
            present, centered, face_bbox, eye_points = _detect_face_and_eyes(
                frame, face_cascade, eye_cascade
            )
            
            # State machine with hysteresis
            if present and centered:
                last_active_time = now
            
            state = detector.update_state(present, centered, now, last_active_time)
            
            
            # Update context 
            with CONTEXT.batch():
                CONTEXT.update("activity_state", state)
                CONTEXT.update("last_active_time", last_active_time)
                CONTEXT.update("user_present", present)
                CONTEXT.update("gaze_centered", centered)
            
            # Debug visualization
            if show_debug:
                _debug_display(
                    frame, state, present, centered, face_bbox, eye_points, now, last_active_time
                )
            
            # Adaptive sleep
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


# ============================================================================
# DEBUG HELPERS
# ============================================================================

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
    """Draw debug visualization on frame."""
    dbg_frame = frame.copy()
    h, w = dbg_frame.shape[:2]
    
    # Draw face box if detected
    if face_bbox:
        x, y, fw, fh = face_bbox
        color = (0, 255, 0) if centered else (0, 0, 255)
        cv2.rectangle(dbg_frame, (x, y), (x + fw, y + fh), color, 2)
        
        # Draw eye points
        if eye_points:
            for ex, ey in eye_points:
                cv2.circle(dbg_frame, (ex, ey), 3, (255, 255, 0), -1)
        
        # Draw face center line (for reference)
        cv2.line(dbg_frame, (x + fw // 2, y), (x + fw // 2, y + fh), (255, 0, 0), 1)
    
    # Status text
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
    
    # FPS counter
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
    """Validate CONFIG values at startup."""
    assert CFG["IDLE_TIMEOUT"] > 0, "IDLE_TIMEOUT must be positive"
    assert CFG["AWAY_TIMEOUT"] > CFG["IDLE_TIMEOUT"], "AWAY_TIMEOUT must be > IDLE_TIMEOUT"
    assert 0 < CFG["GAZE_TOLERANCE"] < 0.5, "GAZE_TOLERANCE should be 0-50%"
    assert CFG["MIN_FACE_WIDTH"] > 10, "MIN_FACE_WIDTH must be > 10px"
    assert CFG["LOOP_INTERVAL"] > 0, "LOOP_INTERVAL must be positive"
    logger.debug("Config validation passed")


# ============================================================================
# THREADING WRAPPER
# ============================================================================

def start_idle_detection_thread(
    camera_manager,
    *,
    show_debug: bool = False,
) -> Tuple[threading.Thread, threading.Event]:
    """Start idle detection in a background thread.
    
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