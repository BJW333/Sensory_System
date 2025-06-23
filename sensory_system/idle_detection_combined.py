from __future__ import annotations
import threading
import time
import logging
from typing import Tuple, Optional, List
import cv2
import numpy as np
import mediapipe as mp
from context_fusion import CONTEXT

#CONFIG

CFG = {
    "IDLE_TIMEOUT": 60,           # seconds: gaze moved away
    "AWAY_TIMEOUT": 300,          # seconds: user not present
    "LOOP_INTERVAL": 0.2,         # seconds between frames (≈5 FPS)
    "GAZE_TOLERANCE": 0.12,       # % of frame width considered centered
}

LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]

logger = logging.getLogger(__name__)

#Helper  presence & gaze per frame

def _presence_and_gaze(
    frame,
    face_mesh: "mp.solutions.face_mesh.FaceMesh",
) -> Tuple[bool, bool, Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """Returns (user_present, gaze_centered, left_pts, right_pts)."""

    user_present = False
    gaze_centered = False
    left_eye_pts = right_eye_pts = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        user_present = True
        landmarks = res.multi_face_landmarks[0]
        h, w, _ = frame.shape

        left_eye_pts = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE_IDX]
        right_eye_pts = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE_IDX]

        l_center_x = np.mean([p[0] for p in left_eye_pts])
        r_center_x = np.mean([p[0] for p in right_eye_pts])
        face_center_x = (l_center_x + r_center_x) / 2
        gaze_centered = abs(face_center_x - w / 2) < w * CFG["GAZE_TOLERANCE"]

    return user_present, gaze_centered, left_eye_pts, right_eye_pts


#Main loop

def idle_detection_loop(
    camera_manager,
    *,
    show_debug: bool = False,
    stop_event: Optional[threading.Event] = None,
):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    last_active_time = time.time()
    state = "active"

    try:
        while stop_event is None or not stop_event.is_set():
            CONTEXT.update("idle_detection_last_beat", time.time())

            frame = camera_manager.get_frame()
            if frame is None:
                with CONTEXT.batch():
                    CONTEXT.update("activity_state", "unknown")
                    CONTEXT.update("user_present", False)
                time.sleep(CFG["LOOP_INTERVAL"])
                continue

            now = time.time()
            present, centered, left_pts, right_pts = _presence_and_gaze(frame, face_mesh)

            #state updater
            if present:
                if centered:
                    last_active_time = now
                    state = "active"
                else:
                    state = "idle" if now - last_active_time > CFG["IDLE_TIMEOUT"] else "distracted"
            else:
                away_for = now - last_active_time
                if away_for > CFG["AWAY_TIMEOUT"]:
                    state = "away"
                elif away_for > CFG["IDLE_TIMEOUT"]:
                    state = "idle"
                else:
                    state = "distracted"

            #CONTEXT updates
            with CONTEXT.batch():
                CONTEXT.update("activity_state", state)
                CONTEXT.update("last_active_time", last_active_time)
                CONTEXT.update("user_present", present)
                CONTEXT.update("gaze_centered", centered)

            #Debug display 
            if show_debug:
                logger.debug(
                    "State: %s | present=%s | centered=%s | idle_for=%ds",
                    state,
                    present,
                    centered,
                    int(now - last_active_time),
                )
                dbg_frame = frame.copy()
                if present and left_pts and right_pts:
                    for (x, y) in left_pts + right_pts:
                        cv2.circle(dbg_frame, (x, y), 4, (0, 255, 0), -1)
                text = f"{state.upper()} | present={present} | gaze={centered}"
                cv2.putText(dbg_frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if centered else (0,0,255), 2)
                cv2.imshow("Idle / Presence", dbg_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(CFG["LOOP_INTERVAL"])
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Idle detection loop crashed: %s", e)
    finally:
        cv2.destroyAllWindows()
        face_mesh.close()

#Threading
def start_idle_detection_thread(
    camera_manager,
    *,
    show_debug: bool = False,
) -> threading.Thread:
    stop_event = threading.Event()
    t = threading.Thread(
        target=idle_detection_loop,
        args=(camera_manager,),
        kwargs={"show_debug": show_debug, "stop_event": stop_event},
        daemon=True,
    )
    t.start()
    return t  # caller can stop with t._kwargs["stop_event"].set()
