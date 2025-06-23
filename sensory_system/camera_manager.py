from __future__ import annotations
import cv2
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_BACKENDS = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

class CameraManager:
    def __init__(self, camera_index: int = 0) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        for be in _BACKENDS:
            cap = cv2.VideoCapture(camera_index, be)
            if cap.isOpened():
                self.cap = cap
                logger.info("Camera %d opened with backend %s", camera_index, be)
                break
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Camera {camera_index} could not be opened with any backend")

        # FPS handling – some webcams return 0.0
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.fps = fps if fps > 1 else 30.0  # sensible default
        self.interval = 1 / self.fps

        self._frame = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._running = True

        self._thread = threading.Thread(target=self._grab_loop, name="CameraFeed", daemon=True)
        self._thread.start()

    # Internal loop for frame grabbing
    # Runs in a separate thread to avoid blocking main thread
    
    def _grab_loop(self) -> None:
        consecutive_fail = 0
        while self._running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                consecutive_fail = 0
                with self._lock:
                    self._frame = frame
                    self._new.set()
            else:
                consecutive_fail += 1
                if consecutive_fail > 30:  # ~1 sec of failures
                    logger.warning("Camera read failed %d times – attempting reopen", consecutive_fail)
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap.open(0)
                    consecutive_fail = 0
                time.sleep(self.interval)

    # Public API

    def get_frame(self, *, wait: bool = True, timeout: float = 0.5):
        """Return a **copy** of the latest frame; optionally block for fresh one."""
        if wait:
            self._new.wait(timeout)
            self._new.clear()
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read(self):
        """Immediate non-blocking frame access (alias for get_frame(wait=False))."""
        return self.get_frame(wait=False)

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        self._frame = None

    #Convenience context-manager

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
