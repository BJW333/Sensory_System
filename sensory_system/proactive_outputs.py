import csv, json, logging, os, threading, time
from pathlib import Path
from queue import SimpleQueue, Empty
from typing import Callable, List, Sequence, Union
import numpy as np

from context_fusion import CONTEXT

log = logging.getLogger(__name__)


class NaturalResponder:
    """Speaks context-aware reminders & writes a CSV change-log."""

    def __init__(
        self,
        *,
        break_delay: int = 300,               # sec idle remind break
        break_cooldown: int = 1800,           # sec between break prompts
        app_cooldown: int = 3600,             # sec between app prompts
        log_file: Union[str, Path] = "context_log.csv",
        speak_fn: Callable[[str], None] = None,
        
    ):
        self.break_delay      = break_delay
        self.break_cooldown   = break_cooldown
        self.app_cooldown     = app_cooldown

        self._idle_timer: threading.Timer | None = None
        self._prev_faces: Sequence[str] = []
        self._last_app_suggestion  = 0.0
        self._last_break_suggestion = 0.0
        
        self._last_face_prompt: dict[str, float] = {}
        self.face_prompt_cooldown = 600
        
        self.speak = speak_fn or (lambda txt: print(f"[TTS] {txt}", flush=True)) #this is where tts would go

        self.log_path = Path(log_file)
        self._ensure_log_header()
        self._q: "SimpleQueue[tuple[float,str,object]]" = SimpleQueue()

        # background writer
        self._writer_th = threading.Thread(target=self._flush_loop, daemon=True)
        self._writer_th.start()

        CONTEXT.register_callback(self._on_ctx_change)
        log.info("NaturalResponder online")

    # ─────────────────────────── private helpers ──────────────────────────
    def _ensure_log_header(self) -> None:
        if not self.log_path.exists():
            self.log_path.write_text("timestamp,key,value\n")

    def _log_async(self, key: str, value) -> None:
        self._q.put((time.time(), key, value))
        
    @staticmethod
    def _json_fallback(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def _flush_loop(self):
        """Write queued rows to CSV; runs as a daemon thread."""
        while True:
            rows = []
            try:
                while True:                     # drain queue
                    rows.append(self._q.get_nowait())
            except Empty:
                if rows:
                    with self.log_path.open("a", newline="") as f:
                        w = csv.writer(f)
                        for ts, k, v in rows:
                            w.writerow([ts, k, json.dumps(v, default=NaturalResponder._json_fallback)])
            time.sleep(0.2)

    def _handle_idle_state(self, new_state: str):
        if new_state == "idle":
            if self._idle_timer:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(
                self.break_delay, self._remind_break
            )
            self._idle_timer.start()
        else:
            if self._idle_timer:
                self._idle_timer.cancel()
                self._idle_timer = None

    def _remind_break(self):
        if time.time() - self._last_break_suggestion >= self.break_cooldown:
            self.speak("You've been idle for a while. Would you like a break?")
            self._last_break_suggestion = time.time()

    # ─────────────────────────── context callback ─────────────────────────
    def _on_ctx_change(self, key: str, old, new) -> None:
        """Registered with CONTEXT; runs in the sensor thread."""
        self._log_async(key, new)

        now = time.time()

        if key == "activity_state":
            self._handle_idle_state(new)

        elif key == "active_app_duration":
            app = CONTEXT.get("active_app")
            if (
                new > 5400                       # > 90-min session
                and CONTEXT.get("activity_state") == "idle"
                and now - self._last_app_suggestion >= self.app_cooldown
            ):
                self.speak(f"You've been in {app} for over 90 minutes. Time for a break?")
                self._last_app_suggestion = now

        elif key == "weather" and isinstance(new, dict):
            if "rain" in new.get("summary", "").lower():
                self.speak("It looks like rain—grab an umbrella if you head out.")

        elif key == "faces_recognized":
            self._face_prompt(new)

    # debounce walk-in prompt
    def _face_prompt(self, faces: List[str]):
        # faces is a list of recognized names (no "Unknown")
        now = time.time()

        new_faces = [f for f in faces if f not in self._prev_faces]
        greeted = []

        for face in new_faces:
            last_greeted = self._last_face_prompt.get(face, 0)
            if now - last_greeted >= self.face_prompt_cooldown:
                greeted.append(face)
                self._last_face_prompt[face] = now

        if greeted:
            self.speak(f"{', '.join(greeted)} just walked in.")

        elif not faces and self._prev_faces:
            self.speak("Everyone just left.")

        self._prev_faces = faces


# instantiate once (no separate thread needed)
NaturalResponder()