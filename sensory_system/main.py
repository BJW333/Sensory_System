"""
sensors/bootstrap.py
────────────────────
Reusable starter for the ARGUS sensory stack.
Call `start_sensor_hub()` from your orchestrator and keep the returned
`hub` object alive; call `hub.stop()` on shutdown.
"""

from __future__ import annotations
import logging
import threading
import time
from contextlib import ExitStack, contextmanager
from typing import List, Tuple, Optional
from pathlib import Path
import traceback
import signal
import sys

from camera_manager          import CameraManager
from context_fusion          import CONTEXT
from idle_detection_combined import start_idle_detection_thread
from obj_face_recog          import start_obj_person_recog_thread
from app_usage_monitor       import start_app_usage_monitor_thread
from proactive_outputs       import NaturalResponder



def speak(speak_text: str):
    """
    Purpose: framework speak function replace with the real one once program production ready
    """
    print(f"Speak: {speak_text}")  # placeholder for actual speech synthesis
    
# end def
# ───────────────────────── SensorHub (with self-restart) ──────────────────
class SensorHub:
    _RESTART_DELAY   = 3.0       # seconds between restart attempts
    _HANG_TIMEOUT    = 6.0       # no heartbeat for this → treat as hang
    _HB_KEY          = "heartbeat"

    def __init__(self,
                 camera_index: int = 0,
                 debug: bool = False,
                 console_interval: Optional[float] = 2.0):
        self._stack      = ExitStack()
        self._debug      = debug
        self._threads: List[str] = []          # names only; per-sensor data in CONTEXT
        self.console_int = console_interval

        # shared camera
        self.camera = CameraManager(camera_index)
        self._stack.callback(self.camera.stop)

    # ---------- public API -----------------------------------------------
    def start(self) -> "SensorHub":
        self._spawn("idle_detection",   start_idle_detection_thread,
                    self.camera, show_debug=self._debug)
        self._spawn("obj_person_recog", start_obj_person_recog_thread,
                    self.camera, show_debug=self._debug)
        self._spawn("app_usage_monitor", start_app_usage_monitor_thread)
        
        self._responder = NaturalResponder(
            break_delay      = 300,        # idle > 5 min → suggest break
            break_cooldown   = 1800,       # 30 min silence after a break prompt
            app_cooldown     = 3600,       # 1 h silence after an app prompt
            log_file         = Path.cwd() / "context_log.csv",
            speak_fn         = speak       # omit or replace with your own
        )
        
        if self.console_int:
            self._stack.enter_context(
                _thread_context(ContextPrinter(self.console_int).run_forever,
                                name="ContextPrinter")
            )
        return self

    def stop(self):
        CONTEXT.update("sensorhub_shutdown", True)
        self._stack.close()

    def any_dead(self) -> bool:
        """Return True if any sensor is in permanent failure state."""
        for name in self._threads:
            status = CONTEXT.get(f"{name}_status")
            if status not in {"running", "restarting", "stopped", "finished"}:
                return True
        return False

    # ---------- internal helpers -----------------------------------------
    def _spawn(self, name: str, func, *args, **kwargs):
        sup = threading.Thread(
            target=self._supervisor_loop,
            name=f"Sup-{name}",
            args=(name, func, args, kwargs),
            daemon=True,
        )
        sup.start()
        self._threads.append(name)
        logging.info("supervising sensor %s", name)

    # supervisor for a single sensor
    def _supervisor_loop(self, name: str, target, args, kwargs):
        restart = 0
        
        while not CONTEXT.get("sensorhub_shutdown"):
            CONTEXT.update(f"{name}_status", "starting")
            CONTEXT.update(f"{name}_restart_count", restart)

            # ------ launch worker --------
            worker_stop = threading.Event()
            thr = threading.Thread(
                target=self._invoke_with_optional_stop,
                name=name,
                args=(target, worker_stop, *args),
                kwargs=kwargs,
                daemon=True,
            )
            thr.start()
            CONTEXT.update(f"{name}_status", "running")
            CONTEXT.update(f"{name}_{self._HB_KEY}", time.time())

            # ------ supervise loop -------
            try:
                while thr.is_alive() and not CONTEXT.get("sensorhub_shutdown"):
                    thr.join(timeout=1.0)

                    # hang check
                    last_hb = CONTEXT.get(f"{name}_{self._HB_KEY}", 0)
                    if time.time() - last_hb > self._HANG_TIMEOUT:
                        raise RuntimeError("heartbeat timeout")

                if CONTEXT.get("sensorhub_shutdown"):
                    worker_stop.set()
                    thr.join(timeout=2.0)
                    CONTEXT.update(f"{name}_status", "stopped")
                    break
                if not thr.is_alive():
                    # exited cleanly
                    CONTEXT.update(f"{name}_status", "finished")
                    break

            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                CONTEXT.update(f"{name}_status", f"crashed: {exc!r}")
                CONTEXT.update(f"{name}_traceback", tb)
                logging.error("%s crashed: %s", name, exc)

            # ------ restart -------------
            restart += 1
            logging.warning("restarting %s in %.1fs (attempt %d)",
                            name, self._RESTART_DELAY, restart)
            CONTEXT.update(f"{name}_status", "restarting")
            time.sleep(self._RESTART_DELAY)

    # inject stop_event if worker accepts it
    @staticmethod
    def _invoke_with_optional_stop(target, stop_ev, *args, **kwargs):
        from inspect import signature
        if "stop_event" in signature(target).parameters:
            return target(*args, stop_event=stop_ev, **kwargs)
        return target(*args, **kwargs)

# ───────────────────────── context printer ────────────────────────────────
class ContextPrinter:
    def __init__(self, interval: float):
        self.iv   = interval
        self._ver = -1
    def run_forever(self):
        try:
            while True:
                ver, snap = CONTEXT.snapshot()
                if ver != self._ver:
                    print(f"\n=== Context v{ver} {time.strftime('%H:%M:%S')} ===",
                          *[f"{k}: {v}" for k, v in snap.items()], sep="\n", flush=True)
                    self._ver = ver
                time.sleep(self.iv)
        except KeyboardInterrupt:
            pass

# ───────────────────────── helper ─────────────────────────────────────────
@contextmanager
def _thread_context(target, name: str):
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    try:
        yield t
    finally:
        if t.is_alive():
            t.join(timeout=2.0)

# ───────────────────────── public helpers ─────────────────────────────────
def start_sensor_hub(*,
                     camera_index: int = 0,
                     debug: bool = False,
                     console_interval: Optional[float] = 2.0) -> SensorHub:
    """
    Returns a running SensorHub.  Keep it alive for as long as ARGUS needs
    sensory data, then call hub.stop().
    """
    _configure_logging(debug)
    hub = SensorHub(camera_index, debug, console_interval).start()
    return hub

def _configure_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

# ───────────────────────── standalone test entry ──────────────────────────
if __name__ == "__main__":
    hub = start_sensor_hub(debug=False, console_interval=2.0)
    signal.signal(signal.SIGINT,  lambda *_: hub.stop() or sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: hub.stop() or sys.exit(0))
    try:
        while not hub.any_dead():
            time.sleep(1)
    finally:
        hub.stop()