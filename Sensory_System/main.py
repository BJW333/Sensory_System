"""
sensors/bootstrap.py
────────────────────
Reusable starter for the sensory stack.
Call `start_sensor_hub()` from your orchestrator and keep the returned
`hub` object alive; call `hub.stop()` on shutdown.
"""
from __future__ import annotations
import logging
import threading
import time
from contextlib import ExitStack, contextmanager
from typing import List, Optional
from pathlib import Path
import traceback
import signal
import sys
from src.camera_manager          import CameraManager
from src.context_fusion          import CONTEXT
from src.proactive_outputs       import NaturalResponder
from src.idle_detection_combined import idle_detection_loop
from src.app_usage_monitor import _AppUsageMonitor
from src.obj_face_recog import obj_person_recog_loop

def speak(speak_text: str):
    """
    Purpose: framework speak function replace with the real one once program production ready
    """
    print(f"Speak: {speak_text}")  #placeholder for actual speech synthesis
    
#SENSOR HUB CLASS
class SensorHub:
    """
    Manages multiple sensor threads with self-restarting supervision.
    Sensors are run in their own threads and monitored for hangs or crashes.
    If a sensor thread crashes or hangs (no heartbeat), it is restarted after a delay.
    ContextFusion is used to share state and status between sensors and the hub.
    Args:
        camera_index (int): Index of the camera to use.
        debug (bool): If True, enable debug logging.
        console_interval (float | None): If set, print CONTEXT to console every
            `console_interval` seconds. If None, disable console output.
    Returns:
        SensorHub: The running sensor hub instance.
    """
    _RESTART_DELAY   = 3.0       #seconds between restart attempts
    _HANG_TIMEOUT    = 6.0       #no heartbeat for this → treat as hang
    _HB_KEY          = "heartbeat"

    def __init__(self,
                 camera_index: int = 0,
                 debug: bool = False,
                 console_interval: Optional[float] = 2.0):
        self._stack      = ExitStack()
        self._debug      = debug
        self._threads: List[str] = []          #names only; per-sensor data in CONTEXT
        self.console_int = console_interval

        #shared camera
        self.camera = CameraManager(camera_index)
        self._stack.callback(self.camera.stop)

    #PUBLIC API
    def start(self) -> "SensorHub":
        """
        Start the sensor hub and all sensors.
        Args:
            None
        Returns:
            SensorHub: The running sensor hub instance.
        """
        #Use loop functions directly with supervisor
        self._spawn(
            "idle_detection",
            idle_detection_loop,
            self.camera,
            show_debug=self._debug,
        )
        
        self._spawn(
            "obj_person_recog",
            obj_person_recog_loop,
            self.camera,
            show_debug=self._debug,
        )
        
        #App monitor create instance and pass run method
        monitor = _AppUsageMonitor()
        self._spawn("app_usage_monitor", lambda stop_event=None: monitor.run(stop_event)) #now has stop_event 


        self._responder = NaturalResponder(
            break_delay=300, # pass break delay time in seconds
            break_cooldown=1800, # pass break cooldown time in seconds
            app_cooldown=3600, # pass app cooldown time in seconds
            log_file=Path.cwd() / "context_log.csv",
            speak_fn=speak, # pass the speak function
            max_log_size=50_000_000,  # 50MB
            privacy_mode=False,  # face/person logging or enable for privacy
        )
        
        if self.console_int:
            self._stack.enter_context(
                _thread_context(
                    ContextPrinter(self.console_int).run_forever,
                    name="ContextPrinter",
                )
            )
        return self

    def stop(self):
        """
        Stop the sensor hub and all sensors.
        Args:
            None
        Returns:
            None
        """
        CONTEXT.update("sensorhub_shutdown", True)
        self._stack.close()

    def any_dead(self) -> bool:
        """
        Return True if any sensor is in permanent failure state.
        Args:
            None
        Returns:
            bool: True if any sensor is dead, False otherwise.
        """
        for name in self._threads:
            status = CONTEXT.get(f"{name}_status")
            if status not in {"running", "restarting", "stopped", "finished"}:
                return True
        return False

    #INTERNAL HELPERS
    def _spawn(self, name: str, func, *args, **kwargs):
        """
        Spawn a supervised thread for a sensor.
        Args:
            name: Name of the sensor/thread.
            func: Function to run in the thread.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        Returns:
            None
        """
        sup = threading.Thread(
            target=self._supervisor_loop,
            name=f"Sup-{name}",
            args=(name, func, args, kwargs),
            daemon=True,
        )
        sup.start()
        self._threads.append(name)
        logging.info("supervising sensor %s", name)

    #supervisor for a single sensor
    def _supervisor_loop(self, name: str, target, args, kwargs):
        """
        Supervise a single sensor thread, restarting it on failure.
        Args:
            name: Name of the sensor/thread.
            target: Function to run in the thread.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
        Returns:
            None
        """
        restart = 0
        
        while not CONTEXT.get("sensorhub_shutdown"):
            CONTEXT.update(f"{name}_status", "starting")
            CONTEXT.update(f"{name}_restart_count", restart)

            #Launch worker thread
            worker_stop = threading.Event()
            thr = threading.Thread(
                target=self._invoke_with_optional_stop,
                args=(target, worker_stop, *args),  #Passes worker_stop
                daemon=True,
            )
            thr.start()
            CONTEXT.update(f"{name}_status", "running")
            CONTEXT.update(f"{name}_{self._HB_KEY}", time.time())

            #Supervise loop
            try:
                while thr.is_alive() and not CONTEXT.get("sensorhub_shutdown"):
                    thr.join(timeout=1.0)

                    #hang check
                    last_hb = CONTEXT.get(f"{name}_last_beat", 0) #default 0
                    if time.time() - last_hb > self._HANG_TIMEOUT:
                        raise RuntimeError("heartbeat timeout")

                if CONTEXT.get("sensorhub_shutdown"):
                    worker_stop.set()
                    thr.join(timeout=2.0)
                    CONTEXT.update(f"{name}_status", "stopped")
                    break
                if not thr.is_alive():
                    #exited cleanly
                    CONTEXT.update(f"{name}_status", "finished")
                    break

            except Exception as exc:  #noqa: BLE001
                tb = traceback.format_exc()
                CONTEXT.update(f"{name}_status", f"crashed: {exc!r}")
                CONTEXT.update(f"{name}_traceback", tb)
                logging.error("%s crashed: %s", name, exc)

            #Restart sensor thread after delay
            restart += 1
            logging.warning("restarting %s in %.1fs (attempt %d)",
                            name, self._RESTART_DELAY, restart)
            CONTEXT.update(f"{name}_status", "restarting")
            time.sleep(self._RESTART_DELAY)

    #inject stop_event if worker accepts it
    @staticmethod
    def _invoke_with_optional_stop(target, stop_ev, *args, **kwargs):
        """
        Inject stop_event if the worker function accepts it.
        Args:
            target: The worker function to invoke.
            stop_ev: The threading.Event used to signal stopping.
            *args: Positional arguments for the worker function.
            **kwargs: Keyword arguments for the worker function.
        Returns:
            The result of the worker function.
        """
        from inspect import signature
        if "stop_event" in signature(target).parameters:
            return target(*args, stop_event=stop_ev, **kwargs)
        return target(*args, **kwargs)

#CONTEXT PRINTER CLASS
class ContextPrinter:
    def __init__(self, interval: float):
        self.iv   = interval
        self._ver = -1
        
    def run_forever(self):
        """
        Continuously print the current context snapshot at the specified interval.
        Args:
            None
        Returns:
            None
        """
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

#HELPER CONTEXT MANAGER
@contextmanager
def _thread_context(target, name: str):
    """
    Context manager to run a thread and ensure it is properly joined.
    Args:
        target: The target function for the thread.
        name: The name of the thread.
    Yields:
        threading.Thread: The started thread.
    """
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    try:
        yield t
    finally:
        if t.is_alive():
            t.join(timeout=2.0)

#START SENSOR HUB FUNCTION
def start_sensor_hub(
    *,
    camera_index: int = 0,
    debug: bool = True,
    console_interval: Optional[float] = 2.0
) -> SensorHub:  
    """
    Returns a running SensorHub. Keep it alive for as long as the AI system needs
    sensory data, then call hub.stop().
    Args:
        camera_index (int): Index of the camera to use (default: 0)
        debug (bool): If True, enable debug logging (default: True)
        console_interval (float | None): If set, print CONTEXT to console every
            `console_interval` seconds. If None, disable console output. (default: 2.0)
    Returns:
        SensorHub: The running sensor hub instance.
    """
    _configure_logging(debug)
    hub = SensorHub(camera_index, debug, console_interval).start()
    return hub

def _configure_logging(debug: bool):
    """
    Configure logging for the application.
    Args:
        debug (bool): If True, set logging level to DEBUG, else INFO.
    Returns:
        None
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
def safe_shutdown(signum, frame):
    """
    Safely shutdown the sensor hub on receiving termination signals.
    Args:
        signum: The signal number.
        frame: The current stack frame.
    Returns:
        None
    """
    global hub
    try:
        if 'hub' in globals() and hub is not None:
            hub.stop()
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")
    finally:
        sys.exit(0)

# MAIN ENTRY POINT
if __name__ == "__main__":
    hub = start_sensor_hub(debug=False, console_interval=2.0)
    #signal.signal(signal.SIGINT,  lambda *_: hub.stop() or sys.exit(0)) #old line
    #signal.signal(signal.SIGTERM, lambda *_: hub.stop() or sys.exit(0)) #old line
    signal.signal(signal.SIGINT, safe_shutdown)
    signal.signal(signal.SIGTERM, safe_shutdown)
    try:
        while not hub.any_dead():
            time.sleep(1)
    finally:
        hub.stop()