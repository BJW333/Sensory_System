from __future__ import annotations
import time
import threading
import logging
from typing import Optional, Tuple
from context_fusion import CONTEXT
import platform
import Quartz

logger = logging.getLogger(__name__)


CHECK_INTERVAL = 5.0  # seconds between polls

def get_foreground_app() -> str:
    system = platform.system()

    if system == "Darwin":
        options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

        for window in window_list:
            if (
                window.get("kCGWindowLayer", 1) == 0
                and window.get("kCGWindowOwnerName")
                and window.get("kCGWindowAlpha", 1) > 0
                and window.get("kCGWindowBounds", {}).get("Height", 0) > 100
            ):
                return window.get("kCGWindowOwnerName")

        return "Unknown"
        
    elif system == "Windows":
        try:
            import win32gui
            return win32gui.GetWindowText(win32gui.GetForegroundWindow())
        except Exception as e:
            return f"[Windows Error: {e}]"

    elif system == "Linux":
        try:
            import subprocess
            result = subprocess.run(["wmctrl", "-lp"], stdout=subprocess.PIPE, text=True)
            return result.stdout.splitlines()[0] if result.stdout else "[None]"
        except Exception as e:
            return f"[Linux Error: {e}]"

    return "[Unsupported OS]"


class _AppUsageMonitor:
    def __init__(self):
        self.last_app: Optional[str] = None
        self.last_app_start: float = time.time()

    def _poll(self) -> Tuple[Optional[str], float]:
        """Query frontmost app; return tuple(name, now)."""
        try:
            app_name = get_foreground_app()
            if not app_name:
                print("No active app detected.")
                CONTEXT.update("active_app", None)
                return None, time.time()
            #print(f"Active app: {app_name}")  # Debug output
            return app_name, time.time()
        except Exception as exc:
            logger.warning("NSWorkspace poll failed: %s", exc)
            return None, time.time()

    def run(self, stop_event: Optional[threading.Event] = None):
        """Run the app usage monitor loop."""

        CONTEXT.update("app_usage_monitor_status", "running")

        try:
            first_app, now = self._poll()
            if first_app:
                CONTEXT.update("active_app", first_app)
                CONTEXT.update("active_app_start_time", now)
                self.last_app, self.last_app_start = first_app, now

            while stop_event is None or not stop_event.is_set():
                CONTEXT.update("app_usage_monitor_last_beat", time.time())

                app, now = self._poll()
                if app is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                if app != self.last_app:
                    CONTEXT.update("active_app", app)
                    CONTEXT.update("active_app_start_time", now)
                    self.last_app, self.last_app_start = app, now

                CONTEXT.update("active_app_duration", now - self.last_app_start)
                time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.exception("App usage monitor crashed: %s", e)

        finally:
            CONTEXT.update("app_usage_monitor_status", "stopped")
            CONTEXT.update("active_app", None)
            CONTEXT.update("active_app_duration", 0)


def start_app_usage_monitor_thread(*, check_interval: float | None = None) -> Tuple[threading.Thread, threading.Event]:
    """Start the app usage monitor in a background thread and return both the thread and its stop_event."""
    global CHECK_INTERVAL
    if check_interval is not None:
        CHECK_INTERVAL = check_interval

    stop_event = threading.Event()
    monitor = _AppUsageMonitor()

    t = threading.Thread(
        target=monitor.run,
        kwargs={"stop_event": stop_event},
        name="AppUsageMonitor",
        daemon=True,
    )
    t.start()
    return t, stop_event  # Use stop_event.set() to stop