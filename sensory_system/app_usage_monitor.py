from __future__ import annotations
import time
import threading
import logging
import sys
from typing import Optional
from context_fusion import CONTEXT

logger = logging.getLogger(__name__)

# macOS-specific import guarded by try/except
try:
    from AppKit import NSWorkspace  # type: ignore
except Exception:  # noqa: BLE001
    NSWorkspace = None

CHECK_INTERVAL = 5.0  # seconds between polls


#Internal helper

class _AppUsageMonitor:
    def __init__(self):
        self.last_app: Optional[str] = None
        self.last_app_start: float = time.time()

    #One poll iteration

    def _poll(self):
        """Query frontmost app; return tuple(name, now)."""
        try:
            app_name = (
                NSWorkspace.sharedWorkspace()  # type: ignore[attr-defined]
                .frontmostApplication()
                .localizedName()
            )
            return app_name, time.time()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NSWorkspace poll failed: %s", exc)
            return None, time.time()

    #Main loop
    

    def run(self, stop_event: Optional[threading.Event] = None):
        #  platform checks 
        if NSWorkspace is None:
            if sys.platform == "darwin":
                raise RuntimeError("PyObjC (AppKit) missing – install with `pip install pyobjc`.")
            CONTEXT.update("app_usage_monitor_status", "unsupported_platform")
            logger.info("App usage monitor disabled: unsupported platform")
            return

        CONTEXT.update("app_usage_monitor_status", "running")

        # initial population so other modules have a value
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
                # Switch detected
                CONTEXT.update("active_app", app)
                CONTEXT.update("active_app_start_time", now)
                self.last_app, self.last_app_start = app, now
            # duration always updated, even on switch (effectively 0 at switch time)
            CONTEXT.update("active_app_duration", now - self.last_app_start)

            time.sleep(CHECK_INTERVAL)

        # graceful shutdown
        CONTEXT.update("app_usage_monitor_status", "stopped")
        CONTEXT.update("active_app", None)
        CONTEXT.update("active_app_duration", 0)


#Threading 
def start_app_usage_monitor_thread(*, check_interval: float | None = None) -> threading.Thread:
    """Fire-and-forget daemon thread; returns Thread so caller may stop it."""
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
    return t  # caller: t._kwargs["stop_event"].set() to stop
