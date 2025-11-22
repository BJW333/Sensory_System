from __future__ import annotations
import time
import threading
import logging
from typing import Optional, Tuple
import platform
import subprocess
from src.context_fusion import CONTEXT

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 5.0  #seconds between polls

def get_foreground_app() -> str:
    """
    Get the name of the currently active (foreground) application.
    Returns:
        str: Name of the active application.
    """
    system = platform.system()

    if system == "Darwin":
        try:
            import Quartz
        except ImportError:
            return "[Quartz Import Error]"
        
        try:
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
        except Exception as e:
            return f"[Darwin Error: {e}]"
        
        return "Unknown"
        
    elif system == "Windows":
        try:
            import win32gui
            return win32gui.GetWindowText(win32gui.GetForegroundWindow())
        except Exception as e:
            return f"[Windows Error: {e}]"

    elif system == "Linux":
        try:
            result = subprocess.run(["wmctrl", "-lp"], stdout=subprocess.PIPE, text=True)
            lines = result.stdout.strip().splitlines()
            if len(lines) > 1:
                #wmctrl format: ID WORKSPACE PID MACHINE NAME
                #Split on first 4 whitespaces only to preserve window name
                parts = lines[1].split(None, 4)  #Split on whitespace, max 5 parts
                if len(parts) >= 5:
                    return parts[4]  #Window name is everything after column 4
                return "[None]"
            return "[None]"
        except Exception as e:
            return f"[Linux Error: {e}]"

    return "[Unsupported OS]"


class _AppUsageMonitor:
    def __init__(self):
        self.last_app: Optional[str] = None
        self.last_app_start: float = time.time()

    def _poll(self) -> Tuple[Optional[str], float]:
        """
        Poll the current active application.
        Returns:
            Tuple[Optional[str], float]: Active application name and current timestamp.
        """
        try:
            app_name = get_foreground_app()
            if not app_name:
                logger.info("No active app detected.")
                CONTEXT.update("active_app", None)
                return None, time.time()
            logger.info(f"Active app: {app_name}")  
            return app_name, time.time()
        except Exception as exc:
            logger.warning("NSWorkspace poll failed: %s", exc)
            return None, time.time()

    def run(self, stop_event: Optional[threading.Event] = None):
        """
        Run the app usage monitor loop.
        If stop_event is provided, the loop will exit when stop_event.is_set() returns True.
        This method updates CONTEXT with the active application and its usage duration.
        Args:
            stop_event (Optional[threading.Event]): Event to signal stopping the loop.
        Returns:
            None
        """
        CONTEXT.update("app_usage_monitor_status", "running")
        
        try:
            while stop_event is None or not stop_event.is_set():
                #Update heartbeat at START of every loop
                CONTEXT.update("app_usage_monitor_last_beat", time.time())
                
                try:
                    app, now = self._poll()
                    
                    if app is None:
                        time.sleep(CHECK_INTERVAL)
                        continue  #Loop back and update heartbeat again
                    
                    #App changed
                    if app != self.last_app:
                        CONTEXT.update("active_app", app)
                        CONTEXT.update("active_app_start_time", now)
                        self.last_app = app
                        self.last_app_start = now
                    
                    #Update duration
                    CONTEXT.update("active_app_duration", now - self.last_app_start)
                    
                    time.sleep(CHECK_INTERVAL)
                    
                except Exception as e:
                    #Handle errors in polling
                    logger.exception("Poll failed: %s", e)
                    CONTEXT.update("app_usage_monitor_last_beat", time.time())
                    time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("App usage monitor interrupted")
        
        except Exception as e:
            logger.exception("App usage monitor crashed: %s", e)
        
        finally:
            #Only runs when exiting the loop
            CONTEXT.update("app_usage_monitor_status", "stopped")
            CONTEXT.update("active_app", None)
            CONTEXT.update("active_app_duration", 0)


def start_app_usage_monitor_thread(*, check_interval: float | None = None) -> Tuple[threading.Thread, threading.Event]:
    """
    Start the app usage monitor in a background thread and return both the thread and its stop_event.     
    Args:
        check_interval (float | None): Optional interval between checks in seconds. If None, uses default.
    Returns:
        Tuple[threading.Thread, threading.Event]: The monitor thread and the event to signal stopping it.
    """
    #global CHECK_INTERVAL #old line
    #if check_interval is not None: #old line
    #    CHECK_INTERVAL = check_interval #old line

    stop_event = threading.Event()
    #monitor = _AppUsageMonitor() #old line
    #new line below revert to old lines if issues arise
    monitor = _AppUsageMonitor(check_interval=check_interval or CHECK_INTERVAL) #new line
    
    t = threading.Thread(
        target=monitor.run,
        kwargs={"stop_event": stop_event},
        name="AppUsageMonitor",
        daemon=True,
    )
    t.start()
    return t, stop_event  #use stop_event.set() to stop