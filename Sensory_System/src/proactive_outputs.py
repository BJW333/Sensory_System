import csv, json, logging, threading, time
from pathlib import Path
from queue import SimpleQueue, Empty
from typing import Callable, List, Sequence, Union, Optional
import numpy as np
from src.context_fusion import CONTEXT

log = logging.getLogger(__name__)

class NaturalResponder:
    """
    Speaks context aware reminders and writes a CSV change log.
    Args:
        break_delay: seconds of idle before break reminder
        break_cooldown: seconds between break reminders
        app_cooldown: seconds between app usage reminders
        log_file: path to CSV log file
        speak_fn: function to call with text to speak
        max_log_size: maximum log file size in bytes before rotation (default 50MB)
        privacy_mode: if True, anonymize face names in logs (default False)
    Returns:
        NaturalResponder instance
    """
    def __init__(
        self,
        *,
        break_delay: int = 300,               #sec idle remind break
        break_cooldown: int = 1800,           #sec between break prompts
        app_cooldown: int = 3600,             #sec between app prompts
        log_file: Union[str, Path] = "context_log.csv",
        speak_fn: Callable[[str], None] = None,
        max_log_size: int = 50_000_000,       #50MB default
        privacy_mode: bool = False,           #Anonymize sensitive data
    ):
    
        self.break_delay = break_delay
        self.break_cooldown = break_cooldown
        self.app_cooldown = app_cooldown
        self.max_log_size = max_log_size
        self.privacy_mode = privacy_mode 

        self._idle_timer: Optional[threading.Timer] = None #track idle timer
        self._prev_faces: Sequence[str] = [] #track previously seen faces
        self._last_app_suggestion = 0.0 #track last app suggestion time
        self._last_break_suggestion = 0.0 #track last break suggestion time
        
        self._last_face_prompt: dict[str, float] = {} #track last prompt times
        self.face_prompt_cooldown = 600 #10 min between same face prompts
            
        #Add delay before "left" announcement
        self._faces_left_time: float = 0
        self.left_announcement_delay = 3.0  #Wait 3 seconds before announcing that someone left

        self.speak = speak_fn or (lambda txt: print(f"[TTS] {txt}", flush=True))

        self.log_path = Path(log_file)
        self._ensure_log_header()
        self._q: SimpleQueue[tuple[float, str, object]] = SimpleQueue() 

        #Track if writer thread should stop
        self._stop_writer = threading.Event()
        
        #Background writer
        self._writer_th = threading.Thread(target=self._flush_loop, daemon=True)
        self._writer_th.start()

        CONTEXT.register_callback(self._on_context_change)
        log.info("NaturalResponder online")

    def _ensure_log_header(self) -> None:
        """
        Ensure the log file has a header row.
        """
        if not self.log_path.exists():
            self.log_path.write_text("timestamp,key,value\n")

    def _anonymize_value(self, key: str, value):
        """
        Anonymize sensitive data if privacy mode is enabled
        Args:
            key: context key
            value: context value
        Returns:
            anonymized value
        """
        if not self.privacy_mode:
            return value
            
        #Anonymize face names
        if key == "faces_recognized" and isinstance(value, list):
            return [f"Person_{hash(name) % 10000}" for name in value]
        
        #Don't log specific app names
        if key == "active_app" and isinstance(value, str):
            return "REDACTED_APP"
            
        return value

    def _log_async(self, key: str, value) -> None:
        """
        Queue a context change for asynchronous logging.
        Args:
            key: context key
            value: context value
        Returns:
            None
        """
        anonymized_value = self._anonymize_value(key, value) #Anonymize if needed
        self._q.put((time.time(), key, anonymized_value)) #Enqueue for logging
        
    def _rotate_log_if_needed(self) -> None:
        """
        Rotate log file if it exceeds max size
        Args:
            None
        Returns:
            None
        """
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > self.max_log_size:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                new_name = self.log_path.parent / f"{self.log_path.stem}_{timestamp}{self.log_path.suffix}"
                self.log_path.rename(new_name)
                self._ensure_log_header()
                log.info(f"Rotated log file to {new_name}")
                
                #Optional: Clean up old logs (keep last 5)
                self._cleanup_old_logs(keep_count=5)
        except Exception as e:
            log.error(f"Failed to rotate log: {e}")
    
    def _cleanup_old_logs(self, keep_count: int = 5) -> None:
        """
        Keep only the most recent N log files
        Args:
            keep_count: number of recent logs to keep
        Returns:
            None
        """
        try:
            pattern = f"{self.log_path.stem}_*{self.log_path.suffix}"
            old_logs = sorted(self.log_path.parent.glob(pattern))
            
            if len(old_logs) > keep_count:
                for old_log in old_logs[:-keep_count]:
                    old_log.unlink()
                    log.info(f"Deleted old log: {old_log}")
        except Exception as e:
            log.error(f"Failed to cleanup old logs: {e}")
            
    @staticmethod
    def _json_fallback(obj):
        """
        JSON serializer fallback for non-serializable objects.
        Args:
            obj: object to serialize
        Returns:
            serializable representation
        """
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def _flush_loop(self):
        """
        Write queued rows to CSV; runs as a daemon thread.
        Args:
            None
        Returns:
            None
        """
        write_counter = 0
        
        while not self._stop_writer.is_set():
            rows = []
            try:
                #Limit batch size to prevent memory issues
                for _ in range(100):  #Process max 100 at a time
                    try:
                        rows.append(self._q.get_nowait())
                    except Empty:
                        break
                        
                if rows:
                    #Check for rotation before writing
                    self._rotate_log_if_needed()
                    
                    with self.log_path.open("a", newline="", encoding='utf-8') as f:
                        w = csv.writer(f)
                        for ts, k, v in rows:
                            w.writerow([ts, k, json.dumps(v, default=NaturalResponder._json_fallback)])
                    
                    write_counter += len(rows)
                    
                    #Periodic rotation check even if file isn't full
                    if write_counter % 1000 == 0:
                        self._rotate_log_if_needed()
                        
            except Exception as e:
                log.error(f"Failed to write CSV: {e}")
                
            #Use wait with timeout for graceful shutdown
            self._stop_writer.wait(timeout=0.2)

    def _handle_idle_state(self, new_state: str):
        """
        Handle transitions in idle state.
        Args:
            new_state: new activity state
        Returns:
            None
        """
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
        """
        Remind the user to take a break if they've been idle for a while.
        Returns:
            None
        """
        if time.time() - self._last_break_suggestion >= self.break_cooldown:
            self.speak("You've been idle for a while. Would you like a break?")
            self._last_break_suggestion = time.time()

    def _on_context_change(self, key: str, old, new) -> None:
        """
        Registered with CONTEXT; runs in the sensor thread.
        Args:
            key: context key that changed
            old: previous value
            new: new value
        Returns:
            None
        """
        #Filter out high frequency updates to prevent log spam
        if key in ["idle_detection_last_beat", "obj_person_recog_last_beat", "app_usage_monitor_last_beat"]:
            return  #Don't log heartbeats
            
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
            #Only process face prompts if not in privacy mode
            if not self.privacy_mode:
                self._face_prompt(new)
            else:
                log.debug("Face prompts disabled in privacy mode")

    def _face_prompt(self, faces: List[str]):
        """
        Handle face recognition announcements
        Args:
            faces: list of recognized face names
        Returns:
            None
        """
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
            self._faces_left_time = 0  #Reset left timer
        
        #Only announce "left" after consistent absence
        elif not faces and self._prev_faces:
            if self._faces_left_time == 0:
                self._faces_left_time = now  #Start timer
            elif now - self._faces_left_time >= self.left_announcement_delay:
                self.speak("Everyone just left.")
                self._faces_left_time = 0
        else:
            self._faces_left_time = 0  #Reset if faces present
        
        self._prev_faces = faces

    def stop(self):
        """
        Gracefully stop the responder
        Args:
            None
        Returns:
            None
        """
        log.info("Stopping NaturalResponder...")
        
        #Stop the idle timer
        if self._idle_timer:
            self._idle_timer.cancel()
            
        #Unregister callback
        CONTEXT.unregister_callback(self._on_context_change)
        
        #Stop writer thread
        self._stop_writer.set()
        self._writer_th.join(timeout=2.0)
        
        #Flush remaining items
        remaining = []
        try:
            while True:
                remaining.append(self._q.get_nowait())
        except Empty:
            pass
            
        if remaining:
            try:
                with self.log_path.open("a", newline="", encoding='utf-8') as f:
                    w = csv.writer(f)
                    for ts, k, v in remaining:
                        w.writerow([ts, k, json.dumps(v, default=NaturalResponder._json_fallback)])
            except Exception as e:
                log.error(f"Failed to flush remaining logs: {e}")
        
        log.info("NaturalResponder stopped")


#Don't instantiate globally let main.py handle it
#This allows for proper configuration and lifecycle management