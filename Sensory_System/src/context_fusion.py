from __future__ import annotations
import logging
import threading
import time
from contextlib import contextmanager, suppress
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator

logger = logging.getLogger(__name__)

Callback = Callable[[str, Any, Any], None]

class ContextFusion:
    """
    Thread safe key value store with change callbacks and version counter.
    Example::
        CONTEXT.update("face_names", ["Alice", "Bob"])
        names = CONTEXT.get("face_names", [])
        with CONTEXT.batch():
            CONTEXT.update("key1", val1)
            CONTEXT.update("key2", val2)
    Callbacks::
        def on_change(key, old_val, new_val):
            print(f"Context key {key} changed from {old_val} to {new_val}")
        CONTEXT.register_callback(on_change)
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Dict[str, Tuple[Any, float]] = {}
        self._ver: int = 0
        self._callbacks: List[Callback] = []
        self._change_event = threading.Event()

    # Public API
    def update(self, key: str, value: Any) -> None:
        """
        Update the value for a given key in the context.
        Args:
            key (str): The key to update.
            value (Any): The new value for the key.
        Returns:
            None
        """
        with self._lock:
            old_val = self._state.get(key, (None, None))[0]
            if old_val == value:
                return
            self._state[key] = (value, time.time())
            self._ver += 1
            self._change_event.set(); self._change_event.clear()
            for cb in list(self._callbacks):
                try:
                    cb(key, old_val, value)
                except Exception as exc:  #noqa: BLE001
                    logger.exception("ContextFusion callback %s raised: %s", cb, exc)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """
        Group several updates into one version tick.
        Example:
            with CONTEXT.batch():
                CONTEXT.update("foo", 1)
                CONTEXT.update("bar", 2)
        """
        with self._lock:
            orig_ver = self._ver
            yield
            if self._ver != orig_ver:
                self._change_event.set(); self._change_event.clear()

    def snapshot(self, *, with_timestamps: bool = False):
        """
        Return a snapshot of the current context state.
        Args:
            with_timestamps (bool): If True, include timestamps with values.
        Returns:
            Tuple[int, Dict[str, Any]]: A tuple containing the current version and a dictionary of key-value pairs.
        """
        with self._lock:
            data = {
                k: (v, ts) if with_timestamps else v
                for k, (v, ts) in self._state.items()
            }
            return self._ver, data

    def get(self, key: str, default: Any = None):
        """
        Return the value for a given key from the context.
        Args:
            key (str): The key to retrieve.
            default (Any): The default value if the key is not found.
        Returns:
            Any: The value associated with the key or the default.
        """
        with self._lock:
            return self._state.get(key, (default, None))[0]

    def get_timestamp(self, key: str) -> Optional[float]:
        """
        Return the timestamp for a given key from the context.
        Args:
            key (str): The key to retrieve the timestamp for.
        Returns:
            Optional[float]: The timestamp associated with the key or None if not found.
        """
        with self._lock:
            return self._state.get(key, (None, None))[1]

    def clear(self) -> None:
        """
        Clear the context state.
        Returns:
            None
        """
        with self._lock:
            self._state.clear(); self._ver = 0
            self._change_event.set(); self._change_event.clear()

    def register_callback(self, callback: Callback) -> None:
        """
        Register a callback to be called on context changes.
        Args:
            callback (Callback): The callback function to register.
        Returns:
            None
        """
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callback) -> None:
        """
        Unregister a previously registered callback.
        Args:
            callback (Callback): The callback function to unregister.
        Returns:
            None
        """
        with self._lock:
            with suppress(ValueError):
                self._callbacks.remove(callback)

    def wait_for_change(self, *, timeout: Optional[float] = None) -> bool:
        """
        Wait for a change in the context.
        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
        Returns:
            bool: True if a change occurred, False if timed out.
        """
        return self._change_event.wait(timeout)

#singleton
CONTEXT = ContextFusion()