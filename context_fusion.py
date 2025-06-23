# context_fusion.py – thread-safe global state bus for ARGUS (rev2)
# ---------------------------------------------------------------------------
# Change log (rev2):
#   • `batch()` is now a true context-manager with @contextmanager decorator –
#     fixes AttributeError: __enter__ seen in idle_detection loop.
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager, suppress
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator

logger = logging.getLogger(__name__)

Callback = Callable[[str, Any, Any], None]

class ContextFusion:
    """Thread-safe key→value store with change callbacks and version counter."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Dict[str, Tuple[Any, float]] = {}
        self._ver: int = 0
        self._callbacks: List[Callback] = []
        self._change_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, key: str, value: Any) -> None:
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
                except Exception as exc:  # noqa: BLE001
                    logger.exception("ContextFusion callback %s raised: %s", cb, exc)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Group several updates into one version tick.

        Example::
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
        with self._lock:
            data = {
                k: (v, ts) if with_timestamps else v
                for k, (v, ts) in self._state.items()
            }
            return self._ver, data

    def get(self, key: str, default: Any = None):
        with self._lock:
            return self._state.get(key, (default, None))[0]

    def get_timestamp(self, key: str) -> Optional[float]:
        with self._lock:
            return self._state.get(key, (None, None))[1]

    def clear(self) -> None:
        with self._lock:
            self._state.clear(); self._ver = 0
            self._change_event.set(); self._change_event.clear()

    def register_callback(self, callback: Callback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callback) -> None:
        with self._lock:
            with suppress(ValueError):
                self._callbacks.remove(callback)

    def wait_for_change(self, *, timeout: Optional[float] = None) -> bool:
        return self._change_event.wait(timeout)

# singleton
CONTEXT = ContextFusion()