"""capture module: global hotkey listener + screenshot trigger.

Public interface
----------------
capture = Capture(callback, monitor_index=1, debounce_seconds=2.0)
capture.start()   # blocks; press hotkey to fire callback(png_bytes)
capture.stop()    # call from another thread to tear down

The callback receives raw PNG bytes of the chosen display.

The `grabber` collaborator (default `src.screen.Screen`) owns the actual
screen capture and the mss-vs-Quartz capture seam.  In v1 the HUD is
excluded by `NSWindowSharingNone` on the panel (ADR-0002); `Capture` only
wires the hotkey, debounce, and the call to `grabber.grab(..., below_window=...)`. The `below_window` argument is the Quartz fallback seam.
"""

import time as _time_module
from typing import Any, Callable

from pynput import keyboard

from src.config import HOTKEYS
from src.screen import Screen


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DEBOUNCE = 2.0   # seconds
_DEFAULT_MONITOR = 1       # mss monitors[1] == first physical display


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Capture:
    """Trigger surface: hotkey → debounce → screenshot → callback.

    Parameters
    ----------
    callback:
        Called with raw PNG bytes every time a non-debounced trigger fires.
    monitor_index:
        Which display index to pass to the grabber (1 = main display).
    debounce_seconds:
        Minimum gap between successive triggers; shorter presses are dropped.
    clock:
        Callable returning a float (monotonic time).  Injectable for tests.
    grabber:
        Object with ``grab(monitor_index, below_window=None) -> bytes``.
        Defaults to a ``Screen`` instance.
    hotkeys:
        List of pynput key-combination strings, all mapped to the same
        ``trigger()``.  Shared debouncing falls out of using one trigger.
    hud:
        Optional HUD collaborator.  Its ``window_id`` is passed to the grabber
        as the exclusion target.
    """

    def __init__(
        self,
        callback: Callable[[bytes], None],
        monitor_index: int = _DEFAULT_MONITOR,
        debounce_seconds: float = _DEFAULT_DEBOUNCE,
        *,
        clock: Callable[[], float] | None = None,
        grabber: Any | None = None,
        hotkeys: list[str] | None = None,
        hud: Any | None = None,
    ):
        self._callback = callback
        self._monitor_index = monitor_index
        self._debounce = debounce_seconds
        self._clock = clock if clock is not None else _time_module.monotonic
        self._grabber = grabber if grabber is not None else Screen()
        self._hotkeys = hotkeys if hotkeys is not None else HOTKEYS
        self._hud = hud
        self._last_trigger: float = -debounce_seconds  # allow first press immediately
        self._listener: keyboard.GlobalHotKeys | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def trigger(self) -> None:
        """Fire a capture cycle; debounces internally."""
        now = self._clock()
        if now - self._last_trigger < self._debounce:
            return
        self._last_trigger = now
        self._do_capture()

    def start(self) -> None:
        """Start the global hotkey listener (blocks until stop() is called)."""
        hotkeys = {hk: self.trigger for hk in self._hotkeys}
        self._listener = keyboard.GlobalHotKeys(hotkeys)
        self._listener.start()
        self._listener.join()

    def stop(self) -> None:
        """Stop the hotkey listener from another thread."""
        if self._listener is not None:
            self._listener.stop()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_capture(self) -> None:
        below_window = self._hud.window_id if self._hud is not None else None
        png_bytes = self._grabber.grab(self._monitor_index, below_window)
        self._callback(png_bytes)
