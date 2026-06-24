"""capture module: global hotkey listener + screenshot.

Public interface
----------------
capture = Capture(callback, monitor_index=1, debounce_seconds=2.0)
capture.start()   # blocks; press hotkey to fire callback(png_bytes)
capture.stop()    # call from another thread to tear down

The callback receives raw PNG bytes of the chosen display.

Collaborators are injectable for testing (clock, mss_factory, to_png).
Hardware-bound behaviour (actual hotkey, real screen grab) is verified
manually, not in unit tests.
"""

import time as _time_module
from typing import Callable

import mss
import mss.tools
from pynput import keyboard


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
        Which entry in ``mss.monitors`` to capture (1 = main display).
    debounce_seconds:
        Minimum gap between successive triggers; shorter presses are dropped.
    clock:
        Callable returning a float (monotonic time).  Injectable for tests.
    mss_factory:
        Callable returning an mss context-manager.  Injectable for tests.
    to_png:
        Callable ``(rgb, size) -> bytes``.  Injectable for tests.
    hotkey:
        Key combination string understood by ``pynput.keyboard.HotKey``
        (e.g. ``"<ctrl>+<alt>+<space>"``).
    """

    def __init__(
        self,
        callback: Callable[[bytes], None],
        monitor_index: int = _DEFAULT_MONITOR,
        debounce_seconds: float = _DEFAULT_DEBOUNCE,
        *,
        clock: Callable[[], float] | None = None,
        mss_factory: Callable | None = None,
        to_png: Callable | None = None,
        hotkey: str = "<ctrl>+<alt>+<space>",
    ):
        self._callback = callback
        self._monitor_index = monitor_index
        self._debounce = debounce_seconds
        self._clock = clock if clock is not None else _time_module.monotonic
        self._mss_factory = mss_factory if mss_factory is not None else mss.MSS
        self._to_png = to_png if to_png is not None else mss.tools.to_png
        self._hotkey_str = hotkey
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
        hotkeys = {self._hotkey_str: self.trigger}
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
        with self._mss_factory() as sct:
            monitor = sct.monitors[self._monitor_index]
            shot = sct.grab(monitor)
            png_bytes = self._to_png(shot.rgb, shot.size)
        self._callback(png_bytes)
