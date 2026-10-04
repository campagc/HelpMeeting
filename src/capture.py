"""Global hotkey capture of the selected display as a Slide.

The HUD is excluded from the Slide by NSWindowSharingNone (ADR-0002).
"""

import time as _time_module
from typing import Any, Callable

from pynput import keyboard

from src.config import HOTKEYS

_DEFAULT_DEBOUNCE = 2.0


class Capture:
    """Translate hotkeys into debounced display captures."""

    def __init__(
        self,
        callback: Callable[[bytes], None],
        *,
        displays: Any,
        display_index: int,
        debounce_seconds: float = _DEFAULT_DEBOUNCE,
        clock: Callable[[], float] | None = None,
        hotkeys: list[str] | None = None,
    ) -> None:
        self._callback = callback
        self._displays = displays
        self._display_index = display_index
        self._debounce = debounce_seconds
        self._clock = clock if clock is not None else _time_module.monotonic
        self._hotkeys = hotkeys if hotkeys is not None else HOTKEYS
        self._last_trigger: float = -debounce_seconds  # allow first press immediately
        self._listener: keyboard.GlobalHotKeys | None = None

    def trigger(self) -> None:
        """Fire a capture cycle; debounces internally."""
        now = self._clock()
        if now - self._last_trigger < self._debounce:
            return
        self._last_trigger = now
        self._do_capture()

    def start(self) -> None:
        """Start the global hotkey listener (blocks until stop() is called)."""
        hotkeys = {hotkey: self.trigger for hotkey in self._hotkeys}
        self._listener = keyboard.GlobalHotKeys(hotkeys)
        self._listener.start()
        self._listener.join()

    def stop(self) -> None:
        """Stop the hotkey listener from another thread."""
        if self._listener is not None:
            self._listener.stop()

    def _do_capture(self) -> None:
        self._callback(self._displays.grab(self._display_index))
