"""screen module: capture a display as PNG bytes.

Public interface
----------------
screen = Screen()
screen.grab(1)                       # PNG bytes of monitor 1
screen.grab(1, below_window=12345)   # PNG bytes, excluding HUD if backend can

Screen owns the mss-vs-Quartz capture decision (ADR-0002).  The v1 backend
uses `mss`; the HUD is excluded by `NSWindowSharingNone` on the panel.  The
`below_window` parameter is the seam for a future Quartz
`kCGWindowListOptionOnScreenBelowWindow` backend: the interface is identical,
so callers do not change if the backend is swapped.

The default `MSSBackend` is a hardware-backed context manager.  Tests inject a
fake backend via `Screen(backend_factory=fake)`.
"""

from typing import Any, Callable, Mapping, Protocol

import mss
import mss.tools


class Backend(Protocol):
    """Screen-capture backend used by `Screen`."""

    def __enter__(self) -> Any:
        ...

    def __exit__(self, *_) -> None:  # noqa: ANN002
        ...

    @property
    def monitors(self) -> list[Mapping[str, int]]:
        ...

    def grab(self, monitor: Mapping[str, int], below_window: int | None = None) -> bytes:
        """Capture `monitor` and return raw PNG bytes.

        `below_window` is the window number to exclude when a backend supports
        per-window exclusion (Quartz).  The mss backend ignores it because the
        HUD is excluded by the panel's sharing type.
        """
        ...


class MSSBackend:
    """mss-backed display capture.

    Excludes the HUD not here but in `hud.py` via `NSWindowSharingNone`
    (ADR-0002).  `below_window` is accepted for the Quartz fallback seam but
    is intentionally unused in this backend.
    """

    def __init__(
        self,
        *,
        mss_factory: Callable[[], Any] | None = None,
        to_png: Callable[..., bytes | None] | None = None,
    ) -> None:
        self._mss_factory = mss_factory if mss_factory is not None else mss.MSS
        self._to_png = to_png if to_png is not None else mss.tools.to_png
        self._sct: Any = None

    def __enter__(self) -> "MSSBackend":
        self._sct = self._mss_factory()
        return self

    def __exit__(self, *_) -> None:  # noqa: ANN002
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
        self._sct = None

    @property
    def monitors(self) -> list[Mapping[str, int]]:
        if self._sct is None:
            raise RuntimeError("MSSBackend used outside a context manager")
        return self._sct.monitors

    def grab(self, monitor: Mapping[str, int], below_window: int | None = None) -> bytes:
        _ = below_window
        shot = self._sct.grab(monitor)
        png = self._to_png(shot.rgb, shot.size)
        if png is None:
            raise RuntimeError("PNG conversion returned no bytes")
        return png


class Screen:
    """Public display capture: selects a monitor and asks a backend for PNG bytes."""

    def __init__(self, *, backend_factory: Callable[[], Backend] | None = None) -> None:
        self._backend_factory = backend_factory if backend_factory is not None else MSSBackend

    def grab(self, monitor_index: int, below_window: int | None = None) -> bytes:
        """Return PNG bytes for `monitor_index`.

        `below_window` is part of the public seam for a Quartz fallback; the v1
        `MSSBackend` ignores it because the HUD is excluded by the panel's
        sharing type (ADR-0002).
        """
        with self._backend_factory() as backend:
            monitor = backend.monitors[monitor_index]
            return backend.grab(monitor, below_window)
