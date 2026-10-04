"""Display enumeration, geometry, coordinate conversion, and capture."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mss
import mss.tools


@dataclass(frozen=True)
class Display:
    index: int
    left: int
    top: int
    width: int
    height: int


class DisplayBackend(Protocol):
    """Adapter seam. Two adapters: MSSDisplayBackend in production, a fake in tests."""

    def __enter__(self) -> "DisplayBackend":
        ...

    def __exit__(self, *_) -> None:
        ...

    @property
    def monitors(self) -> list[Mapping[str, int]]:
        ...

    def grab(self, monitor: Mapping[str, int]) -> bytes:
        ...


class MSSDisplayBackend:
    def __init__(
        self,
        *,
        mss_factory: Callable[[], Any] | None = None,
        to_png: Callable[..., bytes | None] | None = None,
    ) -> None:
        self._mss_factory = mss_factory if mss_factory is not None else mss.MSS
        self._to_png = to_png if to_png is not None else mss.tools.to_png
        self._sct: Any = None

    def __enter__(self) -> "MSSDisplayBackend":
        self._sct = self._mss_factory()
        return self

    def __exit__(self, *_) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
        self._sct = None

    @property
    def monitors(self) -> list[Mapping[str, int]]:
        if self._sct is None:
            raise RuntimeError("MSSDisplayBackend used outside a context manager")
        return self._sct.monitors

    def grab(self, monitor: Mapping[str, int]) -> bytes:
        shot = self._sct.grab(monitor)
        png = self._to_png(shot.rgb, shot.size)
        if png is None:
            raise RuntimeError("PNG conversion returned no bytes")
        return png


def appkit_bottom_right(
    displays: Sequence[Display],
    display: Display,
    size: int,
    padding: int,
) -> tuple[float, float]:
    """Return AppKit coordinates for a square in a display's bottom-right corner."""
    main = next((item for item in displays if item.left == 0 and item.top == 0), displays[0])
    x = display.left + display.width - size - padding
    y = -(display.top + display.height) + main.height + padding
    return float(x), float(y)


class Displays:
    def __init__(
        self,
        *,
        backend_factory: Callable[[], DisplayBackend] | None = None,
    ) -> None:
        self._backend_factory = (
            backend_factory if backend_factory is not None else MSSDisplayBackend
        )

    def list(self) -> list[Display]:
        with self._backend_factory() as backend:
            return [
                Display(
                    index=index,
                    left=monitor["left"],
                    top=monitor["top"],
                    width=monitor["width"],
                    height=monitor["height"],
                )
                for index, monitor in enumerate(backend.monitors[1:], start=1)
            ]

    def get(self, index: int) -> Display:
        with self._backend_factory() as backend:
            monitors = backend.monitors
            if index < 1 or index >= len(monitors):
                raise ValueError(f"No display {index}")
            monitor = monitors[index]
            return Display(
                index=index,
                left=monitor["left"],
                top=monitor["top"],
                width=monitor["width"],
                height=monitor["height"],
            )

    def grab(self, index: int) -> bytes:
        with self._backend_factory() as backend:
            monitors = backend.monitors
            if index < 1 or index >= len(monitors):
                raise ValueError(f"No display {index}")
            return backend.grab(monitors[index])

    def appkit_bottom_right(
        self,
        index: int,
        size: int,
        padding: int,
    ) -> tuple[float, float]:
        displays = self.list()
        display = next((item for item in displays if item.index == index), None)
        if display is None:
            raise ValueError(f"No display {index}")
        return appkit_bottom_right(displays, display, size, padding)
