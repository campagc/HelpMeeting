"""Tests for the capture module (issue #7).

Hardware-bound behaviour (actual hotkey listener, real mss grabs) is validated
manually.  Everything that lives above the hardware boundary is tested here:

  - Debounce: a second trigger arriving within the debounce window is dropped.
  - Debounce: a trigger arriving after the window fires the callback.
  - Screenshot bytes: the callback receives raw PNG bytes taken from the
    startup-selected monitor.
  - Display default: when no monitor index is given, monitor 1 is used.

The capture module accepts injected collaborators so tests never touch the OS.
"""
import pytest

from src.capture import Capture


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------

class FakeClock:
    """Controllable monotonic clock."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


class FakeMss:
    """Minimal stand-in for an mss context manager.

    monitors[0] is the combined virtual display (mss convention);
    monitors[1] is the first physical display.
    """

    def __init__(self, monitors=None, png_bytes=b"\x89PNG fake"):
        self.monitors = monitors or [
            {"left": 0, "top": 0, "width": 2880, "height": 900},   # combined
            {"left": 0, "top": 0, "width": 1440, "height": 900},   # display 1
            {"left": 1440, "top": 0, "width": 1440, "height": 900}, # display 2
        ]
        self._png_bytes = png_bytes
        self.grabbed: list[dict] = []  # records which monitor dicts were grabbed

    def grab(self, monitor: dict) -> "FakeShot":
        self.grabbed.append(monitor)
        return FakeShot(self._png_bytes)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class FakeShot:
    def __init__(self, png_bytes: bytes):
        self._png_bytes = png_bytes
        self.rgb = png_bytes          # not real RGB but enough for the fake
        self.size = (1440, 900)


def fake_to_png(rgb, size) -> bytes:
    """Stand-in for mss.tools.to_png — just return whatever rgb was."""
    return rgb


# ---------------------------------------------------------------------------
# Tracer bullet: debounce suppresses second trigger within 2 s
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_second_trigger_within_window_is_dropped(self):
        clock = FakeClock(start=0.0)
        fired: list[bytes] = []
        capture = Capture(
            callback=fired.append,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=clock,
            mss_factory=lambda: FakeMss(),
            to_png=fake_to_png,
        )

        capture.trigger()           # t=0 → fires
        clock.advance(1.0)
        capture.trigger()           # t=1 → within 2s window → dropped

        assert len(fired) == 1

    def test_trigger_after_window_fires_again(self):
        clock = FakeClock(start=0.0)
        fired: list[bytes] = []
        capture = Capture(
            callback=fired.append,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=clock,
            mss_factory=lambda: FakeMss(),
            to_png=fake_to_png,
        )

        capture.trigger()           # t=0 → fires
        clock.advance(2.1)
        capture.trigger()           # t=2.1 → outside window → fires again

        assert len(fired) == 2


# ---------------------------------------------------------------------------
# Screenshot bytes come from the selected monitor
# ---------------------------------------------------------------------------

class TestScreenshot:
    def test_callback_receives_png_bytes_from_selected_monitor(self):
        clock = FakeClock()
        received: list[bytes] = []
        fake_mss = FakeMss(png_bytes=b"\x89PNG real-fake")
        capture = Capture(
            callback=received.append,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=clock,
            mss_factory=lambda: fake_mss,
            to_png=fake_to_png,
        )

        capture.trigger()

        assert len(received) == 1
        assert received[0] == b"\x89PNG real-fake"

    def test_correct_monitor_dict_is_grabbed(self):
        clock = FakeClock()
        fake_mss = FakeMss()
        capture = Capture(
            callback=lambda _: None,
            monitor_index=2,
            debounce_seconds=2.0,
            clock=clock,
            mss_factory=lambda: fake_mss,
            to_png=fake_to_png,
        )

        capture.trigger()

        assert len(fake_mss.grabbed) == 1
        assert fake_mss.grabbed[0] == fake_mss.monitors[2]


# ---------------------------------------------------------------------------
# Display default: monitor 1 when no index given
# ---------------------------------------------------------------------------

class TestDisplayDefault:
    def test_default_monitor_index_is_one(self):
        clock = FakeClock()
        fake_mss = FakeMss()
        # Omit monitor_index — should default to 1
        capture = Capture(
            callback=lambda _: None,
            debounce_seconds=2.0,
            clock=clock,
            mss_factory=lambda: fake_mss,
            to_png=fake_to_png,
        )

        capture.trigger()

        assert fake_mss.grabbed[0] == fake_mss.monitors[1]
