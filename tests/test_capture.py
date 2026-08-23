"""Tests for the capture module (issue #7/#15).

Hardware-bound behaviour (actual hotkey listener, real screen grab) is validated
manually.  Everything that lives above the hardware boundary is tested here:

  - Debounce: a second trigger arriving within the debounce window is dropped.
  - Debounce: a trigger arriving after the window fires the callback.
  - Screenshot bytes: the callback receives raw PNG bytes from the grabber.
  - Display default: when no monitor index is given, monitor 1 is used.
  - HUD exclusion: the grabber receives the HUD window number when supplied.

The capture module accepts injected collaborators so tests never touch the OS.
"""

import pytest

import src.capture as _capture_module
from src.capture import Capture


TEST_HOTKEYS = ["<ctrl>+<alt>+<space>", "<ctrl>+<alt>+s"]


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


class FakeScreen:
    """Records grab calls and returns a fixed PNG payload."""

    def __init__(self, png_bytes: bytes = b"\x89PNG fake"):
        self._png_bytes = png_bytes
        self.calls: list[tuple[int, int | None]] = []

    def grab(self, monitor_index: int, below_window: int | None = None) -> bytes:
        self.calls.append((monitor_index, below_window))
        return self._png_bytes


class FakeHud:
    """HUD collaborator that exposes a window number."""

    def __init__(self, window_id: int | None = None):
        self.window_id = window_id


class FakeGlobalHotKeys:
    """Records the hotkey map passed to pynput without starting a real listener."""

    instances: list["FakeGlobalHotKeys"] = []

    def __init__(self, hotkeys):
        self.hotkeys = hotkeys
        self.instances.append(self)

    def start(self):
        pass

    def join(self):
        pass

    def stop(self):
        pass


class FakeKeyboardModule:
    """Drop-in replacement for the pynput keyboard module used by capture."""

    GlobalHotKeys = FakeGlobalHotKeys


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
            grabber=FakeScreen(),
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
            grabber=FakeScreen(),
        )

        capture.trigger()           # t=0 → fires
        clock.advance(2.1)
        capture.trigger()           # t=2.1 → outside window → fires again

        assert len(fired) == 2


# ---------------------------------------------------------------------------
# Screenshot bytes come from the injected grabber
# ---------------------------------------------------------------------------

class TestScreenshot:
    def test_callback_receives_png_bytes_from_selected_monitor(self):
        clock = FakeClock()
        received: list[bytes] = []
        fake_screen = FakeScreen(png_bytes=b"\x89PNG real-fake")
        capture = Capture(
            callback=received.append,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=clock,
            grabber=fake_screen,
        )

        capture.trigger()

        assert len(received) == 1
        assert received[0] == b"\x89PNG real-fake"

    def test_correct_monitor_index_is_grabbed(self):
        clock = FakeClock()
        fake_screen = FakeScreen()
        capture = Capture(
            callback=lambda _: None,
            monitor_index=2,
            debounce_seconds=2.0,
            clock=clock,
            grabber=fake_screen,
        )

        capture.trigger()

        assert fake_screen.calls == [(2, None)]


# ---------------------------------------------------------------------------
# Display default: monitor 1 when no index given
# ---------------------------------------------------------------------------

class TestDisplayDefault:
    def test_default_monitor_index_is_one(self):
        clock = FakeClock()
        fake_screen = FakeScreen()
        # Omit monitor_index — should default to 1
        capture = Capture(
            callback=lambda _: None,
            debounce_seconds=2.0,
            clock=clock,
            grabber=fake_screen,
        )

        capture.trigger()

        assert fake_screen.calls[0][0] == 1


# ---------------------------------------------------------------------------
# HUD exclusion
# ---------------------------------------------------------------------------

class TestHudExclusion:
    def test_hud_window_id_is_passed_to_grabber(self):
        fake_screen = FakeScreen()
        capture = Capture(
            callback=lambda _: None,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=FakeClock(),
            grabber=fake_screen,
            hud=FakeHud(window_id=12345),
        )

        capture.trigger()

        assert fake_screen.calls == [(1, 12345)]

    def test_no_hud_means_below_window_is_none(self):
        fake_screen = FakeScreen()
        capture = Capture(
            callback=lambda _: None,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=FakeClock(),
            grabber=fake_screen,
        )

        capture.trigger()

        assert fake_screen.calls == [(1, None)]


# ---------------------------------------------------------------------------
# Multiple hotkeys map to the same trigger and share one debounce window
# ---------------------------------------------------------------------------

class TestMultipleHotkeys:
    def setup_method(self):
        FakeGlobalHotKeys.instances.clear()

    def test_binds_both_hotkeys_to_same_trigger(self, monkeypatch):
        capture = Capture(
            callback=lambda _: None,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=FakeClock(),
            grabber=FakeScreen(),
            hotkeys=TEST_HOTKEYS,
        )
        monkeypatch.setattr(_capture_module, "keyboard", FakeKeyboardModule)

        capture.start()

        listener = FakeGlobalHotKeys.instances[-1]
        bound = listener.hotkeys
        assert set(bound.keys()) == set(TEST_HOTKEYS)
        assert len(set(bound.values())) == 1

    def test_press_on_either_hotkey_drops_within_shared_debounce(self, monkeypatch):
        clock = FakeClock(start=0.0)
        fired: list[bytes] = []
        capture = Capture(
            callback=fired.append,
            monitor_index=1,
            debounce_seconds=2.0,
            clock=clock,
            grabber=FakeScreen(),
            hotkeys=TEST_HOTKEYS,
        )
        monkeypatch.setattr(_capture_module, "keyboard", FakeKeyboardModule)

        capture.start()
        listener = FakeGlobalHotKeys.instances[-1]
        mapping = listener.hotkeys

        mapping[TEST_HOTKEYS[0]]()  # t=0 → fires
        clock.advance(0.5)
        mapping[TEST_HOTKEYS[1]]()  # t=0.5 → within debounce → dropped

        assert len(fired) == 1

        clock.advance(2.1)
        mapping[TEST_HOTKEYS[1]]()  # t=2.6 → outside window → fires again

        assert len(fired) == 2
