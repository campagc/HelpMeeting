"""Tests for debounced hotkey capture through the Displays interface."""

import src.capture as capture_module
from src.capture import Capture


TEST_HOTKEYS = ["<ctrl>+<alt>+<space>", "<ctrl>+<alt>+s"]


class FakeClock:
    def __init__(self, start: float = 0.0):
        self._time = start

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class FakeDisplays:
    def __init__(self, png_bytes: bytes = b"\x89PNG fake"):
        self.png_bytes = png_bytes
        self.calls: list[int] = []

    def grab(self, display_index: int) -> bytes:
        self.calls.append(display_index)
        return self.png_bytes


class FakeGlobalHotKeys:
    instances: list["FakeGlobalHotKeys"] = []

    def __init__(self, hotkeys):
        self.hotkeys = hotkeys
        self.started = False
        self.joined = False
        self.stopped = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def join(self):
        self.joined = True

    def stop(self):
        self.stopped = True


class FakeKeyboardModule:
    GlobalHotKeys = FakeGlobalHotKeys


def test_second_trigger_within_debounce_window_is_dropped():
    clock = FakeClock()
    displays = FakeDisplays()
    received: list[bytes] = []
    capture = Capture(
        received.append,
        displays=displays,
        display_index=1,
        clock=clock,
    )

    capture.trigger()
    clock.advance(1.0)
    capture.trigger()

    assert received == [b"\x89PNG fake"]
    assert displays.calls == [1]


def test_trigger_after_debounce_window_fires_again():
    clock = FakeClock()
    displays = FakeDisplays()
    received: list[bytes] = []
    capture = Capture(
        received.append,
        displays=displays,
        display_index=1,
        clock=clock,
    )

    capture.trigger()
    clock.advance(2.1)
    capture.trigger()

    assert len(received) == 2


def test_callback_receives_png_from_selected_display():
    displays = FakeDisplays(png_bytes=b"\x89PNG selected")
    received: list[bytes] = []
    capture = Capture(
        received.append,
        displays=displays,
        display_index=2,
        clock=FakeClock(),
    )

    capture.trigger()

    assert received == [b"\x89PNG selected"]
    assert displays.calls == [2]


def test_start_binds_hotkeys_and_stop_stops_listener(monkeypatch):
    FakeGlobalHotKeys.instances.clear()
    capture = Capture(
        lambda _: None,
        displays=FakeDisplays(),
        display_index=1,
        clock=FakeClock(),
        hotkeys=TEST_HOTKEYS,
    )
    monkeypatch.setattr(capture_module, "keyboard", FakeKeyboardModule)

    capture.start()
    capture.stop()

    listener = FakeGlobalHotKeys.instances[-1]
    assert set(listener.hotkeys) == set(TEST_HOTKEYS)
    assert len(set(listener.hotkeys.values())) == 1
    assert listener.started and listener.joined and listener.stopped


def test_multiple_hotkeys_share_one_debounce_window(monkeypatch):
    FakeGlobalHotKeys.instances.clear()
    clock = FakeClock()
    received: list[bytes] = []
    capture = Capture(
        received.append,
        displays=FakeDisplays(),
        display_index=1,
        clock=clock,
        hotkeys=TEST_HOTKEYS,
    )
    monkeypatch.setattr(capture_module, "keyboard", FakeKeyboardModule)

    capture.start()
    bindings = FakeGlobalHotKeys.instances[-1].hotkeys
    bindings[TEST_HOTKEYS[0]]()
    clock.advance(0.5)
    bindings[TEST_HOTKEYS[1]]()
    clock.advance(2.1)
    bindings[TEST_HOTKEYS[1]]()

    assert len(received) == 2
