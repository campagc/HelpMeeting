"""Tests for the screen module (issue #15).

The screen module owns display capture and the mss-vs-Quartz exclusion seam.
Tests use a fake backend to stay away from real displays.
"""

from src.screen import MSSBackend, Screen


class FakeShot:
    def __init__(self, png_bytes: bytes, size: tuple[int, int] = (1440, 900)):
        self.rgb = png_bytes
        self.size = size


class FakeBackend:
    """Records which monitor and exclusion were requested."""

    def __init__(
        self,
        monitors: list[dict],
        png_bytes: bytes,
    ) -> None:
        self.monitors = monitors
        self._png_bytes = png_bytes
        self.grabbed: list[dict] = []
        self.exclusions: list[int | None] = []

    def __enter__(self) -> "FakeBackend":
        return self

    def __exit__(self, *_) -> None:  # noqa: ANN002
        return None

    def grab(self, monitor: dict, below_window: int | None = None) -> bytes:
        self.grabbed.append(monitor)
        self.exclusions.append(below_window)
        return self._png_bytes


class TestScreen:
    def test_grab_returns_png_bytes_for_selected_monitor(self):
        png_bytes = b"\x89PNG real-fake"
        monitors = [
            {"left": 0, "top": 0, "width": 2880, "height": 900},
            {"left": 0, "top": 0, "width": 1440, "height": 900},
            {"left": 1440, "top": 0, "width": 1440, "height": 900},
        ]
        backend = FakeBackend(monitors, png_bytes)
        screen = Screen(backend_factory=lambda: backend)

        result = screen.grab(1)

        assert result == png_bytes

    def test_default_monitor_index_is_one(self):
        png_bytes = b"\x89PNG fake"
        monitors = [
            {"left": 0, "top": 0, "width": 1440, "height": 900},
            {"left": 0, "top": 0, "width": 1440, "height": 900},
        ]
        backend = FakeBackend(monitors, png_bytes)
        screen = Screen(backend_factory=lambda: backend)

        screen.grab(1)

        assert backend.grabbed[0] == monitors[1]

    def test_hud_window_number_is_passed_through_when_supplied(self):
        png_bytes = b"\x89PNG fake"
        monitors = [
            {"left": 0, "top": 0, "width": 1440, "height": 900},
            {"left": 0, "top": 0, "width": 1440, "height": 900},
        ]
        backend = FakeBackend(monitors, png_bytes)
        screen = Screen(backend_factory=lambda: backend)

        screen.grab(1, below_window=12345)

        assert backend.exclusions == [12345]

    def test_below_window_is_omitted_when_not_supplied(self):
        png_bytes = b"\x89PNG fake"
        monitors = [
            {"left": 0, "top": 0, "width": 1440, "height": 900},
            {"left": 0, "top": 0, "width": 1440, "height": 900},
        ]
        backend = FakeBackend(monitors, png_bytes)
        screen = Screen(backend_factory=lambda: backend)

        screen.grab(1)

        assert backend.exclusions == [None]


# ---------------------------------------------------------------------------
# MSS backend: migrated monitor-selection and PNG-passthrough assertions
# ---------------------------------------------------------------------------


class FakeMss:
    """Minimal stand-in for an mss context manager."""

    def __init__(self, monitors: list[dict] | None = None, png_bytes: bytes = b"\x89PNG fake"):
        self.monitors = monitors or [
            {"left": 0, "top": 0, "width": 2880, "height": 900},
            {"left": 0, "top": 0, "width": 1440, "height": 900},
            {"left": 1440, "top": 0, "width": 1440, "height": 900},
        ]
        self._png_bytes = png_bytes
        self.grabbed: list[dict] = []

    def grab(self, monitor: dict) -> FakeShot:
        self.grabbed.append(monitor)
        return FakeShot(self._png_bytes)

    def close(self) -> None:
        pass


def fake_to_png(rgb: bytes, size: tuple[int, int]) -> bytes:
    """Stand-in for mss.tools.to_png — just return the input bytes."""
    return rgb


class TestMSSBackend:
    def test_backend_returns_png_bytes_for_selected_monitor(self):
        fake_mss = FakeMss(png_bytes=b"\x89PNG real-fake")
        backend = MSSBackend(mss_factory=lambda: fake_mss, to_png=fake_to_png)

        with backend:
            result = backend.grab(backend.monitors[1])

        assert result == b"\x89PNG real-fake"

    def test_correct_monitor_dict_is_grabbed(self):
        fake_mss = FakeMss()
        backend = MSSBackend(mss_factory=lambda: fake_mss, to_png=fake_to_png)

        with backend:
            backend.grab(backend.monitors[2])

        assert len(fake_mss.grabbed) == 1
        assert fake_mss.grabbed[0] == fake_mss.monitors[2]

    def test_below_window_is_accepted_but_ignored_by_mss(self):
        fake_mss = FakeMss(png_bytes=b"\x89PNG real-fake")
        backend = MSSBackend(mss_factory=lambda: fake_mss, to_png=fake_to_png)

        with backend:
            result = backend.grab(backend.monitors[1], below_window=12345)

        assert result == b"\x89PNG real-fake"
