import pytest

from src.display import (
    Display,
    Displays,
    MSSDisplayBackend,
    appkit_bottom_right,
)


def monitor(left: int, top: int, width: int, height: int) -> dict[str, int]:
    return {"left": left, "top": top, "width": width, "height": height}


class FakeBackend:
    def __init__(
        self,
        monitors: list[dict[str, int]],
        png_bytes: bytes = b"\x89PNG fake",
    ) -> None:
        self._monitors = monitors
        self.png_bytes = png_bytes
        self.grabbed: list[dict[str, int]] = []

    def __enter__(self) -> "FakeBackend":
        return self

    def __exit__(self, *_) -> None:
        return None

    @property
    def monitors(self) -> list[dict[str, int]]:
        return self._monitors

    def grab(self, monitor: dict[str, int]) -> bytes:
        self.grabbed.append(monitor)
        return self.png_bytes


class FakeShot:
    def __init__(self, rgb: bytes, size: tuple[int, int] = (1440, 900)) -> None:
        self.rgb = rgb
        self.size = size


class FakeMss:
    def __init__(self, monitors: list[dict[str, int]] | None = None) -> None:
        self.monitors = monitors or [
            monitor(0, 0, 2880, 900),
            monitor(0, 0, 1440, 900),
            monitor(1440, 0, 1440, 900),
        ]
        self.close_count = 0
        self.grabbed: list[dict[str, int]] = []

    def grab(self, selected: dict[str, int]) -> FakeShot:
        self.grabbed.append(selected)
        return FakeShot(b"\x89PNG fake")

    def close(self) -> None:
        self.close_count += 1


def fake_to_png(rgb: bytes, _size: tuple[int, int]) -> bytes:
    return rgb


def test_list_excludes_virtual_display_and_preserves_one_based_indices():
    monitors = [
        monitor(-1440, 0, 2880, 900),
        monitor(0, 0, 1440, 900),
        monitor(-1440, 0, 1440, 900),
    ]
    displays = Displays(backend_factory=lambda: FakeBackend(monitors))

    assert displays.list() == [
        Display(1, 0, 0, 1440, 900),
        Display(2, -1440, 0, 1440, 900),
    ]


@pytest.mark.parametrize("index", [0, 3, -1])
def test_get_raises_for_an_index_outside_physical_displays(index):
    displays = Displays(
        backend_factory=lambda: FakeBackend(
            [monitor(0, 0, 2880, 900), monitor(0, 0, 1440, 900)]
        )
    )

    with pytest.raises(ValueError, match=f"No display {index}"):
        displays.get(index)


def test_grab_passes_the_selected_physical_monitor_to_backend():
    monitors = [
        monitor(0, 0, 2880, 900),
        monitor(0, 0, 1440, 900),
        monitor(1440, 0, 1440, 900),
    ]
    backend = FakeBackend(monitors, png_bytes=b"\x89PNG display 2")
    displays = Displays(backend_factory=lambda: backend)

    assert displays.grab(2) == b"\x89PNG display 2"
    assert backend.grabbed == [monitors[2]]


def test_mss_backend_returns_png_bytes_for_selected_monitor():
    fake_mss = FakeMss()
    backend = MSSDisplayBackend(
        mss_factory=lambda: fake_mss,
        to_png=fake_to_png,
    )

    with backend:
        result = backend.grab(backend.monitors[2])

    assert result == b"\x89PNG fake"
    assert fake_mss.grabbed == [fake_mss.monitors[2]]


def test_mss_backend_rejects_missing_png_conversion():
    backend = MSSDisplayBackend(
        mss_factory=FakeMss,
        to_png=lambda *_: None,
    )

    with backend, pytest.raises(RuntimeError, match="PNG conversion returned no bytes"):
        backend.grab(backend.monitors[1])


def test_mss_backend_closes_context_on_exit():
    fake_mss = FakeMss()
    backend = MSSDisplayBackend(mss_factory=lambda: fake_mss, to_png=fake_to_png)

    with backend:
        assert fake_mss.close_count == 0

    assert fake_mss.close_count == 1


def test_mss_backend_monitors_require_context_manager():
    backend = MSSDisplayBackend(mss_factory=FakeMss, to_png=fake_to_png)

    with pytest.raises(RuntimeError, match="outside a context manager"):
        _ = backend.monitors


def test_appkit_bottom_right_on_single_display():
    display = Display(1, 0, 0, 1440, 900)

    assert appkit_bottom_right([display], display, 32, 12) == (1396.0, 12.0)


def test_appkit_bottom_right_on_display_to_the_right_of_main():
    main = Display(1, 0, 0, 1440, 900)
    right = Display(2, 1440, 0, 1440, 900)

    assert appkit_bottom_right([main, right], right, 32, 12) == (2836.0, 12.0)


def test_appkit_bottom_right_on_display_to_the_left_of_main():
    main = Display(1, 0, 0, 1440, 900)
    left = Display(2, -1440, 0, 1440, 900)

    assert appkit_bottom_right([main, left], left, 32, 12) == (-44.0, 12.0)


def test_appkit_bottom_right_on_display_above_main():
    main = Display(1, 0, 0, 1440, 900)
    above = Display(2, 0, -900, 1440, 900)

    assert appkit_bottom_right([main, above], above, 32, 12) == (1396.0, 912.0)


def test_appkit_bottom_right_finds_main_when_it_is_not_first():
    right = Display(2, 1440, 0, 1440, 800)
    main = Display(1, 0, 0, 1440, 900)

    assert appkit_bottom_right([right, main], right, 32, 12) == (2836.0, 112.0)


def test_displays_appkit_bottom_right_uses_display_index():
    displays = Displays(
        backend_factory=lambda: FakeBackend(
            [
                monitor(0, 0, 2880, 900),
                monitor(0, 0, 1440, 900),
                monitor(1440, 0, 1440, 800),
            ]
        )
    )

    assert displays.appkit_bottom_right(2, 32, 12) == (2836.0, 112.0)
