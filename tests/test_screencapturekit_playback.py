from collections.abc import Callable

import numpy as np

from src.screencapturekit_playback import ScreenCaptureKitSource


class FakeScreenCaptureKit:
    format_summary = "16000 Hz, mono float32"

    def __init__(self) -> None:
        self.prepared = False
        self.started = False
        self.stopped = False
        self._on_samples: Callable[[np.ndarray], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def prepare(self) -> None:
        self.prepared = True

    def start(self, on_samples, on_error) -> None:
        self.started = True
        self._on_samples = on_samples
        self._on_error = on_error

    def emit(self, samples: np.ndarray) -> None:
        assert self._on_samples is not None
        self._on_samples(samples)

    def fail(self, error: Exception) -> None:
        assert self._on_error is not None
        self._on_error(error)

    def stop(self) -> None:
        self.stopped = True


def test_source_delivers_copied_16khz_mono_float32_blocks():
    framework = FakeScreenCaptureKit()
    source = ScreenCaptureKitSource(framework=framework)
    blocks: list[np.ndarray] = []
    errors: list[Exception] = []

    source.prepare()
    source.start(blocks.append, errors.append)
    native_samples = np.array([[0.25], [-0.5]], dtype=np.float64)
    framework.emit(native_samples)
    native_samples[:] = 0

    assert framework.prepared is True
    assert framework.started is True
    assert blocks[0].shape == (2,)
    assert blocks[0].dtype == np.dtype("float32")
    np.testing.assert_array_equal(blocks[0], np.array([0.25, -0.5], dtype=np.float32))
    assert errors == []


def test_source_stops_framework_cleanly():
    framework = FakeScreenCaptureKit()
    source = ScreenCaptureKitSource(framework=framework)

    source.start(lambda _: None, lambda _: None)
    source.stop()

    assert framework.stopped is True


def test_source_propagates_framework_failure():
    framework = FakeScreenCaptureKit()
    source = ScreenCaptureKitSource(framework=framework)
    errors: list[Exception] = []
    failure = RuntimeError("stream stopped")

    source.start(lambda _: None, errors.append)
    framework.fail(failure)

    assert errors == [failure]
