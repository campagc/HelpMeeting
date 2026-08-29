from collections.abc import Callable

import numpy as np
import pytest

from src.screencapturekit_playback import (
    QuartzScreenRecordingAccess,
    ScreenCaptureKitSource,
    ScreenCapturePermissionError,
)
from src.system_playback import BlackHoleSource, default_system_playback_source


def test_default_system_playback_source_uses_screencapturekit(monkeypatch):
    monkeypatch.delenv("HELPMEETING_AUDIO_SOURCE", raising=False)

    assert isinstance(default_system_playback_source(), ScreenCaptureKitSource)


def test_explicit_blackhole_system_playback_source_bypasses_screencapturekit(monkeypatch):
    monkeypatch.setenv("HELPMEETING_AUDIO_SOURCE", "blackhole")

    assert isinstance(default_system_playback_source(), BlackHoleSource)


def test_unknown_system_playback_source_lists_supported_values(monkeypatch):
    monkeypatch.setenv("HELPMEETING_AUDIO_SOURCE", "automatic")

    with pytest.raises(ValueError, match="blackhole.*screencapturekit"):
        default_system_playback_source()


class FakeScreenCaptureKit:
    format_summary = "16000 Hz, mono float32"

    def __init__(self) -> None:
        self.prepared = False
        self.started = False
        self.stopped = False
        self.stop_count = 0
        self.events: list[str] = []
        self._on_samples: Callable[[np.ndarray], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def prepare(self) -> None:
        self.prepared = True

    def start(self, on_samples, on_error) -> None:
        self.started = True
        self.events.append("started")
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
        self.stop_count += 1
        self.events.append("stopped")


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


def test_source_cleans_up_before_propagating_framework_failure():
    framework = FakeScreenCaptureKit()
    source = ScreenCaptureKitSource(framework=framework)
    events: list[str] = []
    failure = RuntimeError("stream stopped")

    source.start(lambda _: None, lambda error: events.append(str(error)))
    framework.fail(failure)

    assert framework.events == ["started", "stopped"]
    assert events == ["stream stopped"]


def test_source_stop_is_idempotent():
    framework = FakeScreenCaptureKit()
    source = ScreenCaptureKitSource(framework=framework)

    source.start(lambda _: None, lambda _: None)
    source.stop()
    source.stop()

    assert framework.stop_count == 1


class TestQuartzScreenRecordingAccess:
    def test_granted_access_does_nothing_and_allows_success(self):
        calls: list[str] = []

        def preflight() -> bool:
            return True

        def request() -> None:
            calls.append("request")

        access = QuartzScreenRecordingAccess(preflight=preflight, request=request)
        access.ensure_granted()

        assert calls == []

    def test_first_missing_access_requests_and_raises(self):
        calls: list[str] = []

        class FakeAccess:
            def __init__(self):
                self.granted = False

            def preflight(self) -> bool:
                return self.granted

            def request(self) -> None:
                calls.append("request")

        fake = FakeAccess()
        access = QuartzScreenRecordingAccess(preflight=fake.preflight, request=fake.request)

        with pytest.raises(ScreenCapturePermissionError) as exc_info:
            access.ensure_granted()

        assert calls == ["request"]
        assert "required" in str(exc_info.value)
        assert "System Settings" in str(exc_info.value)

    def test_second_call_does_not_request_again(self):
        calls: list[str] = []

        class FakeAccess:
            def __init__(self):
                self.granted = False

            def preflight(self) -> bool:
                return self.granted

            def request(self) -> None:
                calls.append("request")

        fake = FakeAccess()
        access = QuartzScreenRecordingAccess(preflight=fake.preflight, request=fake.request)

        with pytest.raises(ScreenCapturePermissionError):
            access.ensure_granted()

        calls.clear()

        with pytest.raises(ScreenCapturePermissionError) as exc_info:
            access.ensure_granted()

        assert calls == []
        assert "still denied" in str(exc_info.value)
