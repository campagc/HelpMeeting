from typing import Any

import numpy as np
import pytest

from src.system_playback import BlackHoleSource, resolve_audio_device


_DEVICES = [
    {"name": "MacBook Air Microphone", "max_input_channels": 1},
    {"name": "BlackHole 2ch", "max_input_channels": 2},
]


class FakeInputStream:
    def __init__(self, *, start_error: Exception | None = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.start_error = start_error
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stopped = True
        self.kwargs["finished_callback"]()

    def close(self) -> None:
        self.closed = True

    def emit(self, audio: np.ndarray) -> None:
        self.kwargs["callback"](audio, len(audio), None, None)

    def fail(self) -> None:
        self.kwargs["finished_callback"]()


class FakePortAudio:
    def __init__(self, *, start_error: Exception | None = None) -> None:
        self.start_error = start_error
        self.stream: FakeInputStream | None = None

    def query_devices(self):
        return _DEVICES

    def input_stream(self, **kwargs: Any) -> FakeInputStream:
        self.stream = FakeInputStream(start_error=self.start_error, **kwargs)
        return self.stream


def test_resolver_matches_blackhole_case_insensitively():
    devices = [
        {"name": "Some Mic", "max_input_channels": 1},
        {"name": "blackhole 16ch", "max_input_channels": 16},
    ]

    assert resolve_audio_device(devices) == 1


def test_resolver_ignores_output_only_blackhole_devices():
    devices = [
        {"name": "BlackHole 2ch", "max_input_channels": 0},
        {"name": "BlackHole 16ch", "max_input_channels": 16},
    ]

    assert resolve_audio_device(devices) == 1


def test_blackhole_source_delivers_copied_16khz_mono_float32_blocks():
    port_audio = FakePortAudio()
    source = BlackHoleSource(
        query_devices=port_audio.query_devices,
        input_stream_factory=port_audio.input_stream,
    )
    blocks: list[np.ndarray] = []
    errors: list[Exception] = []

    source.prepare()
    source.start(blocks.append, errors.append)
    original = np.array([[0.25], [-0.5]], dtype=np.float32)
    assert port_audio.stream is not None
    port_audio.stream.emit(original)
    original[:] = 0

    assert port_audio.stream.kwargs["samplerate"] == 16_000
    assert port_audio.stream.kwargs["channels"] == 1
    assert port_audio.stream.kwargs["dtype"] == "float32"
    assert port_audio.stream.kwargs["device"] == 1
    assert blocks[0].shape == (2,)
    assert blocks[0].dtype == np.dtype("float32")
    np.testing.assert_array_equal(blocks[0], np.array([0.25, -0.5], dtype=np.float32))
    assert errors == []


def test_blackhole_source_propagates_block_conversion_failure():
    port_audio = FakePortAudio()
    source = BlackHoleSource(
        query_devices=port_audio.query_devices,
        input_stream_factory=port_audio.input_stream,
    )
    blocks: list[np.ndarray] = []
    errors: list[Exception] = []

    source.start(blocks.append, errors.append)
    assert port_audio.stream is not None
    port_audio.stream.emit(np.array([0.25], dtype=np.float32))

    assert blocks == []
    assert len(errors) == 1


def test_blackhole_source_prepare_has_clear_missing_device_guidance():
    devices = [{"name": "MacBook Air Speakers", "max_input_channels": 0}]
    source = BlackHoleSource(
        query_devices=lambda: devices,
        input_stream_factory=lambda **_: None,
    )

    with pytest.raises(RuntimeError, match="BlackHole.*Multi-Output Device"):
        source.prepare()


def test_blackhole_source_owns_its_silence_guidance():
    source = BlackHoleSource(
        query_devices=lambda: _DEVICES,
        input_stream_factory=lambda **_: None,
    )

    assert source.silence_warning == (
        "No audio detected on the capture device. Is your system output routed "
        "to BlackHole (e.g. via a Multi-Output Device)?"
    )


def test_blackhole_source_cleans_up_when_stream_start_fails():
    start_error = RuntimeError("device disconnected")
    port_audio = FakePortAudio(start_error=start_error)
    source = BlackHoleSource(
        query_devices=port_audio.query_devices,
        input_stream_factory=port_audio.input_stream,
    )

    with pytest.raises(RuntimeError, match="device disconnected"):
        source.start(lambda _: None, lambda _: None)

    assert port_audio.stream is not None
    assert port_audio.stream.closed is True


def test_blackhole_source_propagates_unexpected_stream_failure():
    port_audio = FakePortAudio()
    source = BlackHoleSource(
        query_devices=port_audio.query_devices,
        input_stream_factory=port_audio.input_stream,
    )
    errors: list[Exception] = []

    source.start(lambda _: None, errors.append)
    assert port_audio.stream is not None
    port_audio.stream.fail()

    assert len(errors) == 1
    assert "stopped unexpectedly" in str(errors[0])


def test_blackhole_source_stops_cleanly_without_reporting_failure():
    port_audio = FakePortAudio()
    source = BlackHoleSource(
        query_devices=port_audio.query_devices,
        input_stream_factory=port_audio.input_stream,
    )
    errors: list[Exception] = []

    source.start(lambda _: None, errors.append)
    assert port_audio.stream is not None
    stream = port_audio.stream
    source.stop()
    source.stop()

    assert stream.stopped is True
    assert stream.closed is True
    assert errors == []
