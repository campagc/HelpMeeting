import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.audio import (
    TranscriptIntake,
    WhisperTranscriber,
    _SAMPLE_RATE,
    _SILENCE_RMS,
    _SILENCE_WARN_AFTER,
)
from src.screencapturekit_playback import ScreenCaptureKitSource
from src.storage import Storage
from src.system_playback import BlackHoleSource, default_system_playback_source
from src.transcript import Transcript


class FakeSource:
    description = "fake system playback"
    silence_warning = "No audio detected on system playback."

    def __init__(self, *, start_error: Exception | None = None) -> None:
        self.start_count = 0
        self.stop_count = 0
        self.started = threading.Event()
        self._start_error = start_error
        self._on_block: Callable[[np.ndarray], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def prepare(self) -> None:
        pass

    def start(self, on_block, on_error) -> None:
        self.start_count += 1
        self._on_block = on_block
        self._on_error = on_error
        self.started.set()
        if self._start_error is not None:
            error = self._start_error
            self._start_error = None
            raise error

    def emit(self, block: np.ndarray) -> None:
        assert self._on_block is not None
        self._on_block(block)

    def fail(self, error: Exception) -> None:
        assert self._on_error is not None
        self._on_error(error)

    def stop(self) -> None:
        self.stop_count += 1


class FakeTranscriber:
    def __init__(self, return_text: str = "hello world") -> None:
        self.calls: list[np.ndarray] = []
        self.return_text = return_text

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(audio.copy())
        return self.return_text


class FakeScreenCaptureKitFramework:
    format_summary = None

    def __init__(self) -> None:
        self.started = threading.Event()
        self._on_samples: Callable[[np.ndarray], None] | None = None

    def prepare(self) -> None:
        pass

    def start(self, on_samples, on_error) -> None:
        self._on_samples = on_samples
        self.started.set()

    def emit(self, samples: np.ndarray) -> None:
        assert self._on_samples is not None
        self._on_samples(samples)

    def stop(self) -> None:
        pass


def wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not met before timeout")
        time.sleep(0.01)


def make_intake(
    tmp_path: Path,
    *,
    source=None,
    transcriber=None,
    warn: Callable[[str], None] | None = None,
    log_path: Path | None = None,
    restart_delay_seconds: float = 0.01,
) -> tuple[TranscriptIntake, FakeSource | object, FakeTranscriber | object, Transcript, Storage]:
    actual_source = source if source is not None else FakeSource()
    actual_transcriber = transcriber if transcriber is not None else FakeTranscriber()
    transcript = Transcript()
    archive = Storage(base_dir=tmp_path)
    archive.start_meeting("test-audio", ["en"])
    intake = TranscriptIntake(
        source=actual_source,
        transcriber=actual_transcriber,
        transcript=transcript,
        archive=archive,
        chunk_seconds=1,
        log_path=log_path,
        restart_delay_seconds=restart_delay_seconds,
        warn=warn if warn is not None else (lambda _: None),
    )
    return intake, actual_source, actual_transcriber, transcript, archive


def block(frames: int, amplitude: float = 0.1) -> np.ndarray:
    return np.full(frames, amplitude, dtype=np.float32)


def transcript_file(archive: Storage) -> str:
    assert archive.meeting_dir is not None
    return (archive.meeting_dir / "transcript.txt").read_text(encoding="utf-8")


def intake_thread_stopped() -> bool:
    return not any(thread.name == "TranscriptIntake" for thread in threading.enumerate())


def test_playback_adapters_flow_into_transcript_and_archive(tmp_path, monkeypatch):
    devices = [
        {"name": "MacBook Air Microphone", "max_input_channels": 1},
        {"name": "BlackHole 2ch", "max_input_channels": 2},
    ]
    started = threading.Event()
    stream_callbacks = {}

    class FakeInputStream:
        def __init__(self, **kwargs) -> None:
            stream_callbacks.update(kwargs)

        def start(self) -> None:
            started.set()

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setenv("HELPMEETING_AUDIO_SOURCE", "blackhole")
    monkeypatch.setattr("src.system_playback.sd.query_devices", lambda: devices)
    monkeypatch.setattr("src.system_playback.sd.InputStream", FakeInputStream)
    source = default_system_playback_source()
    assert isinstance(source, BlackHoleSource)
    source.prepare()
    transcriber = FakeTranscriber("captured speech")
    intake, _, _, transcript, archive = make_intake(
        tmp_path,
        source=source,
        transcriber=transcriber,
    )

    intake.start()
    assert started.wait(timeout=1.0)
    stream_callbacks["callback"](
        np.full((_SAMPLE_RATE, 1), 0.1, dtype=np.float32),
        _SAMPLE_RATE,
        None,
        None,
    )
    wait_for(lambda: len(transcriber.calls) == 1)
    intake.stop()

    assert transcript.take_delta() == "captured speech"
    assert transcript_file(archive) == "captured speech"


def test_screencapturekit_source_flows_into_transcript_and_archive(tmp_path):
    native_capture = FakeScreenCaptureKitFramework()
    source = ScreenCaptureKitSource(framework=native_capture)
    transcriber = FakeTranscriber("captured speech")
    intake, _, _, transcript, archive = make_intake(
        tmp_path,
        source=source,
        transcriber=transcriber,
    )

    intake.start()
    assert native_capture.started.wait(timeout=1.0)
    native_capture.emit(block(_SAMPLE_RATE))
    wait_for(lambda: len(transcriber.calls) == 1)
    intake.stop()

    assert transcript.take_delta() == "captured speech"
    assert transcript_file(archive) == "captured speech"


def test_blocks_accumulate_until_chunk_boundary_and_persist_text(tmp_path):
    transcriber = FakeTranscriber("spoken text")
    intake, source, _, transcript, archive = make_intake(tmp_path, transcriber=transcriber)

    intake.start()
    assert source.started.wait(timeout=1.0)
    first = block(_SAMPLE_RATE // 2)
    second = block(_SAMPLE_RATE // 2)
    source.emit(first)
    assert transcriber.calls == []
    source.emit(second)
    wait_for(lambda: len(transcriber.calls) == 1)
    intake.stop()

    assert np.array_equal(transcriber.calls[0], np.concatenate([first, second]))
    assert transcript.take_delta() == "spoken text"
    assert transcript_file(archive) == "spoken text"


def test_silent_chunks_warn_once_and_are_not_transcribed(tmp_path):
    warnings: list[str] = []
    log_path = tmp_path / "audio.log"
    transcriber = FakeTranscriber()
    intake, source, _, _, _ = make_intake(
        tmp_path,
        transcriber=transcriber,
        warn=warnings.append,
        log_path=log_path,
    )

    intake.start()
    assert source.started.wait(timeout=1.0)
    for _ in range(_SILENCE_WARN_AFTER + 1):
        source.emit(block(_SAMPLE_RATE, amplitude=_SILENCE_RMS * 0.5))
    wait_for(
        lambda: f"chunk #{_SILENCE_WARN_AFTER + 1}:" in log_path.read_text(
            encoding="utf-8"
        )
    )
    intake.stop()

    assert len(warnings) == 1
    assert source.silence_warning in warnings[0]
    assert transcriber.calls == []


def test_loud_chunk_resets_silence_warning(tmp_path):
    warnings: list[str] = []
    log_path = tmp_path / "audio.log"
    transcriber = FakeTranscriber()
    intake, source, _, _, _ = make_intake(
        tmp_path,
        transcriber=transcriber,
        warn=warnings.append,
        log_path=log_path,
    )

    intake.start()
    assert source.started.wait(timeout=1.0)
    for _ in range(_SILENCE_WARN_AFTER):
        source.emit(block(_SAMPLE_RATE, amplitude=0.0))
    wait_for(lambda: len(warnings) == 1)
    source.emit(block(_SAMPLE_RATE))
    wait_for(lambda: len(transcriber.calls) == 1)
    for _ in range(_SILENCE_WARN_AFTER):
        source.emit(block(_SAMPLE_RATE, amplitude=0.0))
    wait_for(lambda: len(warnings) == 2)
    intake.stop()

    assert len(warnings) == 2


def test_stop_flushes_partial_audio_and_joins_thread(tmp_path):
    transcriber = FakeTranscriber("final words")
    intake, source, _, transcript, archive = make_intake(
        tmp_path,
        transcriber=transcriber,
    )

    intake.start()
    assert source.started.wait(timeout=1.0)
    partial = block(_SAMPLE_RATE // 2)
    source.emit(partial)
    intake.stop()
    wait_for(intake_thread_stopped)

    assert len(transcriber.calls) == 1
    assert np.array_equal(transcriber.calls[0], partial)
    assert transcript.take_delta() == "final words"
    assert transcript_file(archive) == "final words"
    assert source.stop_count >= 1


def test_stop_with_no_audio_does_not_transcribe(tmp_path):
    transcriber = FakeTranscriber()
    intake, source, _, _, _ = make_intake(tmp_path, transcriber=transcriber)

    intake.start()
    assert source.started.wait(timeout=1.0)
    intake.stop()

    assert transcriber.calls == []


def test_empty_transcription_leaves_transcript_and_archive_empty(tmp_path):
    transcriber = FakeTranscriber("")
    intake, source, _, transcript, archive = make_intake(
        tmp_path,
        transcriber=transcriber,
    )

    intake.start()
    assert source.started.wait(timeout=1.0)
    source.emit(block(_SAMPLE_RATE))
    wait_for(lambda: len(transcriber.calls) == 1)
    intake.stop()

    assert transcript.take_delta() == ""
    assert transcript_file(archive) == ""


def test_source_error_is_logged_warned_and_restarted(tmp_path):
    warnings: list[str] = []
    log_path = tmp_path / "audio.log"
    intake, source, _, _, _ = make_intake(
        tmp_path,
        warn=warnings.append,
        log_path=log_path,
    )

    intake.start()
    assert source.started.wait(timeout=1.0)
    source.fail(RuntimeError("device disconnected"))
    wait_for(lambda: source.start_count == 2)
    intake.stop()

    log = log_path.read_text(encoding="utf-8")
    assert any("restarting" in warning for warning in warnings)
    assert "capture loop crashed" in log
    assert "Traceback" in log
    assert "device disconnected" in log


def test_source_start_error_retries_then_stops_cleanly(tmp_path):
    source = FakeSource(start_error=RuntimeError("initial start failed"))
    intake, _, _, _, _ = make_intake(tmp_path, source=source)

    intake.start()
    wait_for(lambda: source.start_count == 2)
    intake.stop()
    wait_for(intake_thread_stopped)

    assert source.start_count == 2


def test_stop_interrupts_restart_wait(tmp_path):
    warnings: list[str] = []
    intake, source, _, _, _ = make_intake(
        tmp_path,
        warn=warnings.append,
        restart_delay_seconds=5.0,
    )

    intake.start()
    assert source.started.wait(timeout=1.0)
    source.fail(RuntimeError("device disconnected"))
    wait_for(lambda: any("restarting" in warning for warning in warnings))

    started_at = time.monotonic()
    intake.stop()

    assert time.monotonic() - started_at < 1.0
    assert source.start_count == 1


def test_error_delivered_after_stop_does_not_restart(tmp_path):
    intake, source, _, _, _ = make_intake(tmp_path)

    intake.start()
    assert source.started.wait(timeout=1.0)
    intake.stop()
    source.fail(RuntimeError("late callback"))

    assert source.start_count == 1


def test_queued_full_chunk_is_drained_when_stopping(tmp_path):
    transcriber = FakeTranscriber("queued words")
    intake, source, _, transcript, archive = make_intake(
        tmp_path,
        transcriber=transcriber,
    )

    intake.start()
    assert source.started.wait(timeout=1.0)
    source.emit(block(_SAMPLE_RATE))
    intake.stop()

    assert len(transcriber.calls) == 1
    assert transcript.take_delta() == "queued words"
    assert transcript_file(archive) == "queued words"


def test_whisper_model_is_loaded_lazily_and_only_once(monkeypatch):
    constructions: list[tuple[tuple, dict]] = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs) -> None:
            constructions.append((args, kwargs))

        def transcribe(self, _audio, **_kwargs):
            return [SimpleNamespace(text="captured speech")], None

    monkeypatch.setattr("src.audio.WhisperModel", FakeWhisperModel)
    logs: list[str] = []
    transcriber = WhisperTranscriber("base", "en", log_fn=logs.append)

    assert constructions == []
    assert transcriber.transcribe(block(1)) == "captured speech"
    assert transcriber.transcribe(block(1)) == "captured speech"

    assert len(constructions) == 1
    assert logs == ["loading Whisper model…", "Whisper model loaded"]
