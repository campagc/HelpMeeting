import numpy as np

from scripts.spike_screencapturekit import run_diagnostic
from src.screencapturekit_playback import ScreenCapturePermissionError


class FakeSource:
    description = "ScreenCaptureKit test source"
    silence_warning = "No playback"
    format_summary = "16000 Hz, 1 channel, float32"

    def __init__(self) -> None:
        self.stopped = False

    def prepare(self) -> None:
        pass

    def start(self, on_block, on_error) -> None:
        _ = on_error
        on_block(np.full(16_000, 0.25, dtype=np.float32))

    def stop(self) -> None:
        self.stopped = True


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray) -> str:
        return "captured speech"


def test_missing_permission_reports_guidance_without_traceback():
    class PermissionDeniedSource(FakeSource):
        def prepare(self) -> None:
            raise ScreenCapturePermissionError("grant access in System Settings, then relaunch")

    output: list[str] = []

    result = run_diagnostic(
        source=PermissionDeniedSource(),
        transcriber=FakeTranscriber(),
        seconds=1,
        output_fn=output.append,
        wait_fn=lambda _: None,
        output_device_fn=lambda: "MacBook Air Speakers",
    )

    report = "\n".join(output)
    assert result == 1
    assert "Permission required" in report
    assert "System Settings" in report
    assert "Traceback" not in report


def test_diagnostic_reports_signal_transcription_and_stable_output():
    source = FakeSource()
    output: list[str] = []

    result = run_diagnostic(
        source=source,
        transcriber=FakeTranscriber(),
        seconds=1,
        output_fn=output.append,
        wait_fn=lambda _: None,
        output_device_fn=lambda: "MacBook Air Speakers",
    )

    report = "\n".join(output)
    assert result == 0
    assert "Selected source: ScreenCaptureKit test source" in report
    assert "Requested duration: 1s" in report
    assert "RMS=0.250000" in report
    assert "peak=0.250000" in report
    assert "Whisper produced text: yes" in report
    assert "MacBook Air Speakers selected: yes" in report
    assert "Output device unchanged: yes" in report
    assert "Capture callback continuity: yes" in report
    assert source.stopped is True
