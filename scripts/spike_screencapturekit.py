#!/usr/bin/env python3
"""Interactive ScreenCaptureKit system-playback hardware spike for issue #23.

Usage:
    python scripts/spike_screencapturekit.py
    python scripts/spike_screencapturekit.py 12
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

from src.audio import Transcriber, WhisperTranscriber
from src.config import WHISPER_MODEL_SIZE
from src.screencapturekit_playback import (
    ScreenCaptureKitSource,
    ScreenCapturePermissionError,
)
from src.system_playback import SAMPLE_RATE, SystemPlaybackSource


class DiagnosticSource(SystemPlaybackSource, Protocol):
    @property
    def format_summary(self) -> str | None:
        ...


class _LazyWhisperTranscriber:
    def __init__(self) -> None:
        self._transcriber: WhisperTranscriber | None = None

    def transcribe(self, audio: np.ndarray) -> str:
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber(
                model_size=WHISPER_MODEL_SIZE,
                language="en",
                log_fn=print,
            )
        return self._transcriber.transcribe(audio)


def active_output_device() -> str:
    try:
        return str(sd.query_devices(kind="output")["name"])
    except Exception as exc:
        return f"unavailable ({exc})"


def run_diagnostic(
    *,
    source: DiagnosticSource,
    transcriber: Transcriber,
    seconds: int,
    output_fn: Callable[[str], None] = print,
    wait_fn: Callable[[float], None] = time.sleep,
    output_device_fn: Callable[[], str] = active_output_device,
    clock_fn: Callable[[], float] = time.monotonic,
    expected_output: str = "MacBook Air Speakers",
) -> int:
    blocks: list[np.ndarray] = []
    arrival_times: list[float] = []
    errors: list[Exception] = []

    def receive_block(block: np.ndarray) -> None:
        blocks.append(block)
        arrival_times.append(clock_fn())

    output_before = output_device_fn()
    output_fn(f"Selected source: {source.description}")
    output_fn(f"Requested duration: {seconds}s")
    output_fn(f"Active output before capture: {output_before}")
    output_fn(
        "Play speech now. Keep the current output selected and change volume with the "
        "normal macOS controls during capture."
    )

    stop_error: Exception | None = None
    try:
        source.prepare()
        source.start(receive_block, errors.append)
        wait_fn(seconds)
    except ScreenCapturePermissionError as exc:
        output_fn(f"Permission required: {exc}")
        return 1
    except Exception as exc:
        output_fn(f"Capture failed: {exc}")
        return 1
    finally:
        try:
            source.stop()
        except Exception as exc:
            stop_error = exc

    output_after = output_device_fn()
    expected_output_selected = expected_output.casefold() in output_before.casefold()
    output_unchanged = output_before == output_after
    audio = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)
    captured_seconds = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    callback_gaps = [later - earlier for earlier, later in zip(arrival_times, arrival_times[1:])]
    largest_callback_gap = max(callback_gaps) if callback_gaps else 0.0
    callback_continuity = (
        captured_seconds >= seconds * 0.8 and largest_callback_gap < 1.0
    )

    output_fn(f"Native format: {source.format_summary or 'no audio format received'}")
    output_fn(
        f"Captured duration: {captured_seconds:.3f}s | frames={len(audio)} | "
        f"RMS={rms:.6f} | peak={peak:.6f}"
    )
    output_fn(f"Active output after capture: {output_after}")
    output_fn(
        f"{expected_output} selected: {'yes' if expected_output_selected else 'no'}"
    )
    output_fn(f"Output device unchanged: {'yes' if output_unchanged else 'no'}")
    output_fn(
        f"Capture callback continuity: {'yes' if callback_continuity else 'no'} "
        f"(largest gap {largest_callback_gap:.3f}s)"
    )
    output_fn(f"Stream stopped cleanly: {'yes' if stop_error is None else 'no'}")

    if errors:
        output_fn(f"Capture callback failed: {errors[0]}")
        return 1
    if stop_error is not None:
        output_fn(f"Capture shutdown failed: {stop_error}")
        return 1

    text = transcriber.transcribe(audio) if len(audio) else ""
    output_fn(f"Whisper produced text: {'yes' if text else 'no'}")
    if text:
        output_fn(f"Transcript: {text}")

    if rms < 1e-4:
        output_fn(f"FAILED: {source.silence_warning}")
        return 1
    if not text:
        output_fn("FAILED: the captured signal did not produce a Whisper transcript.")
        return 1
    if not expected_output_selected:
        output_fn(f"FAILED: select {expected_output} before running this target-Mac spike.")
        return 1
    if not output_unchanged:
        output_fn("FAILED: the active output device changed during capture.")
        return 1
    if not callback_continuity:
        output_fn("FAILED: audio callbacks were interrupted during the volume-control check.")
        return 1

    output_fn("SUCCESS: native system playback is transcription-ready.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        seconds = int(args[0]) if args else 8
    except ValueError:
        print("Duration must be an integer number of seconds.")
        return 2
    if seconds <= 0:
        print("Duration must be greater than zero.")
        return 2
    return run_diagnostic(
        source=ScreenCaptureKitSource(),
        transcriber=_LazyWhisperTranscriber(),
        seconds=seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
