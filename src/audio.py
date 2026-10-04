"""System-audio intake and local transcription.

Public interface
----------------
transcriber = WhisperTranscriber(model_size="small", language="en")
intake = TranscriptIntake(
    source=system_playback_source,
    transcriber=transcriber,
    transcript=transcript,
    archive=archive,
    chunk_seconds=10,
)
intake.start()
intake.stop()
"""

import queue
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel

from src.storage import Storage
from src.system_playback import SAMPLE_RATE, SystemPlaybackSource
from src.transcript import Transcript

_SAMPLE_RATE = SAMPLE_RATE
_SILENCE_RMS = 1e-4         # below this a chunk is treated as silence
_SILENCE_WARN_AFTER = 2     # warn after this many consecutive silent chunks


def append_log(log_path: Path | None) -> Callable[[str], None]:
    """Return a logger appending timestamped lines to log_path; a no-op when log_path is None."""
    if log_path is None:
        return lambda _: None

    def log(message: str) -> None:
        try:
            stamp = time.strftime("%H:%M:%S")
            with log_path.open("a", encoding="utf-8") as file:
                file.write(f"[{stamp}] {message}\n")
        except Exception:  # noqa: BLE001
            pass

    return log


class Transcriber(Protocol):
    """Converts a 1-D float32 audio array into a transcript string."""

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio; return combined text or empty string."""
        ...


class WhisperTranscriber:
    """Concrete Transcriber backed by faster-whisper.WhisperModel."""

    def __init__(
        self,
        model_size: str,
        language: str,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._model_size = model_size
        self._language = language
        self._log = log_fn if log_fn is not None else lambda _: None
        self._model = None

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array; return combined text or empty string."""
        if self._model is None:
            self._log("loading Whisper model…")
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            self._log("Whisper model loaded")
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class TranscriptIntake:
    """Capture system playback, append Transcript text, and persist it to Storage."""

    def __init__(
        self,
        *,
        source: SystemPlaybackSource,
        transcriber: Transcriber,
        transcript: Transcript,
        archive: Storage,
        chunk_seconds: int = 10,
        log_path: Path | None = None,
        restart_delay_seconds: float = 2.0,
        warn: Callable[[str], None] = print,
    ) -> None:
        self._source = source
        self._transcriber = transcriber
        self._transcript = transcript
        self._archive = archive
        self._chunk_seconds = chunk_seconds
        self._log = append_log(log_path)
        self._restart_delay_seconds = restart_delay_seconds
        self._warn = warn
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="TranscriptIntake",
        )

    def start(self) -> None:
        """Start the intake thread without blocking."""
        self._thread.start()

    def stop(self) -> None:
        """Stop capture, drain queued blocks, and join the intake thread."""
        self._stop_event.set()
        try:
            self._source.stop()
        finally:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        self._log(
            f"thread start | source={self._source.description} "
            f"chunk_seconds={self._chunk_seconds}"
        )
        while not self._stop_event.is_set():
            audio_queue: queue.Queue[np.ndarray | Exception] = queue.Queue()
            try:
                try:
                    self._source.start(audio_queue.put, audio_queue.put)
                    remaining = self._consume(audio_queue)
                finally:
                    self._source.stop()
                self._flush(remaining)
            except Exception as exc:  # noqa: BLE001
                if self._stop_event.is_set():
                    break
                self._log("capture loop crashed:\n" + traceback.format_exc())
                self._warn(
                    f"[TranscriptIntake] error — restarting in "
                    f"{self._restart_delay_seconds} s: {exc}"
                )
                self._stop_event.wait(self._restart_delay_seconds)
        self._log("thread exit")

    def _consume(self, audio_queue: queue.Queue[np.ndarray | Exception]) -> list[np.ndarray]:
        frames_per_chunk = _SAMPLE_RATE * self._chunk_seconds
        accumulator: list[np.ndarray] = []
        accumulated_frames = 0
        silent_chunks = 0
        warned_silent = False
        chunk_count = 0

        while not self._stop_event.is_set() or not audio_queue.empty():
            try:
                block = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(block, Exception):
                raise block

            accumulator.append(block)
            accumulated_frames += len(block)
            if accumulated_frames < frames_per_chunk:
                continue

            audio_chunk = np.concatenate(accumulator)
            accumulator = []
            accumulated_frames = 0
            chunk_count += 1

            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
            peak = float(np.max(np.abs(audio_chunk)))
            self._log(
                f"chunk #{chunk_count}: frames={len(audio_chunk)} "
                f"rms={rms:.6f} peak={peak:.6f}"
            )

            if rms < _SILENCE_RMS:
                silent_chunks += 1
                if silent_chunks >= _SILENCE_WARN_AFTER and not warned_silent:
                    warned_silent = True
                    self._log("SILENCE detected — audio is not reaching the device")
                    self._warn(f"[TranscriptIntake] {self._source.silence_warning}")
                continue

            silent_chunks = 0
            warned_silent = False
            text = self._transcriber.transcribe(audio_chunk)
            self._log(f"chunk #{chunk_count} transcribed: {len(text)} chars: {text[:80]!r}")
            if text:
                self._transcript.append(text)
                self._archive.append_transcript(text)

        return accumulator

    def _flush(self, remaining: list[np.ndarray]) -> None:
        if not remaining:
            return
        try:
            audio_chunk = np.concatenate(remaining)
            text = self._transcriber.transcribe(audio_chunk)
            if text:
                self._transcript.append(text)
                self._archive.append_transcript(text)
        except Exception as exc:  # noqa: BLE001
            self._log(f"final flush failed: {exc}")
