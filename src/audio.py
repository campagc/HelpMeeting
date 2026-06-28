"""audio module: system-audio capture and local transcription thread.

Public interface
----------------
thread = AudioThread(
    transcript=transcript,
    storage=storage,
    chunk_seconds=10,
    language="en",
    device=0,
    model_size="small",
)
thread.start()   # non-blocking; returns immediately
thread.stop()    # signals the thread to finish and joins it
"""

import queue
import threading
import time
from typing import Protocol

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

_SAMPLE_RATE = 16_000       # Hz — Whisper expects 16 kHz mono
_CHANNELS = 1
_CALLBACK_BLOCK = 1024      # frames per PortAudio callback invocation
_SILENCE_RMS = 1e-4         # below this a chunk is treated as silence
_SILENCE_WARN_AFTER = 2     # warn after this many consecutive silent chunks


class Transcriber(Protocol):
    """Converts a 1-D float32 audio array into a transcript string."""

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio; return combined text or empty string."""
        ...


class WhisperTranscriber:
    """Concrete Transcriber backed by faster-whisper.WhisperModel.

    The model is loaded eagerly in __init__ so failures are reported at
    startup (matching the original _capture_loop behaviour).
    """

    def __init__(self, model_size: str, language: str, log_fn=None) -> None:
        self._language = language
        _log = log_fn or (lambda _: None)
        _log("loading Whisper model…")
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _log("Whisper model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array; return combined text or empty string."""
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class AudioThread:
    """Background thread: captures system audio, transcribes in ~chunk_seconds windows,
    appends text to the transcript object and persists it via storage."""

    def __init__(
        self,
        *,
        transcript,
        storage,
        chunk_seconds: int = 10,
        language: str = "en",
        device: int | str | None = 0,
        model_size: str = "small",
        log_path=None,
        transcriber: Transcriber | None = None,
    ):
        self._transcript = transcript
        self._storage = storage
        self._chunk_seconds = chunk_seconds
        self._language = language
        self._device = device
        self._model_size = model_size
        self._log_path = log_path
        self._transcriber = transcriber

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_with_restart, daemon=True, name="AudioThread"
        )

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Append a timestamped diagnostic line to the debug log (if configured)."""
        if self._log_path is None:
            return
        try:
            stamp = time.strftime("%H:%M:%S")
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {message}\n")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the transcription thread (non-blocking)."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and block until it exits."""
        self._stop_event.set()
        self._thread.join()

    # ------------------------------------------------------------------
    # Restart-on-exception outer loop
    # ------------------------------------------------------------------

    def _run_with_restart(self) -> None:
        """Outer loop: if the inner capture loop raises, restart after a short delay."""
        import traceback

        self._log(
            f"thread start | device={self._device} model={self._model_size} "
            f"chunk_seconds={self._chunk_seconds} language={self._language}"
        )
        while not self._stop_event.is_set():
            try:
                self._capture_loop()
            except Exception as exc:  # noqa: BLE001
                if self._stop_event.is_set():
                    break
                self._log("capture loop crashed:\n" + traceback.format_exc())
                print(f"[AudioThread] error — restarting in 2 s: {exc}")
                time.sleep(2)
        self._log("thread exit")

    # ------------------------------------------------------------------
    # Inner capture + transcription loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Open the audio stream and transcribe each complete ~chunk_seconds window
        until _stop_event is set."""
        transcriber = self._transcriber
        if transcriber is None:
            transcriber = WhisperTranscriber(
                model_size=self._model_size,
                language=self._language,
                log_fn=self._log,
            )

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        callback_count = 0

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            """PortAudio callback — runs in a separate C thread."""
            nonlocal callback_count
            callback_count += 1
            if status:
                self._log(f"stream status flag: {status}")
            chunk = indata[:, 0].copy()  # flatten to mono
            audio_queue.put(chunk)

        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="float32",
            device=self._device,
            blocksize=_CALLBACK_BLOCK,
            callback=_callback,
        ):
            self._log(
                f"InputStream open (samplerate={_SAMPLE_RATE} channels={_CHANNELS} "
                f"blocksize={_CALLBACK_BLOCK}); waiting for audio…"
            )
            remaining = self._run_chunk_loop(audio_queue, transcriber)

        # Flush any remaining audio that did not fill a full chunk.
        if remaining:
            audio_chunk = np.concatenate(remaining)
            text = transcriber.transcribe(audio_chunk)
            if text:
                self._transcript.append(text)
                self._storage.append_transcript(text)

    # ------------------------------------------------------------------
    # Core chunk-accumulation loop (testable seam)
    # ------------------------------------------------------------------

    def _run_chunk_loop(
        self,
        audio_queue: "queue.Queue[np.ndarray]",
        transcriber: Transcriber,
    ) -> "list[np.ndarray]":
        """Accumulate audio blocks, detect silence, and transcribe complete chunks.

        Returns the leftover accumulator (blocks that did not fill a complete chunk)
        so the caller can flush them after the stream closes.
        """
        frames_per_chunk = _SAMPLE_RATE * self._chunk_seconds
        accumulator: list[np.ndarray] = []
        accumulated_frames = 0
        silent_chunks = 0
        warned_silent = False
        chunk_count = 0

        while not self._stop_event.is_set():
            try:
                block = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            accumulator.append(block)
            accumulated_frames += len(block)

            if accumulated_frames >= frames_per_chunk:
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

                # Warn once if the meeting audio is not reaching us — a common
                # setup mistake (system output not routed into BlackHole).
                if rms < _SILENCE_RMS:
                    silent_chunks += 1
                    if silent_chunks >= _SILENCE_WARN_AFTER and not warned_silent:
                        warned_silent = True
                        self._log("SILENCE detected — audio is not reaching the device")
                        print(
                            "[AudioThread] No audio detected on the capture device. "
                            "Is your system output routed to BlackHole "
                            "(e.g. via a Multi-Output Device)?"
                        )
                    continue
                silent_chunks = 0
                warned_silent = False

                text = transcriber.transcribe(audio_chunk)
                self._log(f"chunk #{chunk_count} transcribed: {len(text)} chars: {text[:80]!r}")
                if text:
                    self._transcript.append(text)
                    self._storage.append_transcript(text)

        return accumulator
