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

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

_SAMPLE_RATE = 16_000       # Hz — Whisper expects 16 kHz mono
_CHANNELS = 1
_CALLBACK_BLOCK = 1024      # frames per PortAudio callback invocation


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
    ):
        self._transcript = transcript
        self._storage = storage
        self._chunk_seconds = chunk_seconds
        self._language = language
        self._device = device
        self._model_size = model_size

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_with_restart, daemon=True, name="AudioThread"
        )

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
        while not self._stop_event.is_set():
            try:
                self._capture_loop()
            except Exception as exc:  # noqa: BLE001
                if self._stop_event.is_set():
                    break
                print(f"[AudioThread] error — restarting in 2 s: {exc}")
                time.sleep(2)

    # ------------------------------------------------------------------
    # Inner capture + transcription loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Load the Whisper model once, open the audio stream, and transcribe
        each complete ~chunk_seconds window until _stop_event is set."""
        model = WhisperModel(self._model_size, device="cpu", compute_type="int8")

        frames_per_chunk = _SAMPLE_RATE * self._chunk_seconds
        audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        accumulator: list[np.ndarray] = []
        accumulated_frames = 0

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            """PortAudio callback — runs in a separate C thread."""
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
                    text = self._transcribe(model, audio_chunk)
                    if text:
                        self._transcript.append(text)
                        self._storage.append_transcript(text)

        # Flush any remaining audio that did not fill a full chunk.
        if accumulator:
            audio_chunk = np.concatenate(accumulator)
            text = self._transcribe(model, audio_chunk)
            if text:
                self._transcript.append(text)
                self._storage.append_transcript(text)

    # ------------------------------------------------------------------
    # Transcription helper
    # ------------------------------------------------------------------

    def _transcribe(self, model: WhisperModel, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array; return combined text or empty string."""
        segments, _info = model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
