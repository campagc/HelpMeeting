"""Tests for the audio module — chunk accumulation, silence detection, and restart logic.

All tests use a FakeTranscriber instead of WhisperModel, and drive the core
accumulation / silence / transcription loop via AudioThread._run_chunk_loop
directly, without opening a real sd.InputStream.
"""

import queue
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from src.audio import (
    AudioThread,
    _SAMPLE_RATE,
    _SILENCE_RMS,
    _SILENCE_WARN_AFTER,
)
from src.storage import Storage
from src.transcript import Transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTranscriber:
    """Test double for Transcriber: records calls and returns fixed text."""

    def __init__(self, return_text: str = "hello world") -> None:
        self.calls: list[np.ndarray] = []
        self.return_text = return_text

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(audio.copy())
        return self.return_text


def _make_thread(tmp_path, transcriber=None, chunk_seconds: int = 1):
    """Wire a minimal AudioThread with an in-memory Transcript and tmp Storage."""
    transcript = Transcript()
    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-audio", ["en"])
    thread = AudioThread(
        transcript=transcript,
        storage=storage,
        chunk_seconds=chunk_seconds,
        transcriber=transcriber,
    )
    return thread, transcript, storage


def _build_blocks(n_blocks: int, amplitude: float, block_size: int = 512) -> list[np.ndarray]:
    """Return a list of mono float32 blocks with the given amplitude."""
    return [np.full(block_size, amplitude, dtype="float32") for _ in range(n_blocks)]


def _blocks_per_chunk(chunk_seconds: int = 1, block_size: int = 512) -> int:
    """Return the minimum number of blocks to fill one complete chunk."""
    return (_SAMPLE_RATE * chunk_seconds + block_size - 1) // block_size


def _feed_and_run(thread: AudioThread, transcriber, blocks, *, join_timeout: float = 2.0) -> None:
    """Pre-populate a queue, run _run_chunk_loop in a thread, then signal stop.

    Sets stop_event after a brief pause so the loop has time to consume all
    pre-loaded blocks before it exits.
    """
    q: queue.Queue = queue.Queue()
    for block in blocks:
        q.put(block)

    runner = threading.Thread(
        target=thread._run_chunk_loop, args=(q, transcriber), daemon=True
    )
    runner.start()
    # Give the loop time to start and consume queued blocks.
    time.sleep(0.1)
    thread._stop_event.set()
    runner.join(timeout=join_timeout)
    assert not runner.is_alive(), "chunk loop did not exit within timeout"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    """_run_chunk_loop accumulates blocks and triggers transcription at chunk boundaries."""

    def test_full_chunk_is_transcribed(self, tmp_path):
        ft = FakeTranscriber("spoken text")
        thread, transcript, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        n = _blocks_per_chunk()
        blocks = _build_blocks(n, amplitude=0.1)  # well above _SILENCE_RMS

        _feed_and_run(thread, ft, blocks)

        assert len(ft.calls) == 1
        assert len(ft.calls[0]) >= _SAMPLE_RATE  # at least one second of audio
        assert transcript._buffer == ["spoken text"]

    def test_two_full_chunks_trigger_two_transcription_calls(self, tmp_path):
        ft = FakeTranscriber("chunk")
        thread, transcript, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        blocks = _build_blocks(_blocks_per_chunk() * 2, amplitude=0.1)

        _feed_and_run(thread, ft, blocks)

        assert len(ft.calls) == 2
        assert transcript._buffer == ["chunk", "chunk"]

    def test_incomplete_chunk_is_not_transcribed(self, tmp_path):
        ft = FakeTranscriber("text")
        thread, transcript, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        # Feed only half a chunk's worth of blocks.
        half = _blocks_per_chunk() // 2
        blocks = _build_blocks(half, amplitude=0.1)

        _feed_and_run(thread, ft, blocks)

        assert len(ft.calls) == 0

    def test_empty_transcription_is_not_appended(self, tmp_path):
        ft = FakeTranscriber("")  # transcriber returns empty string
        thread, transcript, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        blocks = _build_blocks(_blocks_per_chunk(), amplitude=0.1)

        _feed_and_run(thread, ft, blocks)

        assert len(ft.calls) == 1
        assert transcript._buffer == []  # empty text must not be appended

    def test_partial_blocks_accumulate_across_iterations(self, tmp_path):
        """Verify accumulation works even when block boundaries don't align to chunk size."""
        ft = FakeTranscriber("text")
        thread, _, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        # Use odd block size that doesn't divide evenly into _SAMPLE_RATE
        odd_block_size = 300
        n_blocks = (_SAMPLE_RATE + odd_block_size - 1) // odd_block_size + 1
        blocks = [np.full(odd_block_size, 0.1, dtype="float32") for _ in range(n_blocks)]

        _feed_and_run(thread, ft, blocks)

        assert len(ft.calls) == 1


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

class TestSilenceDetection:
    """Silent chunks (rms < _SILENCE_RMS) are skipped; a warning is printed once."""

    def test_silent_chunk_is_not_transcribed(self, tmp_path):
        ft = FakeTranscriber("text")
        thread, transcript, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        blocks = _build_blocks(_blocks_per_chunk(), amplitude=0.0)  # pure silence

        _feed_and_run(thread, ft, blocks)

        assert len(ft.calls) == 0

    def test_silence_warning_printed_after_threshold(self, tmp_path, capsys):
        ft = FakeTranscriber()
        thread, _, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        # Feed enough silent chunks to exceed _SILENCE_WARN_AFTER.
        n = _blocks_per_chunk() * (_SILENCE_WARN_AFTER + 1)
        blocks = _build_blocks(n, amplitude=0.0)

        _feed_and_run(thread, ft, blocks, join_timeout=3.0)

        captured = capsys.readouterr()
        assert "No audio detected" in captured.out

    def test_silence_warning_printed_only_once(self, tmp_path, capsys):
        ft = FakeTranscriber()
        thread, _, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        n = _blocks_per_chunk() * (_SILENCE_WARN_AFTER + 5)
        blocks = _build_blocks(n, amplitude=0.0)

        _feed_and_run(thread, ft, blocks, join_timeout=3.0)

        captured = capsys.readouterr()
        assert captured.out.count("No audio detected") == 1

    def test_loud_chunk_after_silence_resets_counter(self, tmp_path):
        """One silent chunk followed by a loud chunk: only the loud chunk is transcribed."""
        ft = FakeTranscriber("hello")
        thread, transcript, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        n = _blocks_per_chunk()
        silent = _build_blocks(n, amplitude=0.0)
        loud = _build_blocks(n, amplitude=0.5)

        _feed_and_run(thread, ft, silent + loud, join_timeout=3.0)

        assert len(ft.calls) == 1

    def test_just_below_silence_threshold_not_transcribed(self, tmp_path):
        """Amplitude just under _SILENCE_RMS triggers silence path."""
        ft = FakeTranscriber("text")
        thread, _, _ = _make_thread(tmp_path, ft, chunk_seconds=1)

        # _SILENCE_RMS is the RMS threshold; use amplitude slightly below it.
        below = float(_SILENCE_RMS) * 0.5
        blocks = _build_blocks(_blocks_per_chunk(), amplitude=below)

        _feed_and_run(thread, ft, blocks)

        assert len(ft.calls) == 0


# ---------------------------------------------------------------------------
# Restart-on-error
# ---------------------------------------------------------------------------

class TestRestartOnError:
    """_run_with_restart catches exceptions from _capture_loop and retries."""

    def test_restarts_after_capture_loop_exception(self, tmp_path):
        thread, _, _ = _make_thread(tmp_path, FakeTranscriber())
        call_log: list[str] = []

        def fake_capture_loop():
            call_log.append("call")
            if len(call_log) == 1:
                raise RuntimeError("device disconnected")
            thread._stop_event.set()  # allow exit on second call

        thread._capture_loop = fake_capture_loop

        with patch("time.sleep"):  # skip the 2-second restart delay
            thread._run_with_restart()

        assert call_log.count("call") == 2

    def test_does_not_restart_when_stop_event_set_on_error(self, tmp_path):
        """If stop_event is set before the exception propagates, no restart occurs."""
        thread, _, _ = _make_thread(tmp_path, FakeTranscriber())
        call_log: list[str] = []

        def fake_capture_loop():
            call_log.append("call")
            thread._stop_event.set()
            raise RuntimeError("device disconnected")

        thread._capture_loop = fake_capture_loop

        with patch("time.sleep"):
            thread._run_with_restart()

        assert call_log.count("call") == 1

    def test_does_not_run_when_stop_event_already_set(self, tmp_path):
        """_run_with_restart exits immediately if stop_event is set at entry."""
        thread, _, _ = _make_thread(tmp_path, FakeTranscriber())
        thread._stop_event.set()
        call_log: list[str] = []
        thread._capture_loop = lambda: call_log.append("call")

        thread._run_with_restart()

        assert call_log == []

    def test_multiple_restarts_until_stop_event(self, tmp_path):
        """_run_with_restart keeps retrying until stop_event is set."""
        thread, _, _ = _make_thread(tmp_path, FakeTranscriber())
        call_log: list[str] = []

        def fake_capture_loop():
            call_log.append("call")
            if len(call_log) < 4:
                raise RuntimeError("transient error")
            thread._stop_event.set()

        thread._capture_loop = fake_capture_loop

        with patch("time.sleep"):
            thread._run_with_restart()

        assert call_log.count("call") == 4
