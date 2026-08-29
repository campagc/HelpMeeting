"""System-playback source interface and BlackHole adapter."""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # Hz — Whisper expects 16 kHz mono
CHANNELS = 1
CALLBACK_BLOCK = 1024  # frames per PortAudio callback invocation

BlockHandler = Callable[[np.ndarray], None]
ErrorHandler = Callable[[Exception], None]


class _InputStream(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def close(self) -> None:
        ...


class SystemPlaybackSource(Protocol):
    """Prepared source of copied 16 kHz mono float32 system-playback blocks."""

    @property
    def description(self) -> str:
        """Return a human-readable identity for diagnostics."""
        ...

    @property
    def silence_warning(self) -> str:
        """Return source-specific guidance for sustained silence."""
        ...

    def prepare(self) -> None:
        """Resolve source prerequisites, raising a user-actionable error on failure."""
        ...

    def start(self, on_block: BlockHandler, on_error: ErrorHandler) -> None:
        """Start delivery; report asynchronous failures through on_error."""
        ...

    def stop(self) -> None:
        """Stop delivery and release source resources; repeated calls are safe."""
        ...


def resolve_audio_device(
    devices: Sequence[Mapping[str, Any]], preferred_name: str = "BlackHole"
) -> int:
    """Return the named input-device index despite device index changes."""
    for index, device in enumerate(devices):
        name_matches = preferred_name.lower() in str(device["name"]).lower()
        if name_matches and int(device.get("max_input_channels", 0)) > 0:
            return index
    raise RuntimeError(
        f"No '{preferred_name}' audio device found. Install BlackHole and route "
        f"system audio through a Multi-Output Device. Available devices: "
        f"{[device['name'] for device in devices]}"
    )


class BlackHoleSource:
    """PortAudio adapter for system playback routed through BlackHole."""

    def __init__(
        self,
        *,
        preferred_name: str = "BlackHole",
        query_devices: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        input_stream_factory: Callable[..., _InputStream] | None = None,
        log_path=None,
    ) -> None:
        self._preferred_name = preferred_name
        self._query_devices = query_devices if query_devices is not None else sd.query_devices
        self._input_stream_factory = (
            input_stream_factory if input_stream_factory is not None else sd.InputStream
        )
        self._log_path = log_path
        self._device_index: int | None = None
        self._device_name: str | None = None
        self._stream: _InputStream | None = None
        self._stopping = False

    def _log(self, message: str) -> None:
        if self._log_path is None:
            return
        try:
            stamp = time.strftime("%H:%M:%S")
            with open(self._log_path, "a", encoding="utf-8") as file:
                file.write(f"[{stamp}] {message}\n")
        except Exception:
            pass

    @property
    def description(self) -> str:
        if self._device_index is None or self._device_name is None:
            return self._preferred_name
        return f"device {self._device_index}: {self._device_name}"

    @property
    def silence_warning(self) -> str:
        return (
            "No audio detected on the capture device. Is your system output routed "
            "to BlackHole (e.g. via a Multi-Output Device)?"
        )

    def prepare(self) -> None:
        devices = self._query_devices()
        self._device_index = resolve_audio_device(devices, self._preferred_name)
        self._device_name = str(devices[self._device_index]["name"])

    def start(self, on_block: BlockHandler, on_error: ErrorHandler) -> None:
        if self._device_index is None:
            self.prepare()
        self._stopping = False

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            _ = frames, time_info
            if status:
                self._log(f"stream status flag: {status}")
            try:
                block = np.asarray(indata[:, 0], dtype=np.float32).copy()
            except Exception as exc:
                on_error(exc)
                return
            on_block(block)

        def finished_callback() -> None:
            if not self._stopping:
                on_error(RuntimeError("BlackHole system-playback source stopped unexpectedly"))

        stream = self._input_stream_factory(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=self._device_index,
            blocksize=CALLBACK_BLOCK,
            callback=callback,
            finished_callback=finished_callback,
        )
        self._stream = stream
        try:
            stream.start()
        except Exception:
            self._stopping = True
            try:
                stream.close()
            except Exception:
                pass
            self._stream = None
            raise
        self._log(
            f"InputStream open (samplerate={SAMPLE_RATE} channels={CHANNELS} "
            f"blocksize={CALLBACK_BLOCK}); waiting for audio…"
        )

    def stop(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        self._stopping = True
        try:
            stream.stop()
        finally:
            stream.close()
