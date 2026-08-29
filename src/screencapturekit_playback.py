"""Direct ScreenCaptureKit system-playback source."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import CoreMedia as CM  # type: ignore[import-untyped]
import numpy as np
import objc  # type: ignore[import-untyped]
import Quartz  # type: ignore[import-untyped]
import ScreenCaptureKit as SCK  # type: ignore[import-untyped]
import dispatch  # type: ignore[import-untyped]
from Foundation import NSObject  # type: ignore[import-untyped]

from src.system_playback import BlockHandler, ErrorHandler, CHANNELS, SAMPLE_RATE

SampleHandler = Callable[[np.ndarray], None]
_ASYNC_TIMEOUT = 15.0
_LINEAR_PCM = int.from_bytes(b"lpcm", "big")
_FORMAT_FLAG_IS_FLOAT = 1 << 0
_FORMAT_FLAG_IS_BIG_ENDIAN = 1 << 1


@dataclass(frozen=True)
class _AudioFormat:
    sample_rate: float
    format_id: int
    flags: int
    bytes_per_frame: int
    channels: int
    bits: int

    @classmethod
    def from_native(cls, native: Any) -> "_AudioFormat":
        try:
            return cls(
                sample_rate=float(native.mSampleRate),
                format_id=int(native.mFormatID),
                flags=int(native.mFormatFlags),
                bytes_per_frame=int(native.mBytesPerFrame),
                channels=int(native.mChannelsPerFrame),
                bits=int(native.mBitsPerChannel),
            )
        except AttributeError:
            (
                sample_rate,
                format_id,
                flags,
                _bytes_per_packet,
                _frames_per_packet,
                bytes_per_frame,
                channels,
                bits,
                _reserved,
            ) = native
            return cls(
                sample_rate=float(sample_rate),
                format_id=int(format_id),
                flags=int(flags),
                bytes_per_frame=int(bytes_per_frame),
                channels=int(channels),
                bits=int(bits),
            )

    @property
    def summary(self) -> str:
        return (
            f"{self.sample_rate:.0f} Hz, {self.channels} channel(s), "
            f"{self.bits}-bit {_fourcc(self.format_id)}, flags=0x{self.flags:x}, "
            f"bytes/frame={self.bytes_per_frame}"
        )

    def validate(self) -> None:
        supported = (
            self.sample_rate == SAMPLE_RATE
            and self.channels == CHANNELS
            and self.bits == 32
            and self.format_id == _LINEAR_PCM
            and bool(self.flags & _FORMAT_FLAG_IS_FLOAT)
            and not self.flags & _FORMAT_FLAG_IS_BIG_ENDIAN
        )
        if not supported:
            raise RuntimeError(f"ScreenCaptureKit delivered unsupported audio format: {self.summary}")


class ScreenCapturePermissionError(RuntimeError):
    """Screen & System Audio Recording access is unavailable."""


class ScreenCaptureKitFramework(Protocol):
    format_summary: str | None

    def prepare(self) -> None:
        ...

    def start(self, on_samples: SampleHandler, on_error: ErrorHandler) -> None:
        ...

    def stop(self) -> None:
        ...


class ScreenCaptureKitSource:
    """System-playback source backed by direct PyObjC ScreenCaptureKit calls."""

    def __init__(self, *, framework: ScreenCaptureKitFramework | None = None) -> None:
        self._framework = framework if framework is not None else _PyObjCScreenCaptureKit()

    @property
    def description(self) -> str:
        return "ScreenCaptureKit (all system playback, current process excluded)"

    @property
    def silence_warning(self) -> str:
        return "No system playback was detected by ScreenCaptureKit."

    @property
    def format_summary(self) -> str | None:
        return self._framework.format_summary

    def prepare(self) -> None:
        self._framework.prepare()

    def start(self, on_block: BlockHandler, on_error: ErrorHandler) -> None:
        def deliver(samples: np.ndarray) -> None:
            try:
                block = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
            except Exception as exc:
                on_error(exc)
                return
            on_block(block)

        self._framework.start(deliver, on_error)

    def stop(self) -> None:
        self._framework.stop()


class _StreamDelegate(
    NSObject,
    protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")],
):  # type: ignore[call-arg]
    @objc.python_method
    def configure(
        self,
        on_samples: SampleHandler,
        on_error: ErrorHandler,
        sample_converter: Callable[[Any], np.ndarray],
    ) -> None:
        self._on_samples = on_samples
        self._on_error = on_error
        self._sample_converter = sample_converter
        self._stopping = False

    @objc.python_method
    def mark_stopping(self) -> None:
        self._stopping = True

    def stream_didOutputSampleBuffer_ofType_(
        self, stream: Any, sample_buffer: Any, output_type: int
    ) -> None:
        _ = stream
        if output_type != SCK.SCStreamOutputTypeAudio:
            return
        try:
            samples = self._sample_converter(sample_buffer)
        except Exception as exc:
            self._on_error(exc)
            return
        self._on_samples(samples)

    def stream_didStopWithError_(self, stream: Any, error: Any) -> None:
        _ = stream
        if not self._stopping:
            self._on_error(RuntimeError(f"ScreenCaptureKit stream stopped: {_error_text(error)}"))


class _PyObjCScreenCaptureKit:
    format_summary: str | None

    def __init__(self) -> None:
        self.format_summary = None
        self._content: Any = None
        self._filter: Any = None
        self._configuration: Any = None
        self._stream: Any = None
        self._delegate: _StreamDelegate | None = None
        self._sample_queue: Any = None

    def prepare(self) -> None:
        if not Quartz.CGPreflightScreenCaptureAccess():
            Quartz.CGRequestScreenCaptureAccess()
            raise ScreenCapturePermissionError(
                "Screen & System Audio Recording permission is required. Grant it to your "
                "terminal in System Settings > Privacy & Security > Screen & System Audio "
                "Recording, then relaunch the diagnostic."
            )

        completed = threading.Event()
        result: dict[str, Any] = {}

        def content_ready(content: Any, error: Any) -> None:
            result["content"] = content
            result["error"] = error
            completed.set()

        SCK.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
            False, False, content_ready
        )
        if not completed.wait(_ASYNC_TIMEOUT):
            raise RuntimeError("ScreenCaptureKit timed out while enumerating shareable content")
        if result["error"] is not None:
            raise RuntimeError(
                f"ScreenCaptureKit could not enumerate shareable content: "
                f"{_error_text(result['error'])}"
            )

        content = result["content"]
        displays = content.displays()
        if len(displays) == 0:
            raise RuntimeError("ScreenCaptureKit found no display to anchor system audio capture")

        content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
            displays[0], [], []
        )
        configuration = SCK.SCStreamConfiguration.alloc().init()
        configuration.setCapturesAudio_(True)
        configuration.setExcludesCurrentProcessAudio_(True)
        configuration.setSampleRate_(SAMPLE_RATE)
        configuration.setChannelCount_(CHANNELS)
        configuration.setWidth_(2)
        configuration.setHeight_(2)
        configuration.setMinimumFrameInterval_(CM.CMTimeMake(1, 1))
        configuration.setQueueDepth_(3)
        configuration.setShowsCursor_(False)

        self._content = content
        self._filter = content_filter
        self._configuration = configuration

    def start(self, on_samples: SampleHandler, on_error: ErrorHandler) -> None:
        if self._filter is None or self._configuration is None:
            self.prepare()

        delegate = _StreamDelegate.alloc().init()
        delegate.configure(on_samples, on_error, self._copy_samples)
        sample_queue = dispatch.dispatch_queue_create(
            b"com.helpmeeting.screencapturekit.audio", None
        )
        stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            self._filter, self._configuration, delegate
        )
        added, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            delegate, SCK.SCStreamOutputTypeAudio, sample_queue, None
        )
        if not added:
            raise RuntimeError(
                f"ScreenCaptureKit could not add its audio output: {_error_text(error)}"
            )

        self._delegate = delegate
        self._sample_queue = sample_queue
        self._stream = stream
        completed = threading.Event()
        result: dict[str, Any] = {}

        def started(error: Any) -> None:
            result["error"] = error
            completed.set()

        stream.startCaptureWithCompletionHandler_(started)
        if not completed.wait(_ASYNC_TIMEOUT):
            self._clear_stream()
            raise RuntimeError("ScreenCaptureKit timed out while starting capture")
        if result["error"] is not None:
            self._clear_stream()
            raise RuntimeError(
                f"ScreenCaptureKit could not start capture: {_error_text(result['error'])}"
            )

    def stop(self) -> None:
        stream = self._stream
        if stream is None:
            return
        if self._delegate is not None:
            self._delegate.mark_stopping()

        completed = threading.Event()
        result: dict[str, Any] = {}

        def stopped(error: Any) -> None:
            result["error"] = error
            completed.set()

        try:
            stream.stopCaptureWithCompletionHandler_(stopped)
            if not completed.wait(_ASYNC_TIMEOUT):
                raise RuntimeError("ScreenCaptureKit timed out while stopping capture")
            if result["error"] is not None:
                raise RuntimeError(
                    f"ScreenCaptureKit could not stop capture: {_error_text(result['error'])}"
                )
        finally:
            self._clear_stream()

    def _copy_samples(self, sample_buffer: Any) -> np.ndarray:
        if not CM.CMSampleBufferDataIsReady(sample_buffer):
            raise RuntimeError("ScreenCaptureKit delivered an audio buffer before it was ready")

        format_description = CM.CMSampleBufferGetFormatDescription(sample_buffer)
        native_format = CM.CMAudioFormatDescriptionGetStreamBasicDescription(format_description)
        audio_format = _AudioFormat.from_native(native_format)
        self.format_summary = audio_format.summary
        audio_format.validate()

        block_buffer = CM.CMSampleBufferGetDataBuffer(sample_buffer)
        if block_buffer is None:
            raise RuntimeError("ScreenCaptureKit delivered audio without a CoreMedia data buffer")
        data_length = CM.CMBlockBufferGetDataLength(block_buffer)
        status, data = CM.CMBlockBufferCopyDataBytes(block_buffer, 0, data_length, None)
        if status != CM.kCMBlockBufferNoErr:
            raise RuntimeError(f"CoreMedia could not copy audio data (status {status})")

        samples = np.frombuffer(data, dtype=np.float32)
        expected_samples = CM.CMSampleBufferGetNumSamples(sample_buffer) * audio_format.channels
        if len(samples) != expected_samples:
            raise RuntimeError(
                f"ScreenCaptureKit audio layout mismatch: expected {expected_samples} float32 "
                f"samples, copied {len(samples)}"
            )
        return samples

    def _clear_stream(self) -> None:
        self._stream = None
        self._delegate = None
        self._sample_queue = None


def _fourcc(value: int) -> str:
    try:
        return value.to_bytes(4, "big").decode("ascii")
    except (OverflowError, UnicodeDecodeError):
        return f"0x{value:x}"


def _error_text(error: Any) -> str:
    if error is None:
        return "unknown error"
    try:
        return str(error.localizedDescription())
    except Exception:
        return str(error)
