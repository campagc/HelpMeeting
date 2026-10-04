"""Meeting module: own one Meeting from startup through shutdown."""

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.assistant import Assistant
from src.audio import TranscriptIntake, WhisperTranscriber, append_log
from src.capture import Capture
from src.config import Config, format_hotkeys
from src.display import Displays
from src.hud import Hud, HudPanel, NullHud
from src.storage import Storage
from src.system_playback import SystemPlaybackSource, default_system_playback_source
from src.transcript import Transcript
from src.turn import Turn


@dataclass(frozen=True)
class MeetingSettings:
    label: str
    spoken_language: str
    explanation_language: str
    display_index: int


class MeetingStartupError(RuntimeError):
    pass


class MeetingEnvironment(Protocol):
    def create_hud(self, settings: MeetingSettings, config: Config) -> Hud:
        ...

    def create_system_playback_source(self) -> SystemPlaybackSource:
        ...

    def create_archive(self) -> Storage:
        ...

    def create_assistant(self, config: Config) -> Assistant:
        ...

    def create_transcript_intake(
        self,
        *,
        transcript: Transcript,
        archive: Storage,
        source: SystemPlaybackSource,
        settings: MeetingSettings,
        config: Config,
        log_path: Path | None,
    ) -> TranscriptIntake:
        ...

    def create_trigger_capture(
        self,
        *,
        callback: Callable[[bytes], None],
        settings: MeetingSettings,
        config: Config,
    ) -> Capture:
        ...

    def read_question(self, prompt: str) -> str:
        ...

    def write(self, message: str) -> None:
        ...


class ProductionMeetingEnvironment:
    def __init__(self) -> None:
        self._displays = Displays()

    def create_hud(self, settings: MeetingSettings, config: Config) -> Hud:
        try:
            return HudPanel(
                displays=self._displays,
                display_index=settings.display_index,
                outcome_seconds=config.outcome_badge_seconds,
            )
        except Exception:
            return NullHud()

    def create_system_playback_source(self) -> SystemPlaybackSource:
        return default_system_playback_source()

    def create_archive(self) -> Storage:
        return Storage()

    def create_assistant(self, config: Config) -> Assistant:
        from google import genai

        client = genai.Client(api_key=config.api_key)
        return Assistant(
            client=client,
            model=config.gemini_model_name,
            system_prompt=config.system_prompt,
        )

    def create_transcript_intake(
        self,
        *,
        transcript: Transcript,
        archive: Storage,
        source: SystemPlaybackSource,
        settings: MeetingSettings,
        config: Config,
        log_path: Path | None,
    ) -> TranscriptIntake:
        transcriber = WhisperTranscriber(
            model_size=config.whisper_model_size,
            language=settings.spoken_language,
            log_fn=append_log(log_path),
        )
        return TranscriptIntake(
            source=source,
            transcriber=transcriber,
            transcript=transcript,
            archive=archive,
            chunk_seconds=config.audio_chunk_seconds,
            log_path=log_path,
        )

    def create_trigger_capture(
        self,
        *,
        callback: Callable[[bytes], None],
        settings: MeetingSettings,
        config: Config,
    ) -> Capture:
        return Capture(
            callback=callback,
            displays=self._displays,
            display_index=settings.display_index,
            hotkeys=config.hotkeys,
        )

    def read_question(self, prompt: str) -> str:
        return input(prompt)

    def write(self, message: str) -> None:
        print(message)


class Meeting:
    _settings: MeetingSettings
    _config: Config
    _environment: MeetingEnvironment
    _hud: Hud
    _archive: Storage
    _turn: Turn
    _transcript_intake: TranscriptIntake
    _trigger_capture: Capture
    _print_lock: threading.Lock
    _task_queue: queue.Queue[tuple[str, bytes | str] | None]
    _worker_thread: threading.Thread
    _capture_thread: threading.Thread | None
    _shutdown: threading.Event
    _has_run: bool

    def __init__(self) -> None:
        raise TypeError("Use Meeting.open()")

    @classmethod
    def open(
        cls,
        settings: MeetingSettings,
        config: Config,
        *,
        environment: MeetingEnvironment | None = None,
    ) -> "Meeting":
        actual_environment = (
            environment if environment is not None else ProductionMeetingEnvironment()
        )
        try:
            hud = actual_environment.create_hud(settings, config)
            source = actual_environment.create_system_playback_source()
            source.prepare()

            transcript = Transcript()
            archive = actual_environment.create_archive()
            archive.start_meeting(
                settings.label,
                [settings.spoken_language, settings.explanation_language],
            )

            actual_environment.write(f"Capturing audio from {source.description}")
            audio_log = archive.meeting_dir / "audio_debug.log" if archive.meeting_dir else None
            if audio_log is not None:
                actual_environment.write(f"Audio diagnostics: {audio_log}")

            assistant = actual_environment.create_assistant(config)
            turn = Turn(transcript=transcript, archive=archive, assistant=assistant)

            meeting = cls.__new__(cls)
            meeting._settings = settings
            meeting._config = config
            meeting._environment = actual_environment
            meeting._hud = hud
            meeting._archive = archive
            meeting._turn = turn
            meeting._print_lock = threading.Lock()
            meeting._task_queue = queue.Queue()
            meeting._worker_thread = threading.Thread(
                target=meeting._worker_loop,
                daemon=True,
                name="AssistantWorker",
            )
            meeting._capture_thread = None
            meeting._shutdown = threading.Event()
            meeting._has_run = False
            meeting._transcript_intake = actual_environment.create_transcript_intake(
                transcript=transcript,
                archive=archive,
                source=source,
                settings=settings,
                config=config,
                log_path=audio_log,
            )
            meeting._trigger_capture = actual_environment.create_trigger_capture(
                callback=meeting._on_trigger,
                settings=settings,
                config=config,
            )
            return meeting
        except Exception as exc:
            message = str(exc) if isinstance(exc, RuntimeError) else f"HelpMeeting cannot start: {exc}"
            raise MeetingStartupError(message) from exc

    def run(self) -> int:
        if self._has_run:
            raise RuntimeError("A Meeting can only run once")
        self._has_run = True

        self._environment.write(f"HelpMeeting ready: {self._settings.label}")
        self._environment.write(
            f"  spoken: {self._settings.spoken_language}, "
            f"explanation: {self._settings.explanation_language}"
        )
        self._environment.write(f"  display: {self._settings.display_index}")
        self._environment.write(
            f"Press {format_hotkeys(self._config.hotkeys)} to request an explanation, "
            "Ctrl+C to stop."
        )

        self._transcript_intake.start()
        self._capture_thread = threading.Thread(
            target=self._trigger_capture.start,
            daemon=True,
            name="CaptureListener",
        )
        self._capture_thread.start()
        self._worker_thread.start()
        input_thread = threading.Thread(
            target=self._run_input_loop,
            daemon=True,
            name="QuestionInput",
        )
        input_thread.start()
        try:
            self._hud.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._hud.stop()
            self._environment.write("\nShutting down and saving the meeting…")
            self._stop()
            self._environment.write("Done.")
        return 0

    def _run_input_loop(self) -> None:
        self._safe_print("Type a question and press Enter, or press Ctrl+C to stop.")
        while not self._shutdown.is_set():
            try:
                question = self._environment.read_question("Question: ")
            except (EOFError, KeyboardInterrupt):
                self._shutdown.set()
                self._hud.stop()
                break
            if self._shutdown.is_set():
                break
            if question.strip():
                self._queue_turn("question", question)

    def _on_trigger(self, png_bytes: bytes) -> None:
        self._queue_turn("explain", png_bytes)

    def _queue_turn(self, kind: str, payload: bytes | str) -> None:
        self._hud.turn_started()
        self._task_queue.put((kind, payload))

    def _worker_loop(self) -> None:
        while True:
            item = self._task_queue.get()
            if item is None:
                break
            task_type, payload = item
            if task_type == "explain":
                assert isinstance(payload, bytes)
                result = self._turn.explain(payload)
            else:
                assert isinstance(payload, str)
                result = self._turn.ask(payload)
            if result.ok:
                heading = "[Explanation]" if task_type == "explain" else "[Answer]"
                self._safe_print(f"\n{heading}\n{result.text}\n")
            else:
                self._safe_print(result.text)
            self._hud.turn_finished(result.ok)

    def _stop(self) -> None:
        self._shutdown.set()
        self._quietly(self._trigger_capture.stop)

        # Let any in-flight or queued turn finish and be recorded.
        self._task_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=30.0)

        # Critical: persist the machine-readable history before slow cleanup.
        self._quietly(self._archive.finalize)

        # Best-effort: flush the final transcript chunk; may be slow.
        self._quietly(self._transcript_intake.stop)

        capture_thread = self._capture_thread
        if capture_thread is not None:
            self._quietly(lambda: capture_thread.join(timeout=1.0))

    @staticmethod
    def _quietly(func: Callable[[], None]) -> None:
        try:
            func()
        except (Exception, KeyboardInterrupt):
            pass

    def _safe_print(self, message: str) -> None:
        with self._print_lock:
            self._environment.write(message)
