import json
import threading

import pytest

from src.assistant import AssistantUnavailable
from src.config import Config
from src.hud import NullHud
from src.meeting import (
    Meeting,
    MeetingSettings,
    MeetingStartupError,
    ProductionMeetingEnvironment,
)
from src.storage import Storage


class FakeSource:
    description = "fake system playback"
    silence_warning = "No playback"

    def __init__(self, events, prepare_error=None):
        self.events = events
        self.prepare_error = prepare_error

    def prepare(self):
        self.events.append("source.prepare")
        if self.prepare_error is not None:
            raise self.prepare_error

    def start(self, on_block, on_error):
        pass

    def stop(self):
        self.events.append("source.stop")


class FakeAssistant:
    def __init__(self, events, question_error=None):
        self.events = events
        self.question_error = question_error
        self.explain_calls = []
        self.ask_calls = []

    def explain_slide(self, *, image_bytes, delta):
        self.explain_calls.append((image_bytes, delta))
        self.events.append("turn.explain.finished")
        return "fake explanation"

    def ask_question(self, *, text, delta):
        self.ask_calls.append((text, delta))
        if self.question_error is not None:
            raise self.question_error
        self.events.append("turn.ask.finished")
        return "fake answer"


class RecordingArchive(Storage):
    def __init__(self, base_dir, events):
        super().__init__(base_dir=base_dir)
        self.events = events

    def start_meeting(self, label, languages):
        self.events.append("archive.start")
        super().start_meeting(label, languages)

    def finalize(self):
        self.events.append("archive.finalize")
        super().finalize()


class FakeTranscriptIntake:
    def __init__(self, events, stop_error=None):
        self.events = events
        self.stop_error = stop_error

    def start(self):
        self.events.append("transcript-intake.start")

    def stop(self):
        self.events.append("transcript-intake.stop")
        if self.stop_error is not None:
            raise self.stop_error


class FakeTriggerCapture:
    def __init__(self, callback, events, slide=None):
        self.callback = callback
        self.events = events
        self.slide = slide
        self.emitted = threading.Event()
        self.stopped = threading.Event()

    def start(self):
        self.events.append("trigger-capture.start")
        if self.slide is not None:
            self.callback(self.slide)
            self.emitted.set()
        self.stopped.wait(timeout=2.0)

    def stop(self):
        self.events.append("trigger-capture.stop")
        self.stopped.set()


class FakeHud:
    def __init__(self, events):
        self.events = events
        self.started = 0
        self.finished = []
        self.wait_for = None
        self.stopped = threading.Event()

    def run(self):
        self.events.append("hud.run")
        if self.wait_for is not None:
            self.wait_for.wait(timeout=2.0)
        self.stopped.wait(timeout=2.0)

    def stop(self):
        self.events.append("hud.stop")
        self.stopped.set()

    def turn_started(self):
        self.started += 1

    def turn_finished(self, ok):
        self.finished.append(ok)


class InMemoryMeetingEnvironment:
    def __init__(
        self,
        tmp_path,
        *,
        inputs=None,
        slide=None,
        prepare_error=None,
        question_error=None,
        transcript_stop_error=None,
    ):
        self.tmp_path = tmp_path
        self.events = []
        self.outputs = []
        self.inputs = list(inputs or [])
        self.source = FakeSource(self.events, prepare_error=prepare_error)
        self.hud = FakeHud(self.events)
        self.archive = RecordingArchive(tmp_path, self.events)
        self.assistant = FakeAssistant(self.events, question_error=question_error)
        self.transcript_intake = FakeTranscriptIntake(
            self.events, stop_error=transcript_stop_error
        )
        self.slide = slide
        self.trigger_capture = None
        self.archive_create_count = 0

    def create_hud(self, settings, config):
        return self.hud

    def create_system_playback_source(self):
        return self.source

    def create_archive(self):
        self.archive_create_count += 1
        return self.archive

    def create_assistant(self, config):
        return self.assistant

    def create_transcript_intake(
        self, *, transcript, archive, source, settings, config, log_path
    ):
        return self.transcript_intake

    def create_trigger_capture(self, *, callback, settings, config):
        self.trigger_capture = FakeTriggerCapture(callback, self.events, slide=self.slide)
        if self.slide is not None:
            self.hud.wait_for = self.trigger_capture.emitted
        return self.trigger_capture

    def read_question(self, prompt):
        if not self.inputs:
            raise EOFError()
        value = self.inputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def write(self, message):
        self.outputs.append(message)


def make_settings():
    return MeetingSettings(
        label="test-meeting",
        spoken_language="en",
        explanation_language="it",
        display_index=2,
    )


def make_config():
    return Config(
        hotkeys=["<ctrl>+<alt>+s"],
        gemini_model_name="test-model",
        audio_chunk_seconds=10,
        whisper_model_size="base",
        outcome_badge_seconds=1.5,
        meetings_dir="meetings",
        api_key="test-key",
        system_prompt="test prompt",
    )


def open_meeting(environment):
    return Meeting.open(make_settings(), make_config(), environment=environment)


def test_open_prepares_playback_before_starting_archive(tmp_path):
    environment = InMemoryMeetingEnvironment(tmp_path)

    open_meeting(environment)

    assert environment.events[:2] == ["source.prepare", "archive.start"]


def test_failed_preflight_raises_one_startup_error_and_creates_no_archive(tmp_path):
    cause = RuntimeError("ScreenCaptureKit is unavailable")
    environment = InMemoryMeetingEnvironment(tmp_path, prepare_error=cause)

    with pytest.raises(MeetingStartupError, match="ScreenCaptureKit is unavailable") as exc_info:
        open_meeting(environment)

    assert exc_info.value.__cause__ is cause
    assert environment.archive_create_count == 0
    assert not (tmp_path / "meetings").exists()


def test_generic_startup_failure_keeps_the_existing_message_prefix(tmp_path):
    environment = InMemoryMeetingEnvironment(
        tmp_path, prepare_error=ValueError("unsupported playback")
    )

    with pytest.raises(
        MeetingStartupError,
        match="HelpMeeting cannot start: unsupported playback",
    ):
        open_meeting(environment)


def test_open_uses_the_production_environment_adapter_by_default(tmp_path, monkeypatch):
    environment = InMemoryMeetingEnvironment(tmp_path)
    monkeypatch.setattr(
        "src.meeting.ProductionMeetingEnvironment",
        lambda: environment,
    )

    meeting = Meeting.open(make_settings(), make_config())

    assert isinstance(meeting, Meeting)
    assert environment.events[:2] == ["source.prepare", "archive.start"]


def test_production_environment_degrades_when_hud_creation_fails(monkeypatch):
    def fail_hud(**kwargs):
        raise RuntimeError("AppKit failed")

    monkeypatch.setattr("src.meeting.HudPanel", fail_hud)
    environment = ProductionMeetingEnvironment()

    hud = environment.create_hud(make_settings(), make_config())

    assert isinstance(hud, NullHud)


def test_question_turn_runs_through_meeting_interface(tmp_path):
    environment = InMemoryMeetingEnvironment(
        tmp_path, inputs=["What changed?", EOFError()]
    )
    meeting = open_meeting(environment)

    assert meeting.run() == 0

    assert environment.assistant.ask_calls == [("What changed?", "")]
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    assert json.loads(history_path.read_text()) == [
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "fake answer"},
    ]
    assert environment.hud.started == 1
    assert environment.hud.finished == [True]
    assert environment.events.index("turn.ask.finished") < environment.events.index(
        "archive.finalize"
    )
    assert environment.events.index("archive.finalize") < environment.events.index(
        "transcript-intake.stop"
    )


def test_explain_turn_runs_through_meeting_interface(tmp_path):
    slide = b"\x89PNG fake"
    environment = InMemoryMeetingEnvironment(
        tmp_path, inputs=[EOFError()], slide=slide
    )
    meeting = open_meeting(environment)

    assert meeting.run() == 0

    assert environment.assistant.explain_calls == [(slide, "")]
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    history = json.loads(history_path.read_text())
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "fake explanation"
    assert history[0]["slide_path"].endswith("slide_0001.png")
    assert environment.hud.started == 1
    assert environment.hud.finished == [True]


def test_failed_turn_reports_failed_outcome_and_keeps_meeting_running(tmp_path):
    environment = InMemoryMeetingEnvironment(
        tmp_path,
        inputs=["Can you retry?", EOFError()],
        question_error=AssistantUnavailable("[Could not get a response]"),
    )
    meeting = open_meeting(environment)

    assert meeting.run() == 0

    assert environment.hud.finished == [False]
    assert any("Could not get a response" in output for output in environment.outputs)


@pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt()])
def test_terminal_endings_stop_the_meeting(tmp_path, ending):
    environment = InMemoryMeetingEnvironment(tmp_path, inputs=[ending])
    meeting = open_meeting(environment)

    assert meeting.run() == 0

    assert environment.hud.stopped.is_set()
    assert environment.trigger_capture.stopped.is_set()
    assert "archive.finalize" in environment.events


def test_transcript_stop_failure_does_not_skip_completion(tmp_path):
    environment = InMemoryMeetingEnvironment(
        tmp_path,
        inputs=["Record this before shutdown", EOFError()],
        transcript_stop_error=KeyboardInterrupt(),
    )
    meeting = open_meeting(environment)

    assert meeting.run() == 0

    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    assert len(json.loads(history_path.read_text())) == 2
    assert environment.outputs[-1] == "Done."
    assert "archive.finalize" in environment.events


def test_meeting_prints_runtime_output(tmp_path):
    environment = InMemoryMeetingEnvironment(tmp_path, inputs=[EOFError()])
    meeting = open_meeting(environment)

    meeting.run()

    output = "\n".join(environment.outputs)
    assert "Capturing audio from fake system playback" in output
    assert "Audio diagnostics:" in output
    assert "HelpMeeting ready: test-meeting" in output
    assert "spoken: en, explanation: it" in output
    assert "display: 2" in output
    assert "Control+Option+S" in output
    assert "Type a question" in output
    assert "Shutting down and saving the meeting" in output
    assert output.endswith("Done.")


def test_meeting_is_single_use(tmp_path):
    environment = InMemoryMeetingEnvironment(tmp_path, inputs=[EOFError()])
    meeting = open_meeting(environment)
    meeting.run()

    with pytest.raises(RuntimeError, match="only run once"):
        meeting.run()
