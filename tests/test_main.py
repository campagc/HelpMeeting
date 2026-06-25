import json

import pytest

from src.main import main, prompt_settings, MeetingSession, resolve_audio_device
from src.storage import Storage
from src.transcript import Transcript


class TestResolveAudioDevice:
    """The meeting audio comes from BlackHole; we must find it by name because
    device indices shift whenever other audio devices connect/disconnect."""

    DEVICES = [
        {"name": "iPhone di Giuliano Microphone", "max_input_channels": 1},
        {"name": "BlackHole 2ch", "max_input_channels": 2},
        {"name": "MacBook Air Microphone", "max_input_channels": 1},
        {"name": "MacBook Air Speakers", "max_input_channels": 0},
    ]

    def test_finds_blackhole_by_name_not_index_zero(self):
        assert resolve_audio_device(self.DEVICES) == 1

    def test_match_is_case_insensitive_and_substring(self):
        devices = [
            {"name": "Some Mic", "max_input_channels": 1},
            {"name": "blackhole 16ch", "max_input_channels": 16},
        ]
        assert resolve_audio_device(devices) == 1

    def test_raises_clear_error_when_blackhole_absent(self):
        devices = [
            {"name": "MacBook Air Microphone", "max_input_channels": 1},
            {"name": "MacBook Air Speakers", "max_input_channels": 0},
        ]
        with pytest.raises(RuntimeError, match="BlackHole"):
            resolve_audio_device(devices)


class TestPromptSettings:
    def test_collects_label_spoken_explanation_and_monitor(self):
        inputs = iter(["weekly-standup", "it", "en", "2"])

        def fake_input(prompt=""):
            return next(inputs)

        def fake_monitors():
            return [
                {"left": 0, "top": 0, "width": 2880, "height": 900},
                {"left": 0, "top": 0, "width": 1440, "height": 900},
                {"left": 1440, "top": 0, "width": 1440, "height": 900},
            ]

        settings = prompt_settings(input_fn=fake_input, list_monitors_fn=fake_monitors)

        assert settings["label"] == "weekly-standup"
        assert settings["spoken_language"] == "it"
        assert settings["explanation_language"] == "en"
        assert settings["monitor_index"] == 2

    def test_uses_defaults_for_languages_and_single_display(self):
        inputs = iter(["daily-sync", "", ""])

        def fake_input(prompt=""):
            return next(inputs)

        def fake_monitors():
            # Only the virtual combined + one physical display
            return [
                {"left": 0, "top": 0, "width": 1440, "height": 900},
                {"left": 0, "top": 0, "width": 1440, "height": 900},
            ]

        settings = prompt_settings(input_fn=fake_input, list_monitors_fn=fake_monitors)

        assert settings["label"] == "daily-sync"
        assert settings["spoken_language"] == "en"
        assert settings["explanation_language"] == "en"
        assert settings["monitor_index"] == 1

    def test_explanation_language_defaults_to_spoken_language(self):
        inputs = iter(["sync", "fr", ""])

        def fake_input(prompt=""):
            return next(inputs)

        def fake_monitors():
            return [
                {"left": 0, "top": 0, "width": 1440, "height": 900},
                {"left": 0, "top": 0, "width": 1440, "height": 900},
            ]

        settings = prompt_settings(input_fn=fake_input, list_monitors_fn=fake_monitors)

        assert settings["spoken_language"] == "fr"
        assert settings["explanation_language"] == "fr"


# ---------------------------------------------------------------------------
# Fakes for the orchestration tests
# ---------------------------------------------------------------------------

class FakeAssistant:
    def __init__(self, responses=None):
        self._responses = responses or {}
        self.explain_calls = []
        self.ask_calls = []

    def explain_slide(self, *, image_bytes: bytes, delta: str) -> str:
        self.explain_calls.append((image_bytes, delta))
        return self._responses.get("explain", "fake explanation")

    def ask_question(self, *, text: str, delta: str) -> str:
        self.ask_calls.append((text, delta))
        return self._responses.get("ask", "fake answer")


class FakeAudioThread:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeCapture:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeSession:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.input_loop_ran = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def run_input_loop(self):
        self.input_loop_ran = True


class TestHotkeyCallback:
    def test_explain_slide_called_and_turn_persisted(self, tmp_path):
        transcript = Transcript()
        transcript.append("new speech since last turn")
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        assistant = FakeAssistant()
        audio_thread = FakeAudioThread()
        capture = FakeCapture()
        outputs = []

        session = MeetingSession(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=audio_thread,
            capture=capture,
            input_fn=lambda _: "",
            output_fn=outputs.append,
        )
        session.start()
        session.on_hotkey(b"\x89PNG fake")
        session.stop()

        assert assistant.explain_calls == [(b"\x89PNG fake", "new speech since last turn")]
        assert any("fake explanation" in line for line in outputs)

        history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 1
        assert turns[0]["role"] == "assistant"
        assert turns[0]["content"] == "fake explanation"
        assert (tmp_path / "meetings" / "test-meeting" / "slides" / "slide_0001.png").exists()


class TestQuestionCallback:
    def test_ask_question_called_and_turns_persisted(self, tmp_path):
        transcript = Transcript()
        transcript.append("latest speech")
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        assistant = FakeAssistant()
        audio_thread = FakeAudioThread()
        capture = FakeCapture()
        outputs = []

        session = MeetingSession(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=audio_thread,
            capture=capture,
            input_fn=lambda _: "",
            output_fn=outputs.append,
        )
        session.start()
        session.on_question("What does this mean?")
        session.stop()

        assert assistant.ask_calls == [("What does this mean?", "latest speech")]
        assert any("fake answer" in line for line in outputs)

        history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "What does this mean?"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "fake answer"


class TestAIFailure:
    def test_hotkey_ai_failure_prints_inline_message_and_session_keeps_running(self, tmp_path):
        class FailingAssistant:
            def explain_slide(self, *, image_bytes: bytes, delta: str) -> str:
                raise RuntimeError("model unreachable")

            def ask_question(self, *, text: str, delta: str) -> str:
                return "ok"

        transcript = Transcript()
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        assistant = FailingAssistant()
        audio_thread = FakeAudioThread()
        capture = FakeCapture()
        outputs = []

        session = MeetingSession(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=audio_thread,
            capture=capture,
            input_fn=lambda _: "",
            output_fn=outputs.append,
        )
        session.start()
        session.on_hotkey(b"\x89PNG fake")
        # After the hotkey failure, a subsequent question should still work.
        session.on_question("Can you retry?")
        session.stop()

        assert any("model unreachable" in line for line in outputs)
        history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 2  # user question + successful answer
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "ok"


class TestShutdown:
    def test_input_loop_exits_on_eof_and_finalizes(self, tmp_path):
        inputs = iter(["What is this?", ""])

        def fake_input(prompt=""):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError()

        transcript = Transcript()
        transcript.append("latest speech")
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        assistant = FakeAssistant()
        audio_thread = FakeAudioThread()
        capture = FakeCapture()
        outputs = []

        session = MeetingSession(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=audio_thread,
            capture=capture,
            input_fn=fake_input,
            output_fn=outputs.append,
        )
        session.start()
        session.run_input_loop()
        session.stop()

        assert assistant.ask_calls == [("What is this?", "latest speech")]
        assert audio_thread.stopped is True
        assert capture.stopped is True
        history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 2

    def test_input_loop_exits_on_keyboard_interrupt(self, tmp_path):
        """Ctrl+C while blocked on input() raises KeyboardInterrupt; the loop
        must exit promptly instead of waiting for the user to press Enter."""
        def fake_input(prompt=""):
            raise KeyboardInterrupt()

        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        audio_thread = FakeAudioThread()
        capture = FakeCapture()

        session = MeetingSession(
            transcript=Transcript(),
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=audio_thread,
            capture=capture,
            input_fn=fake_input,
            output_fn=lambda _: None,
        )
        session.start()
        session.run_input_loop()  # must return, not propagate KeyboardInterrupt
        session.stop()

        assert audio_thread.stopped is True
        assert capture.stopped is True

    def test_stop_finalizes_even_with_no_turns(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        audio_thread = FakeAudioThread()
        capture = FakeCapture()
        transcript = Transcript()

        session = MeetingSession(
            transcript=transcript,
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=audio_thread,
            capture=capture,
            input_fn=lambda _: "",
            output_fn=lambda _: None,
        )
        session.start()
        session.stop()

        history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
        turns = json.loads(history_path.read_text())
        assert turns == []

    def test_stop_finalizes_even_if_audio_flush_fails(self, tmp_path):
        """A slow/failing final audio flush (or an impatient second Ctrl+C)
        must not prevent history.json from being finalized."""
        class ExplodingAudioThread(FakeAudioThread):
            def stop(self):
                self.stopped = True
                raise KeyboardInterrupt()

        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        audio_thread = ExplodingAudioThread()
        capture = FakeCapture()

        session = MeetingSession(
            transcript=Transcript(),
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=audio_thread,
            capture=capture,
            input_fn=lambda _: "",
            output_fn=lambda _: None,
        )
        session.start()
        session.on_question("recorded before shutdown")
        session.stop()  # must not raise

        history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 2  # the question turn was recorded and finalized


class TestMain:
    def test_main_prints_ready_when_config_loads(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("GEMINI_API_KEY=test-key\n")
        (tmp_path / "system_prompt.md").write_text("persona")

        fake_session = FakeSession()
        monkeypatch.setattr(
            "src.main.prompt_settings",
            lambda **kwargs: {
                "label": "test",
                "spoken_language": "en",
                "explanation_language": "en",
                "monitor_index": 1,
            },
        )
        monkeypatch.setattr("src.main._build_session", lambda *args, **kwargs: fake_session)

        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "ready" in captured.out.lower()
        assert "Control+Option+Space" in captured.out
        assert fake_session.started is True
        assert fake_session.input_loop_ran is True
        assert fake_session.stopped is True

    def test_main_exits_cleanly_with_error_when_api_key_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("# empty\n")
        (tmp_path / "system_prompt.md").write_text("persona")

        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Missing GEMINI_API_KEY" in captured.err
