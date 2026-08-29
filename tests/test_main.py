import json

from src.assistant import AssistantUnavailable
from src.main import main, prompt_settings, MeetingSession
from src.storage import Storage
from src.turn import Turn, TurnResult
from src.transcript import Transcript


def _make_session(*, transcript, storage, assistant, audio_thread, capture,
                  input_fn=lambda _: "", output_fn=lambda _: None, hud=None):
    """Wire a MeetingSession around a real Turn, mirroring _build_session."""
    turn = Turn(transcript=transcript, archive=storage, assistant=assistant)
    return MeetingSession(
        turn=turn,
        storage=storage,
        audio_thread=audio_thread,
        capture=capture,
        input_fn=input_fn,
        output_fn=output_fn,
        hud=hud,
    )


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


class FakeTurn:
    def __init__(self, result=TurnResult(text="fake answer", ok=True)):
        self._result = result
        self.explain_calls = []
        self.ask_calls = []

    def explain(self, slide):
        self.explain_calls.append(slide)
        return self._result

    def ask(self, question):
        self.ask_calls.append(question)
        return self._result


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


class FakeHud:
    """Records turn lifecycle calls and stop() for MeetingSession tests."""

    def __init__(self, **kwargs):
        self.turn_started_calls: list[None] = []
        self.turn_finished_calls: list[bool] = []
        self.stopped = False
        self.window_id = None

    def run(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def turn_started(self) -> None:
        self.turn_started_calls.append(None)

    def turn_finished(self, ok: bool) -> None:
        self.turn_finished_calls.append(ok)


class FailingAssistant:
    """Assistant that gives up on explain_slide, like the real assistant module."""

    def explain_slide(self, *, image_bytes: bytes, delta: str) -> str:
        raise AssistantUnavailable("[Could not get a response from the assistant: model unreachable]")

    def ask_question(self, *, text: str, delta: str) -> str:
        return "ok"


class TestTurnScheduling:
    def test_question_uses_turn_result_for_terminal_and_hud(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        turn = FakeTurn(TurnResult(text="deep answer", ok=True))
        hud = FakeHud()
        outputs = []
        session = MeetingSession(
            turn=turn,
            storage=storage,
            audio_thread=FakeAudioThread(),
            capture=FakeCapture(),
            output_fn=outputs.append,
            hud=hud,
        )

        session.start()
        session.on_question("What changed?")
        session.stop()

        assert turn.ask_calls == ["What changed?"]
        assert any("deep answer" in line for line in outputs)
        assert hud.turn_started_calls == [None]
        assert hud.turn_finished_calls == [True]


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

        session = _make_session(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=audio_thread,
            capture=capture,
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

        session = _make_session(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=audio_thread,
            capture=capture,
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

        session = _make_session(
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

        session = _make_session(
            transcript=Transcript(),
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=audio_thread,
            capture=capture,
            input_fn=fake_input,
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

        session = _make_session(
            transcript=transcript,
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=audio_thread,
            capture=capture,
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

        session = _make_session(
            transcript=Transcript(),
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=audio_thread,
            capture=capture,
        )
        session.start()
        session.on_question("recorded before shutdown")
        session.stop()  # must not raise

        history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 2  # the question turn was recorded and finalized


class TestHudLifecycle:
    """MeetingSession signals turn start/finish to the injected HUD."""

    def test_hotkey_turn_signals_hud(self, tmp_path):
        transcript = Transcript()
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        assistant = FakeAssistant()
        hud = FakeHud()

        session = _make_session(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=FakeAudioThread(),
            capture=FakeCapture(),
            hud=hud,
        )
        session.start()
        session.on_hotkey(b"\x89PNG fake")
        session.stop()

        assert hud.turn_started_calls == [None]
        assert hud.turn_finished_calls == [True]

    def test_question_turn_signals_hud(self, tmp_path):
        transcript = Transcript()
        transcript.append("latest speech")
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        assistant = FakeAssistant()
        hud = FakeHud()

        session = _make_session(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=FakeAudioThread(),
            capture=FakeCapture(),
            hud=hud,
        )
        session.start()
        session.on_question("What does this mean?")
        session.stop()

        assert hud.turn_started_calls == [None]
        assert hud.turn_finished_calls == [True]

    def test_failed_turn_signals_finished_false(self, tmp_path):
        transcript = Transcript()
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        assistant = FailingAssistant()
        hud = FakeHud()

        session = _make_session(
            transcript=transcript,
            storage=storage,
            assistant=assistant,
            audio_thread=FakeAudioThread(),
            capture=FakeCapture(),
            hud=hud,
        )
        session.start()
        session.on_hotkey(b"\x89PNG fake")
        # After the hotkey failure, a subsequent question should still work.
        session.on_question("Can you retry?")
        session.stop()

        assert hud.turn_started_calls == [None, None]
        assert hud.turn_finished_calls == [False, True]

    def test_input_loop_eof_stops_hud(self, tmp_path):
        def fake_input(prompt=""):
            raise EOFError()

        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        hud = FakeHud()

        session = _make_session(
            transcript=Transcript(),
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=FakeAudioThread(),
            capture=FakeCapture(),
            input_fn=fake_input,
            hud=hud,
        )
        session.run_input_loop()

        assert hud.stopped is True
        assert session._shutdown.is_set()

    def test_input_loop_keyboard_interrupt_stops_hud(self, tmp_path):
        def fake_input(prompt=""):
            raise KeyboardInterrupt()

        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("test-meeting", ["en"])
        hud = FakeHud()

        session = _make_session(
            transcript=Transcript(),
            storage=storage,
            assistant=FakeAssistant(),
            audio_thread=FakeAudioThread(),
            capture=FakeCapture(),
            input_fn=fake_input,
            hud=hud,
        )
        session.run_input_loop()

        assert hud.stopped is True
        assert session._shutdown.is_set()


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
        monkeypatch.setattr("src.main.HudPanel", FakeHud)

        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "ready" in captured.out.lower()
        assert "Control+Option+Space" in captured.out
        assert "Control+Option+S" in captured.out
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
