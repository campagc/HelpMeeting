"""Tests for the conversation module — the turn, exercised through its own
interface (no threads, no queue)."""

import json

from src.conversation import Conversation
from src.storage import Storage
from src.transcript import Transcript


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


def _conversation(tmp_path, assistant=None, transcript=None):
    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    transcript = transcript if transcript is not None else Transcript()
    assistant = assistant if assistant is not None else FakeAssistant()
    conv = Conversation(transcript=transcript, storage=storage, assistant=assistant)
    return conv, storage, transcript, assistant


def _history(tmp_path):
    path = tmp_path / "meetings" / "test-meeting" / "history.json"
    return json.loads(path.read_text())


class TestExplain:
    def test_sends_slide_with_delta_saves_slide_and_records_turn(self, tmp_path):
        transcript = Transcript()
        transcript.append("new speech since last turn")
        conv, _storage, _t, assistant = _conversation(tmp_path, transcript=transcript)

        result = conv.explain(b"\x89PNG fake")

        assert result == "fake explanation"
        assert assistant.explain_calls == [(b"\x89PNG fake", "new speech since last turn")]

        turns = _history(tmp_path)
        assert len(turns) == 1
        assert turns[0]["role"] == "assistant"
        assert turns[0]["content"] == "fake explanation"
        assert (tmp_path / "meetings" / "test-meeting" / "slides" / "slide_0001.png").exists()

    def test_takes_delta_before_assistant_so_pointer_advances(self, tmp_path):
        transcript = Transcript()
        transcript.append("first")
        conv, _storage, _t, assistant = _conversation(tmp_path, transcript=transcript)

        conv.explain(b"img")
        # Nothing new appended; a second turn must see an empty delta.
        conv.explain(b"img")

        assert assistant.explain_calls[0][1] == "first"
        assert assistant.explain_calls[1][1] == ""


class TestAsk:
    def test_records_user_then_assistant_turn_with_delta(self, tmp_path):
        transcript = Transcript()
        transcript.append("latest speech")
        conv, _storage, _t, assistant = _conversation(tmp_path, transcript=transcript)

        result = conv.ask("What does this mean?")

        assert result == "fake answer"
        assert assistant.ask_calls == [("What does this mean?", "latest speech")]

        turns = _history(tmp_path)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "What does this mean?"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "fake answer"


class TestFailurePropagates:
    def test_assistant_error_propagates_to_caller(self, tmp_path):
        class FailingAssistant:
            def explain_slide(self, *, image_bytes, delta):
                raise RuntimeError("model unreachable")

            def ask_question(self, *, text, delta):
                raise RuntimeError("model unreachable")

        conv, _storage, _t, _a = _conversation(tmp_path, assistant=FailingAssistant())

        import pytest

        with pytest.raises(RuntimeError, match="model unreachable"):
            conv.explain(b"img")
