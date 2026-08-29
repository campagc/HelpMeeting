import json

from src.assistant import AssistantUnavailable
from src.storage import Storage
from src.transcript import Transcript
from src.turn import Turn, TurnResult


class FakeAssistant:
    def __init__(self):
        self.explain_calls = []
        self.ask_calls = []

    def explain_slide(self, *, image_bytes: bytes, delta: str) -> str:
        self.explain_calls.append((image_bytes, delta))
        return "fake explanation"

    def ask_question(self, *, text: str, delta: str) -> str:
        self.ask_calls.append((text, delta))
        return "fake answer"


class FlakyStorage(Storage):
    def __init__(self, *args, fail_count: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._fail_count = fail_count

    def _flush_history(self) -> None:
        if self._fail_count > 0:
            self._fail_count -= 1
            raise OSError("disk full")
        super()._flush_history()


class BrokenStorage(Storage):
    def record_turn(self, role, content, slide_path=None):
        raise OSError("disk full")


def test_explain_turn_returns_answer_and_records_slide(tmp_path):
    transcript = Transcript()
    transcript.append("new speech since last turn")
    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    assistant = FakeAssistant()
    turn = Turn(transcript=transcript, archive=storage, assistant=assistant)

    result = turn.explain(b"\x89PNG fake")

    assert result == TurnResult(text="fake explanation", ok=True)
    assert assistant.explain_calls == [(b"\x89PNG fake", "new speech since last turn")]
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    assert json.loads(history_path.read_text()) == [
        {
            "role": "assistant",
            "content": "fake explanation",
            "slide_path": str(
                tmp_path / "meetings" / "test-meeting" / "slides" / "slide_0001.png"
            ),
        }
    ]


def test_question_turn_returns_answer_and_records_attendee_text(tmp_path):
    transcript = Transcript()
    transcript.append("latest speech")
    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    assistant = FakeAssistant()
    turn = Turn(transcript=transcript, archive=storage, assistant=assistant)

    result = turn.ask("What does this mean?")

    assert result == TurnResult(text="fake answer", ok=True)
    assert assistant.ask_calls == [("What does this mean?", "latest speech")]
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    assert json.loads(history_path.read_text()) == [
        {"role": "user", "content": "What does this mean?"},
        {"role": "assistant", "content": "fake answer"},
    ]


def test_unavailable_assistant_returns_failure_and_records_one_entry(tmp_path):
    class UnavailableAssistant(FakeAssistant):
        def explain_slide(self, *, image_bytes: bytes, delta: str) -> str:
            raise AssistantUnavailable("[Rate limited — please wait.]")

    transcript = Transcript()
    transcript.append("assigned delta")
    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    turn = Turn(transcript=transcript, archive=storage, assistant=UnavailableAssistant())

    result = turn.explain(b"\x89PNG fake")

    assert result == TurnResult(text="[Rate limited — please wait.]", ok=False)
    assert transcript.take_delta() == ""
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    entries = json.loads(history_path.read_text())
    assert len(entries) == 1
    assert entries[0]["role"] == "assistant"
    assert entries[0]["content"] == "[Rate limited — please wait.]"
    assert entries[0]["slide_path"].endswith("slide_0001.png")


def test_pipeline_failure_returns_failure_and_records_one_note(tmp_path):
    class ExplodingAssistant(FakeAssistant):
        def ask_question(self, *, text: str, delta: str) -> str:
            raise RuntimeError("network broken")

    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    turn = Turn(transcript=Transcript(), archive=storage, assistant=ExplodingAssistant())

    result = turn.ask("Can you explain?")

    assert result == TurnResult(text="[Error processing turn: network broken]", ok=False)
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    assert json.loads(history_path.read_text()) == [
        {"role": "user", "content": "Can you explain?"},
        {"role": "assistant", "content": "[Turn failed: network broken]"},
    ]


def test_failed_archive_write_retries_once_without_duplicate_entry(tmp_path):
    class UnavailableAssistant(FakeAssistant):
        def explain_slide(self, *, image_bytes: bytes, delta: str) -> str:
            raise AssistantUnavailable("[Could not get a response]")

    storage = FlakyStorage(base_dir=tmp_path, fail_count=1)
    storage.start_meeting("test-meeting", ["en"])
    turn = Turn(transcript=Transcript(), archive=storage, assistant=UnavailableAssistant())

    result = turn.explain(b"\x89PNG fake")

    assert result == TurnResult(text="[Error processing turn: disk full]", ok=False)
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    assert json.loads(history_path.read_text()) == [
        {"role": "assistant", "content": "[Turn failed: disk full]"}
    ]


def test_persistent_archive_failure_returns_without_raising(tmp_path):
    storage = BrokenStorage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    turn = Turn(transcript=Transcript(), archive=storage, assistant=FakeAssistant())

    result = turn.ask("Will this be recorded?")

    assert result == TurnResult(text="[Error processing turn: disk full]", ok=False)


def test_question_assistant_failure_records_attendee_then_one_failure(tmp_path):
    class UnavailableAssistant(FakeAssistant):
        def ask_question(self, *, text: str, delta: str) -> str:
            raise AssistantUnavailable("[Could not get a response]")

    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    turn = Turn(transcript=Transcript(), archive=storage, assistant=UnavailableAssistant())

    result = turn.ask("What does this mean?")

    assert result == TurnResult(text="[Could not get a response]", ok=False)
    history_path = tmp_path / "meetings" / "test-meeting" / "history.json"
    assert json.loads(history_path.read_text()) == [
        {"role": "user", "content": "What does this mean?"},
        {"role": "assistant", "content": "[Could not get a response]"},
    ]


def test_consecutive_turns_consume_distinct_deltas(tmp_path):
    transcript = Transcript()
    transcript.append("first")
    storage = Storage(base_dir=tmp_path)
    storage.start_meeting("test-meeting", ["en"])
    assistant = FakeAssistant()
    turn = Turn(transcript=transcript, archive=storage, assistant=assistant)

    turn.explain(b"first slide")
    turn.explain(b"second slide")

    assert assistant.explain_calls == [
        (b"first slide", "first"),
        (b"second slide", ""),
    ]
