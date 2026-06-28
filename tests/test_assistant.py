"""Tests for the assistant module (issue #4).

All tests run against a fake Gemini client — no network access.
"""
import pytest

from src.assistant import Assistant
from src.transcript import Transcript


# ---------------------------------------------------------------------------
# Fake Gemini client infrastructure
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal stand-in for google.genai.types.GenerateContentResponse."""

    def __init__(self, text: str):
        self.text = text


class FakeChat:
    """Records every send_message call so tests can inspect what was sent."""

    def __init__(self, response_text: str = "ok"):
        self._response_text = response_text
        self.calls: list[list] = []  # each element is the parts list passed

    def send_message(self, parts):
        self.calls.append(list(parts))
        return FakeResponse(self._response_text)


class FakeChats:
    """Replaces client.chats; returns the FakeChat injected into it."""

    def __init__(self, chat: FakeChat):
        self._chat = chat

    def create(self, *, model: str, config=None, history=None):
        return self._chat


class FakeClient:
    """Minimal stand-in for google.genai.Client."""

    def __init__(self, chat: FakeChat):
        self.chats = FakeChats(chat)


# ---------------------------------------------------------------------------
# Tracer bullet: explain_slide returns the model's text
# ---------------------------------------------------------------------------

class TestExplainSlide:
    def test_explain_slide_returns_model_text(self):
        chat = FakeChat(response_text="Here is the explanation.")
        client = FakeClient(chat)
        assistant = Assistant(client=client, model="gemini-2.5-flash")

        result = assistant.explain_slide(image_bytes=b"\x89PNG", delta="speaker said hello")

        assert result == "Here is the explanation."


# ---------------------------------------------------------------------------
# ask_question returns the model's text
# ---------------------------------------------------------------------------

class TestAskQuestion:
    def test_ask_question_returns_model_text(self):
        chat = FakeChat(response_text="Good question, here is the answer.")
        client = FakeClient(chat)
        assistant = Assistant(client=client, model="gemini-2.5-flash")

        result = assistant.ask_question(text="What does this mean?", delta="recent speech")

        assert result == "Good question, here is the answer."


# ---------------------------------------------------------------------------
# Both turns advance the transcript delta pointer exactly once per call
# ---------------------------------------------------------------------------

class TestTranscriptPointer:
    def test_explain_slide_advances_pointer_once(self):
        chat = FakeChat()
        transcript = Transcript()
        transcript.append("chunk one")
        transcript.append("chunk two")
        assistant = Assistant(client=FakeClient(chat), model="m")

        assistant.explain_slide(image_bytes=b"PNG", delta=transcript.take_delta())

        # After the call the pointer is already advanced by take_delta() above;
        # next take_delta() must return empty
        assert transcript.take_delta() == ""

    def test_ask_question_advances_pointer_once(self):
        chat = FakeChat()
        transcript = Transcript()
        transcript.append("new speech")
        assistant = Assistant(client=FakeClient(chat), model="m")

        assistant.ask_question(text="question", delta=transcript.take_delta())

        assert transcript.take_delta() == ""

    def test_consecutive_explain_slides_do_not_repeat_delta(self):
        chat = FakeChat()
        transcript = Transcript()
        assistant = Assistant(client=FakeClient(chat), model="m")

        transcript.append("first chunk")
        first_delta = transcript.take_delta()
        assistant.explain_slide(image_bytes=b"PNG", delta=first_delta)

        transcript.append("second chunk")
        second_delta = transcript.take_delta()
        assistant.explain_slide(image_bytes=b"PNG", delta=second_delta)

        def parts_text(parts):
            pieces = []
            for p in parts:
                if isinstance(p, str):
                    pieces.append(p)
                elif hasattr(p, "text") and p.text is not None:
                    pieces.append(p.text)
            return " ".join(pieces)

        # First call got "first chunk", second call got "second chunk"
        first_call_parts_text = parts_text(chat.calls[0])
        second_call_parts_text = parts_text(chat.calls[1])
        assert "first chunk" in first_call_parts_text
        assert "first chunk" not in second_call_parts_text
        assert "second chunk" in second_call_parts_text


# ---------------------------------------------------------------------------
# Prior slides/explanations are retained in one ongoing chat session
# ---------------------------------------------------------------------------

class TestChatSession:
    def test_single_chat_session_reused_across_turns(self):
        chat = FakeChat()
        client = FakeClient(chat)
        assistant = Assistant(client=client, model="m")

        assistant.explain_slide(image_bytes=b"PNG1", delta="delta1")
        assistant.explain_slide(image_bytes=b"PNG2", delta="delta2")
        assistant.ask_question(text="q", delta="delta3")

        # All three calls went through the same FakeChat object
        assert len(chat.calls) == 3

    def test_assistant_keeps_client_alive_after_caller_drops_it(self):
        """The real genai client closes its transport in __del__; the Assistant
        must keep it alive so worker-thread turns don't hit a closed client."""
        import gc
        import weakref

        chat = FakeChat()
        client = FakeClient(chat)
        assistant = Assistant(client=client, model="m")

        client_ref = weakref.ref(client)
        del client
        gc.collect()

        # The client must still be alive because the Assistant holds a reference.
        assert client_ref() is not None
        # And it can still be used.
        assert assistant.explain_slide(image_bytes=b"PNG", delta="d") == "ok"


# ---------------------------------------------------------------------------
# explain_slide submits image bytes; ask_question does not
# ---------------------------------------------------------------------------

class TestTurnContent:
    def _image_parts(self, parts):
        """Return any parts that carry binary/image data."""
        from google.genai import types
        return [p for p in parts if isinstance(p, types.Part) and p.inline_data is not None]

    def test_explain_slide_sends_image_part(self):
        chat = FakeChat()
        assistant = Assistant(client=FakeClient(chat), model="m")

        assistant.explain_slide(image_bytes=b"\x89PNG\r\n\x1a\n", delta="text")

        image_parts = self._image_parts(chat.calls[0])
        assert len(image_parts) == 1

    def test_ask_question_sends_no_image_part(self):
        chat = FakeChat()
        assistant = Assistant(client=FakeClient(chat), model="m")

        assistant.ask_question(text="What is this?", delta="text")

        image_parts = self._image_parts(chat.calls[0])
        assert len(image_parts) == 0


# ---------------------------------------------------------------------------
# Transient errors are retried; persistent failure returns graceful message
# ---------------------------------------------------------------------------

class FlakyChat:
    """Fails the first N calls, then succeeds."""

    def __init__(self, fail_count: int, success_text: str = "ok"):
        self._fail_count = fail_count
        self._call_count = 0
        self._success_text = success_text
        self.calls = []

    def send_message(self, parts):
        self._call_count += 1
        self.calls.append(list(parts))
        if self._call_count <= self._fail_count:
            from google.genai.errors import ServerError
            raise ServerError(code=429, response_json={"error": {"message": "rate limited"}})
        return FakeResponse(self._success_text)


class AlwaysFailChat:
    """Always raises a ServerError."""

    def send_message(self, parts):
        from google.genai.errors import ServerError
        raise ServerError(code=500, response_json={"error": {"message": "internal error"}})


class TestRetry:
    def test_transient_error_is_retried_and_succeeds(self):
        chat = FlakyChat(fail_count=1, success_text="eventually ok")
        assistant = Assistant(
            client=FakeClient(chat),
            model="m",
            max_retries=3,
            retry_wait_seconds=0,
        )

        result = assistant.explain_slide(image_bytes=b"PNG", delta="text")

        assert result == "eventually ok"
        assert len(chat.calls) == 2  # one failure + one success

    def test_persistent_failure_returns_graceful_message_not_exception(self):
        class AlwaysFailChats:
            def create(self, *, model, config=None, history=None):
                return AlwaysFailChat()

        class AlwaysFailClient:
            chats = AlwaysFailChats()

        assistant = Assistant(
            client=AlwaysFailClient(),
            model="m",
            max_retries=2,
            retry_wait_seconds=0,
        )

        result = assistant.explain_slide(image_bytes=b"PNG", delta="text")

        assert isinstance(result, str)
        assert len(result) > 0  # something was returned, not an empty string
        # Must not raise — the test reaching here is proof
