"""assistant module: wraps a single long-lived Gemini chat session per meeting.

Two turn types share the same chat session:
- explain_slide(image_bytes, delta) -> str
- ask_question(text, delta)         -> str

Both accept a pre-consumed transcript delta (caller calls transcript.take_delta()
before passing it in).  Failure is handled with retry + backoff; persistent
failure returns a graceful inline message and never raises.
"""
import time

from google.genai import types
from google.genai.errors import APIError


_EXPLAIN_INSTRUCTION = (
    "Explain only what is new in this slide and the speech below. "
    "Build on what you already told me — do not repeat earlier explanations."
)
_QUESTION_INSTRUCTION = (
    "Answer the question below using the full context of this meeting. "
    "Here is the latest speech for additional context."
)


class Assistant:
    """Single meeting assistant backed by one Gemini chat session."""

    def __init__(
        self,
        *,
        client,
        model: str,
        transcript,
        max_retries: int = 3,
        retry_wait_seconds: float = 2.0,
    ):
        self._model = model
        self._transcript = transcript
        self._max_retries = max_retries
        self._retry_wait = retry_wait_seconds
        self._chat = client.chats.create(model=model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_slide(self, *, image_bytes: bytes, delta: str) -> str:
        """Hotkey turn: submit current slide image + transcript delta."""
        parts = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            types.Part.from_text(text=f"{_EXPLAIN_INSTRUCTION}\n\nTranscript delta:\n{delta}"),
        ]
        return self._send(parts)

    def ask_question(self, *, text: str, delta: str) -> str:
        """Question turn: submit typed question + transcript delta, no image."""
        parts = [
            types.Part.from_text(
                text=f"{_QUESTION_INSTRUCTION}\n\nQuestion: {text}\n\nTranscript delta:\n{delta}"
            ),
        ]
        return self._send(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, parts: list) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._chat.send_message(parts)
                return response.text
            except APIError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_wait)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_wait)
        return self._graceful_message(last_exc)

    @staticmethod
    def _graceful_message(exc: Exception | None) -> str:
        msg = str(exc) if exc else "unknown error"
        if "429" in msg or "rate" in msg.lower():
            return "[Rate limited — please wait a moment and try again.]"
        return f"[Could not get a response from the assistant: {msg}]"
