"""Turn module: resolve Explain and Question turns behind one interface.

The implementation consumes the Delta, saves the Slide when present, calls the
assistant, and records exactly one assistant archive entry when the archive is
writable. Runtime failures return ``TurnResult(ok=False)`` rather than crossing
the Turn seam.
"""

from dataclasses import dataclass

from src.assistant import AssistantUnavailable


@dataclass(frozen=True)
class TurnResult:
    text: str
    ok: bool


class Turn:
    def __init__(self, *, transcript, archive, assistant):
        self._transcript = transcript
        self._archive = archive
        self._assistant = assistant

    def explain(self, slide: bytes) -> TurnResult:
        try:
            delta = self._transcript.take_delta()
            slide_path = self._archive.save_slide(slide)
            try:
                text = self._assistant.explain_slide(image_bytes=slide, delta=delta)
            except AssistantUnavailable as exc:
                text = str(exc)
                self._archive.record_turn(role="assistant", content=text, slide_path=slide_path)
                return TurnResult(text=text, ok=False)
            self._archive.record_turn(role="assistant", content=text, slide_path=slide_path)
            return TurnResult(text=text, ok=True)
        except Exception as exc:
            return self._failed(exc)

    def ask(self, question: str) -> TurnResult:
        try:
            delta = self._transcript.take_delta()
            self._archive.record_turn(role="user", content=question)
            try:
                text = self._assistant.ask_question(text=question, delta=delta)
            except AssistantUnavailable as exc:
                text = str(exc)
                self._archive.record_turn(role="assistant", content=text)
                return TurnResult(text=text, ok=False)
            self._archive.record_turn(role="assistant", content=text)
            return TurnResult(text=text, ok=True)
        except Exception as exc:
            return self._failed(exc)

    def _failed(self, exc: Exception) -> TurnResult:
        try:
            self._archive.record_turn(role="assistant", content=f"[Turn failed: {exc}]")
        except Exception:
            pass
        return TurnResult(text=f"[Error processing turn: {exc}]", ok=False)
