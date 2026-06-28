"""conversation module: the turn — HelpMeeting's central operation.

A turn bundles the steps that must happen together, in order, every time the
user asks for help: take the transcript delta, (for an explain turn) save the
slide, call the assistant, and record the turn(s).  Keeping them in one module
gives the ordering a single home and one test surface.

Public interface
----------------
conversation = Conversation(transcript=transcript, storage=storage, assistant=assistant)
explanation = conversation.explain(slide_bytes)   # explain turn: slide + delta
answer      = conversation.ask(question)           # question turn: text + delta

Both return the assistant's text.  Both advance the transcript delta pointer via
``transcript.take_delta()``.  Assistant failures propagate to the caller, which
decides how to surface them.
"""


class Conversation:
    """One meeting's turn logic, over a shared transcript, storage, and assistant."""

    def __init__(self, *, transcript, storage, assistant):
        self._transcript = transcript
        self._storage = storage
        self._assistant = assistant

    def explain(self, slide_bytes: bytes) -> str:
        """Explain turn: current slide image + transcript delta since the last turn."""
        delta = self._transcript.take_delta()
        slide_path = self._storage.save_slide(slide_bytes)
        explanation = self._assistant.explain_slide(image_bytes=slide_bytes, delta=delta)
        self._storage.record_turn(
            role="assistant",
            content=explanation,
            slide_path=slide_path,
        )
        return explanation

    def ask(self, question: str) -> str:
        """Question turn: typed question + transcript delta, no new slide."""
        delta = self._transcript.take_delta()
        self._storage.record_turn(role="user", content=question)
        answer = self._assistant.ask_question(text=question, delta=delta)
        self._storage.record_turn(role="assistant", content=answer)
        return answer
