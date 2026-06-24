"""main orchestration: startup prompts, threads, question loop, finalize.

Public interface
----------------
settings = prompt_settings()
session = MeetingSession(settings, config, ...)
session.start()
session.run_input_loop()   # runs until Ctrl+C or EOF
session.stop()
"""

import os
import sys
import threading
import queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load, MissingApiKeyError


def _list_monitors():
    import mss

    with mss.MSS() as sct:
        return sct.monitors


def _choose_monitor(monitors, input_fn, output_fn):
    """Return the mss monitor index to capture (1 = first physical display)."""
    if len(monitors) <= 2:
        return 1
    output_fn(f"Found {len(monitors) - 1} displays:")
    for i in range(1, len(monitors)):
        m = monitors[i]
        output_fn(f"  {i}: {m['width']}x{m['height']} at ({m['left']}, {m['top']})")
    choice = input_fn("Choose display for screenshots [1]: ").strip() or "1"
    try:
        idx = int(choice)
        if 1 <= idx < len(monitors):
            return idx
    except ValueError:
        pass
    return 1


def prompt_settings(input_fn=input, output_fn=print, list_monitors_fn=_list_monitors):
    """Prompt the user for the startup settings and return them as a dict."""
    label = input_fn("Meeting label: ").strip()
    if not label:
        label = "meeting"

    spoken_language = input_fn("Spoken language [en]: ").strip() or "en"
    explanation_default = spoken_language
    explanation_language = (
        input_fn(f"Explanation language [{explanation_default}]: ").strip() or explanation_default
    )

    monitor_index = _choose_monitor(list_monitors_fn(), input_fn, output_fn)

    return {
        "label": label,
        "spoken_language": spoken_language,
        "explanation_language": explanation_language,
        "monitor_index": monitor_index,
    }


class MeetingSession:
    """Wires the audio, capture, assistant, and storage modules into the live loop.

    The capture object must be pre-configured with its own callback pointing at
    ``on_hotkey``.  This class owns the threads: audio transcription, capture
    listener, and the assistant worker that processes hotkey / question turns.
    """

    def __init__(
        self,
        *,
        transcript,
        storage,
        assistant,
        audio_thread,
        capture,
        input_fn=input,
        output_fn=print,
    ):
        self._transcript = transcript
        self._storage = storage
        self._assistant = assistant
        self._audio_thread = audio_thread
        self._capture = capture
        self._input_fn = input_fn
        self._output_fn = output_fn
        self._print_lock = threading.Lock()
        self._task_queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="AssistantWorker"
        )
        self._capture_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        """Start the audio, capture, and assistant-worker threads."""
        self._audio_thread.start()
        self._capture_thread = threading.Thread(
            target=self._capture.start, daemon=True, name="CaptureListener"
        )
        self._capture_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        """Signal shutdown, finish queued turns, stop threads, and finalize storage."""
        self._shutdown.set()
        self._capture.stop()
        self._task_queue.put(None)
        self._worker_thread.join()
        self._audio_thread.stop()
        self._storage.finalize()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)

    def run_input_loop(self) -> None:
        """Read questions from the terminal until shutdown or EOF."""
        self._safe_print("Type a question and press Enter, or press Ctrl+C to stop.")
        while not self._shutdown.is_set():
            try:
                question = self._input_fn("Question: ")
            except EOFError:
                break
            if self._shutdown.is_set():
                break
            if question.strip():
                self.on_question(question)

    def on_hotkey(self, png_bytes: bytes) -> None:
        """Queue a slide-explanation turn."""
        self._task_queue.put(("explain", png_bytes))

    def on_question(self, text: str) -> None:
        """Queue a question-and-answer turn."""
        self._task_queue.put(("question", text))

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            item = self._task_queue.get()
            if item is None:
                break
            task_type, payload = item
            try:
                if task_type == "explain":
                    self._do_explain(payload)
                elif task_type == "question":
                    self._do_question(payload)
            except Exception as exc:  # noqa: BLE001
                self._safe_print(f"[Error processing turn: {exc}]")

    def _do_explain(self, png_bytes: bytes) -> None:
        delta = self._transcript.take_delta()
        slide_path = self._storage.save_slide(png_bytes)
        explanation = self._assistant.explain_slide(image_bytes=png_bytes, delta=delta)
        self._safe_print(f"\n[Explanation]\n{explanation}\n")
        self._storage.record_turn(
            role="assistant",
            content=explanation,
            explanation="",
            slide_path=slide_path,
        )

    def _do_question(self, question: str) -> None:
        delta = self._transcript.take_delta()
        self._storage.record_turn(role="user", content=question, explanation="")
        answer = self._assistant.ask_question(text=question, delta=delta)
        self._safe_print(f"\n[Answer]\n{answer}\n")
        self._storage.record_turn(role="assistant", content=answer, explanation="")

    def _safe_print(self, message: str) -> None:
        with self._print_lock:
            self._output_fn(message)


def _build_session(config, settings, *, input_fn=input, output_fn=print):
    """Wire the real dependencies into a MeetingSession."""
    from google import genai

    from src.audio import AudioThread
    from src.assistant import Assistant
    from src.capture import Capture
    from src.storage import Storage
    from src.transcript import Transcript

    transcript = Transcript()
    storage = Storage()
    storage.start_meeting(settings["label"], [settings["spoken_language"], settings["explanation_language"]])

    client = genai.Client(api_key=config.api_key)
    assistant = Assistant(
        client=client,
        model=config.gemini_model_name,
        transcript=transcript,
        system_prompt=config.system_prompt,
    )

    audio_thread = AudioThread(
        transcript=transcript,
        storage=storage,
        chunk_seconds=config.audio_chunk_seconds,
        language=settings["spoken_language"],
        device=0,
    )

    session = MeetingSession(
        transcript=transcript,
        storage=storage,
        assistant=assistant,
        audio_thread=audio_thread,
        capture=None,  # set below after wiring the hotkey callback
        input_fn=input_fn,
        output_fn=output_fn,
    )
    capture = Capture(callback=session.on_hotkey, monitor_index=settings["monitor_index"])
    session._capture = capture
    return session


def main():
    import signal

    try:
        config = load()
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    settings = prompt_settings()
    session = _build_session(config, settings)

    print(f"HelpMeeting ready: {settings['label']}")
    print(f"  spoken: {settings['spoken_language']}, explanation: {settings['explanation_language']}")
    print(f"  display: {settings['monitor_index']}")
    print(f"Press {config.hotkey} to request an explanation, Ctrl+C to stop.")

    def _handle_sigint(signum, frame):
        session._shutdown.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    session.start()
    try:
        session.run_input_loop()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
