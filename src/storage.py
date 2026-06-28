"""storage — per-meeting folder and all persisted artifacts.

Public interface
----------------
storage = Storage(base_dir=Path("."))
storage.start_meeting(label, langs)
path   = storage.save_slide(image_bytes)
storage.append_transcript(text)
storage.record_turn(role, content, slide_path=None)
storage.finalize()
"""

import json
from pathlib import Path


class Storage:
    def __init__(self, base_dir=None):
        self._base_dir = Path(base_dir) if base_dir is not None else Path(".")
        self._meeting_dir: Path | None = None
        self._slide_counter = 0
        self._turns: list[dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def meeting_dir(self) -> Path | None:
        """The current meeting folder, or None before start_meeting()."""
        return self._meeting_dir

    def start_meeting(self, label: str, langs: list[str]) -> None:
        """Create meetings/<label>/ and write skeleton files."""
        self._meeting_dir = self._base_dir / "meetings" / label
        self._meeting_dir.mkdir(parents=True, exist_ok=True)
        (self._meeting_dir / "slides").mkdir(exist_ok=True)

        # transcript.txt — empty to start
        (self._meeting_dir / "transcript.txt").write_text("", encoding="utf-8")

        # history.json — empty list
        (self._meeting_dir / "history.json").write_text("[]", encoding="utf-8")

        # session.md — header
        (self._meeting_dir / "session.md").write_text(
            f"# Meeting: {label}\n\n", encoding="utf-8"
        )

        self._slide_counter = 0
        self._turns = []

    def save_slide(self, image: bytes) -> str:
        """Write a PNG to slides/ and return its absolute path string."""
        self._assert_started()
        self._slide_counter += 1
        slide_path = self._meeting_dir / "slides" / f"slide_{self._slide_counter:04d}.png"
        slide_path.write_bytes(image)
        return str(slide_path)

    def append_transcript(self, text: str) -> None:
        """Append text to transcript.txt."""
        self._assert_started()
        with (self._meeting_dir / "transcript.txt").open("a", encoding="utf-8") as f:
            f.write(text)

    def record_turn(
        self,
        role: str,
        content: str,
        slide_path: str | None = None,
    ) -> None:
        """Append a turn to history.json and session.md."""
        self._assert_started()

        turn: dict = {"role": role, "content": content}
        if slide_path is not None:
            turn["slide_path"] = slide_path

        self._turns.append(turn)
        self._flush_history()
        self._append_session_md(turn)

    def finalize(self) -> None:
        """Flush all state cleanly, leaving no half-written files."""
        if self._meeting_dir is None:
            return
        self._flush_history()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_started(self) -> None:
        if self._meeting_dir is None:
            raise RuntimeError("Call start_meeting() before using Storage.")

    def _flush_history(self) -> None:
        history_path = self._meeting_dir / "history.json"
        # Write atomically via a temp file to avoid half-written JSON
        tmp_path = history_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self._turns, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(history_path)

    def _append_session_md(self, turn: dict) -> None:
        role = turn["role"].capitalize()
        content = turn["content"]
        slide_path = turn.get("slide_path")

        lines = [f"## {role}\n"]
        if slide_path:
            lines.append(f"![slide]({slide_path})\n\n")
        lines.append(f"{content}\n")
        lines.append("\n")

        with (self._meeting_dir / "session.md").open("a", encoding="utf-8") as f:
            f.write("".join(lines))
