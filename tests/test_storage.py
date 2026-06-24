import json
from pathlib import Path

import pytest

from src.storage import Storage


class TestStorage:
    # ------------------------------------------------------------------
    # start_meeting
    # ------------------------------------------------------------------

    def test_start_meeting_creates_named_folder(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        assert (tmp_path / "meetings" / "2026-06-25_14-00_standup").is_dir()

    def test_start_meeting_creates_skeleton_files(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        meeting_dir = tmp_path / "meetings" / "2026-06-25_14-00_standup"
        assert (meeting_dir / "transcript.txt").exists()
        assert (meeting_dir / "history.json").exists()
        assert (meeting_dir / "session.md").exists()
        assert (meeting_dir / "slides").is_dir()

    def test_start_meeting_history_json_is_valid_empty_list(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        history_path = tmp_path / "meetings" / "2026-06-25_14-00_standup" / "history.json"
        data = json.loads(history_path.read_text())
        assert data == []

    # ------------------------------------------------------------------
    # save_slide
    # ------------------------------------------------------------------

    def test_save_slide_writes_png_and_returns_path(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        slide_path = storage.save_slide(fake_png)

        assert Path(slide_path).exists()
        assert Path(slide_path).suffix == ".png"
        assert Path(slide_path).read_bytes() == fake_png

    def test_save_slide_returns_path_not_base64(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        slide_path = storage.save_slide(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        # A path string should not look like base64 (no = padding, no long alphanum run)
        assert "/" in slide_path or "\\" in slide_path
        assert len(slide_path) < 300  # base64 of a PNG would be much longer than a path

    def test_save_slide_increments_filename_per_call(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        path1 = storage.save_slide(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        path2 = storage.save_slide(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        assert path1 != path2

    # ------------------------------------------------------------------
    # append_transcript
    # ------------------------------------------------------------------

    def test_append_transcript_writes_to_transcript_txt(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        storage.append_transcript("Hello world")

        txt = (tmp_path / "meetings" / "2026-06-25_14-00_standup" / "transcript.txt").read_text()
        assert "Hello world" in txt

    def test_append_transcript_accumulates_across_calls(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        storage.append_transcript("First chunk. ")
        storage.append_transcript("Second chunk.")

        txt = (tmp_path / "meetings" / "2026-06-25_14-00_standup" / "transcript.txt").read_text()
        assert "First chunk." in txt
        assert "Second chunk." in txt

    # ------------------------------------------------------------------
    # record_turn
    # ------------------------------------------------------------------

    def test_record_turn_appends_entry_to_history_json(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        storage.record_turn(role="assistant", content="Slide 1 shows…", explanation="This is the intro slide.")

        history_path = tmp_path / "meetings" / "2026-06-25_14-00_standup" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 1
        assert turns[0]["role"] == "assistant"
        assert turns[0]["content"] == "Slide 1 shows…"
        assert turns[0]["explanation"] == "This is the intro slide."

    def test_record_turn_appends_to_session_md(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        storage.record_turn(role="assistant", content="Some explanation", explanation="This is the intro slide.")

        session = (tmp_path / "meetings" / "2026-06-25_14-00_standup" / "session.md").read_text()
        assert "Some explanation" in session

    def test_record_turn_multiple_turns_accumulate_in_history(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        storage.record_turn(role="user", content="What is this?", explanation="")
        storage.record_turn(role="assistant", content="It's a chart.", explanation="Revenue chart Q1.")

        history_path = tmp_path / "meetings" / "2026-06-25_14-00_standup" / "history.json"
        turns = json.loads(history_path.read_text())
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    def test_record_turn_with_slide_stores_path_in_history_not_base64(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        slide_path = storage.save_slide(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        storage.record_turn(
            role="assistant",
            content="See attached slide",
            explanation="Revenue chart.",
            slide_path=slide_path,
        )

        history_path = tmp_path / "meetings" / "2026-06-25_14-00_standup" / "history.json"
        turns = json.loads(history_path.read_text())
        assert turns[0]["slide_path"] == slide_path

    def test_record_turn_with_slide_renders_inline_in_session_md(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])

        slide_path = storage.save_slide(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        storage.record_turn(
            role="assistant",
            content="See attached slide",
            explanation="Revenue chart.",
            slide_path=slide_path,
        )

        session = (tmp_path / "meetings" / "2026-06-25_14-00_standup" / "session.md").read_text()
        # Markdown image syntax must reference the slide path
        assert "![" in session
        assert slide_path in session

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def test_finalize_does_not_leave_partial_files(self, tmp_path):
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])
        storage.append_transcript("some text")
        storage.record_turn(role="assistant", content="OK", explanation="Fine.")

        storage.finalize()

        # All expected files should still be valid after finalize
        history_path = tmp_path / "meetings" / "2026-06-25_14-00_standup" / "history.json"
        turns = json.loads(history_path.read_text())
        assert isinstance(turns, list)

    def test_finalize_makes_history_json_resume_capable(self, tmp_path):
        """history.json must contain enough fields to reconstruct the chat."""
        storage = Storage(base_dir=tmp_path)
        storage.start_meeting("2026-06-25_14-00_standup", langs=["en"])
        storage.record_turn(role="user", content="Hello", explanation="")
        storage.record_turn(role="assistant", content="Hi there", explanation="Greeting.")

        storage.finalize()

        history_path = tmp_path / "meetings" / "2026-06-25_14-00_standup" / "history.json"
        turns = json.loads(history_path.read_text())
        for turn in turns:
            assert "role" in turn
            assert "content" in turn
