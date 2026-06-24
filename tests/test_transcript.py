from src.transcript import Transcript


class TestTranscript:
    def test_append_then_take_delta_returns_appended_text(self):
        transcript = Transcript()
        transcript.append("hello")
        assert transcript.take_delta() == "hello"

    def test_take_delta_with_no_append_returns_empty(self):
        transcript = Transcript()
        assert transcript.take_delta() == ""

    def test_consecutive_take_delta_calls_return_empty(self):
        transcript = Transcript()
        transcript.append("hello")
        transcript.take_delta()
        assert transcript.take_delta() == ""

    def test_interleaved_append_and_take_delta_cycles_never_overlap_or_drop(self):
        transcript = Transcript()
        transcript.append("a")
        assert transcript.take_delta() == "a"
        transcript.append("b")
        transcript.append("c")
        assert transcript.take_delta() == "bc"
        transcript.append("d")
        assert transcript.take_delta() == "d"
        assert transcript.take_delta() == ""
