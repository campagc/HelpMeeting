from types import SimpleNamespace

from src.config import MissingApiKeyError
from src.display import Display
from src.main import main, prompt_settings
from src.meeting import MeetingSettings, MeetingStartupError


class FakeDisplays:
    def __init__(self, displays):
        self.displays = displays

    def list(self):
        return self.displays


class TestPromptSettings:
    def test_collects_label_languages_and_selected_display(self):
        inputs = iter(["weekly-standup", "it", "en", "2"])

        def fake_input(prompt=""):
            return next(inputs)

        output = []
        displays = FakeDisplays(
            [
                Display(1, 0, 0, 1440, 900),
                Display(2, 1440, 0, 1440, 900),
            ]
        )

        settings = prompt_settings(
            input_fn=fake_input,
            output_fn=output.append,
            displays=displays,
        )

        assert settings == MeetingSettings(
            label="weekly-standup",
            spoken_language="it",
            explanation_language="en",
            display_index=2,
        )
        assert output == [
            "Found 2 displays:",
            "  1: 1440x900 at (0, 0)",
            "  2: 1440x900 at (1440, 0)",
        ]

    def test_uses_defaults_for_languages_and_single_display(self):
        inputs = iter(["daily-sync", "", ""])

        def fake_input(prompt=""):
            return next(inputs)

        settings = prompt_settings(
            input_fn=fake_input,
            displays=FakeDisplays([Display(1, 0, 0, 1440, 900)]),
        )

        assert settings.label == "daily-sync"
        assert settings.spoken_language == "en"
        assert settings.explanation_language == "en"
        assert settings.display_index == 1

    def test_explanation_language_defaults_to_spoken_language(self):
        inputs = iter(["sync", "fr", ""])

        def fake_input(prompt=""):
            return next(inputs)

        settings = prompt_settings(
            input_fn=fake_input,
            displays=FakeDisplays([Display(1, 0, 0, 1440, 900)]),
        )

        assert settings.spoken_language == "fr"
        assert settings.explanation_language == "fr"

    def test_invalid_display_choice_falls_back_to_first_display(self):
        inputs = iter(["sync", "en", "it", "2"])

        def fake_input(_prompt=""):
            return next(inputs)

        settings = prompt_settings(
            input_fn=fake_input,
            displays=FakeDisplays(
                [
                    Display(1, 0, 0, 1440, 900),
                    Display(3, 1440, 0, 1440, 900),
                ]
            ),
        )

        assert settings.display_index == 1


class FakeMeeting:
    def __init__(self, exit_code=0):
        self.exit_code = exit_code
        self.run_calls = 0

    def run(self):
        self.run_calls += 1
        return self.exit_code


class TestMain:
    def test_delegates_the_entire_runtime_to_meeting(self, monkeypatch):
        config = SimpleNamespace()
        settings = MeetingSettings("test", "en", "en", 1)
        meeting = FakeMeeting()
        opened = []
        monkeypatch.setattr("src.main.load", lambda: config)
        monkeypatch.setattr("src.main.prompt_settings", lambda: settings)
        monkeypatch.setattr(
            "src.main.Meeting.open",
            lambda actual_settings, actual_config: opened.append(
                (actual_settings, actual_config)
            )
            or meeting,
        )

        assert main() == 0

        assert opened == [(settings, config)]
        assert meeting.run_calls == 1

    def test_exits_cleanly_when_api_key_is_missing(self, monkeypatch, capsys):
        def fail_load():
            raise MissingApiKeyError("Missing GEMINI_API_KEY")

        monkeypatch.setattr("src.main.load", fail_load)

        assert main() == 1

        captured = capsys.readouterr()
        assert "Missing GEMINI_API_KEY" in captured.err

    def test_renders_the_single_meeting_startup_error(self, monkeypatch, capsys):
        settings = MeetingSettings("test", "en", "en", 1)
        monkeypatch.setattr("src.main.load", lambda: SimpleNamespace())
        monkeypatch.setattr("src.main.prompt_settings", lambda: settings)

        def fail_open(*args, **kwargs):
            raise MeetingStartupError(
                "Screen & System Audio Recording permission is required; relaunch."
            )

        monkeypatch.setattr("src.main.Meeting.open", fail_open)

        assert main() == 1

        captured = capsys.readouterr()
        assert "Screen & System Audio Recording" in captured.err
        assert "relaunch" in captured.err.lower()
        assert "Traceback" not in (captured.out + captured.err)
