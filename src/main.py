"""Command-line entry: collect Meeting settings and run the Meeting."""

import os
import sys
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MissingApiKeyError, load
from src.display import Display, Displays
from src.meeting import Meeting, MeetingSettings, MeetingStartupError


def _choose_display(
    displays: list[Display],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> int:
    if len(displays) <= 1:
        return 1
    output_fn(f"Found {len(displays)} displays:")
    for display in displays:
        output_fn(
            f"  {display.index}: {display.width}x{display.height} "
            f"at ({display.left}, {display.top})"
        )
    choice = input_fn("Choose display for screenshots [1]: ").strip() or "1"
    try:
        index = int(choice)
        if index in {display.index for display in displays}:
            return index
    except ValueError:
        pass
    return 1


def prompt_settings(
    input_fn=input,
    output_fn=print,
    displays: Displays | None = None,
):
    label = input_fn("Meeting label: ").strip()
    if not label:
        label = "meeting"

    spoken_language = input_fn("Spoken language [en]: ").strip() or "en"
    explanation_language = (
        input_fn(f"Explanation language [{spoken_language}]: ").strip()
        or spoken_language
    )
    display_index = _choose_display((displays or Displays()).list(), input_fn, output_fn)

    return MeetingSettings(
        label=label,
        spoken_language=spoken_language,
        explanation_language=explanation_language,
        display_index=display_index,
    )


def main():
    try:
        config = load()
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    settings = prompt_settings()
    try:
        meeting = Meeting.open(settings, config)
    except MeetingStartupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return meeting.run()


if __name__ == "__main__":
    sys.exit(main())
