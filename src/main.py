import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load, MissingApiKeyError


def main():
    try:
        config = load()
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"HelpMeeting ready. Press {config.hotkey} to request an explanation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
