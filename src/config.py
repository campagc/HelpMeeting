from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# pynput binding syntax for the explain trigger.  Both map to the same action and
# share one debounce window, so alternating between them cannot bypass the budget
# guard.
# Control+Option+Space is retained even though it collides with the macOS
# "Select next input source" system shortcut. That collision is a known,
# accepted trade-off for existing muscle memory.
HOTKEYS = [
    "<ctrl>+<alt>+<space>",  # Control+Option+Space (collides with system shortcut)
    "<ctrl>+<alt>+s",        # Control+Option+S
]
GEMINI_MODEL_NAME = "gemini-2.5-flash"
AUDIO_CHUNK_SECONDS = 10
# faster-whisper model size. "base" downloads reliably (~145 MB) and runs fast
# on CPU; raise to "small"/"medium" if jargon is garbled and the full weights
# are present in the HF cache.
WHISPER_MODEL_SIZE = "base"
MEETINGS_DIR = "meetings"
SYSTEM_PROMPT_PATH = Path("system_prompt.md")
DOTENV_PATH = Path(".env")


@dataclass
class Config:
    hotkeys: list[str]
    gemini_model_name: str
    audio_chunk_seconds: int
    whisper_model_size: str
    meetings_dir: str
    api_key: str
    system_prompt: str


class MissingApiKeyError(Exception):
    pass


def load(dotenv_path=DOTENV_PATH, system_prompt_path=SYSTEM_PROMPT_PATH):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    api_key = _get_required_env("GEMINI_API_KEY")
    system_prompt = _read_system_prompt(system_prompt_path)
    return Config(
        hotkeys=HOTKEYS,
        gemini_model_name=GEMINI_MODEL_NAME,
        audio_chunk_seconds=AUDIO_CHUNK_SECONDS,
        whisper_model_size=WHISPER_MODEL_SIZE,
        meetings_dir=MEETINGS_DIR,
        api_key=api_key,
        system_prompt=system_prompt,
    )


def _get_required_env(name):
    value = _get_env(name)
    if not value:
        raise MissingApiKeyError(
            f"Missing {name}. Set it in your .env file (e.g. {name}=your_key)."
        )
    return value


def _get_env(name):
    import os

    return os.environ.get(name, "").strip()


def _read_system_prompt(path):
    return Path(path).read_text(encoding="utf-8").strip()


def _token_display(token: str) -> str:
    """Convert one pynput key token into a human-readable name."""
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        token = token[1:-1]
    name_map = {
        "ctrl": "Control",
        "alt": "Option",
        "space": "Space",
        "cmd": "Command",
        "shift": "Shift",
    }
    if token in name_map:
        return name_map[token]
    if len(token) == 1:
        return token.upper()
    return token.capitalize()


def format_hotkeys(hotkeys: list[str]) -> str:
    """Derive a human-readable startup line from a list of pynput bindings."""
    pretty = []
    for combo in hotkeys:
        parts = combo.split("+")
        pretty.append("+".join(_token_display(part) for part in parts))
    if len(pretty) == 1:
        return pretty[0]
    if len(pretty) == 2:
        return f"{pretty[0]} or {pretty[1]}"
    return ", ".join(pretty[:-1]) + f", or {pretty[-1]}"
