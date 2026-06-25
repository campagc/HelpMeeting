from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

HOTKEY = "Control+Option+Space"
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
    hotkey: str
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
        hotkey=HOTKEY,
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
