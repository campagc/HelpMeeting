import pytest

from src.config import load, MissingApiKeyError


class TestConfig:
    def test_load_returns_config_with_system_prompt(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("GEMINI_API_KEY=test-key\n")
        prompt = tmp_path / "system_prompt.md"
        prompt.write_text("You are a helpful assistant.")

        config = load(dotenv_path=dotenv, system_prompt_path=prompt)

        assert config.api_key == "test-key"
        assert config.system_prompt == "You are a helpful assistant."
        assert config.hotkey == "Control+Option+Space"

    def test_load_raises_clear_error_when_api_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text("# empty env\n")
        prompt = tmp_path / "system_prompt.md"
        prompt.write_text("You are a helpful assistant.")

        with pytest.raises(MissingApiKeyError, match="Missing GEMINI_API_KEY"):
            load(dotenv_path=dotenv, system_prompt_path=prompt)

    def test_load_reflects_edits_to_system_prompt_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        dotenv = tmp_path / ".env"
        dotenv.write_text("GEMINI_API_KEY=test-key\n")
        prompt = tmp_path / "system_prompt.md"
        prompt.write_text("Original persona.")

        assert load(dotenv_path=dotenv, system_prompt_path=prompt).system_prompt == "Original persona."

        prompt.write_text("Updated persona.")
        assert load(dotenv_path=dotenv, system_prompt_path=prompt).system_prompt == "Updated persona."
