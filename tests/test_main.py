from src.main import main


class TestMain:
    def test_main_prints_ready_when_config_loads(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("GEMINI_API_KEY=test-key\n")
        (tmp_path / "system_prompt.md").write_text("persona")

        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "ready" in captured.out.lower()
        assert "Control+Option+Space" in captured.out

    def test_main_exits_cleanly_with_error_when_api_key_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("# empty\n")
        (tmp_path / "system_prompt.md").write_text("persona")

        exit_code = main()

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Missing GEMINI_API_KEY" in captured.err
