"""Commands without options must still parse arguments, so --help or a typo never starts a long run."""

import pytest

from movie_genre import eda, train


def refuse_to_load(*args, **kwargs):
    raise AssertionError("argument parsing should stop before any data is loaded")


def test_help_exits_without_running(monkeypatch, capsys):
    for module, prog in ((eda, "movie-genre-eda"), (train, "movie-genre-train")):
        monkeypatch.setattr(module, "load_data", refuse_to_load)

        with pytest.raises(SystemExit) as exit_info:
            module.main(["--help"])

        assert exit_info.value.code == 0
        assert capsys.readouterr().out.startswith(f"usage: {prog}")


def test_unknown_argument_is_rejected(monkeypatch, capsys):
    for module in (eda, train):
        monkeypatch.setattr(module, "load_data", refuse_to_load)

        with pytest.raises(SystemExit) as exit_info:
            module.main(["--epochs", "5"])

        assert exit_info.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err
