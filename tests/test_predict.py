import joblib
import pytest

from movie_genre import predict


@pytest.fixture
def artifact(tmp_path, tiny_model):
    model, label_encoder = tiny_model
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "label_encoder": label_encoder}, path)
    return path


def test_file_option_predicts_one_plot_per_line(tmp_path, artifact, capsys):
    plots = tmp_path / "plots.txt"
    plots.write_text("alien space battle\n\na couple falls in love at a wedding\n", encoding="utf-8")

    assert predict.main(["--file", str(plots), "--artifact", str(artifact), "--top-k", "2"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "row,title,genre_1,genre_2"
    assert [line.split(",")[2] for line in lines[1:]] == ["sci-fi", "romance"]


def test_file_option_reads_csv_with_titles(tmp_path, artifact, capsys):
    plots = tmp_path / "plots.csv"
    plots.write_text("Title,Description\nNight Cop (1987),detective hunts a killer\n", encoding="utf-8")

    assert predict.main(["--file", str(plots), "--artifact", str(artifact), "--top-k", "1"]) == 0

    assert capsys.readouterr().out.splitlines() == ["row,title,genre_1", "1,Night Cop (1987),thriller"]


def test_file_option_rejects_csv_without_description(tmp_path, artifact):
    plots = tmp_path / "plots.csv"
    plots.write_text("title\nHeat (1995)\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        predict.main(["--file", str(plots), "--artifact", str(artifact)])
