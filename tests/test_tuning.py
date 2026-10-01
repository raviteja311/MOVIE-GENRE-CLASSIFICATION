from movie_genre.config import DEFAULT_SVC_C, load_svc_c
from movie_genre.tune import save_tuning


def test_tuned_c_round_trips_and_defaults(tmp_path):
    path = tmp_path / "config" / "tuning.json"
    assert load_svc_c(path) == DEFAULT_SVC_C

    save_tuning(path, 2.0, 0.581234567, folds=5, grid=[1, 2])

    assert load_svc_c(path) == 2.0
