# Movie Genre Classification

[![tests](https://github.com/raviteja311/MOVIE-GENRE-CLASSIFICATION/actions/workflows/tests.yml/badge.svg)](https://github.com/raviteja311/MOVIE-GENRE-CLASSIFICATION/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predict a movie's genre (27 classes) from its plot description and title using a scikit-learn pipeline: regex text cleaning, TF-IDF features, and a linear classifier.

<p align="center">
  <img src="docs/images/streamlit_demo.png" alt="Streamlit demo predicting crime for a detective thriller plot, with the top five genres plotted by score" width="640">
</p>
<p align="center"><em>The Streamlit demo (<code>streamlit run streamlit_app.py</code>)</em></p>

## Problem

Given a short plot summary, predict which of 27 genres (drama, comedy, thriller, documentary, ...) the movie belongs to. The classes are very imbalanced: drama and documentary together cover about half of the data, while genres like war and news have fewer than 200 examples each.

## Approach

1. **Data cleaning** (`data.py`): drop training rows whose description also appears in the test set (164), rows with conflicting labels for the same description (11), and exact duplicates (49).
2. **Text cleaning** (`preprocessing.py`): lowercase, strip emails, tags, numbers, punctuation and single letters.
3. **Features** (`model.py`):
   * Description: word unigrams and bigrams, English stopwords removed, `sublinear_tf=True`, `min_df=2`, `max_df=0.9`, capped at 100,000 features.
   * Title: character 2-4 grams (up to 50,000), which pick up the release year and the quoting used for TV episodes.
4. **Models**: Logistic Regression, Complement Naive Bayes and Linear SVC, with balanced class weights where supported. The SVC's `C=0.5` was chosen by 5-fold cross-validation (`movie-genre-tune`, which writes it to `config/tuning.json`).
5. **Selection**: a stratified 80/20 split of the training data. The model with the best **validation** weighted F1 is selected; the labelled test set is only used to report the final score.
6. **Export**: the selected model is refit on all 53,990 cleaned training rows and saved as a single pipeline that accepts raw text. This refit model is the one scored on the test set and reported below.

## Dataset

Three `:::` delimited files live in `data/raw/` (see [data/README.md](data/README.md) for the format):

* `train_data.txt` - 54,214 labelled movies
* `test_data.txt` - 54,200 unlabelled movies
* `test_data_solution.txt` - the same test movies with their `GENRE`, used for held-out evaluation

## Results

Shipped model (Linear SVC refit on all cleaned training rows) on the held-out test set (54,200 movies):

* Majority-class baseline: **25.11% accuracy**
* Accuracy: **59.43%**
* Weighted F1: **59.35%**
* Macro F1: **41.30%**
* Top-3 accuracy: **81.88%**

Model selection, with every model fit on the 80% train split:

| Model | Val accuracy | Val weighted F1 | Test accuracy | Test weighted F1 | Test macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear SVC | 59.10% | 58.90% | 59.07% | 58.86% | 40.74% |
| Logistic Regression | 54.02% | 55.10% | 53.99% | 55.09% | 39.22% |
| Complement NB | 54.06% | 47.16% | 53.94% | 46.88% | 21.91% |

The test columns here are for comparison only; selection uses validation weighted F1. `reports/metrics.json` keeps both sets of numbers, with a `fit` field on every row (`train split (80%)` or `all cleaned rows (shipped)`) and the shipped scores under `shipped_model`.

Without a title the shipped model still reaches 59.39% accuracy, 57.95% weighted F1 and 38.47% macro F1 on the test set.

Balanced class weights trade some overall accuracy for better recall on rare genres (higher macro F1).

### Tried and not kept

Each was compared on the validation split:

* **NLTK stopwords and lemmatization** (earlier versions; NLTK is no longer a dependency): no gain over plain regex cleaning and much slower.
* **Title as word tokens**: no gain; character n-grams worked better.
* **Calibrated probabilities** (`CalibratedClassifierCV`): higher accuracy and top-3, but lower weighted and macro F1, because calibration undoes the balanced class weights.

## Project Structure

```
MOVIE-GENRE-CLASSIFICATION/
├── .github/workflows/tests.yml    # CI: ruff, then pytest on Python 3.12 and 3.14
├── config/tuning.json             # C chosen by movie-genre-tune, read by training
├── data/
│   ├── README.md                  # file format and source
│   └── raw/                       # train, test and solution files
├── docs/images/                   # README screenshot
├── models/
│   └── movie_genre_classifier.joblib
├── reports/
│   ├── metrics.json               # shipped-model and selection metrics, per-class report
│   └── figures/                   # EDA plots and confusion matrix
├── src/movie_genre/
│   ├── config.py                  # paths, constants, data checksums, tuned C loader
│   ├── data.py                    # loading, cleaning and splitting
│   ├── preprocessing.py           # text cleaning shared by training and inference
│   ├── model.py                   # pipeline definition and top-k prediction
│   ├── eda.py                     # exploratory analysis
│   ├── tune.py                    # cross-validation for C
│   ├── train.py                   # training, selection, evaluation and export
│   ├── predict.py                 # command-line inference, single plot or --file batch
│   └── download_data.py           # restore and SHA-256 verify the data files
├── tests/                         # pytest suite, including headless app tests
├── streamlit_app.py               # web demo
├── Dockerfile                     # container for the web demo
├── pyproject.toml                 # package, pytest and ruff settings
├── requirements.txt               # pinned runtime dependencies
└── requirements-dev.txt           # adds pytest and ruff
```

## Usage

Create an environment (Python 3.11+) and install the package with pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Explore the data (prints summaries, saves plots to `reports/figures/`):

```bash
movie-genre-eda
```

Tune `C` with cross-validation (about 6 minutes). The best value is written to `config/tuning.json`, which `movie-genre-train` reads; without that file `C` defaults to 0.5. Pass `--no-save` to only print the result:

```bash
movie-genre-tune
```

Train, select, refit, evaluate and export the model (about 6 minutes):

```bash
movie-genre-train
```

Predict from the command line. The title is optional but improves accuracy:

```bash
movie-genre-predict "A small-town detective investigates a string of unsettling disappearances." --title "Hollow Creek (2015)" --top-k 3
```

Each call loads the model, which takes a few seconds, so classify many plots in one run with `--file`. It accepts a text file with one description per line, or a CSV with a `description` column and an optional `title` column, and prints CSV (`row,title,genre_1,...`) to stdout:

```bash
movie-genre-predict --file plots.csv --top-k 3 > predictions.csv
```

Launch the web demo (opens at http://localhost:8501):

```bash
streamlit run streamlit_app.py
```

The demo takes a plot description (up to 5,000 characters) and optional title (up to 250), shows the top 1, 3 or 5 genres with their scores, and includes one-click examples.

Restore the data files from another location. Each file is checked against a SHA-256 checksum stored in `config.py` (computed with LF line endings, so a Windows checkout matches too):

```bash
movie-genre-download --source-dir <path-to-data>
movie-genre-download --verify-only
```

Each command is also available as a module, for example `python -m movie_genre.train`.

## Deployment

### Docker

The image (Python 3.14 slim, non-root user) contains only the app, the package, the model and its metrics, and runs the demo on port 8501 with a health check on `/_stcore/health`:

```bash
docker build -t movie-genre .
docker run --rm -p 8501:8501 movie-genre
```

Then open http://localhost:8501.

### Streamlit Community Cloud

The repository is laid out for Community Cloud: `streamlit_app.py` at the root, pinned `requirements.txt` (which installs the package with `-e .`), and the model committed at `models/movie_genre_classifier.joblib` (about 21 MB), which the app loads by a path relative to the repository.

1. Push the branch to GitHub.
2. Sign in at https://share.streamlit.io with GitHub and choose **Create app**.
3. Pick this repository and branch, and set the main file path to `streamlit_app.py`.
4. Under **Advanced settings**, choose Python 3.12 or 3.14 (the versions tested in CI).
5. Deploy. No secrets are needed.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest                            # 18 tests
```

The suite has 18 tests: unit tests for cleaning, data loading, the pipeline, tuning output, checksums and the `--file` CLI, headless Streamlit tests, and a minimum-quality regression test. The regression test fails if the shipped model's recorded test accuracy or weighted F1 in `reports/metrics.json` drops below 58%, or if the saved model scores below 56% on a fixed 2,000-row test sample. Tests that need the model or data skip when those files are missing. CI runs `ruff check` and then `pytest` on Python 3.12 and 3.14.

## License

MIT, see [LICENSE](LICENSE).

## Author

**Jetti Raviteja** · [Portfolio](https://jettiraviteja.vercel.app) · [GitHub](https://github.com/raviteja311) · [LinkedIn](https://www.linkedin.com/in/jettiraviteja/)
