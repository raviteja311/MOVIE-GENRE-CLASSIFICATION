# 🎬 Movie Genre Classification

Predict a movie's genre (27 classes) from its plot description using a scikit-learn text pipeline: NLTK text cleaning, TF-IDF over words and word pairs, and a linear classifier.

## Problem

Given a short plot summary, predict which of 27 genres (drama, comedy, thriller, documentary, ...) the movie belongs to. The classes are very imbalanced: drama and documentary together cover about half of the data, while genres like war and news have fewer than 200 examples each.

## Approach

1. **Cleaning** (`preprocessing.py`): lowercase, strip emails, tags, numbers and punctuation, remove stopwords, lemmatize. The text is cleaned once, in parallel, and reused by every model.
2. **Features**: `TfidfVectorizer` with unigrams and bigrams, `sublinear_tf=True`, `min_df=2`, `max_df=0.9`, capped at 100,000 features.
3. **Models**: Logistic Regression, Complement Naive Bayes and Linear SVC (`C=0.3`), with balanced class weights where supported.
4. **Selection**: a stratified 80/20 split of the training data. The model with the best **validation** weighted F1 is selected; the labelled test set is only used to report the final score.
5. **Export**: the selected model is refit on all training data and saved as a single pipeline that accepts raw plot text.

## Dataset

Three `:::` delimited files live in `data/raw/` (see [data/README.md](data/README.md) for the format):

* `train_data.txt` - 54,214 labelled movies
* `test_data.txt` - 54,200 unlabelled movies
* `test_data_solution.txt` - the same test movies with their `GENRE`, used for held-out evaluation

## Results

Held-out test set (54,200 movies):

* Majority-class baseline: **25.11% accuracy**
* Selected model: **Linear SVC**
* Accuracy: **55.21%**
* Weighted F1: **55.81%**
* Macro F1: **37.47%**
* Top-3 accuracy: **79.12%**

| Model | Val accuracy | Val weighted F1 | Test accuracy | Test weighted F1 | Test macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear SVC | 55.34% | 55.87% | 55.21% | 55.81% | 37.47% |
| Logistic Regression | 50.39% | 52.24% | 50.43% | 52.45% | 37.43% |
| Complement NB | 54.36% | 47.29% | 54.77% | 47.76% | 23.15% |

Balanced class weights trade some overall accuracy for better recall on rare genres (higher macro F1). Removing them raises accuracy to about 60% but lowers macro F1 to about 32%.

## Project Structure

```
MOVIE-GENRE-CLASSIFICATION/
├── data/
│   ├── README.md                  # file format and source
│   └── raw/                       # train, test and solution files
├── models/
│   └── movie_genre_classifier.joblib
├── reports/
│   ├── metrics.json               # validation and test metrics, per-class report
│   └── figures/                   # EDA plots and confusion matrix
├── src/movie_genre/
│   ├── config.py                  # paths and constants
│   ├── data.py                    # loading and merging the data files
│   ├── preprocessing.py           # text cleaning shared by training and inference
│   ├── model.py                   # pipeline definition and top-k prediction
│   ├── eda.py                     # exploratory analysis
│   ├── train.py                   # training, selection, evaluation and export
│   ├── predict.py                 # command-line inference
│   └── download_data.py           # restore the data files
├── tests/                         # pytest suite
├── pyproject.toml
├── requirements.txt               # pinned runtime dependencies
└── requirements-dev.txt           # adds pytest
```

## Usage

Create an environment (Python 3.11+) and install the package with pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Explore the data (prints summaries, saves plots to `reports/figures/`):

```bash
movie-genre-eda
```

Train, evaluate and export the model (about 4 minutes):

```bash
movie-genre-train
```

Predict from the command line:

```bash
movie-genre-predict "A small-town detective investigates a string of unsettling disappearances." --top-k 3
```

Restore the data files from another location:

```bash
movie-genre-download --source-dir <path-to-data>
```

Each command is also available as a module, for example `python -m movie_genre.train`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Notes

* `nltk` is pinned to 3.10.3. Version 3.10.1 ships an import guard that blocks imports whenever the virtual environment lives inside the working directory (the usual `.venv` layout).
* 91 descriptions appear in both the training and test files, and 15 training descriptions carry conflicting labels.

## License

MIT, see [LICENSE](LICENSE).
