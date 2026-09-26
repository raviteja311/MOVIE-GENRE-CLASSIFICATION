# 🎬 Movie Genre Classification

Predict a movie's genre (27 classes) from its plot description using a scikit-learn text pipeline: NLTK text cleaning, TF-IDF over words and word pairs, and a linear classifier.

## Problem

Given a short plot summary, predict which of 27 genres (drama, comedy, thriller, documentary, ...) the movie belongs to. The classes are very imbalanced: drama and documentary together cover about half of the data, while genres like war and news have fewer than 200 examples each.

## Approach

1. **Cleaning** (`text_utils.py`): lowercase, strip emails, tags, numbers and punctuation, remove stopwords, lemmatize. The text is cleaned once, in parallel, and reused by every model.
2. **Features**: `TfidfVectorizer` with unigrams and bigrams, `sublinear_tf=True`, `min_df=2`, `max_df=0.9`, capped at 100,000 features.
3. **Models**: Logistic Regression, Complement Naive Bayes and Linear SVC (`C=0.3`), with balanced class weights where supported.
4. **Selection**: a stratified 80/20 split of the training data. The model with the best **validation** weighted F1 is selected; the labelled test set is only used to report the final score.
5. **Export**: the selected model is refit on all training data and saved as a single pipeline that accepts raw plot text.

## Dataset

The repository expects three text files in the project root, delimited by `:::`:

* `train_data.txt` - `ID ::: TITLE ::: GENRE ::: DESCRIPTION` (54,214 rows)
* `test_data.txt` - `ID ::: TITLE ::: DESCRIPTION` (54,200 rows)
* `test_data_solution.txt` - the same test rows with `GENRE`, used for held-out evaluation

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

## Usage

Create an environment (Python 3.12+) and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Explore the data (prints summaries, saves plots to `artifacts/plots/`):

```bash
python eda.py
```

Train, evaluate and export the model (about 4 minutes):

```bash
python evaluate_models.py
```

Predict from the command line:

```bash
python predict.py "A small-town detective investigates a string of unsettling disappearances."
```

If you need to restore the data files from another location:

```bash
python download_data.py --source-dir <path-to-data>
```

## Project Files

* `eda.py` - dataset summaries, genre distribution, description length and word cloud plots
* `evaluate_models.py` - training, model selection, evaluation and export
* `predict.py` - command-line inference entry point
* `text_utils.py` - text cleaning shared by training and inference
* `download_data.py` - helper for restoring the data files
* `artifacts/movie_genre_classifier.joblib` - saved pipeline and label encoder
* `artifacts/metrics.json` - validation and test metrics, plus the per-class test report
* `artifacts/plots/` - EDA plots and the confusion matrix of the selected model

## Notes

* `nltk` is pinned to 3.10.3. Version 3.10.1 ships an import guard that blocks imports whenever the virtual environment lives inside the working directory (the usual `.venv` layout).
* 91 descriptions appear in both the training and test files, and 15 training descriptions carry conflicting labels.
