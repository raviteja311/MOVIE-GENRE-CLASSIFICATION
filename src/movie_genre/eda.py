"""Exploratory data analysis: print dataset summaries and save plots to reports/figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from movie_genre.config import FIGURES_DIR
from movie_genre.data import load_data
from movie_genre.preprocessing import preprocess_text


def save_plot(name):
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_data, test_eval_data = load_data()

    print("Shape of the training data :", train_data.shape)
    print("Shape of the held-out test data :", test_eval_data.shape)
    print()
    train_data.info()
    print()
    print(train_data.describe(include="object").T)
    print()
    print("Missing values (train):\n", train_data.isnull().sum(), sep="")
    print("Missing values (test):\n", test_eval_data.isnull().sum(), sep="")
    print("Duplicate rows (train):", train_data.duplicated().sum())
    print("Duplicate descriptions (train):", train_data["DESCRIPTION"].duplicated().sum())
    shared = set(train_data["DESCRIPTION"]) & set(test_eval_data["DESCRIPTION"])
    print("Descriptions shared by train and test:", len(shared))

    genre_counts = train_data["GENRE"].value_counts()
    print("\nGenre counts:\n", genre_counts, sep="")

    genre_counts.plot(kind="bar", figsize=(10, 5))
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.title("Genre Distribution")
    save_plot("genre_distribution.png")

    train_data["DESC_LENGTH"] = train_data["DESCRIPTION"].apply(lambda x: len(str(x).split()))
    print("\nDescription length (words):\n", train_data["DESC_LENGTH"].describe(), sep="")

    plt.figure(figsize=(10, 5))
    train_data["DESC_LENGTH"].hist(bins=30, color="salmon")
    plt.title("Description Length Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Movies")
    save_plot("description_length.png")

    plt.figure(figsize=(15, 10))
    sns.barplot(x="GENRE", y="DESC_LENGTH", data=train_data)
    plt.title("Description Length by Genre")
    plt.xticks(rotation=45)
    plt.xlabel("Genre")
    plt.ylabel("Description Length")
    save_plot("description_length_by_genre.png")

    print("\nCleaning text for the word cloud...")
    text = " ".join(train_data["DESCRIPTION"].map(preprocess_text))
    wordcloud = WordCloud(width=1000, height=500, background_color="black", colormap="spring").generate(text)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words in Descriptions", fontsize=18)
    save_plot("wordcloud.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
