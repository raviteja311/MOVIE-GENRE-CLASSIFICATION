# 🎬 Movie Genre Classification using NLP

This project leverages Natural Language Processing (NLP) and Machine Learning techniques to automatically classify movies into their respective genres based on their plot summaries or descriptions. The core of the project lies within the accompanying Jupyter Notebook, which details the data exploration, text preprocessing, feature engineering, model training, and evaluation steps.

---

## ✨ Features

*   **Data Handling:** Loads movie datasets containing ID, Title, Genre, and Description.
*   **Exploratory Data Analysis (EDA) 📊:**
    *   Analyzes data structure and checks for missing values or duplicates.
    *   Visualizes genre distribution using bar charts.
    *   Investigates description length patterns with histograms.
*   **Text Preprocessing ⚙️:**
    *   Applies standard NLP cleaning techniques: lowercasing, removal of emails, mentions, hashtags, HTML tags, numbers, punctuation, and single characters.
    *   Utilizes NLTK for:
        *   Tokenization
        *   Stopword removal
        *   Lemmatization
*   **Visualization:** Creates a WordCloud to highlight the most frequent terms in movie descriptions.
*   **Feature Engineering:** Transforms cleaned text descriptions into numerical feature vectors using the **TF-IDF** (Term Frequency-Inverse Document Frequency) method, incorporating n-grams (1, 2).
*   **Label Encoding:** Converts categorical genre labels into a numerical format suitable for ML models.
*   **Model Training & Optimization 🧠:**
    *   Splits the training data into training and validation sets.
    *   Trains multiple classification models:
        *   Logistic Regression (Optimized using GridSearchCV for hyperparameter tuning)
        *   Multinomial Naive Bayes
        *   Linear Support Vector Classifier (SVM)
*   **Model Evaluation 📈:**
    *   Assesses model performance on the validation set using detailed Classification Reports.
    *   Visualizes model predictions with Confusion Matrices.
    *   Identifies the best-performing model based on evaluation metrics (weighted F1-score).
*   **Prediction 🚀:** Demonstrates how to use the best-trained model (SVM) to predict genres for new, unseen movie descriptions.

---

## 💾 Dataset

The project utilizes three primary datasets provided as text files:

1.  **`train_data.txt`**: Training data including `ID`, `TITLE`, `GENRE`, and `DESCRIPTION`.
2.  **`test_data.txt`**: Test data containing `ID`, `TITLE`, and `DESCRIPTION` (for prediction).
3.  **`test_data_solution.txt`**: Ground truth `GENRE` labels corresponding to the test data.

*Fields within these files are delimited by `:::`.*

---

## 📝 Workflow Overview

> 1.  **Library Imports:** Load necessary Python packages (`pandas`, `numpy`, `sklearn`, `nltk`, `matplotlib`, `seaborn`, `wordcloud`).
> 2.  **Data Loading:** Read the datasets into pandas DataFrames.
> 3.  **EDA:** Explore and visualize the data characteristics.
> 4.  **Preprocessing:** Clean and normalize the movie descriptions using NLP techniques.
> 5.  **Feature Engineering:** Create TF-IDF vectors from text and encode labels.
> 6.  **Data Splitting:** Divide the training data for model training and validation.
> 7.  **Model Training & Tuning:** Build and optimize Logistic Regression, Naive Bayes, and SVM models.
> 8.  **Evaluation:** Compare model performance using classification metrics and confusion matrices.
> 9.  **Prediction:** Apply the best model to make genre predictions on test samples.

---

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Install Dependencies:** Ensure you have the required Python libraries.
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud jupyter
    ```
3.  **Download NLTK Resources:** Run the following in Python or the notebook:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    # nltk.download('punkt_tab') # May also be needed
    ```
4.  **Data Files:** Place `train_data.txt`, `test_data.txt`, and `test_data_solution.txt` in the project directory or adjust the file paths within the notebook.

---

## 📊 Evaluation Highlights

Three models were evaluated on the validation set. The SVM model demonstrated the best performance based on the weighted average F1-score:

*   Logistic Regression (Optimized): ~0.54 F1-score
*   Naive Bayes: ~0.43 F1-score
*   **SVM: ~0.54 F1-score (Selected Model)**

*(Note: Performance metrics might slightly differ based on specific data splits or library versions.)*

---

## 🚀 Sample Prediction

The notebook concludes by predicting genres for the first two samples from the test set using the trained SVM model:
