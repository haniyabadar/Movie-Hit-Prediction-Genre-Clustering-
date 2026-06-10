# Movie-Hit-Prediction-Genre-Clustering-
# 🎬 Movie Genre Recommendation & Analysis System

A data science project built on the TMDB 5000 Movies dataset that applies multiple machine learning models to analyze movies, predict hits/flops, and recommend by genre using clustering.

---

## 📌 About the Project

This project was developed as part of **Data Science Fundamentals** coursework. It explores a real-world movie dataset using a full ML pipeline — from preprocessing and feature engineering to applying and comparing five different models.

The core idea: given a movie's budget, revenue, runtime, and popularity, can we predict whether it'll be a **hit or a flop**? And can we group movies into meaningful clusters by genre similarity?

---

## 📂 Dataset

- **Source:** [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **File:** `tmdb_5000_movies.csv`
- **Size:** ~5000 movies with features like budget, revenue, genres, popularity, runtime, language

---

## ⚙️ Preprocessing & Feature Engineering

- Filled missing values for genres, language, and numeric columns
- Applied **TF-IDF Vectorization** on genre lists to capture genre similarity
- Used **One-Hot Encoding** for original language
- Applied **MinMax Scaling** on numeric features (budget, revenue, popularity, runtime)
- Engineered new features:
  - `profit = revenue - budget`
  - `is_hit = 1 if profit > 0 else 0` (used for classification)
- Applied **PCA** (2 components) for dimensionality reduction and visualization

---

## 🤖 Models Applied

| Model | Task | Result |
|---|---|---|
| **K-Means Clustering** | Unsupervised / Genre Grouping | Silhouette Score: ~0.66 — Best performing model for this dataset |
| **KNN Classifier** | Classification (Hit/Flop) | Accuracy: ~88%, F1: ~83% |
| **Decision Tree** | Classification (Hit/Flop) | Reduced overfitting via max-depth tuning → ~71% accuracy |
| **Polynomial Regression** | Regression (Predict Popularity) | Degree-2 polynomial; better than linear but errors remain |
| **ANN** | Regression (Predict Popularity) | Lower MSE than Polynomial; handles complex patterns better |

---

## 📊 Visualizations

- Correlation heatmap of numeric features
- Train vs Test distribution histograms (classification & regression)
- K-Means cluster plot (PCA 2D scatter)
- Decision Tree structure plot
- Accuracy vs Max Depth line graph (overfitting reduction)
- Scatter plots for Polynomial Regression and ANN predictions
- Final model performance comparison bar chart

---

## 🔍 Key Findings

- **K-Means** worked best as the dataset naturally lends itself to cluster-based grouping
- **Decision Tree** was initially overfitting at ~99% accuracy — fixed using hyperparameter tuning (max depth)
- **KNN** gave solid results at 88% accuracy without overfitting
- **ANN** outperformed Polynomial Regression for popularity prediction

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
python movie_analysis.py
```

Make sure `tmdb_5000_movies.csv` is in the same folder as the script.

---

## 👩‍💻 Author

**Haniya Badar** — Software Engineering Student, COMSATS University Lahore
