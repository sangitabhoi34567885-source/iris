# iris
# Iris Flower Classification 🌸

**CodSoft Data Science Internship — Task 3**
**By:** Sangita

## Overview

The Iris dataset is one of the most well-known datasets in machine learning. It contains measurements of 150 iris flowers from three species — **Setosa**, **Versicolor**, and **Virginica** — along with their sepal and petal dimensions.

The goal of this project is to build a machine learning model that can accurately classify an iris flower into its correct species based on four measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Dataset

- **Source:** [Iris Flower Dataset (Kaggle)](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- **Rows:** 150 (50 samples per species)
- **Features:** 4 numeric measurements (in cm)
- **Target:** Species (setosa / versicolor / virginica)

> Note: The script uses scikit-learn's built-in copy of the Iris dataset by default, so it runs without needing to download anything. If you'd rather use the Kaggle CSV directly, just uncomment the `pd.read_csv("IRIS.csv")` line in the script.

## Project Workflow

1. **Data Loading** — Load the dataset and inspect its structure.
2. **Exploratory Data Analysis (EDA)**
   - Summary statistics and class distribution
   - Pairplot to visualize relationships between features by species
   - Correlation heatmap between features
3. **Preprocessing**
   - Label encoding of the species column
   - Train/test split (80/20, stratified)
   - Feature scaling with `StandardScaler`
4. **Model Training & Comparison**
   Four models are trained and compared on accuracy:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
5. **Evaluation**
   - Accuracy, precision, recall, and F1-score for each model
   - Confusion matrix for the best-performing model
   - Feature importance plot (if Random Forest is selected as best)
6. **Prediction** — Demonstrates predicting the species of a new, unseen flower sample.

## Results

All four models perform well on this dataset since the species are fairly well separated by petal measurements in particular. The script automatically selects the best-performing model based on test accuracy and evaluates it in more detail.

| Model | Accuracy |
|---|---|
| Logistic Regression | ~93% |
| Decision Tree | ~93% |
| Random Forest | ~90% |
| SVM | ~100% |

*(Exact numbers may vary slightly by random seed / train-test split.)*

## Files in This Repository

| File | Description |
|---|---|
| `iris_classification.py` | Main script — loads data, trains models, evaluates, and predicts |
| `pairplot.png` | Pairwise feature relationships by species |
| `correlation_heatmap.png` | Correlation between numeric features |
| `confusion_matrix.png` | Confusion matrix for the best model |
| `feature_importance.png` | Feature importance (only if Random Forest wins) |
| `README.md` | Project documentation |

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python iris_classification.py
```

The script will print EDA summaries and model comparison metrics to the console, and save all plots as PNG files in the working directory.

## Key Takeaways

- Petal length and petal width are the most discriminative features for classifying iris species.
- Setosa is linearly separable from the other two species, while Versicolor and Virginica have some overlap.
- Simple models like Logistic Regression and SVM perform extremely well on this dataset due to its small size and clear class separation.

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---
*Part of the CodSoft Data Science Internship task series.*
