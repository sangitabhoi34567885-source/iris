
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)


from sklearn.datasets import load_iris

iris_raw = load_iris()
df = pd.DataFrame(iris_raw.data, columns=iris_raw.feature_names)
df["species"] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

# Rename columns to simpler names
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSpecies distribution:\n", df["species"].value_counts())
print("\nSummary statistics:\n", df.describe())


sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Pairwise Feature Relationships by Species", y=1.02)
plt.savefig("pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved pairplot.png and correlation_heatmap.png")

X = df.drop(columns="species")
y = df["species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # setosa=0, versicolor=1, virginica=2

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")

best_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved confusion_matrix.png")

if best_model_name == "Random Forest":
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    importances.plot(kind="bar", color="teal")
    plt.title("Feature Importance - Random Forest")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    print("\nFeature Importances:\n", importances)
    print("Saved feature_importance.png")

sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # example measurements
sample_scaled = scaler.transform(sample)
sample_pred = best_model.predict(sample_scaled)
print("\nPrediction for sample", sample.tolist(), "->", le.inverse_transform(sample_pred)[0])

print("\nDone.")
