"""
TASK 4 - SALES PREDICTION USING PYTHON
CodSoft Data Science Internship

Goal: Predict product Sales based on advertising spend across
TV, Radio, and Newspaper channels using Linear Regression and
Random Forest Regression.

Dataset expected columns: TV, Radio, Newspaper, Sales
(the classic "Advertising.csv" dataset)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------
df = pd.read_csv("advertising.csv")   # <-- update path/filename if different

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe())

# Drop an unnamed index column if the CSV has one
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# -----------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Scatter plots: each ad channel vs Sales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
    sns.scatterplot(x=df[col], y=df["Sales"], ax=ax)
    ax.set_title(f"{col} vs Sales")
plt.tight_layout()
plt.savefig("scatter_ad_channels_vs_sales.png")
plt.close()

# Pairplot for a full picture
sns.pairplot(df)
plt.savefig("pairplot.png")
plt.close()

print("\nEDA plots saved: correlation_heatmap.png, "
      "scatter_ad_channels_vs_sales.png, pairplot.png")

# -----------------------------------------------------------
# 3. FEATURE / TARGET SPLIT
# -----------------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (mainly helps Linear Regression's coefficients be comparable;
# Random Forest is unaffected by scaling but it doesn't hurt)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------
# 4. MODEL 1: LINEAR REGRESSION
# -----------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

print("\n===== Linear Regression =====")
print("Coefficients:", dict(zip(X.columns, lr_model.coef_)))
print("Intercept:", lr_model.intercept_)
print("MAE :", mean_absolute_error(y_test, lr_preds))
print("MSE :", mean_squared_error(y_test, lr_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))
print("R2  :", r2_score(y_test, lr_preds))

# -----------------------------------------------------------
# 5. MODEL 2: RANDOM FOREST REGRESSOR
# -----------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)   # tree models don't need scaling
rf_preds = rf_model.predict(X_test)

print("\n===== Random Forest Regressor =====")
print("MAE :", mean_absolute_error(y_test, rf_preds))
print("MSE :", mean_squared_error(y_test, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("R2  :", r2_score(y_test, rf_preds))

# Feature importance from Random Forest
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("\nFeature Importances (Random Forest):\n", importances)

plt.figure(figsize=(5, 4))
importances.plot(kind="bar", color="teal")
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.close()

# -----------------------------------------------------------
# 6. ACTUAL VS PREDICTED PLOTS
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, lr_preds, alpha=0.7, color="steelblue")
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--")
axes[0].set_xlabel("Actual Sales")
axes[0].set_ylabel("Predicted Sales")
axes[0].set_title("Linear Regression: Actual vs Predicted")

axes[1].scatter(y_test, rf_preds, alpha=0.7, color="seagreen")
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--")
axes[1].set_xlabel("Actual Sales")
axes[1].set_ylabel("Predicted Sales")
axes[1].set_title("Random Forest: Actual vs Predicted")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

# -----------------------------------------------------------
# 7. MODEL COMPARISON SUMMARY
# -----------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mean_absolute_error(y_test, lr_preds),
            mean_absolute_error(y_test, rf_preds)],
    "RMSE": [np.sqrt(mean_squared_error(y_test, lr_preds)),
             np.sqrt(mean_squared_error(y_test, rf_preds))],
    "R2 Score": [r2_score(y_test, lr_preds),
                 r2_score(y_test, rf_preds)]
})
print("\n===== Model Comparison =====")
print(results)
results.to_csv("model_comparison.csv", index=False)

# -----------------------------------------------------------
# 8. PREDICT ON NEW / CUSTOM DATA (example)
# -----------------------------------------------------------
new_data = pd.DataFrame({
    "TV": [150],
    "Radio": [25],
    "Newspaper": [10]
})

new_data_scaled = scaler.transform(new_data)
lr_new_pred = lr_model.predict(new_data_scaled)
rf_new_pred = rf_model.predict(new_data)

print("\nPrediction for new ad spend (TV=150, Radio=25, Newspaper=10):")
print(f"Linear Regression predicted Sales: {lr_new_pred[0]:.2f}")
print(f"Random Forest predicted Sales:     {rf_new_pred[0]:.2f}")
