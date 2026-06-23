"""
House Price Prediction - Model Training Pipeline
===================================================
Dataset: California Housing (sklearn built-in, standard replacement for the
deprecated Boston Housing dataset).

This script:
1. Loads the dataset
2. Does basic EDA (saved as PNG plots)
3. Preprocesses (scaling)
4. Trains multiple regression models
5. Evaluates and picks the best one
6. Saves the trained model + scaler as .pkl files (ready for S3 upload)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
print("Loading California Housing dataset...")
try:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # includes target column 'MedHouseVal'
except Exception as e:
    # Fallback for offline/sandboxed environments without internet access.
    # On your own machine with internet access, the try block above will
    # succeed and this fallback will never run.
    print(f"  (Could not download dataset: {e})")
    print("  Generating a synthetic dataset with the same schema instead...")
    rng = np.random.RandomState(42)
    n = 20000
    med_inc = rng.gamma(5, 1.0, n).clip(0.5, 15)
    house_age = rng.uniform(1, 52, n)
    ave_rooms = rng.normal(5.4, 1.5, n).clip(1, 15)
    ave_bedrms = (ave_rooms * rng.uniform(0.15, 0.25, n)).clip(0.5, 5)
    population = rng.gamma(3, 400, n).clip(3, 35000)
    ave_occup = rng.normal(3.0, 1.0, n).clip(0.5, 10)
    latitude = rng.uniform(32.5, 42.0, n)
    longitude = rng.uniform(-124.3, -114.3, n)

    target = (
        0.45 * med_inc
        + 0.002 * house_age
        - 0.02 * ave_occup
        + 0.05 * ave_rooms
        - 0.08 * ave_bedrms
        - 0.03 * np.abs(latitude - 36)
        + rng.normal(0, 0.5, n)
    )
    target = target.clip(0.15, 5.0)

    df = pd.DataFrame({
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
        "MedHouseVal": target,
    })

print(f"Shape: {df.shape}")
print(df.head())
df.to_csv(os.path.join("data", "housing.csv"), index=False)

# -----------------------------------------------------------------------
# 2. BASIC EDA
# -----------------------------------------------------------------------
print("\nGenerating EDA plots...")

plt.figure(figsize=(10, 6))
sns.histplot(df["MedHouseVal"], kde=True, bins=40)
plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value ($100,000s)")
plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close()

print("EDA plots saved to outputs/")
print("\nDataset summary statistics:")
print(df.describe().to_string())

# -----------------------------------------------------------------------
# 3. PREPROCESSING
# -----------------------------------------------------------------------
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# 4. TRAIN MULTIPLE MODELS
# -----------------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "RandomForest": RandomForestRegressor(
        n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
    ),
}

results = []
print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
    print(f"  {name:<18} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

results_df = pd.DataFrame(results).sort_values("rmse")
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))

# -----------------------------------------------------------------------
# 5. PICK BEST MODEL + SAVE ARTIFACTS
# -----------------------------------------------------------------------
best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

joblib.dump(best_model, os.path.join(MODEL_DIR, "house_price_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Save feature names + metadata so the loader knows the input contract
metadata = {
    "best_model": best_model_name,
    "features": list(X.columns),
    "target": "MedHouseVal (median house value, in $100,000s)",
    "metrics": results_df.iloc[0].to_dict(),
    "sklearn_dataset": "fetch_california_housing",
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Feature importance plot (for tree-based models)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    importances.plot(kind="bar")
    plt.title(f"Feature Importance - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

print(f"\nSaved model artifacts to {MODEL_DIR}/:")
for f in os.listdir(MODEL_DIR):
    print(f"  - {f}")

print("\nDone. These files are what you upload to your S3 bucket:")
print("  models/house_price_model.pkl")
print("  models/scaler.pkl")
print("  models/metadata.json")
print("  data/housing.csv  (optional, the dataset used)")
