"""
Task 3: Baseline Model Training + Evaluation Harness
=====================================================
Target: Exam_Score (continuous, 0-100) → Regression task.
Rationale: Exam_Score is the most natural outcome variable —
           it directly measures student performance, which is
           what all other features are meant to explain.

Models:
  1. Dummy Regressor (mean baseline) — sanity floor
  2. Ridge Regression              — linear baseline
  3. Random Forest Regressor       — stronger non-linear baseline

Split: 80/20 stratified by Exam_Score quintile (no leakage).

Metrics: MAE, RMSE, R², MAPE
Success: R² > 0.20 for Ridge; > 0.50 for Random Forest.
         All artifacts (metrics, config, predictions) saved.
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── logging ──────────────────────────────────────────────────────────────────
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("outputs/logs/task3.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── load ──────────────────────────────────────────────────────────────────────
DF_PATH = "outputs/reports/cleaned_data.csv"
log.info(f"Loading: {DF_PATH}")
df = pd.read_csv(DF_PATH)
log.info(f"Shape: {df.shape}")

TARGET = "Exam_Score"
log.info(f"Target column: {TARGET}  (regression task)")

# ── feature / target split ────────────────────────────────────────────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

NUMERIC_FEATURES = X.select_dtypes(include=[np.number]).columns.tolist()
CAT_FEATURES     = X.select_dtypes(include="object").columns.tolist()
log.info(f"Numeric features  ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
log.info(f"Categorical features ({len(CAT_FEATURES)}): {CAT_FEATURES}")

# ── train / test split (stratified by score quintile) ─────────────────────────
# Stratify to ensure balanced score distribution in both splits
y_quintile = pd.qcut(y, q=5, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y_quintile,
)
log.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
log.info(f"Train y mean={y_train.mean():.2f} std={y_train.std():.2f}")
log.info(f"Test  y mean={y_test.mean():.2f} std={y_test.std():.2f}")

# ── preprocessing pipeline ────────────────────────────────────────────────────
# Categorical ordinal encoding — order matters for many features
CAT_ORDERS = {
    "Parental_Involvement":     ["Low", "Medium", "High"],
    "Access_to_Resources":      ["Low", "Medium", "High"],
    "Motivation_Level":         ["Low", "Medium", "High"],
    "Family_Income":            ["Low", "Medium", "High"],
    "Teacher_Quality":          ["Low", "Medium", "High"],
    "Peer_Influence":           ["Negative", "Neutral", "Positive"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home":       ["Near", "Moderate", "Far"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Internet_Access":          ["No", "Yes"],
    "Learning_Disabilities":    ["No", "Yes"],
    "School_Type":              ["Public", "Private"],
    "Gender":                   ["Female", "Male"],
}

# Build per-column category lists in feature order
cat_order_lists = []
for col in CAT_FEATURES:
    if col in CAT_ORDERS:
        cat_order_lists.append(CAT_ORDERS[col])
    else:
        # fallback: alphabetical
        cat_order_lists.append(sorted(df[col].unique().tolist()))
        log.warning(f"No explicit order for categorical '{col}' — using alphabetical.")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(
            categories=cat_order_lists,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ), CAT_FEATURES),
    ],
    remainder="drop",
)

# ── metric helper ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    log.info(f"[{name}] MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "MAPE": round(mape, 4)}

all_metrics = []

# ── Model 1: Dummy (mean) ─────────────────────────────────────────────────────
log.info("─" * 50)
log.info("Training Model 1: DummyRegressor (mean strategy)")
dummy = Pipeline([
    ("pre", preprocessor),
    ("model", DummyRegressor(strategy="mean")),
])
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
all_metrics.append(evaluate("DummyRegressor", y_test, y_pred_dummy))

# ── Model 2: Ridge Regression ─────────────────────────────────────────────────
log.info("─" * 50)
log.info("Training Model 2: Ridge Regression (alpha=1.0)")
ridge = Pipeline([
    ("pre", preprocessor),
    ("model", Ridge(alpha=1.0, random_state=SEED)),
])
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
all_metrics.append(evaluate("Ridge", y_test, y_pred_ridge))

# 5-fold CV on train set
cv_r2 = cross_val_score(ridge, X_train, y_train, cv=5, scoring="r2")
log.info(f"Ridge 5-fold CV R²: {cv_r2.round(4)} | mean={cv_r2.mean():.4f}")

# ── Model 3: Random Forest ────────────────────────────────────────────────────
log.info("─" * 50)
log.info("Training Model 3: RandomForestRegressor (n_estimators=200)")
rf = Pipeline([
    ("pre", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=SEED,
    )),
])
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
all_metrics.append(evaluate("RandomForest", y_test, y_pred_rf))

cv_r2_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
log.info(f"RF 5-fold CV R²: {cv_r2_rf.round(4)} | mean={cv_r2_rf.mean():.4f}")

# ── Feature importance (RF) ───────────────────────────────────────────────────
feature_names = NUMERIC_FEATURES + CAT_FEATURES
importances = rf.named_steps["model"].feature_importances_
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
fi_df.to_csv("outputs/metrics/feature_importances.csv", index=False)
log.info(f"Top-5 RF features:\n{fi_df.head().to_string()}")

# ── Save predictions ──────────────────────────────────────────────────────────
preds_df = pd.DataFrame({
    "y_true":         y_test.values,
    "pred_dummy":     y_pred_dummy,
    "pred_ridge":     y_pred_ridge,
    "pred_rf":        y_pred_rf,
    "residual_ridge": y_test.values - y_pred_ridge,
    "residual_rf":    y_test.values - y_pred_rf,
})
preds_df.to_csv("outputs/metrics/test_predictions.csv", index=False)
log.info("Saved test_predictions.csv")

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("outputs/metrics/evaluation_metrics.csv", index=False)
log.info("Saved evaluation_metrics.csv")
log.info(f"\n{metrics_df.to_string()}")

# ── Save config ───────────────────────────────────────────────────────────────
config = {
    "target":            TARGET,
    "task_type":         "regression",
    "seed":              SEED,
    "test_size":         0.20,
    "stratify_by":       "quintile(Exam_Score)",
    "n_train":           len(X_train),
    "n_test":            len(X_test),
    "numeric_features":  NUMERIC_FEATURES,
    "categorical_features": CAT_FEATURES,
    "models": {
        "DummyRegressor": {"strategy": "mean"},
        "Ridge":          {"alpha": 1.0, "cv_folds": 5},
        "RandomForest":   {"n_estimators": 200, "min_samples_leaf": 4, "cv_folds": 5},
    },
}
with open("outputs/metrics/model_config.json", "w") as f:
    json.dump(config, f, indent=2)
log.info("Saved model_config.json")

# ── Save models ───────────────────────────────────────────────────────────────
joblib.dump(ridge, "outputs/models/ridge_pipeline.pkl")
joblib.dump(rf,    "outputs/models/rf_pipeline.pkl")
log.info("Saved model pipelines (joblib)")

# ── Success criteria check ────────────────────────────────────────────────────
ridge_r2 = metrics_df.loc[metrics_df["model"] == "Ridge", "R2"].values[0]
rf_r2    = metrics_df.loc[metrics_df["model"] == "RandomForest", "R2"].values[0]

log.info("─" * 50)
log.info(f"SUCCESS CHECK — Ridge R²={ridge_r2:.4f} (threshold 0.20): {'PASS ✓' if ridge_r2 > 0.20 else 'FAIL ✗'}")
log.info(f"SUCCESS CHECK — RF    R²={rf_r2:.4f} (threshold 0.50): {'PASS ✓' if rf_r2 > 0.50 else 'FAIL ✗'}")

log.info("═" * 60)
log.info("TASK 3 COMPLETE")
log.info("═" * 60)
