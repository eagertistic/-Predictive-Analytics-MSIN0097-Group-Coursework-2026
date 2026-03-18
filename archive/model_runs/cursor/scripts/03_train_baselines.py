from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from _common import OUTPUTS_DIR, RANDOM_SEED, save_run_metadata, setup_logging


def load_clean_dataset(logger) -> pd.DataFrame:
    cleaned = OUTPUTS_DIR / "task1_ingestion" / "cleaned_student_performance.csv"
    if not cleaned.exists():
        raise FileNotFoundError(
            f"Expected cleaned dataset at {cleaned}. Run scripts/01_ingest_and_clean.py first."
        )
    logger.info("loading_cleaned_dataset path=%s", cleaned.as_posix())
    return pd.read_csv(cleaned, low_memory=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def eval_regression(y_true, y_pred) -> dict:
    # Compatibility note: some sklearn builds don't support mean_squared_error(..., squared=False).
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    out_dir = OUTPUTS_DIR / "task3_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "task3.log")
    save_run_metadata(out_dir, extra={"task": "3_baseline_modeling"})

    df = load_clean_dataset(logger)

    # Target selection (explicit + justified):
    # - `Exam_Score` is numeric and appears to be the main outcome in the dataset.
    target = "Exam_Score"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target].astype(float)

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    logger.info("split_done train=%d test=%d target=%s", X_train.shape[0], X_test.shape[0], target)

    pre = build_preprocessor(X_train)

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(random_state=RANDOM_SEED),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
    }

    metrics = {
        "task_type": "regression",
        "target": target,
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "models": {},
    }

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        logger.info("training_model name=%s", name)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        m = eval_regression(y_test, preds)
        metrics["models"][name] = m
        logger.info("eval name=%s mae=%.4f rmse=%.4f r2=%.4f", name, m["mae"], m["rmse"], m["r2"])

        pred_df = pd.DataFrame(
            {
                "row_id": X_test.index,
                "y_true": y_test.values,
                "y_pred": preds,
                "model": name,
            }
        )
        pred_df.to_csv(out_dir / f"predictions_{name}.csv", index=False, encoding="utf-8")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "target": target,
                "task_type": "regression",
                "random_seed": RANDOM_SEED,
                "test_size": test_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("task3_complete")


if __name__ == "__main__":
    main()

