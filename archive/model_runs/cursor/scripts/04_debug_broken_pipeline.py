from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _common import OUTPUTS_DIR, RANDOM_SEED, save_exception_artifacts, save_run_metadata, setup_logging, write_text


def load_clean_dataset(logger) -> pd.DataFrame:
    cleaned = OUTPUTS_DIR / "task1_ingestion" / "cleaned_student_performance.csv"
    if not cleaned.exists():
        raise FileNotFoundError(
            f"Expected cleaned dataset at {cleaned}. Run scripts/01_ingest_and_clean.py first."
        )
    logger.info("loading_cleaned_dataset path=%s", cleaned.as_posix())
    return pd.read_csv(cleaned, low_memory=False)


def main() -> None:
    out_dir = OUTPUTS_DIR / "task4_debugging"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "task4.log")
    save_run_metadata(out_dir, extra={"task": "4_debug_broken_pipeline"})

    df = load_clean_dataset(logger)
    target = "Exam_Score"
    X = df.drop(columns=[target])
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # ---- Deliberately broken pipeline ----
    # Intentional bug: applying StandardScaler directly to a mixed-type DataFrame
    # (will fail when it hits string/categorical columns).
    broken_dir = out_dir / "broken_run"
    broken_dir.mkdir(parents=True, exist_ok=True)

    logger.info("running_broken_pipeline (expected_to_fail)=true")
    try:
        broken_pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                ("model", Ridge(random_state=RANDOM_SEED)),
            ]
        )
        broken_pipe.fit(X_train, y_train)
        preds = broken_pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        write_text(broken_dir / "unexpected_success.txt", f"Broken pipeline unexpectedly succeeded. RMSE={rmse}\n")
        logger.warning("broken_pipeline_unexpectedly_succeeded rmse=%.4f", rmse)
    except Exception as exc:  # noqa: BLE001
        logger.error("broken_pipeline_failed type=%s message=%s", type(exc).__name__, str(exc))
        save_exception_artifacts(broken_dir, exc)

    # ---- Fixed pipeline ----
    fixed_dir = out_dir / "fixed_run"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("running_fixed_pipeline")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    fixed_pipe = Pipeline(steps=[("pre", pre), ("model", Ridge(random_state=RANDOM_SEED))])
    fixed_pipe.fit(X_train, y_train)
    fixed_preds = fixed_pipe.predict(X_test)
    fixed_rmse = float(np.sqrt(mean_squared_error(y_test, fixed_preds)))
    write_text(fixed_dir / "rmse.txt", f"{fixed_rmse:.6f}\n")

    pd.DataFrame(
        {
            "row_id": X_test.index,
            "y_true": y_test.values,
            "y_pred": fixed_preds,
        }
    ).to_csv(fixed_dir / "predictions.csv", index=False, encoding="utf-8")

    debug_md = []
    debug_md.append("## Broken pipeline debugging\n\n")
    debug_md.append("- **Failure shown**: scaling mixed numeric + categorical features with `StandardScaler`.\n")
    debug_md.append("- **Detection**: exception + saved traceback under `broken_run/traceback.txt`.\n")
    debug_md.append("- **Root cause**: `StandardScaler` cannot convert string/object columns to float.\n")
    debug_md.append(
        "- **Fix**: split numeric vs categorical columns and apply `OneHotEncoder` for categoricals via `ColumnTransformer`.\n"
    )
    debug_md.append(f"- **Re-run evidence**: fixed pipeline RMSE = {fixed_rmse:.4f} saved in `fixed_run/rmse.txt`.\n")
    write_text(out_dir / "debug_report.md", "".join(debug_md))

    logger.info("task4_complete fixed_rmse=%.4f", fixed_rmse)


if __name__ == "__main__":
    main()

