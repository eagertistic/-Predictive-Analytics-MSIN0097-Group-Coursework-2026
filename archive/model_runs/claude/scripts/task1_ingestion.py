"""
Task 1: Dataset Ingestion + Schema Checks + Missingness Handling
================================================================
Spec:   Load the CSV, report structure, detect schema issues,
        handle missing values, save reports.
Success: Clean DataFrame saved, schema/missingness reports written,
         no unhandled nulls remain in the working copy.
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import os

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── logging ──────────────────────────────────────────────────────────────────
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("outputs/logs/task1.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── 1. Load ───────────────────────────────────────────────────────────────────
DATA_PATH = "StudentPerformanceFactors.csv"
log.info(f"Loading dataset from: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    log.info(f"Loaded successfully — shape: {df.shape}")
except FileNotFoundError:
    log.error(f"File not found: {DATA_PATH}")
    sys.exit(1)

# ── 2. Basic structure ────────────────────────────────────────────────────────
n_rows, n_cols = df.shape
log.info(f"Rows: {n_rows} | Columns: {n_cols}")
log.info(f"Column names: {df.columns.tolist()}")

# ── 3. Schema checks ──────────────────────────────────────────────────────────
issues = []

# Unnamed columns
unnamed = [c for c in df.columns if c.startswith("Unnamed")]
if unnamed:
    issues.append(f"Unnamed columns detected: {unnamed}")
    log.warning(f"Unnamed columns: {unnamed}")
    df.drop(columns=unnamed, inplace=True)
    log.info("Dropped unnamed columns.")
else:
    log.info("No unnamed columns found.")

# dtype summary
dtype_map = {col: str(df[col].dtype) for col in df.columns}
log.info(f"Dtypes:\n{pd.Series(dtype_map).to_string()}")

# Numeric columns that look categorical / vice versa
numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
log.info(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
log.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Expected types based on domain knowledge
EXPECTED_NUMERIC = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity", "Exam_Score",
]
EXPECTED_CATEGORICAL = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]
for col in EXPECTED_NUMERIC:
    if col in df.columns and col not in numeric_cols:
        issues.append(f"Expected numeric but got {df[col].dtype}: {col}")
        log.warning(f"Type mismatch (expected numeric): {col}")

for col in EXPECTED_CATEGORICAL:
    if col in df.columns and col not in categorical_cols:
        issues.append(f"Expected categorical but got {df[col].dtype}: {col}")
        log.warning(f"Type mismatch (expected categorical): {col}")

# ── 4. Duplicates ─────────────────────────────────────────────────────────────
n_dup = df.duplicated().sum()
log.info(f"Duplicate rows: {n_dup}")
if n_dup > 0:
    issues.append(f"{n_dup} duplicate rows detected.")
    df.drop_duplicates(inplace=True)
    log.info(f"Dropped {n_dup} duplicates. New shape: {df.shape}")
else:
    log.info("No duplicate rows.")

# ── 5. Missingness ────────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missingness_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct,
}).query("missing_count > 0")

if missingness_df.empty:
    log.info("No missing values found — no imputation required.")
else:
    log.warning(f"Missing values detected:\n{missingness_df.to_string()}")
    issues.append(f"Missing values in columns: {missingness_df.index.tolist()}")

    # Strategy:
    #   Numeric  → median (robust to skew)
    #   Categorical → mode (most frequent)
    for col in missingness_df.index:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            fill_val = df[col].median()
            strategy = "median"
        else:
            fill_val = df[col].mode()[0]
            strategy = "mode"
        df[col] = df[col].fillna(fill_val)
        log.info(f"  Filled '{col}' with {strategy} = {fill_val!r}")

remaining_nulls = df.isnull().sum().sum()
assert remaining_nulls == 0, "Nulls still present after imputation!"
log.info("All missing values handled. Zero nulls remain.")

# ── 6. Value-range sanity checks ──────────────────────────────────────────────
range_checks = {
    "Attendance":        (0, 100),
    "Exam_Score":        (0, 100),
    "Sleep_Hours":       (0, 24),
    "Hours_Studied":     (0, 168),
    "Tutoring_Sessions": (0, 100),
}
for col, (lo, hi) in range_checks.items():
    if col not in df.columns:
        continue
    out = df[(df[col] < lo) | (df[col] > hi)]
    if not out.empty:
        issues.append(f"Out-of-range values in '{col}': {len(out)} rows")
        log.warning(f"Out-of-range in '{col}': {len(out)} rows — clipping to [{lo},{hi}]")
        df[col] = df[col].clip(lo, hi)
    else:
        log.info(f"Range OK: '{col}' within [{lo},{hi}]")

# ── 7. Reports ────────────────────────────────────────────────────────────────
schema_report = {
    "source_file":       DATA_PATH,
    "rows_raw":          n_rows,
    "rows_clean":        len(df),
    "columns":           n_cols,
    "column_names":      df.columns.tolist(),
    "dtypes":            dtype_map,
    "numeric_columns":   numeric_cols,
    "categorical_columns": categorical_cols,
    "duplicates_removed": int(n_dup),
    "schema_issues":     issues,
    "seed":              SEED,
}

with open("outputs/reports/schema_report.json", "w") as f:
    json.dump(schema_report, f, indent=2)
log.info("Saved schema_report.json")

if missingness_df.empty:
    miss_out = pd.DataFrame(columns=["missing_count", "missing_pct"])
else:
    miss_out = missingness_df
miss_out.to_csv("outputs/reports/missingness_report.csv")
log.info("Saved missingness_report.csv")

df.to_csv("outputs/reports/cleaned_data.csv", index=False)
log.info(f"Saved cleaned_data.csv — shape: {df.shape}")

log.info("═" * 60)
log.info("TASK 1 COMPLETE")
log.info(f"  Final shape: {df.shape}")
log.info(f"  Issues found/handled: {len(issues)}")
for iss in issues:
    log.info(f"    • {iss}")
log.info("═" * 60)
