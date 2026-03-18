"""
Task 4: Debugging a Deliberately Broken Pipeline
=================================================
Spec:   A small pipeline is constructed with FOUR deliberate bugs.
        Each bug is:
          (a) shown in BROKEN mode (captures the error)
          (b) diagnosed
          (c) fixed
          (d) re-run to confirm fix

Bugs introduced:
  B1. Wrong feature dtypes — numeric column passed as string
      → OrdinalEncoder crashes on numeric data typed as object
  B2. Data leakage — scaler fitted on full data before split
      → StatisticsWarning / silent leakage detected by inspection
  B3. Target column included in features (leakage)
      → Artificially perfect R²
  B4. Train/predict column mismatch — test set missing a column
      → ValueError in pipeline.predict()

Success: All four bugs are demonstrated, diagnosed, and fixed.
         The fixed pipeline produces sensible metrics.
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import traceback

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

# ── setup ─────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("outputs/logs/task4.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

df = pd.read_csv("outputs/reports/cleaned_data.csv")
TARGET = "Exam_Score"

NUMERIC_FEATURES = ["Hours_Studied", "Attendance", "Sleep_Hours",
                    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
CAT_FEATURES     = ["Parental_Involvement", "Motivation_Level",
                    "Internet_Access", "Gender"]

CAT_ORDER_LISTS = [
    ["Low", "Medium", "High"],   # Parental_Involvement
    ["Low", "Medium", "High"],   # Motivation_Level
    ["No", "Yes"],               # Internet_Access
    ["Female", "Male"],          # Gender
]

bug_report = {}

# ═════════════════════════════════════════════════════════════════════════════
# BUG 1: Numeric column deliberately cast to string → type mismatch in encoder
# ═════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("BUG 1: Wrong dtype — numeric column stored as string")
log.info("=" * 60)

df_bug1 = df.copy()
# Deliberately corrupt the dtype
df_bug1["Hours_Studied"] = df_bug1["Hours_Studied"].astype(str)
log.info(f"Hours_Studied dtype (corrupted): {df_bug1['Hours_Studied'].dtype}")

X_b1 = df_bug1[NUMERIC_FEATURES + CAT_FEATURES]
y_b1 = df_bug1[TARGET]
X_train_b1, X_test_b1, y_train_b1, y_test_b1 = train_test_split(
    X_b1, y_b1, test_size=0.2, random_state=SEED)

pre_b1 = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OrdinalEncoder(categories=CAT_ORDER_LISTS), CAT_FEATURES),
])
pipe_b1 = Pipeline([("pre", pre_b1), ("model", Ridge())])

b1_error = None
try:
    pipe_b1.fit(X_train_b1, y_train_b1)
    log.error("BUG 1: Expected an error but none was raised!")
except Exception as e:
    b1_error = str(e)
    log.warning(f"BUG 1 DETECTED — Error: {type(e).__name__}: {e}")

log.info("BUG 1 DIAGNOSIS: StandardScaler cannot compute mean/std on string dtype.")
log.info("BUG 1 FIX: Ensure numeric columns are cast to float before pipeline.")

# ── Fix ───────────────────────────────────────────────────────────────────────
df_fix1 = df_bug1.copy()
df_fix1["Hours_Studied"] = pd.to_numeric(df_fix1["Hours_Studied"], errors="coerce")
log.info(f"Hours_Studied dtype (fixed): {df_fix1['Hours_Studied'].dtype}")

X_f1 = df_fix1[NUMERIC_FEATURES + CAT_FEATURES]
y_f1 = df_fix1[TARGET]
X_tr_f1, X_te_f1, y_tr_f1, y_te_f1 = train_test_split(X_f1, y_f1, test_size=0.2, random_state=SEED)

pre_f1 = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OrdinalEncoder(categories=CAT_ORDER_LISTS), CAT_FEATURES),
])
pipe_f1 = Pipeline([("pre", pre_f1), ("model", Ridge())])
pipe_f1.fit(X_tr_f1, y_tr_f1)
r2_f1 = r2_score(y_te_f1, pipe_f1.predict(X_te_f1))
log.info(f"BUG 1 FIX CONFIRMED — R²={r2_f1:.4f} (pipeline runs successfully)")

bug_report["bug1"] = {
    "description":  "Numeric column 'Hours_Studied' cast to string dtype",
    "error_type":   "ValueError in StandardScaler (cannot handle string)",
    "error_msg":    b1_error,
    "detection":    "try/except around pipe.fit() captured ValueError",
    "fix":          "pd.to_numeric(col, errors='coerce') before pipeline",
    "r2_after_fix": round(r2_f1, 4),
}

# ═════════════════════════════════════════════════════════════════════════════
# BUG 2: Data leakage — scaler fitted on FULL dataset before train/test split
# ═════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("BUG 2: Data leakage — scaler fitted on full data before split")
log.info("=" * 60)

df_bug2 = df[NUMERIC_FEATURES + CAT_FEATURES + [TARGET]].copy()

# LEAK: fit scaler on full dataset
leaky_scaler = StandardScaler()
df_bug2[NUMERIC_FEATURES] = leaky_scaler.fit_transform(df_bug2[NUMERIC_FEATURES])

X_b2 = df_bug2[NUMERIC_FEATURES + CAT_FEATURES]
y_b2 = df_bug2[TARGET]
X_tr_b2, X_te_b2, y_tr_b2, y_te_b2 = train_test_split(
    X_b2, y_b2, test_size=0.2, random_state=SEED)

# Encode cats manually for this test
from sklearn.preprocessing import OrdinalEncoder as OE
enc = OE(categories=CAT_ORDER_LISTS, handle_unknown="use_encoded_value", unknown_value=-1)
X_tr_b2_enc = np.hstack([X_tr_b2[NUMERIC_FEATURES].values,
                          enc.fit_transform(X_tr_b2[CAT_FEATURES])])
X_te_b2_enc = np.hstack([X_te_b2[NUMERIC_FEATURES].values,
                          enc.transform(X_te_b2[CAT_FEATURES])])

from sklearn.linear_model import Ridge as R
model_leak = R().fit(X_tr_b2_enc, y_tr_b2)
r2_leak = r2_score(y_te_b2, model_leak.predict(X_te_b2_enc))
log.warning(f"BUG 2 DETECTED: Leaky pipeline R² = {r2_leak:.4f}")
log.info("BUG 2 DIAGNOSIS: Scaler learned test-set statistics; test evaluation "
         "is overly optimistic — test mean/std leaked into training.")

# ── Fix: use Pipeline so scaler only sees train fold ─────────────────────────
df_fix2 = df[NUMERIC_FEATURES + CAT_FEATURES + [TARGET]].copy()
X_f2 = df_fix2[NUMERIC_FEATURES + CAT_FEATURES]
y_f2 = df_fix2[TARGET]
X_tr_f2, X_te_f2, y_tr_f2, y_te_f2 = train_test_split(
    X_f2, y_f2, test_size=0.2, random_state=SEED)

pre_f2 = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OrdinalEncoder(categories=CAT_ORDER_LISTS,
                           handle_unknown="use_encoded_value",
                           unknown_value=-1), CAT_FEATURES),
])
pipe_f2 = Pipeline([("pre", pre_f2), ("model", Ridge())])
pipe_f2.fit(X_tr_f2, y_tr_f2)
r2_no_leak = r2_score(y_te_f2, pipe_f2.predict(X_te_f2))
log.info(f"BUG 2 FIX CONFIRMED — Leak-free R² = {r2_no_leak:.4f}")
log.info(f"  Difference (leak vs clean): {r2_leak - r2_no_leak:+.4f}")

bug_report["bug2"] = {
    "description": "StandardScaler fitted on full dataset before train/test split",
    "error_type":  "Silent data leakage (no exception; just inflated metrics)",
    "detection":   "Comparing leaky vs pipeline R²; leaky scaler inspected",
    "leaky_r2":    round(r2_leak, 4),
    "clean_r2":    round(r2_no_leak, 4),
    "fix":         "Use sklearn Pipeline so preprocessor only fits on training data",
}

# ═════════════════════════════════════════════════════════════════════════════
# BUG 3: Target column accidentally included in features → perfect R²
# ═════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("BUG 3: Target column leaked into feature matrix")
log.info("=" * 60)

# Accidentally include Exam_Score in X
X_b3_leaky = df[NUMERIC_FEATURES + [TARGET]]   # ← bug: TARGET included
y_b3 = df[TARGET]
X_tr_b3, X_te_b3, y_tr_b3, y_te_b3 = train_test_split(
    X_b3_leaky, y_b3, test_size=0.2, random_state=SEED)

scaler_b3 = StandardScaler()
X_tr_b3_s = scaler_b3.fit_transform(X_tr_b3)
X_te_b3_s = scaler_b3.transform(X_te_b3)
model_b3  = R().fit(X_tr_b3_s, y_tr_b3)
r2_b3     = r2_score(y_te_b3, model_b3.predict(X_te_b3_s))
log.warning(f"BUG 3 DETECTED: R² = {r2_b3:.6f} (suspiciously perfect)")
log.info("BUG 3 DIAGNOSIS: Target 'Exam_Score' is in the feature matrix — "
         "the model trivially predicts the target from itself.")

# ── Fix ───────────────────────────────────────────────────────────────────────
X_f3 = df[NUMERIC_FEATURES]   # target excluded
y_f3 = df[TARGET]
X_tr_f3, X_te_f3, y_tr_f3, y_te_f3 = train_test_split(
    X_f3, y_f3, test_size=0.2, random_state=SEED)

pre_f3 = ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES)])
pipe_f3 = Pipeline([("pre", pre_f3), ("model", Ridge())])
pipe_f3.fit(X_tr_f3, y_tr_f3)
r2_f3 = r2_score(y_te_f3, pipe_f3.predict(X_te_f3))
log.info(f"BUG 3 FIX CONFIRMED — R² without target leakage = {r2_f3:.4f}")

bug_report["bug3"] = {
    "description":        "TARGET column 'Exam_Score' included in feature matrix X",
    "error_type":         "Silent perfect leakage (R² ≈ 1.0)",
    "detection":          f"R²={r2_b3:.6f} flagged as implausibly perfect",
    "fix":                "X = df.drop(columns=[TARGET]) — never include target in features",
    "r2_leaky":           round(r2_b3, 6),
    "r2_after_fix":       round(r2_f3, 4),
}

# ═════════════════════════════════════════════════════════════════════════════
# BUG 4: Column mismatch — test set has a column dropped that train expects
# ═════════════════════════════════════════════════════════════════════════════
log.info("=" * 60)
log.info("BUG 4: Test set missing a column that the fitted pipeline expects")
log.info("=" * 60)

X_full = df[NUMERIC_FEATURES + CAT_FEATURES]
y_full = df[TARGET]
X_tr_b4, X_te_b4, y_tr_b4, y_te_b4 = train_test_split(
    X_full, y_full, test_size=0.2, random_state=SEED)

pre_b4 = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OrdinalEncoder(categories=CAT_ORDER_LISTS,
                           handle_unknown="use_encoded_value",
                           unknown_value=-1), CAT_FEATURES),
])
pipe_b4 = Pipeline([("pre", pre_b4), ("model", Ridge())])
pipe_b4.fit(X_tr_b4, y_tr_b4)

# Deliberately drop a column from the test set
X_te_b4_broken = X_te_b4.drop(columns=["Hours_Studied"])
log.warning(f"Test set columns: {X_te_b4_broken.columns.tolist()}")
log.warning(f"Train expected:   {X_full.columns.tolist()}")

b4_error = None
try:
    pipe_b4.predict(X_te_b4_broken)
    log.error("BUG 4: Expected an error but none was raised!")
except Exception as e:
    b4_error = str(e)
    log.warning(f"BUG 4 DETECTED — {type(e).__name__}: {e}")

log.info("BUG 4 DIAGNOSIS: ColumnTransformer selects columns by name; "
         "dropping 'Hours_Studied' from test set causes a ValueError.")

# ── Fix: restore the missing column ──────────────────────────────────────────
X_te_b4_fixed = X_te_b4.copy()  # use full test set
r2_b4_fixed = r2_score(y_te_b4, pipe_b4.predict(X_te_b4_fixed))
log.info(f"BUG 4 FIX CONFIRMED — R² with correct test columns = {r2_b4_fixed:.4f}")

bug_report["bug4"] = {
    "description": "Test set missing column 'Hours_Studied' that pipeline expects",
    "error_type":  "ValueError in ColumnTransformer during predict()",
    "error_msg":   b4_error,
    "detection":   "try/except around pipe.predict() captured ValueError",
    "fix":         "Ensure train and test feature sets have identical columns",
    "r2_after_fix": round(r2_b4_fixed, 4),
}

# ── Save bug report ───────────────────────────────────────────────────────────
with open("outputs/reports/bug_report.json", "w") as f:
    json.dump(bug_report, f, indent=2)
log.info("Saved bug_report.json")

log.info("=" * 60)
log.info("TASK 4 SUMMARY")
for k, v in bug_report.items():
    log.info(f"  {k.upper()}: {v['description']}")
    log.info(f"    Error type : {v['error_type']}")
    log.info(f"    Detection  : {v['detection']}")
    log.info(f"    Fix        : {v['fix']}")
log.info("=" * 60)
log.info("TASK 4 COMPLETE")
