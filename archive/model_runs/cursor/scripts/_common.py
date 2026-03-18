from __future__ import annotations

import json
import logging
import os
import platform
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "StudentPerformanceFactors.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def setup_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("logging_started path=%s", log_path.as_posix())
    return logger


def save_run_metadata(out_dir: Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {
        "utc_timestamp": utc_now_iso(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "data_path": str(DATA_PATH),
        "random_seed": RANDOM_SEED,
    }
    if extra:
        meta.update(extra)
    write_json(out_dir / "run_metadata.json", meta)
    return meta


def safe_read_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    logger.info("reading_csv path=%s", path.as_posix())
    df = pd.read_csv(path, low_memory=False)

    # Normalize any empty-string-like entries to NaN for object columns.
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df[obj_cols] = df[obj_cols].replace(
            to_replace=[r"^\s*$", r"^\s*(na|n/a|null|none)\s*$"],
            value=np.nan,
            regex=True,
        )
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    return df


def schema_checks(df: pd.DataFrame) -> Dict[str, Any]:
    unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "unnamed_columns": unnamed,
        "duplicate_column_names": duplicate_cols,
        "duplicate_row_count": int(df.duplicated().sum()),
        "missing_values": {
            str(c): {
                "missing_count": int(df[c].isna().sum()),
                "missing_pct": float(df[c].isna().mean()),
            }
            for c in df.columns
        },
    }


@dataclass(frozen=True)
class ImputationSummary:
    numeric_strategy: str
    categorical_strategy: str
    created_missing_indicators: int
    rows_before: int
    rows_after: int


def infer_column_types(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def impute_missing(
    df: pd.DataFrame, *, logger: logging.Logger, add_missing_indicators: bool = True
) -> Tuple[pd.DataFrame, ImputationSummary, Dict[str, Any]]:
    df2 = df.copy()
    rows_before = int(df2.shape[0])

    numeric_cols, categorical_cols = infer_column_types(df2)

    indicator_cols = []
    if add_missing_indicators:
        for c in df2.columns:
            if df2[c].isna().any():
                ind = f"{c}__was_missing"
                df2[ind] = df2[c].isna().astype(int)
                indicator_cols.append(ind)
        logger.info("missing_indicators_created count=%d", len(indicator_cols))

    numeric_fill_values = {}
    for c in numeric_cols:
        if df2[c].isna().any():
            val = float(df2[c].median())
            numeric_fill_values[c] = val
            df2[c] = df2[c].fillna(val)

    categorical_fill_values = {}
    for c in categorical_cols:
        if df2[c].isna().any():
            mode = df2[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            categorical_fill_values[c] = str(fill)
            df2[c] = df2[c].fillna(fill)

    rows_after = int(df2.shape[0])
    summary = ImputationSummary(
        numeric_strategy="median",
        categorical_strategy="mode (or 'Unknown' if empty)",
        created_missing_indicators=len(indicator_cols),
        rows_before=rows_before,
        rows_after=rows_after,
    )

    details = {
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "indicator_columns": indicator_cols,
    }
    return df2, summary, details


def save_exception_artifacts(out_dir: Path, exc: BaseException) -> None:
    ensure_dir(out_dir)
    write_text(out_dir / "exception_type.txt", type(exc).__name__)
    write_text(out_dir / "exception_message.txt", str(exc))
    write_text(out_dir / "traceback.txt", traceback.format_exc())

