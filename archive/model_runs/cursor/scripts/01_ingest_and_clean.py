from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import (
    DATA_PATH,
    OUTPUTS_DIR,
    impute_missing,
    safe_read_csv,
    save_run_metadata,
    schema_checks,
    setup_logging,
    write_json,
)


def main() -> None:
    out_dir = OUTPUTS_DIR / "task1_ingestion"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "task1.log")
    save_run_metadata(out_dir, extra={"task": "1_ingestion_schema_missingness"})

    df_raw = safe_read_csv(DATA_PATH, logger)

    schema = schema_checks(df_raw)
    write_json(out_dir / "schema_report.json", schema)

    logger.info(
        "dataset_loaded rows=%d cols=%d dup_rows=%d",
        schema["row_count"],
        schema["column_count"],
        schema["duplicate_row_count"],
    )

    missing_df = pd.DataFrame(
        {
            "column": list(schema["missing_values"].keys()),
            "missing_count": [v["missing_count"] for v in schema["missing_values"].values()],
            "missing_pct": [v["missing_pct"] for v in schema["missing_values"].values()],
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])
    missing_df.to_csv(out_dir / "missingness_report.csv", index=False, encoding="utf-8")

    warnings = []
    if schema["unnamed_columns"]:
        warnings.append({"type": "unnamed_columns", "columns": schema["unnamed_columns"]})
        logger.warning("schema_issue unnamed_columns=%s", schema["unnamed_columns"])
    if schema["duplicate_column_names"]:
        warnings.append(
            {"type": "duplicate_column_names", "columns": schema["duplicate_column_names"]}
        )
        logger.warning("schema_issue duplicate_column_names=%s", schema["duplicate_column_names"])

    # Missingness handling (documented, deterministic)
    df_clean, summary, details = impute_missing(df_raw, logger=logger, add_missing_indicators=True)

    # Deduplicate (keep first) as a safe default.
    dup_count = int(df_clean.duplicated().sum())
    if dup_count:
        logger.warning("dropping_duplicate_rows count=%d", dup_count)
        df_clean = df_clean.drop_duplicates(keep="first").reset_index(drop=True)

    cleaned_path = out_dir / "cleaned_student_performance.csv"
    df_clean.to_csv(cleaned_path, index=False, encoding="utf-8")
    logger.info("cleaned_dataset_saved path=%s rows=%d cols=%d", cleaned_path.as_posix(), *df_clean.shape)

    write_json(
        out_dir / "missingness_handling.json",
        {
            "warnings": warnings,
            "imputation_summary": summary.__dict__,
            "imputation_details": details,
            "deduplicated_rows_dropped": dup_count,
        },
    )

    logger.info("task1_complete")


if __name__ == "__main__":
    main()

