from __future__ import annotations

import json
from pathlib import Path

from _common import OUTPUTS_DIR, PROJECT_ROOT, setup_logging, write_text


def read_text_if_exists(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "benchmark_report.log")

    task1_dir = out_dir / "task1_ingestion"
    task2_dir = out_dir / "task2_eda"
    task3_dir = out_dir / "task3_modeling"
    task4_dir = out_dir / "task4_debugging"

    schema = read_json_if_exists(task1_dir / "schema_report.json")
    miss = read_text_if_exists(task1_dir / "missingness_report.csv")
    metrics = read_json_if_exists(task3_dir / "metrics.json")
    debug_report = read_text_if_exists(task4_dir / "debug_report.md")

    lines: list[str] = []
    lines.append("## Benchmark / run report\n\n")
    lines.append(f"- **Project root**: `{PROJECT_ROOT}`\n")
    lines.append("- **All artifacts**: saved under `outputs/`.\n\n")

    # ---- Task 1 ----
    lines.append("## Task 1) Dataset ingestion + schema checks + missingness handling\n\n")
    lines.append("### Specification\n")
    lines.append(
        "- Load `data/StudentPerformanceFactors.csv` safely; report shape, columns, dtypes, duplicates, missingness; "
        "run schema sanity checks; impute missing values deterministically; save reports and cleaned CSV.\n\n"
    )
    lines.append("### Success criteria\n")
    lines.append(
        "- `outputs/task1_ingestion/schema_report.json` exists and contains counts/dtypes.\n"
        "- `outputs/task1_ingestion/missingness_report.csv` exists.\n"
        "- `outputs/task1_ingestion/cleaned_student_performance.csv` exists.\n\n"
    )
    if schema:
        lines.append("### Results\n")
        lines.append(
            f"- **Rows / cols**: {schema.get('row_count')} / {schema.get('column_count')}\n"
            f"- **Duplicate rows**: {schema.get('duplicate_row_count')}\n"
            f"- **Unnamed columns**: {schema.get('unnamed_columns')}\n\n"
        )
    else:
        lines.append("### Results\n- (missing) `schema_report.json` not found.\n\n")
    lines.append("### Evidence\n")
    lines.append("- Reports: `outputs/task1_ingestion/schema_report.json`, `outputs/task1_ingestion/missingness_report.csv`\n")
    lines.append("- Cleaned data: `outputs/task1_ingestion/cleaned_student_performance.csv`\n")
    lines.append("- Logs: `outputs/task1_ingestion/task1.log`\n\n")
    lines.append("### Failures + detection + correction\n")
    lines.append("- None observed if the artifacts above exist; if missing, check `outputs/task1_ingestion/task1.log`.\n\n")
    lines.append("### Reproducibility\n")
    lines.append(
        "- Deterministic imputations (median/mode) + deterministic deduplication (keep first). "
        "Run metadata saved in `outputs/task1_ingestion/run_metadata.json`.\n\n"
    )

    # ---- Task 2 ----
    lines.append("## Task 2) EDA and insight generation\n\n")
    lines.append("### Specification\n")
    lines.append(
        "- Produce summary stats and informative plots; save them; write concise insights grounded in the outputs.\n\n"
    )
    lines.append("### Success criteria\n")
    lines.append(
        "- Summary CSVs exist under `outputs/task2_eda/`.\n"
        "- Plots exist under `outputs/task2_eda/plots/`.\n"
        "- `outputs/task2_eda/insights.md` exists.\n\n"
    )
    lines.append("### Evidence\n")
    lines.append("- Summaries: `outputs/task2_eda/numeric_describe.csv`, `outputs/task2_eda/numeric_correlation.csv`\n")
    lines.append("- Plots: `outputs/task2_eda/plots/*.png`\n")
    lines.append("- Insights: `outputs/task2_eda/insights.md`\n")
    lines.append("- Logs: `outputs/task2_eda/task2.log`\n\n")
    lines.append("### Failures + detection + correction\n")
    lines.append(
        "- If `cleaned_student_performance.csv` is missing, task2 falls back to raw+impute and logs a warning.\n\n"
    )
    lines.append("### Reproducibility\n")
    lines.append("- Plots/summaries are deterministic given the cleaned input; run metadata saved in `outputs/task2_eda/run_metadata.json`.\n\n")

    # ---- Task 3 ----
    lines.append("## Task 3) Baseline model training + evaluation harness\n\n")
    lines.append("### Specification\n")
    lines.append(
        "- Select a sensible target; train a simple baseline and a stronger baseline using a train/test split; "
        "avoid leakage via preprocessing inside a pipeline; save metrics/config/predictions.\n\n"
    )
    lines.append("### Success criteria\n")
    lines.append(
        "- `outputs/task3_modeling/metrics.json` exists with at least two models.\n"
        "- `outputs/task3_modeling/predictions_*.csv` exist.\n\n"
    )
    lines.append("### Target choice\n")
    lines.append("- **Chosen target**: `Exam_Score` (numeric) → regression.\n\n")
    lines.append("### Results\n")
    if metrics and "models" in metrics:
        for model_name, m in metrics["models"].items():
            lines.append(
                f"- **{model_name}**: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}\n"
            )
        lines.append("\n")
    else:
        lines.append("- (missing) `metrics.json` not found.\n\n")
    lines.append("### Evidence\n")
    lines.append("- Metrics/config: `outputs/task3_modeling/metrics.json`, `outputs/task3_modeling/config.json`\n")
    lines.append("- Predictions: `outputs/task3_modeling/predictions_dummy_mean.csv`, `...ridge.csv`, `...rf.csv`\n")
    lines.append("- Logs: `outputs/task3_modeling/task3.log`\n\n")
    lines.append("### Failures + detection + correction\n")
    lines.append("- None expected; if training fails, errors will be in `outputs/task3_modeling/task3.log`.\n\n")
    lines.append("### Reproducibility\n")
    lines.append(
        "- Fixed `random_seed=42` for split and models; preprocessing encapsulated in pipelines to avoid leakage.\n\n"
    )

    # ---- Task 4 ----
    lines.append("## Task 4) Debugging a deliberately broken pipeline\n\n")
    lines.append("### Specification\n")
    lines.append(
        "- Create/use a broken pipeline that fails on this dataset; capture failure evidence; diagnose cause; fix; re-run and confirm.\n\n"
    )
    lines.append("### Success criteria\n")
    lines.append(
        "- Broken run produces a saved traceback under `outputs/task4_debugging/broken_run/`.\n"
        "- Fixed run produces predictions + a metric under `outputs/task4_debugging/fixed_run/`.\n\n"
    )
    lines.append("### Evidence\n")
    lines.append("- Failure: `outputs/task4_debugging/broken_run/traceback.txt`\n")
    lines.append("- Fix + rerun: `outputs/task4_debugging/fixed_run/rmse.txt`, `.../predictions.csv`\n")
    lines.append("- Logs: `outputs/task4_debugging/task4.log`\n\n")
    lines.append("### Failures + detection + correction\n")
    if debug_report:
        lines.append(debug_report.strip() + "\n\n")
    else:
        lines.append("- Debug report missing; run `scripts/04_debug_broken_pipeline.py`.\n\n")
    lines.append("### Reproducibility\n")
    lines.append("- Fixed pipeline uses deterministic seed and a fully specified preprocessing graph; artifacts saved for verification.\n\n")

    report_path = out_dir / "BENCHMARK_REPORT.md"
    write_text(report_path, "".join(lines))
    logger.info("benchmark_report_written path=%s", report_path.as_posix())


if __name__ == "__main__":
    main()

