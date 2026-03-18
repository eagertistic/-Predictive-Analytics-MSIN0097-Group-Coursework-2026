# Antigravity Run Overview

This archived model run uses the following native structure:

- `src/` - executable task scripts
- `outputs/` - generated reports, plots, metrics, predictions, cleaned data, and debugging artifacts
- `data/` - dataset copy used by the run
- `benchmark_report.md` - top-level benchmark report

For comparison against other model runs, interpret this folder using the common categories below:

- `code` -> `src/`
- `reports` -> `benchmark_report.md`, `outputs/schema_report.txt`, `outputs/missingness_report.txt`, `outputs/eda_insights.txt`
- `plots` -> `outputs/plot1_exam_score_dist.png`, `outputs/plot2_correlation_matrix.png`, `outputs/plot3_score_by_parental_inv.png`
- `metrics` -> `outputs/metrics.json`
- `logs` -> `outputs/debugging_log.txt`
- `data` -> `data/StudentPerformanceFactors.csv`

This run is valid for comparison, though its internal structure is lighter and less segmented than the Cursor, Copilot, or root Codex workflows.
