# Copilot Run Overview

This archived model run uses the following native structure:

- `scripts/` - executable task scripts
- `outputs/` - generated reports, plots, metrics, logs, predictions, and cleaned data
- `data/` - dataset copy used by the run
- `screenshots/` - supporting process screenshots
- `experiment_log.md` - narrative audit log for the run

For comparison against other model runs, interpret this folder using the common categories below:

- `code` -> `scripts/`
- `reports` -> `outputs/benchmark_report.txt`, `experiment_log.md`
- `plots` -> `.png` files under `outputs/`
- `metrics` -> `outputs/model_metrics.json`, `outputs/model_config.json`
- `logs` -> `outputs/task1_ingestion.log`, `outputs/task2_eda.log`, `outputs/task3_modeling.log`, `outputs/task4_debug.log`
- `data` -> `data/StudentPerformanceFactors.csv`

The artifact naming differs from the root Codex workflow, but the evidence needed for cross-model comparison is present in this folder.
