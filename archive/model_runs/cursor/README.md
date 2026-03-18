# Cursor Run Overview

This archived model run uses the following native structure:

- `scripts/` - executable task scripts
- `outputs/` - generated reports, plots, metrics, logs, predictions, and debug artifacts
- `data/` - dataset copy used by the run

For comparison against other model runs, interpret this folder using the common categories below:

- `code` -> `scripts/`
- `reports` -> `outputs/BENCHMARK_REPORT.md`, `outputs/task4_debugging/debug_report.md`
- `plots` -> `outputs/task2_eda/plots/`
- `metrics` -> `outputs/task3_modeling/metrics.json`
- `logs` -> `outputs/task1_ingestion/task1.log`, `outputs/task2_eda/task2.log`, `outputs/task3_modeling/task3.log`, `outputs/task4_debugging/task4.log`
- `data` -> `data/StudentPerformanceFactors.csv`

Environment note:

- `scripts/02_eda.py` was adjusted to use the non-interactive Matplotlib `Agg` backend.
- `scripts/03_train_baselines.py` uses `n_jobs=1` for the random forest so it runs reliably in this Windows environment.
