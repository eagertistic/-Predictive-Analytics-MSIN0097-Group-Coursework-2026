# MSIN0097 Group Coursework

This repository is organized around one canonical Python workflow for the four coursework tasks, using `data/StudentPerformanceFactors.csv` as the source dataset.

## Structure

- `data/`: input dataset
- `scripts/`: runnable task scripts plus orchestration scripts
- `src/`: shared utility code used by the scripts
- `outputs/`: generated logs, reports, plots, metrics, predictions, and benchmark summary
- `archive/`: preserved alternate AI-generated work that is no longer the primary project structure

## Main entry points

- `python scripts/task1_ingestion_schema_missingness.py`
- `python scripts/task2_eda_insights.py`
- `python scripts/task3_baseline_modeling.py`
- `python scripts/task4_debug_broken_pipeline.py`
- `python scripts/run_all.py`

All generated artifacts are written under `outputs/`.
