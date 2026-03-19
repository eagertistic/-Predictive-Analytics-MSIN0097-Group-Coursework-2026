# MSIN0097 Group Coursework

This repository is the final submission and comparison workspace for an AI-assisted predictive analytics coursework project built on `data/StudentPerformanceFactors.csv`.

The root of the repository contains the canonical `Codex` workflow. It completes the same four coursework tasks end to end:

- Task 1: dataset ingestion, schema checks, and missingness handling
- Task 2: exploratory data analysis with saved plots and written insights
- Task 3: baseline modeling for `Exam_Score` with saved metrics and predictions
- Task 4: debugging a deliberately broken pipeline and confirming the fix

The project is also structured as a controlled benchmark across multiple AI systems. Alternate runs are preserved under `archive/model_runs/` so they can be compared against the same dataset, prompt structure, and deliverable expectations.

## Repository Structure

- `data/`: source dataset used by all runs
- `scripts/`: canonical Codex scripts for the four tasks plus orchestration/report generation
- `src/`: shared utilities used by the canonical workflow
- `outputs/`: generated reports, logs, metrics, predictions, plots, and task artifacts for the canonical workflow
- `archive/model_runs/`: archived outputs from other AI systems, plus a copied Codex snapshot for side-by-side comparison
- `MODEL_COMPARISON.md`: written comparison of the different model runs and the evaluation rubric

## Canonical Codex Workflow

Main entry points:

- `python scripts/task1_ingestion_schema_missingness.py`
- `python scripts/task2_eda_insights.py`
- `python scripts/task3_baseline_modeling.py`
- `python scripts/task4_debug_broken_pipeline.py`
- `python scripts/run_all.py`

Key canonical outputs:

- `outputs/benchmark_report.md`
- `outputs/task1_ingestion_schema_missingness/`
- `outputs/task2_eda_insights/`
- `outputs/task3_baseline_modeling/`
- `outputs/task4_debug_broken_pipeline/`

In the current saved results, `ridge_regression` is the strongest Task 3 baseline with MAE `0.452`, RMSE `1.804`, and R2 `0.770`.

## Comparison Layout

Archived model runs live under:

- `archive/model_runs/codex/`
- `archive/model_runs/claude/`
- `archive/model_runs/cursor/`
- `archive/model_runs/copilot/`
- `archive/model_runs/antigravity/`

Each folder preserves that model's artifacts in its own native layout, with a local `README.md` to map the contents into the common comparison categories.

## Project Purpose

The goal of this repository is not only to submit a working coursework pipeline, but also to compare how different AI systems perform when asked to solve the same practical analytics tasks. The comparison is based on saved evidence such as scripts, logs, plots, metrics, predictions, and debug artifacts rather than writing style alone.
