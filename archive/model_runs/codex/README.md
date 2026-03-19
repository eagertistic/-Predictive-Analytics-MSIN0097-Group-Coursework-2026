# Codex Archived Snapshot

This folder is a copied snapshot of the canonical `Codex` coursework workflow from the repository root.

It is stored here so `Codex` can be compared side by side with the other archived model runs using the same `archive/model_runs/<model>/` pattern.

## Structure

- `data/` - source dataset used by the Codex run
- `scripts/` - canonical task scripts and orchestration
- `src/` - shared utility code
- `outputs/` - benchmark report, logs, plots, metrics, predictions, and debugging artifacts

## Comparison Mapping

- `scripts` -> runnable workflow for Tasks 1-4
- `outputs/benchmark_report.md` -> benchmark summary
- `outputs/task1_ingestion_schema_missingness/` -> ingestion and cleaning evidence
- `outputs/task2_eda_insights/` -> EDA summaries, plots, and insights
- `outputs/task3_baseline_modeling/` -> modeling metrics, predictions, and config
- `outputs/task4_debug_broken_pipeline/` -> broken-run traceback and fix confirmation

The repository root remains the primary submission version. This archived folder is only a comparison copy.
