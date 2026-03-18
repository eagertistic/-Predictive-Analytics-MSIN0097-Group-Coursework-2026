# Model Comparison

This repository now uses a consistent comparison layout for multiple AI model runs on the same coursework prompt and dataset.

## Standard Comparison Layout

### Canonical Submission

The repository root is the main `Codex` submission:

- `data/`
- `scripts/`
- `src/`
- `outputs/`
- `README.md`
- `MODEL_COMPARISON.md`

### Archived Alternate Model Runs

All non-canonical model runs live under:

- `archive/model_runs/<model_name>/`

The model folders currently tracked are:

- `archive/model_runs/claude/`
- `archive/model_runs/cursor/`
- `archive/model_runs/copilot/`
- `archive/model_runs/antigravity/`

Claude, Cursor, Copilot, and Antigravity now contain archived run artifacts.
Each archived model folder also includes a `README.md` that maps its native layout into the same comparison categories.

## Why This Structure Works

- `Codex` remains the clean canonical coursework workflow at the repository root.
- Every alternate AI run is stored in one predictable archive location.
- New models can be added without changing the comparison method.
- `MODEL_COMPARISON.md` can refer to the same folder pattern for every model.

## Current Evidence By Model

### Codex

Primary workflow at repository root:

- `scripts/run_all.py`
- `scripts/task1_ingestion_schema_missingness.py`
- `scripts/task2_eda_insights.py`
- `scripts/task3_baseline_modeling.py`
- `scripts/task4_debug_broken_pipeline.py`
- `outputs/benchmark_report.md`
- `outputs/task1_ingestion_schema_missingness.log`
- `outputs/task2_eda_insights.log`
- `outputs/task3_baseline_modeling.log`
- `outputs/task4_debug_broken_pipeline.log`
- `outputs/task2_eda_insights/plots/exam_score_distribution.png`
- `outputs/task2_eda_insights/plots/previous_scores_vs_exam_score.png`
- `outputs/task2_eda_insights/plots/exam_score_by_motivation.png`
- `outputs/task2_eda_insights/plots/correlation_heatmap.png`

### Claude

Archived workflow under:

- `archive/model_runs/claude/scripts/`
- `archive/model_runs/claude/reports/`
- `archive/model_runs/claude/plots/`
- `archive/model_runs/claude/metrics/`
- `archive/model_runs/claude/models/`

Key files currently present:

- `archive/model_runs/claude/README.md`
- `archive/model_runs/claude/Read Me (Claude Code version).md`
- `archive/model_runs/claude/BENCHMARK_REPORT.md.docx`
- `archive/model_runs/claude/reports/run_summary.json`
- `archive/model_runs/claude/reports/bug_report.json`

### Cursor

Archived workflow under:

- `archive/model_runs/cursor/scripts/`
- `archive/model_runs/cursor/outputs/`
- `archive/model_runs/cursor/data/`

Key files currently present:

- `archive/model_runs/cursor/README.md`
- `archive/model_runs/cursor/outputs/BENCHMARK_REPORT.md`
- `archive/model_runs/cursor/outputs/task1_ingestion/schema_report.json`
- `archive/model_runs/cursor/outputs/task3_modeling/metrics.json`
- `archive/model_runs/cursor/outputs/task4_debugging/debug_report.md`

### Copilot

Archived workflow under:

- `archive/model_runs/copilot/scripts/`
- `archive/model_runs/copilot/outputs/`
- `archive/model_runs/copilot/data/`
- `archive/model_runs/copilot/screenshots/`

Key files currently present:

- `archive/model_runs/copilot/README.md`
- `archive/model_runs/copilot/experiment_log.md`
- `archive/model_runs/copilot/outputs/benchmark_report.txt`
- `archive/model_runs/copilot/outputs/schema_report.json`
- `archive/model_runs/copilot/outputs/model_metrics.json`

### Antigravity

Archived workflow under:

- `archive/model_runs/antigravity/src/`
- `archive/model_runs/antigravity/outputs/`
- `archive/model_runs/antigravity/data/`

Key files currently present:

- `archive/model_runs/antigravity/README.md`
- `archive/model_runs/antigravity/benchmark_report.md`
- `archive/model_runs/antigravity/outputs/schema_report.txt`
- `archive/model_runs/antigravity/outputs/metrics.json`
- `archive/model_runs/antigravity/outputs/eda_insights.txt`

## Scoring Rubric

Use the same rubric for every model:

- `5` = excellent
- `4` = strong
- `3` = acceptable
- `2` = weak
- `1` = poor

Judge each model against the same prompt, dataset, and deliverable expectations.

## Summary Table

This table reflects the current repository contents in this workspace.

| Criterion | Codex | Claude | Cursor | Copilot | Antigravity | Notes |
|---|---:|---:|---:|---:|---:|---|
| Folder structure | 5 | 4 | 4 | 4 | 4 | Codex is the canonical root submission. Claude, Cursor, Copilot, and Antigravity are all now consistently archived under `archive/model_runs/<model>/`, though their internal layouts still differ from each other. |
| Prompt compliance | 5 | 4 | 5 | 4 | 4 | Codex clearly covers all four tasks. Cursor shows strong evidence for ingestion, EDA, modeling, debugging, and benchmark reporting. Claude, Copilot, and Antigravity also cover the required workflow. |
| Reproducibility | 5 | 3 | 5 | 4 | 4 | Codex and Cursor are straightforward to rerun from scripts plus saved outputs. Copilot and Antigravity are also reproducible, while Claude is more dependent on mixed file formats. |
| Logging and evidence | 5 | 4 | 5 | 4 | 3 | Codex keeps logs and outputs neatly together. Cursor includes per-task logs, metadata, benchmark output, plots, predictions, and debug artifacts. |
| Task 1 quality | 5 | 4 | 5 | 4 | 4 | Cursor provides a clean schema report, missingness handling summary, cleaned dataset, and metadata. |
| Task 2 EDA quality | 4 | 5 | 5 | 4 | 3 | Claude has the broadest saved EDA plot set. Cursor also provides a strong EDA package with multiple summaries, grouped analyses, and many saved plots. |
| Task 3 modeling quality | 5 | 4 | 5 | 4 | 4 | Cursor includes multiple baselines, metrics, saved predictions, and a clear configuration file. |
| Task 4 debugging quality | 5 | 4 | 5 | 4 | 4 | Codex has the clearest task-specific traceability. Cursor also preserves broken-run evidence, a debug report, and fixed-run outputs cleanly. |
| Report clarity | 5 | 4 | 5 | 4 | 4 | Codex and Cursor are easiest to inspect directly in repo using markdown, JSON, CSV, and logs. Claude relies more on `.docx` and `.xlsx` artifacts. |
| Overall professionalism | 5 | 4 | 5 | 4 | 4 | The repository is now much cleaner for cross-model comparison, and Cursor is now fully represented with runnable archived artifacts. |

## Interpretation Notes

- `N/A` means the model does not yet have enough real run artifacts in this workspace to score fairly.
- In the current snapshot, all four alternate model folders now contain real artifacts.
- The fairest comparison is root `Codex` versus `archive/model_runs/<model>/` for every alternate run.

## Recommendation

Use the repository root as the canonical deliverable.

Use `archive/model_runs/` as the consistent holding area for every alternative model run:

- `claude`
- `cursor`
- `copilot`
- `antigravity`

## Update Rule For Future Runs

When you add a new model run, place all of its artifacts inside exactly one folder:

- `archive/model_runs/<model_name>/`

Then update these parts of this file:

1. The "Current Evidence By Model" section
2. The relevant score column in the summary table
