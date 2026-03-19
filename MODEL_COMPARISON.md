# Model Comparison

This repository now uses a consistent comparison layout for multiple AI model runs on the same coursework prompt and dataset.

## Experimental Design

Our practical exploration used one common dataset, `StudentPerformanceFactors.csv`, and one common multi-step prompt so that outputs from different AI agents could be compared fairly rather than impressionistically. The dataset contains 6,607 rows and 20 columns, with a mix of numeric and categorical student-performance variables such as `Hours_Studied`, `Attendance`, `Previous_Scores`, `Motivation_Level`, and the final outcome `Exam_Score`. Initial profiling showed no duplicate rows, but it did reveal limited missingness in three categorical fields: `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home`. This mattered because every agent had to work from the same raw evidence before generating insights or models.

The shared prompt throughout the pipeline was effectively the same four-part instruction: first, load the CSV safely, inspect schema and missingness, and save a cleaned dataset; second, produce exploratory data analysis with summary statistics, plots, and concise evidence-based insights; third, train a leakage-safe baseline modeling pipeline using `Exam_Score` as the target and save metrics, predictions, and configuration; fourth, create a deliberately broken pipeline, capture the failure, diagnose it, fix it, and confirm that the repaired version runs successfully. Using the same structure across Codex, Claude, Cursor, Copilot, and Antigravity made the benchmark much more controlled, because success was judged against identical task expectations rather than different writing styles.

The success criteria were task-specific. For Task 1, success meant the dataset loaded without manual intervention, schema and missingness reports were saved, and a cleaned CSV was produced. For Task 2, success meant numeric and categorical summaries existed, plots were saved, and the written insights could be traced back to those artifacts. For Task 3, success meant the target choice was justified, preprocessing stayed inside the sklearn pipeline to avoid leakage, a fixed seed was used, and metrics plus predictions were saved. For Task 4, success meant there was explicit failure evidence, a clear diagnosis of the root cause, and proof that the corrected pipeline ran afterwards.

The main limitations we observed were not only model-quality issues, but also environment and tooling issues that affected different agents in different ways. In the Codex run, Matplotlib initially failed in a headless environment because it tried to use a GUI backend; this was fixed by forcing the `Agg` backend before importing `pyplot`. The modeling task also surfaced a pandas `pd.NA` compatibility issue with `SimpleImputer`, which we fixed by normalizing categorical values while preserving `np.nan` compatibility. A sandbox-related `RandomForestRegressor(n_jobs=-1)` permission problem was solved by switching to `n_jobs=1`. In Copilot’s archived run, the main failures were dependency and version compatibility issues, including missing scikit-learn modules and an older `mean_squared_error` interface that rejected the `squared` argument; that run was repaired by installing dependencies and computing RMSE from MSE manually. Cursor’s archived output was generally strong, but it still had to build in fallbacks, for example rerouting Task 2 to use raw-plus-impute logic if the cleaned dataset was missing. Claude and Antigravity produced useful artifacts, but their limitations were more about inspectability and debugging traceability: Claude relied more heavily on mixed formats such as `.docx`, while Antigravity’s debugging write-up focused on leakage logic rather than a concrete runtime crash.

Most importantly, we did not treat polished language as evidence of correctness. We judged AI output quality by verification against saved artifacts. A high-quality answer had to be reproducible, internally consistent, and supported by files we could inspect. For data preparation, correctness was measured by schema accuracy, missingness counts, and whether remaining nulls were reduced to zero after documented handling. For EDA, quality was measured by whether claims matched saved summaries, correlations, and plots, and whether the insights avoided unsupported causal statements. For modeling, we used objective regression metrics such as MAE, RMSE, and R2 on a fixed train/test split; in the Codex run, for example, ridge regression clearly outperformed the dummy baseline and random forest with MAE 0.452, RMSE 1.804, and R2 0.770. For debugging, quality was measured by whether the failure was real, captured in logs or traceback files, correctly explained, and followed by a successful rerun of the repaired pipeline.

Overall, the benchmark shows that the best AI output is not simply the most fluent report. It is the output that preserves traceability from prompt to code to artifact to conclusion. In this project, that meant using task definitions, saved metrics, plots, logs, and rerunnable scripts as the basis for judging whether an AI system was actually correct, robust, and useful in practice.

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

- `archive/model_runs/codex/`
- `archive/model_runs/claude/`
- `archive/model_runs/cursor/`
- `archive/model_runs/copilot/`
- `archive/model_runs/antigravity/`

Codex, Claude, Cursor, Copilot, and Antigravity now contain archived run artifacts.
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
