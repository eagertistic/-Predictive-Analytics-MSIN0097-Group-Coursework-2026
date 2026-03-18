## Benchmark / run report

- **Project root**: `C:\Users\jackp\Desktop\MSIN0097-Group-Coursework\archive\model_runs\cursor`
- **All artifacts**: saved under `outputs/`.

## Task 1) Dataset ingestion + schema checks + missingness handling

### Specification
- Load `data/StudentPerformanceFactors.csv` safely; report shape, columns, dtypes, duplicates, missingness; run schema sanity checks; impute missing values deterministically; save reports and cleaned CSV.

### Success criteria
- `outputs/task1_ingestion/schema_report.json` exists and contains counts/dtypes.
- `outputs/task1_ingestion/missingness_report.csv` exists.
- `outputs/task1_ingestion/cleaned_student_performance.csv` exists.

### Results
- **Rows / cols**: 6607 / 20
- **Duplicate rows**: 0
- **Unnamed columns**: []

### Evidence
- Reports: `outputs/task1_ingestion/schema_report.json`, `outputs/task1_ingestion/missingness_report.csv`
- Cleaned data: `outputs/task1_ingestion/cleaned_student_performance.csv`
- Logs: `outputs/task1_ingestion/task1.log`

### Failures + detection + correction
- None observed if the artifacts above exist; if missing, check `outputs/task1_ingestion/task1.log`.

### Reproducibility
- Deterministic imputations (median/mode) + deterministic deduplication (keep first). Run metadata saved in `outputs/task1_ingestion/run_metadata.json`.

## Task 2) EDA and insight generation

### Specification
- Produce summary stats and informative plots; save them; write concise insights grounded in the outputs.

### Success criteria
- Summary CSVs exist under `outputs/task2_eda/`.
- Plots exist under `outputs/task2_eda/plots/`.
- `outputs/task2_eda/insights.md` exists.

### Evidence
- Summaries: `outputs/task2_eda/numeric_describe.csv`, `outputs/task2_eda/numeric_correlation.csv`
- Plots: `outputs/task2_eda/plots/*.png`
- Insights: `outputs/task2_eda/insights.md`
- Logs: `outputs/task2_eda/task2.log`

### Failures + detection + correction
- If `cleaned_student_performance.csv` is missing, task2 falls back to raw+impute and logs a warning.

### Reproducibility
- Plots/summaries are deterministic given the cleaned input; run metadata saved in `outputs/task2_eda/run_metadata.json`.

## Task 3) Baseline model training + evaluation harness

### Specification
- Select a sensible target; train a simple baseline and a stronger baseline using a train/test split; avoid leakage via preprocessing inside a pipeline; save metrics/config/predictions.

### Success criteria
- `outputs/task3_modeling/metrics.json` exists with at least two models.
- `outputs/task3_modeling/predictions_*.csv` exist.

### Target choice
- **Chosen target**: `Exam_Score` (numeric) → regression.

### Results
- **dummy_mean**: MAE=2.8235, RMSE=3.7611, R2=-0.0007
- **ridge**: MAE=0.4499, RMSE=1.8033, R2=0.7699
- **rf**: MAE=1.0755, RMSE=2.1470, R2=0.6739

### Evidence
- Metrics/config: `outputs/task3_modeling/metrics.json`, `outputs/task3_modeling/config.json`
- Predictions: `outputs/task3_modeling/predictions_dummy_mean.csv`, `...ridge.csv`, `...rf.csv`
- Logs: `outputs/task3_modeling/task3.log`

### Failures + detection + correction
- None expected; if training fails, errors will be in `outputs/task3_modeling/task3.log`.

### Reproducibility
- Fixed `random_seed=42` for split and models; preprocessing encapsulated in pipelines to avoid leakage.

## Task 4) Debugging a deliberately broken pipeline

### Specification
- Create/use a broken pipeline that fails on this dataset; capture failure evidence; diagnose cause; fix; re-run and confirm.

### Success criteria
- Broken run produces a saved traceback under `outputs/task4_debugging/broken_run/`.
- Fixed run produces predictions + a metric under `outputs/task4_debugging/fixed_run/`.

### Evidence
- Failure: `outputs/task4_debugging/broken_run/traceback.txt`
- Fix + rerun: `outputs/task4_debugging/fixed_run/rmse.txt`, `.../predictions.csv`
- Logs: `outputs/task4_debugging/task4.log`

### Failures + detection + correction
## Broken pipeline debugging

- **Failure shown**: scaling mixed numeric + categorical features with `StandardScaler`.
- **Detection**: exception + saved traceback under `broken_run/traceback.txt`.
- **Root cause**: `StandardScaler` cannot convert string/object columns to float.
- **Fix**: split numeric vs categorical columns and apply `OneHotEncoder` for categoricals via `ColumnTransformer`.
- **Re-run evidence**: fixed pipeline RMSE = 1.8033 saved in `fixed_run/rmse.txt`.

### Reproducibility
- Fixed pipeline uses deterministic seed and a fully specified preprocessing graph; artifacts saved for verification.

