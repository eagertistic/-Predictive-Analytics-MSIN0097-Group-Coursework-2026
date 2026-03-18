## Broken pipeline debugging

- **Failure shown**: scaling mixed numeric + categorical features with `StandardScaler`.
- **Detection**: exception + saved traceback under `broken_run/traceback.txt`.
- **Root cause**: `StandardScaler` cannot convert string/object columns to float.
- **Fix**: split numeric vs categorical columns and apply `OneHotEncoder` for categoricals via `ColumnTransformer`.
- **Re-run evidence**: fixed pipeline RMSE = 1.8033 saved in `fixed_run/rmse.txt`.
