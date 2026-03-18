from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _common import (
    DATA_PATH,
    OUTPUTS_DIR,
    impute_missing,
    safe_read_csv,
    save_run_metadata,
    setup_logging,
    write_text,
)


def load_clean_or_impute(logger) -> pd.DataFrame:
    cleaned = OUTPUTS_DIR / "task1_ingestion" / "cleaned_student_performance.csv"
    if cleaned.exists():
        logger.info("loading_cleaned_dataset path=%s", cleaned.as_posix())
        return pd.read_csv(cleaned, low_memory=False)
    logger.warning("cleaned_dataset_missing;_falling_back_to_raw_plus_impute")
    df = safe_read_csv(DATA_PATH, logger)
    df2, _, _ = impute_missing(df, logger=logger, add_missing_indicators=True)
    return df2


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    out_dir = OUTPUTS_DIR / "task2_eda"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "task2.log")
    save_run_metadata(out_dir, extra={"task": "2_eda_plots_insights"})

    sns.set_theme(style="whitegrid")
    df = load_clean_or_impute(logger)

    # Choose target (explicit, and used throughout the EDA).
    target = "Exam_Score" if "Exam_Score" in df.columns else None
    if target is None:
        raise ValueError("Could not find expected target column 'Exam_Score' in dataset.")
    logger.info("eda_target_selected target=%s dtype=%s", target, str(df[target].dtype))

    # Summary statistics.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    df[numeric_cols].describe().T.to_csv(out_dir / "numeric_describe.csv", encoding="utf-8")
    if categorical_cols:
        cat_summary = []
        for c in categorical_cols:
            vc = df[c].value_counts(dropna=False)
            cat_summary.append(
                {
                    "column": c,
                    "unique": int(df[c].nunique(dropna=False)),
                    "top": str(vc.index[0]) if len(vc) else "",
                    "top_count": int(vc.iloc[0]) if len(vc) else 0,
                }
            )
        pd.DataFrame(cat_summary).sort_values("column").to_csv(
            out_dir / "categorical_summary.csv", index=False, encoding="utf-8"
        )

    # Plot: target distribution.
    plt.figure(figsize=(7, 4))
    sns.histplot(df[target], kde=True, bins=30)
    plt.title(f"Distribution of {target}")
    save_fig(plots_dir / "target_distribution.png")

    # Plot: correlation heatmap for numerics.
    corr = df[numeric_cols].corr(numeric_only=True)
    corr.to_csv(out_dir / "numeric_correlation.csv", encoding="utf-8")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Numeric feature correlation heatmap")
    save_fig(plots_dir / "numeric_correlation_heatmap.png")

    # Plot: top numeric correlations with target.
    if target in corr.columns:
        target_corr = (
            corr[target]
            .drop(labels=[target])
            .dropna()
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        topk = target_corr.head(8)
        topk.to_csv(out_dir / "top_numeric_correlations_with_target.csv", header=["corr"], encoding="utf-8")

        plt.figure(figsize=(8, 4))
        sns.barplot(x=topk.values, y=topk.index, orient="h")
        plt.title(f"Top numeric correlations with {target}")
        plt.xlabel("Pearson correlation")
        save_fig(plots_dir / "top_numeric_correlations.png")

        # Scatter plots for the top 3 correlated numeric features.
        for feat in topk.index[:3]:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=feat, y=target, alpha=0.35)
            plt.title(f"{target} vs {feat}")
            save_fig(plots_dir / f"scatter_{target}_vs_{feat}.png")

    # Plot: target vs selected categorical features (boxplots).
    for c in [col for col in categorical_cols if col != target][:6]:
        plt.figure(figsize=(9, 4))
        sns.boxplot(data=df, x=c, y=target)
        plt.title(f"{target} by {c}")
        plt.xticks(rotation=20, ha="right")
        save_fig(plots_dir / f"box_{target}_by_{c}.png")

    # Data-grounded insights (computed from saved summaries).
    insights_lines = []
    insights_lines.append("## EDA insights (data-grounded)\n")
    insights_lines.append(f"- **Target chosen**: `{target}` (numeric), treated as a regression target.\n")
    insights_lines.append(f"- **Rows / columns used**: {df.shape[0]} / {df.shape[1]}.\n")

    if target in corr.columns:
        abs_sorted = target_corr.abs().sort_values(ascending=False)
        top_name = abs_sorted.index[0] if len(abs_sorted) else None
        if top_name is not None:
            insights_lines.append(
                f"- **Strongest linear numeric association with `{target}`**: `{top_name}` "
                f"(corr={target_corr[top_name]:.3f}).\n"
            )

    # Group means for a few categoricals (if present).
    for c in ["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "School_Type", "Gender"]:
        if c in df.columns:
            means = df.groupby(c, dropna=False)[target].mean().sort_values(ascending=False)
            means.to_csv(out_dir / f"mean_{target}_by_{c}.csv", encoding="utf-8")
            top_group = means.index[0]
            bottom_group = means.index[-1]
            insights_lines.append(
                f"- **Mean `{target}` differs by `{c}`**: highest group `{top_group}` "
                f"({means.iloc[0]:.2f}) vs lowest group `{bottom_group}` ({means.iloc[-1]:.2f}).\n"
            )

    write_text(out_dir / "insights.md", "".join(insights_lines))
    logger.info("task2_complete plots_dir=%s", plots_dir.as_posix())


if __name__ == "__main__":
    main()

