# comparison/timing_comparison.py
"""
Timing comparison — receives timing records collected during pipeline execution.
Produces a single DataFrame with one row per task + model.
"""

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def run_timing_comparison(timing_records):
    """
    Compute timing statistics from pipeline timing records.

    Args:
        timing_records: list of dicts, each with keys:
            article_id, task, model_name, elapsed_seconds

    Returns DataFrame with columns:
        task, model_name, total_seconds, mean_per_article,
        std_per_article, min_per_article, max_per_article
    """
    if not timing_records:
        logger.warning("No timing records provided.")
        return pd.DataFrame(columns=[
            "task", "model_name", "total_seconds", "mean_per_article",
            "std_per_article", "min_per_article", "max_per_article",
        ])

    df = pd.DataFrame(timing_records)
    logger.info(f"Processing {len(df)} timing records")

    rows = []
    for (task, model), group in df.groupby(["task", "model_name"]):
        elapsed = group["elapsed_seconds"]
        rows.append({
            "task": task,
            "model_name": model,
            "total_seconds": round(elapsed.sum(), 2),
            "mean_per_article": round(elapsed.mean(), 4),
            "std_per_article": round(elapsed.std(), 4),
            "min_per_article": round(elapsed.min(), 4),
            "max_per_article": round(elapsed.max(), 4),
        })

    result = pd.DataFrame(rows)
    logger.info(f"\n=== TIMING COMPARISON ===\n{result.to_string(index=False)}")
    return result