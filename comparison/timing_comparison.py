# comparison/timing_comparison.py
"""
Timing comparison — receives timing records collected during pipeline execution.

UPDATED (new schema):
- timing_records contain: mode, task, model_version, article_id, elapsed_seconds
- Produces one row per (task, model_version) with CPU stats, GPU stats, and speedup.
- Adds model_name for readability (0=arabert, 1=camel, 2=mbert, -1=topic/other).

Timing must represent prediction time only (no MySQL write time).
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME_MAP = {
    -1: "topic",
    0: "arabert",
    1: "camel",
    2: "mbert",
}


def _stats(group: pd.Series) -> dict:
    return {
        "total_seconds": round(group.sum(), 2),
        "mean_per_article": round(group.mean(), 4),
        "std_per_article": round(group.std(), 4) if len(group) > 1 else 0.0,
        "min_per_article": round(group.min(), 4),
        "max_per_article": round(group.max(), 4),
    }


def run_timing_comparison(timing_records):
    """
    Args:
        timing_records: list[dict] with keys:
            - mode: "cpu" or "gpu" (if missing -> defaults to "cpu")
            - task: "ner" / "sentiment" / "topic" (etc.)
            - model_version: int (0/1/2) or -1 for non-3-model tasks like topic
            - elapsed_seconds: float
            - (optional) article_id

    Returns DataFrame with columns:
        task, model_version, model_name,
        cpu_total_seconds, cpu_mean_per_article, cpu_std_per_article, cpu_min_per_article, cpu_max_per_article,
        gpu_total_seconds, gpu_mean_per_article, gpu_std_per_article, gpu_min_per_article, gpu_max_per_article,
        speedup_cpu_over_gpu
    """
    if not timing_records:
        logger.warning("No timing records provided.")
        return pd.DataFrame(columns=[
            "task", "model_version", "model_name",
            "cpu_total_seconds", "cpu_mean_per_article", "cpu_std_per_article", "cpu_min_per_article", "cpu_max_per_article",
            "gpu_total_seconds", "gpu_mean_per_article", "gpu_std_per_article", "gpu_min_per_article", "gpu_max_per_article",
            "speedup_cpu_over_gpu",
        ])

    df = pd.DataFrame(timing_records).copy()
    logger.info(f"Processing {len(df)} timing records")

    # Backward compatible: if mode missing, assume cpu
    if "mode" not in df.columns:
        df["mode"] = "cpu"

    # REQUIRED new field: model_version
    if "model_version" not in df.columns:
        raise KeyError("timing_records must include 'model_version' (new pipeline format).")

    # Normalize
    df["mode"] = df["mode"].astype(str).str.lower().str.strip()
    df["task"] = df["task"].astype(str).str.lower().str.strip()
    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce")
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
    df = df.dropna(subset=["mode", "task", "model_version", "elapsed_seconds"])

    if df.empty:
        logger.warning("Timing records are empty after cleaning.")
        return pd.DataFrame()

    df["model_version"] = df["model_version"].astype(int)
    df["model_name"] = df["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # Aggregate per (mode, task, model_version)
    agg_rows = []
    for (mode, task, mv), g in df.groupby(["mode", "task", "model_version"], dropna=False):
        s = g["elapsed_seconds"]
        stats = _stats(s)
        agg_rows.append({
            "mode": mode,
            "task": task,
            "model_version": int(mv),
            "model_name": MODEL_NAME_MAP.get(int(mv), "unknown"),
            **stats,
        })

    agg = pd.DataFrame(agg_rows)

    # Split CPU/GPU, rename columns, merge
    cpu = agg[agg["mode"] == "cpu"].drop(columns=["mode"], errors="ignore").copy()
    gpu = agg[agg["mode"] == "gpu"].drop(columns=["mode"], errors="ignore").copy()

    cpu = cpu.rename(columns={
        "total_seconds": "cpu_total_seconds",
        "mean_per_article": "cpu_mean_per_article",
        "std_per_article": "cpu_std_per_article",
        "min_per_article": "cpu_min_per_article",
        "max_per_article": "cpu_max_per_article",
    })
    gpu = gpu.rename(columns={
        "total_seconds": "gpu_total_seconds",
        "mean_per_article": "gpu_mean_per_article",
        "std_per_article": "gpu_std_per_article",
        "min_per_article": "gpu_min_per_article",
        "max_per_article": "gpu_max_per_article",
    })

    result = pd.merge(
        cpu, gpu,
        on=["task", "model_version", "model_name"],
        how="outer",
    )

    # Speedup (based on mean per article)
    def _speedup(row):
        c = row.get("cpu_mean_per_article")
        g = row.get("gpu_mean_per_article")
        if pd.isna(c) or pd.isna(g) or g == 0:
            return None
        return round(float(c) / float(g), 3)

    result["speedup_cpu_over_gpu"] = result.apply(_speedup, axis=1)

    # Sort output
    result = result.sort_values(["task", "model_version"]).reset_index(drop=True)

    logger.info(f"\n=== TIMING COMPARISON (CPU vs GPU) ===\n{result.to_string(index=False)}")
    return result