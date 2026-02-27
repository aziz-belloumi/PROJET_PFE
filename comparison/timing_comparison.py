import logging
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME_MAP = {
    -1: "topic",
    0: "arabert",
    1: "camel",
    2: "en_bert",
    3: "fr_bert",
}


def _stats(group: pd.Series) -> dict:
    return {
        "total_seconds": round(float(group.sum()), 2),
        "mean_per_article": round(float(group.mean()), 4),
        "std_per_article": round(float(group.std()), 4) if len(group) > 1 else 0.0,
        "min_per_article": round(float(group.min()), 4),
        "max_per_article": round(float(group.max()), 4),
    }


def run_timing_comparison(timing_records):
    
    out_cols = [
        "task", "model_version", "model_name",
        "cpu_total_seconds", "cpu_mean_per_article", "cpu_std_per_article", "cpu_min_per_article", "cpu_max_per_article",
        "gpu_total_seconds", "gpu_mean_per_article", "gpu_std_per_article", "gpu_min_per_article", "gpu_max_per_article",
        "speedup_cpu_over_gpu",
    ]

    if not timing_records:
        logger.warning("No timing records provided.")
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame(timing_records).copy()
    logger.info(f"Processing {len(df)} timing records")

    # Backward compatible: if mode missing, assume cpu
    if "mode" not in df.columns:
        df["mode"] = "cpu"

    # Required fields
    if "model_version" not in df.columns:
        raise KeyError("timing_records must include 'model_version'.")
    if "task" not in df.columns or "elapsed_seconds" not in df.columns:
        raise KeyError("timing_records must include 'task' and 'elapsed_seconds'.")

    # Normalize types
    df["mode"] = df["mode"].astype(str).str.lower().str.strip()
    df["task"] = df["task"].astype(str).str.lower().str.strip()
    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce")
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["mode", "task", "model_version", "elapsed_seconds"]).copy()
    df = df[df["mode"].isin(["cpu", "gpu"])].copy()

    if df.empty:
        logger.warning("Timing records are empty after cleaning.")
        return pd.DataFrame(columns=out_cols)

    df["model_version"] = df["model_version"].astype(int)
    df["model_name"] = df["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # Aggregate per (mode, task, model_version)
    agg_rows = []
    for (mode, task, mv), g in df.groupby(["mode", "task", "model_version"], dropna=False):
        stats = _stats(g["elapsed_seconds"])
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

    # Merge on task+model_version only (model_name is derived from model_version)
    result = pd.merge(cpu, gpu, on=["task", "model_version"], how="outer")
    result["model_name"] = result["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # Speedup (based on mean per article)
    def _speedup(row):
        c = row.get("cpu_mean_per_article")
        g = row.get("gpu_mean_per_article")
        if pd.isna(c) or pd.isna(g) or float(g) == 0.0:
            return None
        return round(float(c) / float(g), 3)

    result["speedup_cpu_over_gpu"] = result.apply(_speedup, axis=1)

    # Sort output
    result = result.sort_values(["task", "model_version"]).reset_index(drop=True)

    # Ensure consistent column order
    for col in out_cols:
        if col not in result.columns:
            result[col] = None
    result = result[out_cols]

    logger.info(f"\n=== TIMING COMPARISON (CPU vs GPU) ===\n{result.to_string(index=False)}")
    return result