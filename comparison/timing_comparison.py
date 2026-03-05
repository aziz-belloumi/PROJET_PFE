import logging
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME_MAP = {
    0: "arabert",
    1: "camel",
    2: "en_bert",
    3: "fr_bert",
}


def run_timing_comparison(engine) -> pd.DataFrame:
    """
    Reads cpu_processing_time and gpu_processing_time from articles_enriched
    and computes per-model timing statistics cumulatively across all runs.

    Note: processing times represent NER + Sentiment combined (per model_version).
    Topic timing is not tracked here (single shared model, not per model_version).

    Output: one row per model_version with CPU/GPU stats and speedup ratio.
    """
    query = """
        SELECT
            model_version,
            cpu_processing_time,
            gpu_processing_time
        FROM articles_enriched
        WHERE cpu_processing_time IS NOT NULL
           OR gpu_processing_time IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"[timing_comparison] Loaded {len(df)} rows from articles_enriched")

    if df.empty:
        logger.warning("[timing_comparison] No timing data found in articles_enriched.")
        return pd.DataFrame()

    # Convert milliseconds -> seconds
    df["cpu_sec"] = pd.to_numeric(df["cpu_processing_time"], errors="coerce") / 1000.0
    df["gpu_sec"] = pd.to_numeric(df["gpu_processing_time"], errors="coerce") / 1000.0
    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce").dropna().astype(int)

    rows = []
    for mv, g in df.groupby("model_version", dropna=True):
        mv = int(mv)
        cpu = g["cpu_sec"].dropna()
        gpu = g["gpu_sec"].dropna()

        def _stats(s: pd.Series, prefix: str) -> dict:
            if s.empty:
                return {f"{prefix}_total_seconds": None, f"{prefix}_mean_per_article": None,
                        f"{prefix}_std_per_article": None, f"{prefix}_min_per_article": None,
                        f"{prefix}_max_per_article": None}
            return {
                f"{prefix}_total_seconds":    round(float(s.sum()), 2),
                f"{prefix}_mean_per_article": round(float(s.mean()), 4),
                f"{prefix}_std_per_article":  round(float(s.std()), 4) if len(s) > 1 else 0.0,
                f"{prefix}_min_per_article":  round(float(s.min()), 4),
                f"{prefix}_max_per_article":  round(float(s.max()), 4),
            }

        row = {
            "model_version": mv,
            "model_name":    MODEL_NAME_MAP.get(mv, "unknown"),
            "articles_count": int(len(g)),
        }
        row.update(_stats(cpu, "cpu"))
        row.update(_stats(gpu, "gpu"))

        # Speedup: how many times faster GPU is vs CPU (per article mean)
        c_mean = row.get("cpu_mean_per_article")
        g_mean = row.get("gpu_mean_per_article")
        if c_mean and g_mean and float(g_mean) > 0:
            row["speedup_cpu_over_gpu"] = round(float(c_mean) / float(g_mean), 3)
        else:
            row["speedup_cpu_over_gpu"] = None

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("model_version").reset_index(drop=True)
    logger.info(f"\n=== TIMING COMPARISON (CPU vs GPU, NER+Sentiment combined) ===\n{result.to_string(index=False)}")
    return result