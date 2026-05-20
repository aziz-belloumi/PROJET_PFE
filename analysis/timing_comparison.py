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
            cpu_time_ner,
            cpu_time_sentiment,
            cpu_time_topic,
            gpu_time_ner,
            gpu_time_sentiment,
            gpu_time_topic
        FROM articles_enriched
        WHERE cpu_time_ner IS NOT NULL
           OR gpu_time_ner IS NOT NULL
           OR cpu_time_sentiment IS NOT NULL
           OR gpu_time_sentiment IS NOT NULL
           OR cpu_time_topic IS NOT NULL
           OR gpu_time_topic IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"[timing_comparison] Loaded {len(df)} rows from articles_enriched")

    if df.empty:
        logger.warning("[timing_comparison] No timing data found in articles_enriched.")
        return pd.DataFrame()

    # Convert milliseconds -> seconds
    for t_col in ["cpu_time_ner", "cpu_time_sentiment", "cpu_time_topic", "gpu_time_ner", "gpu_time_sentiment", "gpu_time_topic"]:
        df[t_col] = pd.to_numeric(df[t_col], errors="coerce") / 1000.0

    df["cpu_sec"] = df[["cpu_time_ner", "cpu_time_sentiment", "cpu_time_topic"]].sum(axis=1, skipna=True)
    df["gpu_sec"] = df[["gpu_time_ner", "gpu_time_sentiment", "gpu_time_topic"]].sum(axis=1, skipna=True)

    # If all components are NaN, the sum will be 0.0, which we should revert to NA to avoid skews
    df.loc[df[["cpu_time_ner", "cpu_time_sentiment", "cpu_time_topic"]].isna().all(axis=1), "cpu_sec"] = pd.NA
    df.loc[df[["gpu_time_ner", "gpu_time_sentiment", "gpu_time_topic"]].isna().all(axis=1), "gpu_sec"] = pd.NA

    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce").dropna().astype(int)

    rows = []
    for mv, g in df.groupby("model_version", dropna=True):
        mv = int(mv)
        cpu = g["cpu_sec"].dropna()
        gpu = g["gpu_sec"].dropna()
        c_ner = g["cpu_time_ner"].dropna()
        c_sent = g["cpu_time_sentiment"].dropna()
        c_top = g["cpu_time_topic"].dropna()
        g_ner = g["gpu_time_ner"].dropna()
        g_sent = g["gpu_time_sentiment"].dropna()
        g_top = g["gpu_time_topic"].dropna()

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
        row.update(_stats(cpu, "cpu_total"))
        row.update(_stats(gpu, "gpu_total"))
        row.update(_stats(c_ner, "cpu_ner"))
        row.update(_stats(g_ner, "gpu_ner"))
        row.update(_stats(c_sent, "cpu_sent"))
        row.update(_stats(g_sent, "gpu_sent"))
        row.update(_stats(c_top, "cpu_top"))
        row.update(_stats(g_top, "gpu_top"))

        # Speedup: how many times faster GPU is vs CPU (per article mean)
        c_mean = row.get("cpu_total_mean_per_article")
        g_mean = row.get("gpu_total_mean_per_article")
        if c_mean and g_mean and float(g_mean) > 0:
            row["speedup_cpu_over_gpu"] = round(float(c_mean) / float(g_mean), 3)
        else:
            row["speedup_cpu_over_gpu"] = None

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("model_version").reset_index(drop=True)
    logger.info(f"\n=== TIMING COMPARISON (CPU vs GPU, NER+Sentiment combined) ===\n{result.to_string(index=False)}")
    return result
