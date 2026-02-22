# comparison/sentiment_comparison.py
"""
Sentiment comparison — reads from articles_enriched (LONG FORMAT).

NEW SCHEMA:
articles_enriched columns:
- article_id
- model_version (0=arabert, 1=camel, 2=mbert)
- sentiment_label
- sentiment_score
(+ language, dominant_topic, processing_time)

Produces a single DataFrame with one row per model_version.
Also computes:
- label distribution per model
- confidence statistics
- pairwise agreement between models (requires pivot by article_id)
- unanimous agreement across the 3 models (requires pivot by article_id)
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME_MAP = {
    0: "arabert",
    1: "camel",
    2: "mbert",
}

LABELS = ["POS", "NEG", "NEU"]  # MIX handled separately


def run_sentiment_comparison(engine):
    # Load long-format sentiment
    query = """
        SELECT
            article_id,
            model_version,
            sentiment_label,
            sentiment_score
        FROM articles_enriched
        WHERE sentiment_label IS NOT NULL
           OR sentiment_score IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from articles_enriched")

    if df.empty:
        return pd.DataFrame()

    # Normalize
    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce")
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.upper().str.strip()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["article_id", "model_version"])
    df["model_version"] = df["model_version"].astype(int)
    df["model"] = df["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # Total articles in this run (unique article_id)
    total_articles = int(df["article_id"].nunique())
    if total_articles == 0:
        return pd.DataFrame()

    # Pivot labels by article_id to compute agreement/unanimous
    labels_wide = (
        df.dropna(subset=["sentiment_label"])
          .pivot_table(index="article_id", columns="model_version", values="sentiment_label", aggfunc="first")
    )

    # Unanimous: all 3 models present and equal
    unanimous_pct = 0.0
    if not labels_wide.empty and all(mv in labels_wide.columns for mv in [0, 1, 2]):
        complete = labels_wide.dropna(subset=[0, 1, 2])
        if len(complete) > 0:
            unanimous_count = int((complete[0].eq(complete[1]) & complete[0].eq(complete[2])).sum())
            unanimous_pct = round(100 * unanimous_count / len(complete), 2)

    # Pairwise agreement pct helper
    def _pairwise_agree_pct(a: int, b: int) -> float:
        if labels_wide.empty or a not in labels_wide.columns or b not in labels_wide.columns:
            return 0.0
        pair = labels_wide[[a, b]].dropna()
        if pair.empty:
            return 0.0
        return round(100 * float((pair[a] == pair[b]).mean()), 2)

    agree_0_1 = _pairwise_agree_pct(0, 1)
    agree_0_2 = _pairwise_agree_pct(0, 2)
    agree_1_2 = _pairwise_agree_pct(1, 2)

    # Per-model stats
    rows = []
    for mv, g in df.groupby("model_version"):
        model_name = MODEL_NAME_MAP.get(int(mv), "unknown")

        labels = g["sentiment_label"].dropna()
        scores = g["sentiment_score"].dropna()
        label_counts = labels.value_counts()

        row = {
            "model_version": int(mv),
            "model": model_name,
            "total_articles": total_articles,
            "unanimous_pct": unanimous_pct,
        }

        # Label distribution
        for lbl in LABELS:
            c = int(label_counts.get(lbl, 0))
            row[f"{lbl}_count"] = c
            row[f"{lbl}_pct"] = round(100 * c / total_articles, 2)

        mix_c = int(label_counts.get("MIX", 0))
        row["MIX_count"] = mix_c
        row["MIX_pct"] = round(100 * mix_c / total_articles, 2)

        # Confidence stats
        row.update({
            "mean_confidence": round(float(scores.mean()), 4) if not scores.empty else 0.0,
            "std_confidence": round(float(scores.std()), 4) if len(scores) > 1 else 0.0,
            "min_confidence": round(float(scores.min()), 4) if not scores.empty else 0.0,
            "median_confidence": round(float(scores.median()), 4) if not scores.empty else 0.0,
            "max_confidence": round(float(scores.max()), 4) if not scores.empty else 0.0,
        })

        # Agreement with others (pairwise)
        if int(mv) == 0:
            row["agree_with_camel_pct"] = agree_0_1
            row["agree_with_mbert_pct"] = agree_0_2
        elif int(mv) == 1:
            row["agree_with_arabert_pct"] = agree_0_1
            row["agree_with_mbert_pct"] = agree_1_2
        elif int(mv) == 2:
            row["agree_with_arabert_pct"] = agree_0_2
            row["agree_with_camel_pct"] = agree_1_2

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("model_version").reset_index(drop=True)
    logger.info(f"\n=== SENTIMENT COMPARISON (LONG FORMAT) ===\n{result.to_string(index=False)}")
    return result