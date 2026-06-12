# analysis/sentiment_comparison.py
"""
Sentiment comparison — reads from articles_enriched.

CURRENT SCHEMA (as used in your pipeline):
articles_enriched columns (relevant):
- article_id
- language (ar/en/fr)
- sentiment_label (POS/NEG/NEU[/MIX])

What this report does (updated to your current situation):
- Produces one row per language for the unified sentiment output
- Label distribution only; no numeric sentiment_score exists in the current schema
- Agreement columns are not meaningful for this unified schema
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME_MAP = {
    0: "unified"
}

LABELS = ["POS", "NEG", "NEU"]  # MIX handled separately (mainly Arabic)


def run_sentiment_comparison(engine):
    query = """
        SELECT
            article_id,
            language,
            sentiment_label
        FROM articles_enriched
        WHERE sentiment_label IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from articles_enriched")

    if df.empty:
        return pd.DataFrame()

    # -----------------------
    # Normalize / clean
    # -----------------------
    df["language"] = df.get("language", "").astype(str).str.lower().str.strip()
    df["model_version"] = 0
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.upper().str.strip()

    df = df.dropna(subset=["article_id", "model_version"]).copy()
    df["model_version"] = df["model_version"].astype(int)
    df["model"] = df["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # Keep only languages you actually handle
    df = df[df["language"].isin(["ar", "en", "fr"])].copy()

    if df.empty:
        return pd.DataFrame()

    # -----------------------
    # Per-language stats for the unified sentiment output
    # -----------------------
    rows = []
    for lang, g in df.groupby("language", dropna=False):
        mv = 0
        model_name = MODEL_NAME_MAP.get(mv, "unknown")

        # total unique articles for this model in this language
        total_articles = int(g["article_id"].nunique())

        labels = g["sentiment_label"].dropna()
        label_counts = labels.value_counts()

        row = {
            "language": str(lang),
            "model_version": mv,
            "model": model_name,
            "total_articles": total_articles,
        }

        # Label distribution (based on total_articles for this model/lang)
        # If somehow total_articles == 0, keep 0%
        denom = total_articles if total_articles > 0 else 1

        for lbl in LABELS:
            c = int(label_counts.get(lbl, 0))
            row[f"{lbl}_count"] = c
            row[f"{lbl}_pct"] = round(100.0 * c / denom, 2)

        mix_c = int(label_counts.get("MIX", 0))
        row["MIX_count"] = mix_c
        row["MIX_pct"] = round(100.0 * mix_c / denom, 2)

        # Unified schema does not include sentiment score confidence statistics
        row.update({
            "mean_confidence": None,
            "std_confidence": None,
            "min_confidence": None,
            "median_confidence": None,
            "max_confidence": None,
        })

        # Unified schema does not compute cross-model agreement
        row["agree_with_other_pct"] = None

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sort: ar models first, then en, then fr (optional)
    lang_order = {"ar": 0, "en": 1, "fr": 2}
    result["_lang_order"] = result["language"].map(lang_order).fillna(99)
    result = result.sort_values(["_lang_order", "model_version"]).drop(columns=["_lang_order"]).reset_index(drop=True)

    logger.info(f"\n=== SENTIMENT COMPARISON (LONG FORMAT) ===\n{result.to_string(index=False)}")
    return result
