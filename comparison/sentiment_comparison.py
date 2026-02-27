# comparison/sentiment_comparison.py
"""
Sentiment comparison — reads from articles_enriched (LONG FORMAT).

CURRENT SCHEMA (as used in your pipeline):
articles_enriched columns (relevant):
- article_id
- model_version (0=arabert, 1=camel, 2=en_bert, 3=fr_bert)
- language (ar/en/fr)
- sentiment_label (POS/NEG/NEU[/MIX])
- sentiment_score

What this report does (updated to your current situation):
- Produces one row per (language, model_version)  [so EN/FR are not mixed with AR]
- Label distribution per model
- Confidence statistics
- Agreement is computed ONLY where it makes sense:
    * Arabic: agreement between model_version 0 and 1
    * English/French: only one model -> agreement columns are None
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME_MAP = {
    0: "arabert",
    1: "camel",
    2: "en_bert",
    3: "fr_bert",
}

LABELS = ["POS", "NEG", "NEU"]  # MIX handled separately (mainly Arabic)


def run_sentiment_comparison(engine):
    query = """
        SELECT
            article_id,
            model_version,
            language,
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

    # -----------------------
    # Normalize / clean
    # -----------------------
    df["language"] = df.get("language", "").astype(str).str.lower().str.strip()
    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce")
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.upper().str.strip()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    df = df.dropna(subset=["article_id", "model_version"]).copy()
    df["model_version"] = df["model_version"].astype(int)
    df["model"] = df["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # If language is missing for some reason, try to infer from model_version
    # (safe fallback; your pipeline usually populates language)
    missing_lang = df["language"].isin(["", "none", "nan"])
    if missing_lang.any():
        mv_to_lang = {0: "ar", 1: "ar", 2: "en", 3: "fr"}
        df.loc[missing_lang, "language"] = df.loc[missing_lang, "model_version"].map(mv_to_lang).fillna("xx")

    # Keep only languages you actually handle
    df = df[df["language"].isin(["ar", "en", "fr"])].copy()

    if df.empty:
        return pd.DataFrame()

    # -----------------------
    # Agreement (only Arabic: mv 0 vs 1)
    # -----------------------
    arabic_df = df[(df["language"] == "ar") & (df["model_version"].isin([0, 1]))].copy()
    agree_0_1 = None

    if not arabic_df.empty:
        labels_wide_ar = (
            arabic_df.dropna(subset=["sentiment_label"])
                     .pivot_table(index="article_id", columns="model_version", values="sentiment_label", aggfunc="first")
        )
        if 0 in labels_wide_ar.columns and 1 in labels_wide_ar.columns:
            pair = labels_wide_ar[[0, 1]].dropna()
            if not pair.empty:
                agree_0_1 = round(100.0 * float((pair[0] == pair[1]).mean()), 2)

    # -----------------------
    # Per-(language, model_version) stats
    # IMPORTANT: percentages are computed over the number of articles for that model+language.
    # -----------------------
    rows = []
    for (lang, mv), g in df.groupby(["language", "model_version"], dropna=False):
        mv = int(mv)
        model_name = MODEL_NAME_MAP.get(mv, "unknown")

        # total unique articles for this model in this language
        total_articles = int(g["article_id"].nunique())

        labels = g["sentiment_label"].dropna()
        scores = g["sentiment_score"].dropna()
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

        # Confidence stats (sentiment_score)
        row.update({
            "mean_confidence": round(float(scores.mean()), 4) if not scores.empty else None,
            "std_confidence": round(float(scores.std()), 4) if len(scores) > 1 else (0.0 if len(scores) == 1 else None),
            "min_confidence": round(float(scores.min()), 4) if not scores.empty else None,
            "median_confidence": round(float(scores.median()), 4) if not scores.empty else None,
            "max_confidence": round(float(scores.max()), 4) if not scores.empty else None,
        })

        # Agreement columns
        # Only Arabic has 2 models; EN/FR have one model in your current setup.
        if lang == "ar":
            if mv == 0:
                row["agree_with_camel_pct"] = agree_0_1
            elif mv == 1:
                row["agree_with_arabert_pct"] = agree_0_1
        else:
            # Keep explicit columns as None for consistency
            row["agree_with_other_pct"] = None

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sort: ar models first, then en, then fr (optional)
    lang_order = {"ar": 0, "en": 1, "fr": 2}
    result["_lang_order"] = result["language"].map(lang_order).fillna(99)
    result = result.sort_values(["_lang_order", "model_version"]).drop(columns=["_lang_order"]).reset_index(drop=True)

    logger.info(f"\n=== SENTIMENT COMPARISON (LONG FORMAT) ===\n{result.to_string(index=False)}")
    return result