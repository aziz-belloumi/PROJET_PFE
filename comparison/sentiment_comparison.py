# comparison/sentiment_comparison.py
"""
Sentiment comparison — reads from sentiment_results MySQL table.
Produces a single DataFrame with one row per model.
"""

import logging

import pandas as pd


logger = logging.getLogger(__name__)

MODELS = ["arabert", "camel", "mbert"]
LABELS = ["POS", "NEG", "NEU"]


def run_sentiment_comparison(engine):
    """
    Read sentiment_results table and compute comparison metrics.

    Returns DataFrame with one row per model and columns:
        model, POS_count, POS_pct, NEG_count, NEG_pct, NEU_count, NEU_pct,
        MIX_count, MIX_pct,
        mean_confidence, std_confidence, min_confidence, median_confidence,
        agree_with_others_pct, unanimous_pct
    """
    df = pd.read_sql("SELECT * FROM sentiment_results", engine)
    logger.info(f"Loaded {len(df)} rows from sentiment_results")

    total_articles = len(df)

    # Unanimous: all 3 models agree on same label
    label_cols = [f"{m}_label" for m in MODELS]
    valid = df[label_cols].dropna()
    unanimous_count = int(valid.apply(lambda r: r.nunique() == 1, axis=1).sum())
    unanimous_pct = round(100 * unanimous_count / len(valid), 2) if len(valid) > 0 else 0

    rows = []
    for model in MODELS:
        label_col = f"{model}_label"
        score_col = f"{model}_score"

        labels = df[label_col].dropna()
        scores = df[score_col].dropna()
        label_counts = labels.value_counts()

        # Agreement with the other two models (average pairwise)
        others = [m for m in MODELS if m != model]
        agree_count = 0
        compare_count = 0
        for other in others:
            other_col = f"{other}_label"
            pair = df[[label_col, other_col]].dropna()
            agree_count += int((pair[label_col] == pair[other_col]).sum())
            compare_count += len(pair)
        agree_pct = round(100 * agree_count / compare_count, 2) if compare_count > 0 else 0

        # Build row
        row = {"model": model}

        for lbl in LABELS:
            c = int(label_counts.get(lbl, 0))
            row[f"{lbl}_count"] = c
            row[f"{lbl}_pct"] = round(100 * c / total_articles, 2) if total_articles > 0 else 0

        # MIX (only arabert produces this)
        mix_c = int(label_counts.get("MIX", 0))
        row["MIX_count"] = mix_c
        row["MIX_pct"] = round(100 * mix_c / total_articles, 2) if total_articles > 0 else 0

        row.update({
            "mean_confidence": round(scores.mean(), 4) if len(scores) > 0 else 0,
            "std_confidence": round(scores.std(), 4) if len(scores) > 0 else 0,
            "min_confidence": round(scores.min(), 4) if len(scores) > 0 else 0,
            "median_confidence": round(scores.median(), 4) if len(scores) > 0 else 0,
            "agree_with_others_pct": agree_pct,
            "unanimous_pct": unanimous_pct,
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info(f"\n=== SENTIMENT COMPARISON ===\n{result.to_string(index=False)}")
    return result