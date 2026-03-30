# pipeline/exporter.py
"""
Stage 4 — CSV Exporter

Exports per-language review samples (EN / FR) to CSV files in the run directory.
These are used for manual evaluation or as input to run_evaluations.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd


def export_lang_samples_csv(
    run_dir: Path,
    work_df: pd.DataFrame,
    sent_results_buffer: Dict[tuple, dict],
    topic_results_buffer: Dict[tuple, dict],
    logger: logging.Logger,
) -> None:
    """
    Write processed_articles.csv to run_dir with all processed articles.

    Includes all languages (AR, EN, FR) with their preprocessing results and NLP outputs.
    Files are only created when there are processed articles.

    Args:
        run_dir:               Directory to write CSVs into.
        work_df:               Articles DataFrame (must have 'id', 'lang', 'text_raw',
                               'text_sentiment', 'text_ner', 'text_topic' columns).
        sent_results_buffer:   {(article_id, model_version): {"label": ..., "score": ...}}
        topic_results_buffer:  {(article_id, model_version): {"label": ..., "score": ...}}
        logger:                Logger instance.
    """
    rows = []

    # Get a distinct set of model versions from the topic buffer (or default to 0)
    topic_mvs = {k[1] for k in topic_results_buffer.keys() if isinstance(k, tuple) and len(k) == 2}
    if not topic_mvs:
        topic_mvs = {0}

    for r in work_df.itertuples(index=False):
        aid  = int(r.id)
        lang = str(r.lang)

        # Get sentiment results (model_version 0 for all languages)
        sent_info = sent_results_buffer.get((aid, 0), {})

        row = {
            "article_id":           aid,
            "lang":                 lang,
            "text_raw":             r.text_raw,
            "text_sentiment_preprocessed": r.text_sentiment,
            "text_ner_preprocessed": r.text_ner,
            "text_topic_preprocessed": r.text_topic,
            "sentiment_label":      sent_info.get("label"),
        }
        
        # Add topic results for each model version
        for mv in sorted(topic_mvs):
            topic_info = topic_results_buffer.get((aid, mv), {})
            row[f"topic_label_mv{mv}"] = topic_info.get("label")

        rows.append(row)

    if not rows:
        logger.info("No processed articles to export — skipping results CSV.")
        return

    out_df = pd.DataFrame(rows)
    run_dir = Path(run_dir)

    # Create comprehensive results CSV
    path = run_dir / "processed_articles.csv"
    out_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(out_df)} processed articles → {path.name}")
