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
    topic_results_buffer: Dict[int, dict],
    logger: logging.Logger,
) -> None:
    """
    Write english_samples.csv and french_samples.csv to run_dir.

    Only EN and FR articles are exported; Arabic is skipped.
    Files are only created when the respective language has at least one row.

    Args:
        run_dir:               Directory to write CSVs into.
        work_df:               Articles DataFrame (must have 'id', 'lang', 'text_raw',
                               'text_sentiment', 'text_ner', 'text_topic' columns).
        sent_results_buffer:   {(article_id, model_version): {"label": ..., "score": ...}}
        topic_results_buffer:  {article_id: {"label": ..., "score": ...}}
        logger:                Logger instance.
    """
    rows = []

    for r in work_df.itertuples(index=False):
        aid  = int(r.id)
        lang = str(r.lang)

        if lang not in {"en", "fr"}:
            continue

        mv = 2 if lang == "en" else 3  # model_version mapping
        sent_info  = sent_results_buffer.get((aid, mv), {})
        topic_info = topic_results_buffer.get(aid, {})

        rows.append({
            "article_id":           aid,
            "lang":                 lang,
            "text_raw":             r.text_raw,
            "sentiment_preprocessed": r.text_sentiment,
            "ner_preprocessed":     r.text_ner,
            "topic_preprocessed":   r.text_topic,
            "sentiment_label":      sent_info.get("label"),
            "sentiment_score":      sent_info.get("score"),
            "dominant_topic":       topic_info.get("label"),
            "topic_score":          topic_info.get("score"),
        })

    if not rows:
        logger.info("No EN/FR articles to export — skipping sample CSVs.")
        return

    out_df = pd.DataFrame(rows)
    run_dir = Path(run_dir)

    en_df = out_df[out_df["lang"] == "en"].copy()
    fr_df = out_df[out_df["lang"] == "fr"].copy()

    if not en_df.empty:
        path = run_dir / "english_samples.csv"
        en_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved {len(en_df)} EN rows → {path.name}")

    if not fr_df.empty:
        path = run_dir / "french_samples.csv"
        fr_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved {len(fr_df)} FR rows → {path.name}")
