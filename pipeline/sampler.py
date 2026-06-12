# pipeline/sampler.py
"""
Stage 1 — Sampling & Preprocessing

Fetches unprocessed articles from the DB, runs fastText language detection,
filters to supported languages, preprocesses text for NER / Sentiment / Topic,
and pre-inserts placeholder rows into articles_enriched.

Returns:
    work_df   — DataFrame of valid articles with preprocessed text columns
    skipped_df — DataFrame of unsupported/low-score language rows (already written to DB)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from src.config import Config
from src.db_config import DatabaseConnection
from src.language_detection import FastTextLanguageDetector
from src.preprocessing import PreprocessRouter


def run_sampling(
    db: DatabaseConnection,
    sample_size: int,
    ner_models_by_lang: Dict[str, list],
    logger: logging.Logger,
    raw_table: str = "article",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Query DB, detect language, filter, preprocess.

    Args:
        db:                 Active DatabaseConnection.
        sample_size:        Max number of articles to fetch.
        ner_models_by_lang: Dict of lang -> [(model_version, name)] used to
                            determine supported languages and pre-insert placeholders.
        logger:             Logger instance.
        raw_table:          Name of the source article table.

    Returns:
        (work_df, skipped_df)
    """
    engine = db.get_engine()
    lang_threshold = Config.LANG_THRESHOLD
    supported_langs = set(ner_models_by_lang.keys())

    # ------------------------------------------------------------------
    # 1) Fetch unprocessed articles
    # ------------------------------------------------------------------
    query = f"""
        SELECT a.id, a.body, a.id_language, a.id_categories
        FROM {raw_table} a
        LEFT JOIN articles_enriched ae ON a.id = ae.article_id
        WHERE a.body IS NOT NULL
          AND (ae.article_id IS NULL OR ae.sentiment_label IS NULL)
        GROUP BY a.id
        ORDER BY a.id DESC
        LIMIT {sample_size}
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Fetched {len(df)} unprocessed rows from {raw_table}")

    # ------------------------------------------------------------------
    # 2) Language detection
    # ------------------------------------------------------------------
    preproc = PreprocessRouter(logger=logger)
    detector = FastTextLanguageDetector(
        model="auto",
        logger=logger,
        preprocessor=None,
    )

    df["text_raw"] = df["body"].fillna("")
    df["text_langdetect"] = df["text_raw"].apply(
        lambda t: preproc.preprocess(t, "xx", "lang_detect")
    )

    lang_rows = []
    for r in df.itertuples(index=False):
        res = detector.detect(r.text_langdetect)
        lang_rows.append({"article_id": int(r.id), "lang": res.lang, "score": res.score})

    lang_df = pd.DataFrame(lang_rows)
    merged = df.merge(lang_df[["article_id", "lang", "score"]], left_on="id", right_on="article_id", how="left")

    # ------------------------------------------------------------------
    # 3) Filter
    # ------------------------------------------------------------------
    valid_mask = (merged["lang"].isin(supported_langs)) & (merged["score"] >= lang_threshold)
    work_df    = merged[valid_mask].copy()
    skipped_df = merged[~valid_mask].copy()

    logger.info(
        f"Lang filter (supported={sorted(supported_langs)} & score>={lang_threshold}): "
        f"{len(work_df)} valid, {len(skipped_df)} skipped."
    )

    # ------------------------------------------------------------------
    # 4) Mark skipped articles in DB (won't be re-fetched)
    # ------------------------------------------------------------------
    for r in skipped_df.itertuples(index=False):
        db.upsert_articles_enriched(
            article_id=int(r.id),
            language=str(r.lang),
            sentiment_label="SKIPPED",
        )

    if work_df.empty:
        logger.warning("No rows passed language filter.")
        return work_df, skipped_df

    # ------------------------------------------------------------------
    # 5) Task-specific preprocessing
    # ------------------------------------------------------------------
    work_df["text_ner"] = work_df.apply(
        lambda r: preproc.preprocess(r.text_raw, str(r.lang), "ner"), axis=1
    )
    work_df["text_sentiment"] = work_df.apply(
        lambda r: preproc.preprocess(r.text_raw, str(r.lang), "sentiment"), axis=1
    )
    work_df["text_topic"] = work_df.apply(
        lambda r: preproc.preprocess(r.text_raw, str(r.lang), "topic"), axis=1
    )

    # ------------------------------------------------------------------
    # 6) Pre-insert placeholder rows in articles_enriched
    # ------------------------------------------------------------------
    for r in work_df.itertuples(index=False):
        aid  = int(r.id)
        lang = str(r.lang)
        db.upsert_articles_enriched(article_id=aid, language=lang)

    return work_df, skipped_df
