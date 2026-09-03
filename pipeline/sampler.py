# pipeline/sampler.py
"""
Stage 1 — Sampling & Preprocessing

Fetches unprocessed articles from the DB, runs fastText language detection,
filters to supported languages, preprocesses text for NER / Sentiment / Topic,
and pre-inserts placeholder rows into article_sentiments & qwen_article_sentiments.

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
    lang_threshold: float | None = None,
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
    lang_threshold = lang_threshold if lang_threshold is not None else Config.LANG_THRESHOLD
    supported_langs = set(ner_models_by_lang.keys()) if ner_models_by_lang else set(Config.NER_MODELS_BY_LANG.keys())

    # ------------------------------------------------------------------
    # 1) Fetch unprocessed articles
    # ------------------------------------------------------------------
    is_qwen_only = Config.RUN_QWEN and not (Config.RUN_NER or Config.RUN_SENTIMENT or Config.RUN_TOPIC)
    enrich_table = "qwen_article_sentiments" if is_qwen_only else "article_sentiments"

    query = f"""
        SELECT a.id, a.body, a.id_language, a.id_categories
        FROM {raw_table} a
        LEFT JOIN {enrich_table} ae ON a.id = ae.article_id
        WHERE a.body IS NOT NULL
          AND (ae.article_id IS NULL OR ae.sentiment_label IS NULL)
        GROUP BY a.id
        ORDER BY a.id DESC
        LIMIT {sample_size}
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Fetched {len(df)} unprocessed rows from {raw_table} (tracking table: {enrich_table})")

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
        aid = int(r.id)
        db.upsert_article_sentiments(
            article_id=aid,
            language=None,
            sentiment_label="SKIPPED",
        )
        db.upsert_qwen_article_sentiments(
            article_id=aid,
            language=None,
            sentiment_label="SKIPPED",
        )
        db.upsert_article_topic(
            article_id=aid,
            language=None,
            topic_label="SKIPPED",
            confidence_score=None,
        )
        db.upsert_qwen_article_topic(
            article_id=aid,
            language=None,
            topic_label="SKIPPED",
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

    # Filter articles that become empty after preprocessing
    empty_mask = (
        work_df["text_sentiment"].fillna("").str.strip().eq("")
        | work_df["text_topic"].fillna("").str.strip().eq("")
    )
    empty_df = work_df[empty_mask].copy()
    work_df  = work_df[~empty_mask].copy()

    if not empty_df.empty:
        logger.info(f"Marking {len(empty_df)} empty preprocessed articles as SKIPPED (language=NULL)")
        for r in empty_df.itertuples(index=False):
            aid = int(r.id)
            db.upsert_article_sentiments(
                article_id=aid,
                language=None,
                sentiment_label="SKIPPED",
            )
            db.upsert_qwen_article_sentiments(
                article_id=aid,
                language=None,
                sentiment_label="SKIPPED",
            )
            db.upsert_article_topic(
                article_id=aid,
                language=None,
                topic_label="SKIPPED",
                confidence_score=None,
            )
            db.upsert_qwen_article_topic(
                article_id=aid,
                language=None,
                topic_label="SKIPPED",
            )

    if work_df.empty:
        logger.warning("All articles became empty after preprocessing.")
        return work_df, pd.concat([skipped_df, empty_df], ignore_index=True)

    # ------------------------------------------------------------------
    # 6) Pre-insert placeholder rows in article_sentiments & qwen_article_sentiments
    # ------------------------------------------------------------------
    for r in work_df.itertuples(index=False):
        aid  = int(r.id)
        lang = str(r.lang)
        db.upsert_article_sentiments(article_id=aid, language=lang)
        db.upsert_qwen_article_sentiments(article_id=aid, language=lang)
        db.upsert_article_topic(article_id=aid, language=lang)
        db.upsert_qwen_article_topic(article_id=aid, language=lang)

    return work_df, skipped_df
