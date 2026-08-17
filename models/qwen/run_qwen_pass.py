import time
import logging
import pandas as pd

from .sentiment_analysis import LLMSentiment
from .topic_generation import LLMTopic


def run_qwen_pass(work_df: pd.DataFrame, db, logger: logging.Logger):
    """
    Runs Qwen (Ollama) on the sampled articles for Sentiment and Topic extraction.
    Updates the results in the database.

    Args:
        work_df: DataFrame with columns: id, lang, text_sentiment[, text_topic]
        db:      DatabaseConnection instance (from the main application project)
        logger:  Python logger instance
    """
    if work_df.empty:
        return

    logger.info("=== GPU PASS: QWEN (OLLAMA) ===")

    try:
        sentiment_model = LLMSentiment(logger=logger)
        topic_model = LLMTopic(logger=logger)
    except Exception as e:
        logger.error(f"[QWEN] Failed to initialize Qwen models: {e}")
        return

    for r in work_df.itertuples(index=False):
        aid = int(r.id)
        lang = str(r.lang)

        # 1. Sentiment
        t0 = time.perf_counter()
        try:
            sent_res = sentiment_model.predict(r.text_sentiment, lang)
            sent_label = sent_res.label
        except Exception as e:
            logger.error(f"[QWEN] Sentiment failed for article {aid}: {e}")
            sent_label = "NEUTRAL"
        gpu_time_sent = int((time.perf_counter() - t0) * 1000)

        # 2. Topic
        t0 = time.perf_counter()
        try:
            topic_res = topic_model.predict(getattr(r, "text_topic", r.text_sentiment), lang)
            topic_label = topic_res.label
        except Exception as e:
            logger.error(f"[QWEN] Topic failed for article {aid}: {e}")
            topic_label = "General"
        gpu_time_topic = int((time.perf_counter() - t0) * 1000)

        # 3. Update DB
        db.upsert_article_topic(article_id=aid, topic_label=topic_label)
        db.upsert_articles_enriched(
            article_id=aid,
            language=lang,
            sentiment_label=sent_label,
            gpu_time_sentiment=gpu_time_sent,
            gpu_time_topic=gpu_time_topic
        )

    logger.info("=== QWEN PASS COMPLETE ===")
