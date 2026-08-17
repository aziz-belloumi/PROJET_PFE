import time
import logging
import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.db_config import DatabaseConnection

from .sentiment_extraction import LLMSentiment
from .topic_extraction import LLMTopic

def run_qwen_pass(
    work_df: pd.DataFrame,
    db: DatabaseConnection,
    logger: logging.Logger,
    skipped_df: pd.DataFrame | None = None,
):
    """
    Runs Qwen (Ollama) on the sampled articles for Sentiment and Topic extraction.
    Updates the results in the dedicated Qwen database tables.
    """
    if skipped_df is not None and not skipped_df.empty:
        for r in skipped_df.itertuples(index=False):
            aid = int(r.id)
            db.upsert_qwen_articles_enriched(
                article_id=aid,
                language=None,
                sentiment_label="SKIPPED",
            )
            db.upsert_qwen_article_topic(
                article_id=aid,
                topic_label="SKIPPED",
                confidence_score=None,
            )

    if work_df.empty:
        return

    logger.info("=== GPU PASS: QWEN (OLLAMA) ===")
    
    try:
        sentiment_model = LLMSentiment(logger=logger)
        topic_model = LLMTopic(logger=logger)
    except Exception as e:
        logger.error(f"[QWEN] Failed to initialize Qwen models: {e}")
        return

    model_id = Config.MODEL_ID_MAP.get(Config.QWEN_BENCHMARK_MODEL, 7)

    pbar = tqdm(
        work_df.itertuples(index=False),
        total=len(work_df),
        desc="[QWEN] Inference",
        unit="art",
    )
    for r in pbar:
        aid = int(r.id)
        lang = str(r.lang)
        pbar.set_postfix({"id": aid, "lang": lang})
        
        # 1. Sentiment
        t0 = time.perf_counter()
        try:
            sent_res = sentiment_model.predict(r.text_sentiment, lang)
            sent_label = sent_res.label
            sent_score = sent_res.score
        except Exception as e:
            logger.error(f"[QWEN] Sentiment failed for article {aid}: {e}")
            sent_label = "NEUTRAL"
            sent_score = 0.0
        gpu_time_sent = int((time.perf_counter() - t0) * 1000)
        
        # 2. Topic
        t0 = time.perf_counter()
        try:
            topic_res = topic_model.predict(getattr(r, "text_topic", r.text_sentiment), lang)
            topic_label = topic_res.label
            topic_score = topic_res.score
        except Exception as e:
            logger.error(f"[QWEN] Topic failed for article {aid}: {e}")
            topic_label = "General"
            topic_score = 0.0
        gpu_time_topic = int((time.perf_counter() - t0) * 1000)
        
        # 3. Update DB
        # Update topic in qwen_article_topics
        db.upsert_qwen_article_topic(
            article_id=aid,
            topic_label=topic_label,
            confidence_score=topic_score,
        )
        
        # Update sentiment in qwen_articles_enriched
        db.upsert_qwen_articles_enriched(
            article_id=aid,
            language=lang,
            sentiment_label=sent_label,
            sentiment_score=sent_score,
        )

        # Update Qwen benchmarks
        db.upsert_qwen_benchmark_result(
            article_id=aid,
            task="sentiment",
            model_id=model_id,
            language=lang,
            device="GPU",
            total_inf_time_sec=round(gpu_time_sent / 1000.0, 6),
            avg_ms_per_doc=float(gpu_time_sent),
        )
        db.upsert_qwen_benchmark_result(
            article_id=aid,
            task="topic",
            model_id=model_id,
            language=lang,
            device="GPU",
            total_inf_time_sec=round(gpu_time_topic / 1000.0, 6),
            avg_ms_per_doc=float(gpu_time_topic),
        )
        
    logger.info("=== QWEN PASS COMPLETE ===")
