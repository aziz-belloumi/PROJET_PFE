# pipeline/cpu_pass.py
"""
Stage 2 — CPU Timing Pass

Loads all NER, Sentiment, and Topic models on CPU and runs inference
on every article in work_df purely to record timing. Results are discarded.

Returns:
    cpu_time_ner_ms       — dict[(article_id, model_version), int]  milliseconds
    cpu_time_sentiment_ms — dict[(article_id, model_version), int]
    cpu_time_topic_ms     — dict[article_id, int]
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Dict, List, Tuple

import pandas as pd
import torch

from src.config import Config
from src.ner_extraction import TransformersNER, DEFAULT_NER_PARAMS
from src.sentiment_analysis import TransformersSentiment, DEFAULT_SENTIMENT_PARAMS
from src.topic_classification import TransformersTopic, DEFAULT_TOPIC_PARAMS


def run_cpu_pass(
    work_df: pd.DataFrame,
    ner_models_by_lang: Dict[str, List[Tuple[int, str]]],
    sent_models_by_lang: Dict[str, list],
    ner_params: dict,
    sentiment_params: dict,
    topic_params: dict,
    logger: logging.Logger,
) -> tuple[dict, dict, dict]:
    """
    Timed CPU inference across all models and articles.

    Args:
        work_df:            Articles DataFrame with preprocessed text columns.
        ner_models_by_lang: {lang: [(model_version, model_name), ...]}
        sent_models_by_lang:{lang: [(model_version, model_name, norm_fn), ...]}
        ner_params:         NER hyperparameters.
        sentiment_params:   Sentiment hyperparameters.
        topic_params:       Topic hyperparameters.
        logger:             Logger instance.

    Returns:
        (cpu_time_ner_ms, cpu_time_sentiment_ms, cpu_time_topic_ms)
    """
    cpu_device = Config.CPU_DEVICE

    cpu_time_ner_ms:       Dict[Tuple[int, int], int] = {}
    cpu_time_sentiment_ms: Dict[Tuple[int, int], int] = {}
    cpu_time_topic_ms:     Dict[int, int]             = {}

    logger.info("=== CPU PASS (TIMING ONLY) ===")

    # ---- Load models ----
    ner_cpu_models_by_lang = {
        lang: [
            (mv, TransformersNER(model_name=name, logger=logger, preprocessor=None, device=cpu_device, **ner_params))
            for mv, name in models
        ]
        for lang, models in ner_models_by_lang.items()
    }

    sent_cpu_models_by_lang = {
        lang: [
            (mv, TransformersSentiment(model_name=name, logger=logger, preprocessor=None, device=cpu_device, probs_normalizer=norm, **sentiment_params))
            for mv, name, norm in models
        ]
        for lang, models in sent_models_by_lang.items()
    }

    topic_cpu = TransformersTopic(
        model_name=Config.TOPIC_MODEL,
        logger=logger,
        preprocessor=None,
        device=cpu_device,
        **topic_params,
    )

    # ---- Timed inference ----
    with torch.inference_mode():
        for r in work_df.itertuples(index=False):
            aid  = int(r.id)
            lang = str(r.lang)

            for mv, model in ner_cpu_models_by_lang.get(lang, []):
                t0 = time.perf_counter()
                _ = model.predict(r.text_ner)
                cpu_time_ner_ms[(aid, mv)] = cpu_time_ner_ms.get((aid, mv), 0) + int((time.perf_counter() - t0) * 1000)

            for mv, model in sent_cpu_models_by_lang.get(lang, []):
                t0 = time.perf_counter()
                _ = model.predict(r.text_sentiment)
                cpu_time_sentiment_ms[(aid, mv)] = cpu_time_sentiment_ms.get((aid, mv), 0) + int((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            _ = topic_cpu.predict(r.text_topic, lang=lang)
            cpu_time_topic_ms[aid] = int((time.perf_counter() - t0) * 1000)

    # ---- Cleanup ----
    del ner_cpu_models_by_lang, sent_cpu_models_by_lang, topic_cpu
    gc.collect()

    return cpu_time_ner_ms, cpu_time_sentiment_ms, cpu_time_topic_ms
