# pipeline/gpu_pass.py
"""
Stage 3 — GPU Inference Pass

Runs NER, Sentiment, and Topic models on GPU (model-by-model to avoid OOM).
Writes all results to the DB (entities, article_entities, article_topics,
articles_enriched). Returns result buffers for the exporter stage.

Returns:
    sent_results_buffer   — {(article_id, model_version): {"label", "score"}}
    topic_results_buffer  — {(article_id, model_version): {"label", "score"}}
    gpu_time_ner_ms       — {(article_id, model_version): int}
    gpu_time_sentiment_ms — {(article_id, model_version): int}
    gpu_time_topic_ms     — {(article_id, model_version): int}
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Dict, List, Tuple

import pandas as pd
import torch

from src.config import Config
from src.db_config import DatabaseConnection
from src.ner_extraction import TransformersNER, DEFAULT_NER_PARAMS
from src.sentiment_analysis import TransformersSentiment, DEFAULT_SENTIMENT_PARAMS
from src.topic_classification import TransformersTopic, DEFAULT_TOPIC_PARAMS, CATEGORY_DISPLAY
from src.preprocessing import PreprocessRouter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cuda_sync(device: int) -> None:
    if device >= 0 and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_cleanup(logger: logging.Logger, cooldown_sec: float = 1.0) -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    if cooldown_sec > 0:
        logger.info(f"[GPU] cooldown {cooldown_sec:.1f}s")
        time.sleep(cooldown_sec)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_gpu_pass(
    work_df: pd.DataFrame,
    db: DatabaseConnection,
    ner_models_by_lang: Dict[str, List[Tuple[int, str]]],
    sent_models_by_lang: Dict[str, list],
    topic_models_by_lang: Dict[str, list],
    ner_params: dict,
    sentiment_params: dict,
    topic_params: dict,
    cpu_time_ner_ms: dict,
    cpu_time_sentiment_ms: dict,
    cpu_time_topic_ms: dict,
    logger: logging.Logger,
) -> tuple[dict, dict, dict, dict, dict]:
    """
    Model-by-model GPU inference with DB writes.

    Args:
        work_df:                Articles DataFrame.
        db:                     Active DatabaseConnection.
        ner_models_by_lang:     {lang: [(mv, name), ...]}
        sent_models_by_lang:    {lang: [(mv, name, norm_fn), ...]}
        ner_params:             NER hyperparameters.
        sentiment_params:       Sentiment hyperparameters.
        topic_params:           Topic hyperparameters.
        cpu_time_ner_ms:        Timing results from CPU pass (written to DB).
        cpu_time_sentiment_ms:  Timing results from CPU pass (written to DB).
        cpu_time_topic_ms:      Timing results from CPU pass (written to DB).
        logger:                 Logger instance.

    Returns:
        (sent_results_buffer, topic_results_buffer,
         gpu_time_ner_ms, gpu_time_sentiment_ms, gpu_time_topic_ms)
    """
    gpu_device   = Config.GPU_DEVICE
    cooldown_sec = Config.GPU_COOLDOWN_SEC
    preproc      = PreprocessRouter(logger=logger)

    gpu_time_ner_ms:       Dict[Tuple[int, int], int] = {}
    gpu_time_sentiment_ms: Dict[Tuple[int, int], int] = {}
    sent_results_buffer:   Dict[Tuple[int, int], dict] = {}

    logger.info("=== GPU PASS (MODEL-BY-MODEL, WRITE RESULTS TO DB) ===")

    # ------------------------------------------------------------------ NER --
    with torch.inference_mode():
        for lang, models in ner_models_by_lang.items():
            lang_subset = work_df[work_df["lang"] == lang]
            if lang_subset.empty:
                continue

            for mv, name in models:
                logger.info(f"[GPU][NER] lang={lang} model_version={mv}: {name}")
                ner = TransformersNER(
                    model_name=name, logger=logger, preprocessor=None,
                    device=gpu_device, **ner_params,
                )

                for r in lang_subset.itertuples(index=False):
                    aid = int(r.id)
                    t0  = time.perf_counter()
                    ents = ner.predict(r.text_ner)
                    _cuda_sync(gpu_device)
                    gpu_time_ner_ms[(aid, mv)] = gpu_time_ner_ms.get((aid, mv), 0) + int((time.perf_counter() - t0) * 1000)

                    for e in ents:
                        nname = preproc.normalize_entity(e.text, lang)
                        if nname:
                            eid = db.upsert_entity(e.text.strip(), (e.label or "UNK").upper(), nname)
                            db.upsert_article_entity(
                                article_id=aid, entity_id=eid,
                                model_version=mv, confidence_score=e.score,
                            )

                del ner
                _gpu_cleanup(logger, cooldown_sec)

    # ----------------------------------------------------------- Sentiment --- DISABLED
    # with torch.inference_mode():
    #     for lang, models in sent_models_by_lang.items():
    #         lang_subset = work_df[work_df["lang"] == lang]
    #         if lang_subset.empty:
    #             continue
    #
    #         for mv, name, norm in models:
    #             logger.info(f"[GPU][SENT] lang={lang} model_version={mv}: {name}")
    #             sent = TransformersSentiment(
    #                 model_name=name, logger=logger, preprocessor=None,
    #                 device=gpu_device, probs_normalizer=norm, **sentiment_params,
    #             )
    #
    #             for r in lang_subset.itertuples(index=False):
    #                 aid = int(r.id)
    #                 t0  = time.perf_counter()
    #                 res = sent.predict(r.text_sentiment)
    #                 _cuda_sync(gpu_device)
    #                 gpu_time_sentiment_ms[(aid, mv)] = gpu_time_sentiment_ms.get((aid, mv), 0) + int((time.perf_counter() - t0) * 1000)
    #                 sent_results_buffer[(aid, mv)] = {"label": res.label, "score": float(res.score)}
    #
    #             del sent
    #             _gpu_cleanup(logger, cooldown_sec)

    # ------------------------------------------------------------- Topic ----
    topic_results_buffer: Dict[Tuple[int, int], dict] = {}
    gpu_time_topic_ms:    Dict[Tuple[int, int], int]  = {}

    with torch.inference_mode():
        for lang, models in topic_models_by_lang.items():
            lang_subset = work_df[work_df["lang"] == lang]
            if lang_subset.empty:
                continue

            for mv, name in models:
                logger.info(f"[GPU][TOPIC] lang={lang} model_version={mv}: {name}")
                topic_gpu = TransformersTopic(
                    model_name=name, logger=logger, preprocessor=None,
                    device=gpu_device, **topic_params,
                )

                for r in lang_subset.itertuples(index=False):
                    aid  = int(r.id)

                    t0   = time.perf_counter()
                    tres = topic_gpu.predict(r.text_topic, lang=lang)
                    _cuda_sync(gpu_device)
                    gpu_time_topic_ms[(aid, mv)] = gpu_time_topic_ms.get((aid, mv), 0) + int((time.perf_counter() - t0) * 1000)

                    if tres.category_id is not None:
                        topic_label = CATEGORY_DISPLAY.get(tres.category_id, {}).get(lang, "Unknown")
                    else:
                        topic_label = "Unknown"
                        
                    topic_results_buffer[(aid, mv)] = {"label": topic_label, "score": float(tres.score)}
                    db.upsert_article_topic(article_id=aid, model_version=mv, topic_label=topic_label, topic_score=tres.score)

                del topic_gpu
                _gpu_cleanup(logger, cooldown_sec)

    # ------------------------------------------------ Persist enriched rows -
    logger.info("Persisting articles_enriched rows...")
    
    # Extract unique model versions across all passes
    all_mvs = set()
    for buffer in (sent_results_buffer, topic_results_buffer, gpu_time_ner_ms):
        for k in buffer.keys():
            if isinstance(k, tuple) and len(k) == 2:
                all_mvs.add(k[1])
    
    # If no inferences happened, provide empty range to avoid skipping loop,
    # but practically all_mvs should populate during inference
    model_versions = all_mvs if all_mvs else {0, 1, 2, 3}
    
    for r in work_df.itertuples(index=False):
        aid  = int(r.id)
        lang = str(r.lang)

        for mv in model_versions:
            sent_info = sent_results_buffer.get((aid, mv), {})
            db.upsert_articles_enriched(
                article_id=aid,
                model_version=mv,
                language=lang,
                sentiment_label=sent_info.get("label"),
                sentiment_score=sent_info.get("score"),
                cpu_time_ner=cpu_time_ner_ms.get((aid, mv)),
                cpu_time_sentiment=cpu_time_sentiment_ms.get((aid, mv)),
                cpu_time_topic=cpu_time_topic_ms.get((aid, mv)),
                gpu_time_ner=gpu_time_ner_ms.get((aid, mv)),
                gpu_time_sentiment=gpu_time_sentiment_ms.get((aid, mv)),
                gpu_time_topic=gpu_time_topic_ms.get((aid, mv)),
            )

    return (
        sent_results_buffer,
        topic_results_buffer,
        gpu_time_ner_ms,
        gpu_time_sentiment_ms,
        gpu_time_topic_ms,
    )
