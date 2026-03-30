# pipeline/gpu_pass.py
"""
Stage 3 — GPU Inference Pass

Runs NER, Sentiment, and Topic models on GPU (model-by-model to avoid OOM).
Writes all results to the DB (entities, article_entities, article_topics,
articles_enriched). Returns result buffers for the exporter stage.

Topic extraction:
- Runs per article using predict(text, lang).
- BARTTopic  : zero-shot classifier (facebook/bart-large-mnli).
               Selects the best label from the language-specific candidate list.
               Returns the top label and its confidence score directly.
- FlanT5Topic: generative text-to-text extractor (google/flan-t5-large).
               Generates a short topic label (1-3 words) from a prompt.
               Score is a documented generation-confidence proxy (mean token probability).

NER stays per-language and per-article.
Sentiment remains disabled as in the current project state.

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
from src.ner_extraction import TransformersNER, DEFAULT_NER_PARAMS, ModelLoadError
from src.sentiment_analysis import LLMSentiment, DEFAULT_SENTIMENT_PARAMS
from src.preprocessing import PreprocessRouter
from src.topic_generation import LLMTopic


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


def _is_valid_topic_text(text: object, min_chars: int = 30, min_words: int = 5) -> bool:
    """
    Basic quality gate for topic-model input text.
    Rejects missing, empty, whitespace-only, and very short texts.
    """
    if text is None:
        return False
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return False
    if len(text) < min_chars:
        return False
    if len(text.split()) < min_words:
        return False
    return True


def _fallback_topic_label(reason: str, lang: str) -> str:
    """
    Standardized topic fallback labels for downstream debugging and analytics.
    """
    reason = (reason or "unknown").strip().lower()
    lang = (lang or "unknown").strip().lower()
    return f"Topic_Fallback:{reason}:{lang}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_gpu_pass(
    work_df: pd.DataFrame,
    db: DatabaseConnection,
    ner_models_by_lang: Dict[str, List[Tuple[int, str]]],
    sent_models_by_lang: Dict[str, list],
    topic_extractors: List[Tuple[int, str]],
    ner_params: dict,
    sentiment_params: dict,
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
        topic_extractors:       [(mv, extractor_type), ...] where extractor_type
                                is 'bart' or 'flant5'.
        ner_params:             NER hyperparameters.
        sentiment_params:       Sentiment hyperparameters.
        cpu_time_ner_ms:        Timing results from CPU pass (written to DB).
        cpu_time_sentiment_ms:  Timing results from CPU pass (written to DB).
        cpu_time_topic_ms:      Timing results from CPU pass (written to DB).
        logger:                 Logger instance.

    Returns:
        (sent_results_buffer, topic_results_buffer,
         gpu_time_ner_ms, gpu_time_sentiment_ms, gpu_time_topic_ms)
    """
    gpu_device = Config.GPU_DEVICE
    cooldown_sec = Config.GPU_COOLDOWN_SEC
    preproc = PreprocessRouter(logger=logger)

    gpu_time_ner_ms: Dict[Tuple[int, int], int] = {}
    gpu_time_sentiment_ms: Dict[Tuple[int, int], int] = {}
    sent_results_buffer: Dict[Tuple[int, int], dict] = {}

    logger.info("=== GPU PASS (MODEL-BY-MODEL, WRITE RESULTS TO DB) ===")

    # ------------------------------------------------------------------ NER --
    with torch.inference_mode():
        for lang, models in ner_models_by_lang.items():
            lang_subset = work_df[work_df["lang"] == lang]
            if lang_subset.empty:
                continue

            for mv, name in models:
                logger.info(f"[GPU][NER] lang={lang} model_version={mv}: {name}")
                try:
                    ner = TransformersNER(
                        model_name=name,
                        logger=logger,
                        preprocessor=None,
                        device=gpu_device,
                        **ner_params,
                    )
                except ModelLoadError as e:
                    logger.error(f"Skipping model {name} due to load error: {e}")
                    continue

                for r in lang_subset.itertuples(index=False):
                    aid = int(r.id)
                    t0 = time.perf_counter()
                    ents = ner.predict(r.text_ner)
                    _cuda_sync(gpu_device)
                    gpu_time_ner_ms[(aid, mv)] = gpu_time_ner_ms.get((aid, mv), 0) + int(
                        (time.perf_counter() - t0) * 1000
                    )

                    for e in ents:
                        nname = preproc.normalize_entity(e.text, lang)
                        if nname:
                            eid = db.upsert_entity(
                                e.text.strip(),
                                (e.label or "UNK").upper(),
                                nname,
                            )
                            db.upsert_article_entity(
                                article_id=aid,
                                entity_id=eid,
                                model_version=mv,
                                confidence_score=e.score,
                            )

                del ner
                _gpu_cleanup(logger, cooldown_sec)

    # ----------------------------------------------------------- Sentiment ---
    gpu_time_sentiment_ms: Dict[Tuple[int, int], int] = {}
    sent_results_buffer: Dict[Tuple[int, int], dict] = {}

    with torch.inference_mode():
        # Load LLM once for all sentiment processing
        logger.info(f"[GPU][SENT] Loading LLM model_version=0: {Config.LLM_MODEL}")
        sent = LLMSentiment(logger=logger, preprocessor=None)

        for r in work_df.itertuples(index=False):
            aid = int(r.id)
            lang = str(r.lang)
            t0 = time.perf_counter()
            res = sent.predict(r.text_sentiment, lang)
            gpu_time_sentiment_ms[(aid, 0)] = gpu_time_sentiment_ms.get((aid, 0), 0) + int((time.perf_counter() - t0) * 1000)
            sent_results_buffer[(aid, 0)] = {"label": res.label, "score": float(res.score)}

        del sent
        _gpu_cleanup(logger, cooldown_sec)

    # ------------------------------------------------------------- Topic ----
    topic_results_buffer: Dict[Tuple[int, int], dict] = {}
    gpu_time_topic_ms: Dict[Tuple[int, int], int] = {}

    if not work_df.empty and topic_extractors:
        min_topic_chars = int(getattr(Config, "TOPIC_MIN_TEXT_CHARS", 30))
        min_topic_words = int(getattr(Config, "TOPIC_MIN_TEXT_WORDS", 5))

        # Load LLM once for all topic processing
        logger.info(f"[GPU][TOPIC] Loading LLM model_version=0: {Config.LLM_MODEL}")
        extractor = LLMTopic(logger=logger)

        for r in work_df.itertuples(index=False):
            aid = int(r.id)
            lang = str(r.lang)
            text_topic = getattr(r, "text_topic", None)

            if not _is_valid_topic_text(
                text_topic,
                min_chars=min_topic_chars,
                min_words=min_topic_words,
            ):
                label = _fallback_topic_label("invalid_text", lang)
                gpu_time_topic_ms[(aid, 0)] = 0
                topic_results_buffer[(aid, 0)] = {"label": label, "score": 0.0}
                db.upsert_article_topic(
                    article_id=aid,
                    topic_label=label,
                )
                continue

            t0 = time.perf_counter()
            try:
                result = extractor.predict(str(text_topic), lang)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)

                safe_label = str(result.label) if result.label else _fallback_topic_label("null_label", lang)
                safe_score = float(result.score) if result.score is not None else 0.0

            except Exception as e:
                logger.exception(
                    f"[GPU][TOPIC] Predict failed for article_id={aid} "
                    f"model_version=0: {e}"
                )
                safe_label = _fallback_topic_label("predict_failed", lang)
                safe_score = 0.0
                elapsed_ms = int((time.perf_counter() - t0) * 1000)

            gpu_time_topic_ms[(aid, 0)] = elapsed_ms
            topic_results_buffer[(aid, 0)] = {
                "label": safe_label,
                "score": safe_score,
            }
            db.upsert_article_topic(
                article_id=aid,
                topic_label=safe_label,
            )

        del extractor
        _gpu_cleanup(logger, cooldown_sec)

    # ------------------------------------------------ Persist enriched rows -
    logger.info("Persisting articles_enriched rows...")

    # Extract unique model versions (only for NER since sentiment/topic use unified model)
    all_mvs: set[int] = set()
    for k in gpu_time_ner_ms.keys():
        if isinstance(k, tuple) and len(k) == 2:
            all_mvs.add(k[1])

    model_versions = all_mvs if all_mvs else {0}

    for r in work_df.itertuples(index=False):
        aid = int(r.id)
        lang = str(r.lang)

        # Insert/update sentiment and topic results (unified model)
        sent_info = sent_results_buffer.get((aid, 0), {})
        db.upsert_articles_enriched(
            article_id=aid,
            language=lang,
            sentiment_label=sent_info.get("label"),
            cpu_time_ner=cpu_time_ner_ms.get((aid, 0)),
            cpu_time_sentiment=cpu_time_sentiment_ms.get((aid, 0)),
            cpu_time_topic=cpu_time_topic_ms.get((aid, 0)),
            gpu_time_ner=gpu_time_ner_ms.get((aid, 0)),
            gpu_time_sentiment=gpu_time_sentiment_ms.get((aid, 0)),
            gpu_time_topic=gpu_time_topic_ms.get((aid, 0)),
        )

    return (
        sent_results_buffer,
        topic_results_buffer,
        gpu_time_ner_ms,
        gpu_time_sentiment_ms,
        gpu_time_topic_ms,
    )