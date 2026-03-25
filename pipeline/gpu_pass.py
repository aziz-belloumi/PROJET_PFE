# pipeline/gpu_pass.py
"""
Stage 3 — GPU Inference Pass

Runs NER, Sentiment, and Topic models on GPU (model-by-model to avoid OOM).
Writes all results to the DB (entities, article_entities, article_topics,
articles_enriched). Returns result buffers for the exporter stage.

Topic modeling note:
- NER remains per-language and per-article.
- Sentiment remains disabled as in the current project state.
- Topic modeling is batch-level and is now executed PER LANGUAGE SUBSET
  instead of on one mixed multilingual batch.
- Invalid / too-short topic texts receive explicit fallback labels.
- Very small language subsets do not force BERTopic/Top2Vec fitting.

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
from src.preprocessing import PreprocessRouter
from src.topic.bertopic_wrapper import BERTopicWrapper
from src.topic.top2vec_wrapper import Top2VecWrapper


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
    Rejects missing, empty, whitespace-only, too-short, and very low-token texts.
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


def _instantiate_topic_model(name: str, logger: logging.Logger, topic_params: dict):
    """
    Creates the requested batch topic model wrapper.
    """
    lname = (name or "").strip().lower()
    if lname == "bertopic":
        return BERTopicWrapper(logger=logger, **topic_params)
    if lname == "top2vec":
        return Top2VecWrapper(logger=logger, **topic_params)

    raise ValueError(f"Unknown topic model: {name}")


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
    batch_topic_models: List[Tuple[int, str]],
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
        batch_topic_models:     [(mv, name), ...] for BERTopic / Top2Vec.
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
                ner = TransformersNER(
                    model_name=name,
                    logger=logger,
                    preprocessor=None,
                    device=gpu_device,
                    **ner_params,
                )

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
    gpu_time_topic_ms: Dict[Tuple[int, int], int] = {}

    if not work_df.empty and batch_topic_models:
        min_topic_chars = int(getattr(Config, "TOPIC_MIN_TEXT_CHARS", 30))
        min_topic_words = int(getattr(Config, "TOPIC_MIN_TEXT_WORDS", 5))
        min_docs_per_lang = int(getattr(Config, "TOPIC_MIN_DOCS_PER_LANG", 15))

        for mv, name in batch_topic_models:
            logger.info(f"[GPU][TOPIC] Starting topic model_version={mv}: {name}")

            try:
                topic_model = _instantiate_topic_model(name=name, logger=logger, topic_params=topic_params)
            except Exception as e:
                logger.exception(f"[GPU][TOPIC] Failed to initialize model {name}: {e}")
                # Apply failure fallback to every article for this model version
                for r in work_df.itertuples(index=False):
                    aid = int(r.id)
                    lang = str(r.lang)
                    label = _fallback_topic_label("model_init_failed", lang)
                    score = 0.0
                    gpu_time_topic_ms[(aid, mv)] = 0
                    topic_results_buffer[(aid, mv)] = {"label": label, "score": score}
                    db.upsert_article_topic(
                        article_id=aid,
                        model_version=mv,
                        topic_label=label,
                        topic_score=score,
                    )
                _gpu_cleanup(logger, cooldown_sec)
                continue

            try:
                # Process per language subset instead of fitting on one mixed-language batch
                for lang in sorted(work_df["lang"].dropna().astype(str).unique().tolist()):
                    lang_subset = work_df[work_df["lang"] == lang].copy()
                    if lang_subset.empty:
                        continue

                    logger.info(
                        f"[GPU][TOPIC] model_version={mv} model={name} lang={lang} "
                        f"docs={len(lang_subset)}"
                    )

                    valid_rows = []
                    invalid_rows = []

                    for r in lang_subset.itertuples(index=False):
                        text_topic = getattr(r, "text_topic", None)
                        if _is_valid_topic_text(
                            text_topic,
                            min_chars=min_topic_chars,
                            min_words=min_topic_words,
                        ):
                            valid_rows.append(r)
                        else:
                            invalid_rows.append(r)

                    logger.info(
                        f"[GPU][TOPIC] model_version={mv} model={name} lang={lang} "
                        f"valid_docs={len(valid_rows)} invalid_docs={len(invalid_rows)}"
                    )

                    # Explicit fallback for invalid / weak input texts
                    for r in invalid_rows:
                        aid = int(r.id)
                        label = _fallback_topic_label("invalid_text", lang)
                        score = 0.0
                        gpu_time_topic_ms[(aid, mv)] = 0
                        topic_results_buffer[(aid, mv)] = {"label": label, "score": score}
                        db.upsert_article_topic(
                            article_id=aid,
                            model_version=mv,
                            topic_label=label,
                            topic_score=score,
                        )

                    # If too few valid docs exist for meaningful topic discovery, avoid forcing model fit
                    if len(valid_rows) < min_docs_per_lang:
                        logger.warning(
                            f"[GPU][TOPIC] Skipping fit for model_version={mv} model={name} lang={lang} "
                            f"because valid_docs={len(valid_rows)} < min_docs_per_lang={min_docs_per_lang}"
                        )
                        for r in valid_rows:
                            aid = int(r.id)
                            label = _fallback_topic_label("too_few_docs", lang)
                            score = 0.0
                            gpu_time_topic_ms[(aid, mv)] = 0
                            topic_results_buffer[(aid, mv)] = {"label": label, "score": score}
                            db.upsert_article_topic(
                                article_id=aid,
                                model_version=mv,
                                topic_label=label,
                                topic_score=score,
                            )
                        continue

                    batch_article_ids = [int(r.id) for r in valid_rows]
                    batch_texts = [str(r.text_topic) for r in valid_rows]

                    t0 = time.perf_counter()
                    try:
                        labels, scores = topic_model.fit_predict(batch_texts)
                        _cuda_sync(gpu_device)
                        elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    except Exception as e:
                        logger.exception(
                            f"[GPU][TOPIC] Fit failed for model_version={mv} model={name} lang={lang}: {e}"
                        )
                        labels = [_fallback_topic_label("fit_failed", lang)] * len(batch_texts)
                        scores = [0.0] * len(batch_texts)
                        elapsed_ms = 0

                    if len(labels) != len(batch_article_ids) or len(scores) != len(batch_article_ids):
                        logger.error(
                            f"[GPU][TOPIC] Output length mismatch for model_version={mv} model={name} lang={lang}: "
                            f"article_ids={len(batch_article_ids)} labels={len(labels)} scores={len(scores)}"
                        )
                        labels = [_fallback_topic_label("length_mismatch", lang)] * len(batch_article_ids)
                        scores = [0.0] * len(batch_article_ids)
                        elapsed_ms = 0

                    ms_per_article = elapsed_ms // max(1, len(batch_article_ids))

                    for aid, label, score in zip(batch_article_ids, labels, scores):
                        safe_label = str(label) if label is not None else _fallback_topic_label("null_label", lang)
                        safe_score = float(score) if score is not None else 0.0

                        gpu_time_topic_ms[(aid, mv)] = ms_per_article
                        topic_results_buffer[(aid, mv)] = {
                            "label": safe_label,
                            "score": safe_score,
                        }
                        db.upsert_article_topic(
                            article_id=aid,
                            model_version=mv,
                            topic_label=safe_label,
                            topic_score=safe_score,
                        )

                del topic_model
                _gpu_cleanup(logger, cooldown_sec)

            except Exception as e:
                logger.exception(f"[GPU][TOPIC] Unhandled error for model_version={mv} model={name}: {e}")
                # Conservative fallback for any article not already written for this model version
                for r in work_df.itertuples(index=False):
                    aid = int(r.id)
                    lang = str(r.lang)
                    if (aid, mv) not in topic_results_buffer:
                        label = _fallback_topic_label("unhandled_error", lang)
                        score = 0.0
                        gpu_time_topic_ms[(aid, mv)] = 0
                        topic_results_buffer[(aid, mv)] = {"label": label, "score": score}
                        db.upsert_article_topic(
                            article_id=aid,
                            model_version=mv,
                            topic_label=label,
                            topic_score=score,
                        )

                try:
                    del topic_model
                except Exception:
                    pass
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
        aid = int(r.id)
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