from __future__ import annotations

import gc
import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from src.config import Config
from src.db_config import DatabaseConnection
from src.ner_extraction import TransformersNER, GLiNERNER, DEFAULT_NER_PARAMS, ModelLoadError
from src.sentiment_extraction import LLMSentiment, DEFAULT_SENTIMENT_PARAMS
from src.preprocessing import PreprocessRouter
from src.topic_extraction import LLMTopic, TransformerTopic


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
        time.sleep(cooldown_sec)


def _reset_peak(device: int, logger: logging.Logger, tag: str) -> bool:
    """Reset CUDA peak-memory counter. Returns True if tracking is available."""
    if torch.cuda.is_available() and 0 <= device < torch.cuda.device_count():
        try:
            torch.cuda.reset_peak_memory_stats(device)
            return True
        except RuntimeError as e:
            logger.warning(f"[{tag}] reset_peak_memory_stats failed: {e}")
    return False


def _read_peak_mb(device: int) -> float | None:
    """Read peak GPU memory in MB, or None if unavailable."""
    if torch.cuda.is_available() and 0 <= device < torch.cuda.device_count():
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return None


def _is_valid_topic_text(text: object, min_chars: int = 30, min_words: int = 5) -> bool:
    """Basic quality gate for topic-model input text."""
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
    """Return the language-specific 'General' label as a topic fallback."""
    lang = (lang or "en").strip().lower()
    return Config.CATEGORY_DISPLAY[17].get(lang, "General")


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
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """
    Model-by-model GPU inference with DB writes.

    Args:
        work_df:                Articles DataFrame.
        db:                     Active DatabaseConnection.
        ner_models_by_lang:     {lang: [(mv, name), ...]}
        sent_models_by_lang:    {lang: [(mv, name), ...]} (reserved for future use)
        topic_extractors:       [(mv, extractor_type, lang), ...]
        ner_params:             NER hyperparameters.
        sentiment_params:       Sentiment hyperparameters.
        cpu_time_ner_ms:        Timing results from CPU pass.
        cpu_time_sentiment_ms:  Timing results from CPU pass.
        cpu_time_topic_ms:      Timing results from CPU pass.
        logger:                 Logger instance.

    Returns:
        (sent_results_buffer, topic_results_buffer,
         gpu_time_ner_ms, gpu_time_sentiment_ms, gpu_time_topic_ms,
         ner_results_buffer)
    """
    gpu_device   = Config.GPU_DEVICE
    cooldown_sec = Config.GPU_COOLDOWN_SEC
    preproc      = PreprocessRouter(logger=logger)

    gpu_time_ner_ms:       Dict[Tuple[int, int], int]  = {}
    gpu_time_sentiment_ms: Dict[Tuple[int, int], int]  = {}
    sent_results_buffer:   Dict[Tuple[int, int], dict] = {}
    ner_results_buffer:    Dict[int, dict]              = {}
    peak_ner_mb:           Dict[int, float | None]      = {}
    peak_sentiment_mb:     Dict[int, float | None]      = {}
    peak_topic_mb:         Dict[int, float | None]      = {}


    # ------------------------------------------------------------------ NER --
    with torch.inference_mode():
        for lang, models in ner_models_by_lang.items():
            lang_subset = work_df[work_df["lang"] == lang]
            if lang_subset.empty:
                continue

            for mv, name in models:
                logger.info(f"[GPU][NER] lang={lang} model_version={mv}: {name}")
                try:
                    is_gliner = "gliner" in name.lower()
                    if is_gliner:
                        ner = GLiNERNER(model_name=name, logger=logger, device=gpu_device)
                    else:
                        ner = TransformersNER(
                            model_name=name,
                            logger=logger,
                            preprocessor=None,
                            device=gpu_device,
                            **ner_params,
                        )
                except Exception as e:
                    logger.error(f"Skipping model {name} due to load error: {e}")
                    continue

                rows = list(lang_subset.itertuples(index=False))
                pbar = tqdm(rows, desc=f"[NER] {lang.upper()} | {name.split('/')[-1]}", unit="art", leave=False)
                for r in pbar:
                    aid = int(r.id)

                    cuda_ok = _reset_peak(gpu_device, logger, "NER")
                    t0 = time.perf_counter()
                    if is_gliner:
                        ents = ner.predict(r.text_ner, language=lang)
                    else:
                        ents = ner.predict(r.text_ner)
                    _cuda_sync(gpu_device)
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)

                    gpu_time_ner_ms[(aid, mv)] = gpu_time_ner_ms.get((aid, mv), 0) + elapsed_ms
                    peak_ner_mb[aid] = _read_peak_mb(gpu_device) if cuda_ok else None

                    if is_gliner:
                        if aid not in ner_results_buffer:
                            ner_results_buffer[aid] = {"text_ner": r.text_ner, "entities": []}
                        ner_results_buffer[aid]["entities"].extend(
                            [f"{e.text} ({e.label})" for e in ents]
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

                pbar.close()
                del ner
                _gpu_cleanup(logger, cooldown_sec)

    # ----------------------------------------------------------- Sentiment ---
    gpu_time_sentiment_ms = {}
    sent_results_buffer   = {}

    with torch.inference_mode():
        logger.info("[GPU][SENT] Initializing LLMSentiment analyzer...")
        sent = LLMSentiment(logger=logger, preprocessor=None, device=gpu_device)

        rows = list(work_df.itertuples(index=False))
        pbar = tqdm(rows, desc="[SENTIMENT]", unit="art", leave=False)
        for r in pbar:
            aid  = int(r.id)
            lang = str(r.lang)

            cuda_ok = _reset_peak(gpu_device, logger, "SENT")

            t0  = time.perf_counter()
            res = sent.predict(r.text_sentiment, lang)
            _cuda_sync(gpu_device)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            peak_sentiment_mb[aid] = _read_peak_mb(gpu_device) if cuda_ok else None

            sent_results_buffer[(aid, 0)]   = {"label": res.label, "score": float(res.score)}
            gpu_time_sentiment_ms[(aid, 0)] = elapsed_ms

            # Log UNK labels to file only – no console prints
            if res.label == "UNK" or res.label not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
                logger.warning(
                    f"[SENTIMENT_UNK] article_id={aid} | lang={lang} | "
                    f"label={res.label} | score={res.score:.4f} | "
                    f"text_len={len(r.text_sentiment) if r.text_sentiment else 0}"
                )

        pbar.close()
        del sent
        _gpu_cleanup(logger, cooldown_sec)

    # ------------------------------------------------------------- Topic ----
    topic_results_buffer: Dict[Tuple[int, int], dict] = {}
    gpu_time_topic_ms:    Dict[Tuple[int, int], int]  = {}

    if not work_df.empty and topic_extractors:
        min_topic_chars = Config.TOPIC_MIN_TEXT_CHARS
        min_topic_words = Config.TOPIC_MIN_TEXT_WORDS

        # Initialise extractors per language
        extractors_by_lang: Dict[str, Dict[int, dict]] = {"ar": {}, "en": {}, "fr": {}}

        for mv, extractor_type, *extra_args in topic_extractors:
            lang = extra_args[0] if extra_args else "en"
            logger.info(f"[GPU][TOPIC] Initializing {extractor_type} extractor v{mv} for lang={lang}")

            if extractor_type.lower() == "llm":
                extractor = LLMTopic(logger=logger, device=gpu_device)
            elif extractor_type.lower() == "transformer":
                try:
                    extractor = TransformerTopic(lang=lang, logger=logger, device=gpu_device)
                except Exception as e:
                    logger.error(f"[GPU][TOPIC] Failed to load TransformerTopic for lang={lang}: {e}")
                    continue
            else:
                logger.warning(f"[GPU][TOPIC] Unknown extractor type: {extractor_type}")
                continue

            extractors_by_lang[lang][mv] = {"extractor": extractor, "type": extractor_type.lower()}

        rows = list(work_df.itertuples(index=False))
        pbar = tqdm(rows, desc="[TOPIC]", unit="art", leave=False)
        for r in pbar:
            aid        = int(r.id)
            lang       = str(r.lang)
            text_topic = getattr(r, "text_topic", None)

            # Fallback for invalid / too-short text
            if not _is_valid_topic_text(text_topic, min_chars=min_topic_chars, min_words=min_topic_words):
                label = _fallback_topic_label("invalid_text", lang)
                gpu_time_topic_ms[(aid, 0)]   = 0
                peak_topic_mb[aid]            = None
                topic_results_buffer[(aid, 0)] = {"label": label, "score": 0.0}
                db.upsert_article_topic(article_id=aid, language=lang, topic_label=label, confidence_score=0.0)
                continue

            lang_extractors = extractors_by_lang.get(lang, {})
            if not lang_extractors:
                label = _fallback_topic_label("no_extractor", lang)
                gpu_time_topic_ms[(aid, 0)]   = 0
                peak_topic_mb[aid]            = None
                topic_results_buffer[(aid, 0)] = {"label": label, "score": 0.0}
                db.upsert_article_topic(article_id=aid, language=lang, topic_label=label, confidence_score=0.0)
                continue

            mv, ext_info = next(iter(lang_extractors.items()))
            extractor    = ext_info["extractor"]

            cuda_ok = _reset_peak(gpu_device, logger, "TOPIC")
            t0 = time.perf_counter()
            try:
                result = extractor.predict(str(text_topic), lang)
                _cuda_sync(gpu_device)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)

                safe_label = str(result.label) if result.label else _fallback_topic_label("null_label", lang)
                safe_score = float(result.score) if result.score is not None else 0.0

            except Exception as e:
                logger.exception(
                    f"[GPU][TOPIC] Predict failed for article_id={aid} model_version={mv}: {e}"
                )
                safe_label = _fallback_topic_label("predict_failed", lang)
                safe_score = 0.0
                elapsed_ms = int((time.perf_counter() - t0) * 1000)

            gpu_time_topic_ms[(aid, mv)]   = elapsed_ms
            peak_topic_mb[aid]             = _read_peak_mb(gpu_device) if cuda_ok else None
            topic_results_buffer[(aid, mv)] = {"label": safe_label, "score": safe_score}
            db.upsert_article_topic(article_id=aid, language=lang, topic_label=safe_label, confidence_score=safe_score)

        pbar.close()

        # Cleanup extractors
        for lang_exts in extractors_by_lang.values():
            for ext_info in lang_exts.values():
                try:
                    if hasattr(ext_info["extractor"], "unload"):
                        ext_info["extractor"].unload()
                except Exception:
                    pass

    # ----------------------------------- Write article_sentiments rows --------
    for r in work_df.itertuples(index=False):
        aid  = int(r.id)
        lang = str(r.lang)

        sent_info  = sent_results_buffer.get((aid, 0), {})

        db.upsert_article_sentiments(
            article_id=aid,
            language=lang,
            sentiment_label=sent_info.get("label"),
            sentiment_score=sent_info.get("score"),
        )

    # ----------------------------------------- Write benchmark_results rows --
    SENT_MODEL_ID_KEYS  = {"ar": "ar_sentiment_ft", "en": "en_sentiment_ft", "fr": "fr_sentiment_ft"}
    TOPIC_MODEL_ID_KEYS = {"ar": "ar_topic_ft",     "en": "en_topic_ft",     "fr": "fr_topic_ft"}

    sent_times_by_lang:  dict = defaultdict(list)
    topic_times_by_lang: dict = defaultdict(list)

    for r in work_df.itertuples(index=False):
        aid  = int(r.id)
        lang = str(r.lang)

        t_sent = gpu_time_sentiment_ms.get((aid, 0))
        if t_sent is not None:
            sent_times_by_lang[lang].append((aid, t_sent))

        t_top = None
        for (k_aid, k_mv), t_val in gpu_time_topic_ms.items():
            if k_aid == aid:
                t_top = t_val
                break
        if t_top is not None:
            topic_times_by_lang[lang].append((aid, t_top))

    # NER benchmark rows  (keyed by lang + mv — one row per article per model)
    _aid_to_lang = dict(zip(work_df["id"].astype(int), work_df["lang"].astype(str)))
    ner_times_by_lang_mv: dict = defaultdict(list)  # (lang, mv) -> [(aid, t_ms)]

    for (aid, mv), t_ms in gpu_time_ner_ms.items():
        lang = _aid_to_lang.get(aid)
        if lang is not None:
            ner_times_by_lang_mv[(lang, mv)].append((aid, t_ms))

    for (lang, mv), items in ner_times_by_lang_mv.items():
        model_id = int(mv)
        total_ms = sum(t for _, t in items)
        avg_ms   = total_ms / len(items) if items else 0.0
        for aid, t_ms in items:
            try:
                db.upsert_benchmark_result(
                    article_id=aid,
                    task="ner",
                    model_id=model_id,
                    language=lang,
                    device="GPU",
                    total_inf_time_sec=round(t_ms / 1000.0, 6),
                    avg_ms_per_doc=round(avg_ms, 4),
                    peak_gpu_mb=peak_ner_mb.get(aid),
                )
            except Exception as e:
                logger.warning(f"[BENCHMARK] ner write failed for article_id={aid}: {e}")

    # Sentiment benchmark rows
    for lang, items in sent_times_by_lang.items():
        mid_key  = Config.SENT_MODEL_ID_KEYS.get(lang)
        model_id = Config.MODEL_ID_MAP.get(mid_key, -1) if mid_key else -1
        total_ms = sum(t for _, t in items)
        avg_ms   = total_ms / len(items) if items else 0.0
        for aid, t_ms in items:
            try:
                db.upsert_benchmark_result(
                    article_id=aid,
                    task="sentiment",
                    model_id=model_id,
                    language=lang,
                    device="GPU",
                    total_inf_time_sec=round(t_ms / 1000.0, 6),
                    avg_ms_per_doc=round(avg_ms, 4),
                    peak_gpu_mb=peak_sentiment_mb.get(aid),
                )
            except Exception as e:
                logger.warning(f"[BENCHMARK] sentiment write failed for article_id={aid}: {e}")

    # Topic benchmark rows
    topic_times_by_lang_mv: dict = defaultdict(list)  # (lang, mv) -> [(aid, t_ms)]
    for (aid, mv), t_ms in gpu_time_topic_ms.items():
        lang = _aid_to_lang.get(aid)
        if lang is not None:
            topic_times_by_lang_mv[(lang, mv)].append((aid, t_ms))

    for (lang, mv), items in topic_times_by_lang_mv.items():
        model_id = int(mv)
        total_ms = sum(t for _, t in items)
        avg_ms   = total_ms / len(items) if items else 0.0
        for aid, t_ms in items:
            try:
                db.upsert_benchmark_result(
                    article_id=aid,
                    task="topic",
                    model_id=model_id,
                    language=lang,
                    device="GPU",
                    total_inf_time_sec=round(t_ms / 1000.0, 6),
                    avg_ms_per_doc=round(avg_ms, 4),
                    peak_gpu_mb=peak_topic_mb.get(aid),
                )
            except Exception as e:
                logger.warning(f"[BENCHMARK] topic write failed for article_id={aid}: {e}")

    return (
        sent_results_buffer,
        topic_results_buffer,
        gpu_time_ner_ms,
        gpu_time_sentiment_ms,
        gpu_time_topic_ms,
        ner_results_buffer,
    )