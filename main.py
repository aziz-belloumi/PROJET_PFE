from datetime import datetime
from pathlib import Path
import logging
import json
import sys
import platform
import importlib.metadata as md
import time
import re
import unicodedata
import gc

import pandas as pd
import torch

from src.config import Config
from src.logger import setup_run
from src.db_config import DatabaseConnection

from src.preprocessing import PreprocessRouter
from src.preprocessing.router import (
    PREPROCESS_LANG_DETECT_PARAMS,
    PREPROCESS_NER_PARAMS,
    PREPROCESS_SENTIMENT_PARAMS,
    LATIN_LANG_DETECT_PARAMS,
    LATIN_NER_PARAMS,
    LATIN_SENTIMENT_PARAMS,
    LATIN_TOPIC_PARAMS,
)

from src.language_detection import FastTextLanguageDetector
from src.ner_extraction import TransformersNER, DEFAULT_NER_PARAMS
from src.sentiment_analysis import (
    TransformersSentiment,
    DEFAULT_SENTIMENT_PARAMS,
    probs_norm_prali22_4class,
    probs_norm_3class_posnegneu,
    probs_norm_camel_3class,
)
from src.topic_classification import (
    TransformersTopic,
    DEFAULT_TOPIC_PARAMS,
    CATEGORY_MAP,
)
from comparison.report_generator import generate_comparison_report
from analysis.report_generator import generate_analytics_reports


def export_lang_samples_csv(
    run_dir: Path,
    work_df: pd.DataFrame,
    sent_results_buffer: dict,
    topic_results_buffer: dict,
):
    rows = []
    for r in work_df.itertuples(index=False):
        aid = int(r.id)
        lang = str(r.lang)

        if lang not in {"en", "fr"}:
            continue

        mv = 2 if lang == "en" else 3  # model_version mapping used in main
        sent_info = sent_results_buffer.get((aid, mv), {})
        topic_info = topic_results_buffer.get(aid, {})

        rows.append({
            "article_id": aid,
            "lang": lang,
            "text_raw": r.text_raw,
            "sentiment_preprocessed": r.text_sentiment,
            "ner_preprocessed": r.text_ner,
            "topic_preprocessed": r.text_topic,
            "sentiment_label": sent_info.get("label"),
            "sentiment_score": sent_info.get("score"),
            "dominant_topic": topic_info.get("label"),
            "topic_score": topic_info.get("score"),
        })

    out_df = pd.DataFrame(rows)
    en_df = out_df[out_df["lang"] == "en"].copy()
    fr_df = out_df[out_df["lang"] == "fr"].copy()

    if not en_df.empty:
        en_df.to_csv(Path(run_dir) / "english_samples.csv", index=False, encoding="utf-8-sig")

    if not fr_df.empty:
        fr_df.to_csv(Path(run_dir) / "french_samples.csv", index=False, encoding="utf-8-sig")


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "unknown"


def _cuda_sync_if_needed(device: int):
    if device >= 0 and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_cleanup(logger: logging.Logger, cooldown_sec: float = 2.0):
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    if cooldown_sec and cooldown_sec > 0:
        logger.info(f"[GPU] cooldown {cooldown_sec:.1f}s")
        time.sleep(cooldown_sec)


def normalize_entity_name(raw: str) -> tuple[str, str]:
    """
    Keep as-is for now (simple + safe).
    Strong normalization can be moved into preprocessing later.
    """
    if not raw:
        return "", ""
    s = unicodedata.normalize("NFC", raw).strip()
    s = re.sub(r"\u0640", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    n = re.sub(r"[\u064B-\u065F\u0670]", "", s)
    n = re.sub(r"[^\u0600-\u06FF0-9A-Za-z\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return s, n


def parse_category_ids(val) -> list[int]:
    if val is None:
        return []
    try:
        if isinstance(val, float) and pd.isna(val):
            return []
    except Exception:
        pass

    if isinstance(val, int):
        return [val]

    s = str(val).strip()
    if not s:
        return []

    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out


def main():
    pipeline_t0 = time.perf_counter()

    # =========================
    # PARAMETERS
    # =========================
    sample_size = 100
    raw_table = getattr(Config, "RAW_TABLE", "article")
    LANG_THRESHOLD = 0.60

    CPU_DEVICE = -1
    GPU_DEVICE = 0
    GPU_COOLDOWN_SEC = 2.0

    NER_PARAMS = dict(DEFAULT_NER_PARAMS)
    SENTIMENT_PARAMS = dict(DEFAULT_SENTIMENT_PARAMS)
    TOPIC_PARAMS = dict(DEFAULT_TOPIC_PARAMS)

    NER_MODELS_BY_LANG = {
        "ar": [
            (0, Config.ARABERT_NER_MODEL),
            (1, Config.CAMEL_NER_MODEL),
        ],
        "en": [
            (2, Config.EN_NER_MODEL),
        ],
        "fr": [
            (3, Config.FR_NER_MODEL),
        ],
    }

    SENT_MODELS_BY_LANG = {
        "ar": [
            (0, Config.ARABERT_SENTIMENT_MODEL, probs_norm_prali22_4class),
            (1, Config.CAMEL_SENTIMENT_MODEL, probs_norm_camel_3class),
        ],
        "en": [
            (2, Config.EN_SENTIMENT_MODEL, probs_norm_3class_posnegneu),
        ],
        "fr": [
            (3, Config.FR_SENTIMENT_MODEL, probs_norm_3class_posnegneu),
        ],
    }

    timing_records = []

    # =========================
    # RUN CONFIG (build first)
    # =========================
    run_config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "pandas": _pkg_version("pandas"),
                "sqlalchemy": _pkg_version("SQLAlchemy"),
                "transformers": _pkg_version("transformers"),
                "torch": _pkg_version("torch"),
                "fasttext-wheel": _pkg_version("fasttext-wheel"),
            },
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "benchmark": {
            "cpu_device": CPU_DEVICE,
            "gpu_device": GPU_DEVICE,
            "gpu_cooldown_sec": GPU_COOLDOWN_SEC,
            "timing_scope": "predict() only (excludes MySQL writes)",
            "cpu_strategy": "article-by-article with all models loaded (per-language)",
            "gpu_strategy": "model-by-model to avoid OOM",
        },
        "sampling": {
            "sample_size": sample_size,
            "filters": [
                "body IS NOT NULL",
                "no language restriction (fastText decides)",
            ],
        },
        "preprocessing_presets": {
            "lang_detect": {
                "ar": PREPROCESS_LANG_DETECT_PARAMS,
                "latin": LATIN_LANG_DETECT_PARAMS,
            },
            "ner": {
                "ar": PREPROCESS_NER_PARAMS,
                "latin": LATIN_NER_PARAMS,
            },
            "sentiment": {
                "ar": PREPROCESS_SENTIMENT_PARAMS,
                "latin": LATIN_SENTIMENT_PARAMS,
            },
            "topic": {
                # router uses Arabic sentiment preset for topic
                "ar": PREPROCESS_SENTIMENT_PARAMS,
                "latin": LATIN_TOPIC_PARAMS,
            },
        },
        "topic_classification": {
            "model": Config.TOPIC_MODEL,
            "params": TOPIC_PARAMS,
            "categories": {str(k): v for k, v in CATEGORY_MAP.items()},
            "ground_truth_mode": "multi_label (correct if predicted_category_id in id_categories list)",
        },
        "models": {
            "ner": NER_MODELS_BY_LANG,
            "sentiment": {
                k: [(mv, name) for (mv, name, _) in v]
                for k, v in SENT_MODELS_BY_LANG.items()
            },
        },
        "analytics_outputs": [
            "topics_by_month.csv",
            "top_entities_by_country.csv",
            "entities_by_month.csv",
            "topic_peaks.csv",
        ],
    }

    # Create logger + run folder, and save run_config.json
    ctx = setup_run(run_config=run_config, save_run_config=True)
    logger = ctx.logger
    run_dir = ctx.run_dir

    # Now that run_dir exists, add run_id for completeness (optional)
    run_config["run_id"] = run_dir.name
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=== RUN CONFIG ===\n" + json.dumps(run_config, indent=2, ensure_ascii=False))

    # =========================
    # DB INIT
    # =========================
    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()
    db.init_result_tables()

    # =========================
    # SAMPLE + PREPROCESS ONCE
    # =========================
    query = f"""
        SELECT id, body, id_language, id_categories
        FROM {raw_table}
        WHERE body IS NOT NULL
        ORDER BY crawl_date DESC, id DESC
        LIMIT {sample_size}
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Fetched {len(df)} rows from {raw_table}")

    cat_ref = pd.read_sql("SELECT id, label_en FROM category", engine)
    cat_map = dict(zip(cat_ref["id"], cat_ref["label_en"]))

    # Ensure category labels exist even if DB category table is incomplete
    for cid, labels in CATEGORY_MAP.items():
        try:
            if isinstance(labels, dict) and "en" in labels:
                cat_map.setdefault(int(cid), str(labels["en"]))
        except Exception:
            continue

    preproc = PreprocessRouter(logger=logger)

    df["text_raw"] = df["body"].fillna("")
    df["text_langdetect"] = df["text_raw"].apply(lambda t: preproc.preprocess(t, "xx", "lang_detect"))

    # Language detection
    detector = FastTextLanguageDetector(
        model_path=Config.FASTTEXT_MODEL_PATH,
        logger=logger,
        preprocessor=None,
    )

    lang_rows = []
    for r in df.itertuples(index=False):
        res = detector.detect(r.text_langdetect)
        lang_rows.append({"article_id": int(r.id), "lang": res.lang, "score": res.score})

    lang_df = pd.DataFrame(lang_rows)
    merged = df.merge(
        lang_df[["article_id", "lang", "score"]],
        left_on="id",
        right_on="article_id",
        how="left",
    )

    supported_langs = set(NER_MODELS_BY_LANG.keys())
    work_df = merged[(merged["lang"].isin(supported_langs)) & (merged["score"] >= LANG_THRESHOLD)].copy()
    logger.info(
        f"Subset after lang filter (supported={sorted(supported_langs)} & score>={LANG_THRESHOLD}): "
        f"{len(work_df)} rows"
    )

    if work_df.empty:
        logger.warning("No rows passed language filter; stopping.")
        db.close()
        return

    # Language-aware preprocessing via router
    work_df["text_ner"] = work_df.apply(lambda r: preproc.preprocess(r.text_raw, str(r.lang), "ner"), axis=1)
    work_df["text_sentiment"] = work_df.apply(lambda r: preproc.preprocess(r.text_raw, str(r.lang), "sentiment"), axis=1)
    work_df["text_topic"] = work_df.apply(lambda r: preproc.preprocess(r.text_raw, str(r.lang), "topic"), axis=1)

    # Pre-insert articles_enriched rows
    for r in work_df.itertuples(index=False):
        aid = int(r.id)
        lang = str(r.lang)
        for mv, _ in NER_MODELS_BY_LANG.get(lang, []):
            db.upsert_articles_enriched(article_id=aid, model_version=mv, language=lang)

    # =========================
    # CPU PASS (TIMING ONLY)
    # =========================
    cpu_time_ms_by_article_model: dict[tuple[int, int], int] = {}
    logger.info("=== CPU PASS (TIMING ONLY) ===")

    ner_cpu_models_by_lang = {
        lang: [
            (mv, TransformersNER(model_name=name, logger=logger, preprocessor=None, device=CPU_DEVICE, **NER_PARAMS))
            for mv, name in models
        ]
        for lang, models in NER_MODELS_BY_LANG.items()
    }
    sent_cpu_models_by_lang = {
        lang: [
            (
                mv,
                TransformersSentiment(
                    model_name=name,
                    logger=logger,
                    preprocessor=None,
                    device=CPU_DEVICE,
                    probs_normalizer=norm,
                    **SENTIMENT_PARAMS,
                ),
            )
            for mv, name, norm in models
        ]
        for lang, models in SENT_MODELS_BY_LANG.items()
    }

    topic_cpu = TransformersTopic(
        model_name=Config.TOPIC_MODEL,
        logger=logger,
        preprocessor=None,
        device=CPU_DEVICE,
        **TOPIC_PARAMS,
    )

    with torch.inference_mode():
        for r in work_df.itertuples(index=False):
            aid = int(r.id)
            lang = str(r.lang)

            for mv, model in ner_cpu_models_by_lang.get(lang, []):
                t0 = time.perf_counter()
                _ = model.predict(r.text_ner)
                dt = time.perf_counter() - t0
                cpu_time_ms_by_article_model[(aid, mv)] = cpu_time_ms_by_article_model.get((aid, mv), 0) + int(dt * 1000)
                timing_records.append({
                    "mode": "cpu",
                    "task": "ner",
                    "model_version": mv,
                    "article_id": aid,
                    "elapsed_seconds": round(dt, 6),
                })

            for mv, model in sent_cpu_models_by_lang.get(lang, []):
                t0 = time.perf_counter()
                _ = model.predict(r.text_sentiment)
                dt = time.perf_counter() - t0
                cpu_time_ms_by_article_model[(aid, mv)] = cpu_time_ms_by_article_model.get((aid, mv), 0) + int(dt * 1000)
                timing_records.append({
                    "mode": "cpu",
                    "task": "sentiment",
                    "model_version": mv,
                    "article_id": aid,
                    "elapsed_seconds": round(dt, 6),
                })

            t0 = time.perf_counter()
            _ = topic_cpu.predict(r.text_topic, lang=lang)
            dt = time.perf_counter() - t0
            timing_records.append({
                "mode": "cpu",
                "task": "topic",
                "model_version": -1,
                "article_id": aid,
                "elapsed_seconds": round(dt, 6),
            })

    del ner_cpu_models_by_lang, sent_cpu_models_by_lang, topic_cpu
    gc.collect()

    # =========================
    # GPU PASS (MODEL-BY-MODEL) - write to DB
    # =========================
    if not torch.cuda.is_available():
        logger.warning("CUDA not available -> GPU pass skipped.")
        sent_results_buffer = {}
        topic_results_buffer = {}
        gpu_time_ms_by_article_model = {}
    else:
        logger.info("=== GPU PASS (MODEL-BY-MODEL, WRITE RESULTS TO DB) ===")

        gpu_time_ms_by_article_model: dict[tuple[int, int], int] = {}
        sent_results_buffer: dict[tuple[int, int], dict] = {}

        # ---- NER GPU ----
        with torch.inference_mode():
            for lang, models in NER_MODELS_BY_LANG.items():
                lang_subset = work_df[work_df["lang"] == lang]
                if lang_subset.empty:
                    continue

                for mv, name in models:
                    logger.info(f"[GPU][NER] lang={lang} model_version={mv}: {name}")
                    ner = TransformersNER(
                        model_name=name,
                        logger=logger,
                        preprocessor=None,
                        device=GPU_DEVICE,
                        **NER_PARAMS,
                    )

                    for r in lang_subset.itertuples(index=False):
                        aid = int(r.id)
                        t0 = time.perf_counter()
                        ents = ner.predict(r.text_ner)
                        _cuda_sync_if_needed(GPU_DEVICE)
                        dt = time.perf_counter() - t0

                        gpu_time_ms_by_article_model[(aid, mv)] = gpu_time_ms_by_article_model.get((aid, mv), 0) + int(dt * 1000)
                        timing_records.append({
                            "mode": "gpu",
                            "task": "ner",
                            "model_version": mv,
                            "article_id": aid,
                            "elapsed_seconds": round(dt, 6),
                        })

                        for e in ents:
                            ename, nname = normalize_entity_name(e.text)
                            if ename and nname:
                                eid = db.upsert_entity(ename, (e.label or "UNK").upper(), nname)
                                db.upsert_article_entity(
                                    article_id=aid,
                                    entity_id=eid,
                                    model_version=mv,
                                    confidence_score=e.score,
                                )

                    del ner
                    _gpu_cleanup(logger, cooldown_sec=GPU_COOLDOWN_SEC)

        # ---- Sentiment GPU ----
        with torch.inference_mode():
            for lang, models in SENT_MODELS_BY_LANG.items():
                lang_subset = work_df[work_df["lang"] == lang]
                if lang_subset.empty:
                    continue

                for mv, name, norm in models:
                    logger.info(f"[GPU][SENT] lang={lang} model_version={mv}: {name}")
                    sent = TransformersSentiment(
                        model_name=name,
                        logger=logger,
                        preprocessor=None,
                        device=GPU_DEVICE,
                        probs_normalizer=norm,
                        **SENTIMENT_PARAMS,
                    )

                    for r in lang_subset.itertuples(index=False):
                        aid = int(r.id)
                        t0 = time.perf_counter()
                        res = sent.predict(r.text_sentiment)
                        _cuda_sync_if_needed(GPU_DEVICE)
                        dt = time.perf_counter() - t0

                        gpu_time_ms_by_article_model[(aid, mv)] = gpu_time_ms_by_article_model.get((aid, mv), 0) + int(dt * 1000)
                        timing_records.append({
                            "mode": "gpu",
                            "task": "sentiment",
                            "model_version": mv,
                            "article_id": aid,
                            "elapsed_seconds": round(dt, 6),
                        })

                        sent_results_buffer[(aid, mv)] = {"label": res.label, "score": float(res.score)}

                    del sent
                    _gpu_cleanup(logger, cooldown_sec=GPU_COOLDOWN_SEC)

        # ---- Topic GPU ----
        topic_results_buffer: dict[int, dict] = {}
        with torch.inference_mode():
            logger.info(f"[GPU][TOPIC] loading: {Config.TOPIC_MODEL}")
            topic_gpu = TransformersTopic(
                model_name=Config.TOPIC_MODEL,
                logger=logger,
                preprocessor=None,
                device=GPU_DEVICE,
                **TOPIC_PARAMS,
            )

            for r in work_df.itertuples(index=False):
                aid = int(r.id)
                lang = str(r.lang)

                expected_cat_ids = parse_category_ids(r.id_categories)
                expected_cat_labels = [cat_map.get(cid, "Unknown") for cid in expected_cat_ids]

                t0 = time.perf_counter()
                tres = topic_gpu.predict(r.text_topic, lang=lang)
                _cuda_sync_if_needed(GPU_DEVICE)
                dt = time.perf_counter() - t0

                timing_records.append({
                    "mode": "gpu",
                    "task": "topic",
                    "model_version": -1,
                    "article_id": aid,
                    "elapsed_seconds": round(dt, 6),
                })

                topic_results_buffer[aid] = {
                    "label": tres.label,
                    "score": float(tres.score),
                    "expected_category_ids": expected_cat_ids,
                    "expected_category_labels": expected_cat_labels,
                }

                db.upsert_article_topic(article_id=aid, topic_label=tres.label, topic_score=tres.score)

            del topic_gpu
            _gpu_cleanup(logger, cooldown_sec=GPU_COOLDOWN_SEC)

        # ---- Persist articles_enriched rows ----
        logger.info("Persisting articles_enriched rows...")
        for r in work_df.itertuples(index=False):
            aid = int(r.id)
            lang = str(r.lang)
            dom_topic = topic_results_buffer.get(aid, {}).get("label")

            for mv, _ in NER_MODELS_BY_LANG.get(lang, []):
                sent_info = sent_results_buffer.get((aid, mv), {})
                cpu_ptime = cpu_time_ms_by_article_model.get((aid, mv))
                gpu_ptime = gpu_time_ms_by_article_model.get((aid, mv))

                db.upsert_articles_enriched(
                    article_id=aid,
                    model_version=mv,
                    language=lang,
                    sentiment_label=sent_info.get("label"),
                    sentiment_score=sent_info.get("score"),
                    dominant_topic=dom_topic,
                    cpu_processing_time=cpu_ptime,
                    gpu_processing_time=gpu_ptime,
                )

        # Export EN/FR review CSVs
        export_lang_samples_csv(
            run_dir=run_dir,
            work_df=work_df,
            sent_results_buffer=sent_results_buffer,
            topic_results_buffer=topic_results_buffer,
        )
        logger.info("Saved english_samples.csv and french_samples.csv (if EN/FR rows exist).")

    # =========================
    # COMPARISON REPORT
    # =========================
    try:
        generate_comparison_report(run_dir=run_dir, timing_records=timing_records)
        logger.info("Comparison CSVs generated successfully.")
    except Exception as e:
        logger.error(f"Comparison report generation failed: {e}")

    # =========================
    # ANALYTICS REPORTS (NEW)
    # =========================
    try:
        generate_analytics_reports(
            run_dir=run_dir,
            raw_table=raw_table,
            top_entities_k=50,
            top_entities_by_month_k=50,
            peaks_window=6,
            peaks_z_threshold=2.5,
            default_country_id=8,
        )
        logger.info("Analytics CSVs generated successfully.")
    except Exception as e:
        logger.error(f"Analytics report generation failed: {e}")

    total = time.perf_counter() - pipeline_t0
    logger.info(f"Pipeline complete in {total:.2f}s")
    db.close()


if __name__ == "__main__":
    main()