from datetime import datetime
from pathlib import Path
import logging
import json
import sys
import platform
import importlib.metadata as md
import time
import gc

import pandas as pd
import torch

from src.config import Config
from src.db_config import DatabaseConnection
from src.preprocessing import (
    ArabicPreprocessor,
    PREPROCESS_LANG_DETECT_PARAMS,
    PREPROCESS_NER_PARAMS,
    PREPROCESS_SENTIMENT_PARAMS,
    PREPROCESS_KEYWORDS_PARAMS,
)
from src.language_detection import FastTextLanguageDetector
from src.ner_extraction import TransformersNER, DEFAULT_NER_PARAMS
from src.sentiment_analysis import (
    TransformersSentiment,
    DEFAULT_SENTIMENT_PARAMS,
    probs_norm_prali22_4class,
    probs_norm_camel_3class,
    probs_norm_nlptown_to_3class,
)
from src.keyword_extraction import TFIDFKeywordExtractor

from comparison.report_generator import generate_comparison_report


def setup_logger():
    run_dir = Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("nlp_pipeline")
    logger.setLevel(Config.LOG_LEVEL)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, run_dir


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "unknown"


def _round_probs(probs: dict, decimals: int = 4) -> dict:
    return {k: round(v, decimals) for k, v in probs.items()}


def _format_entities(entities, decimals: int = 2) -> str:
    if not entities:
        return ""
    return ", ".join(f"{e.text} ({e.label}, {e.score:.{decimals}f})" for e in entities)


def _cuda_sync_if_needed(device: int):
    if device >= 0 and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_cleanup(logger: logging.Logger, cooldown_sec: float = 2.0):
    # Free VRAM + cooldown to protect laptop GPU thermals
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


def main():
    logger, run_dir = setup_logger()
    pipeline_t0 = time.perf_counter()

    # =========================
    # PARAMETERS
    # =========================
    sample_size = 10000
    raw_table = getattr(Config, "RAW_TABLE", "article")
    LANG_THRESHOLD = 0.60

    # Benchmark devices
    CPU_DEVICE = -1
    GPU_DEVICE = 0
    GPU_COOLDOWN_SEC = 2.0

    NER_PARAMS = dict(DEFAULT_NER_PARAMS)
    SENTIMENT_PARAMS = dict(DEFAULT_SENTIMENT_PARAMS)

    # Keywords (CPU once)
    KW_MAX_FEATURES = 1000
    KW_MIN_K = 5
    KW_MAX_K = 15
    KW_REL_THRESHOLD = 0.30
    KW_COVERAGE_TARGET = 0.70

    # Timing records -> timing_comparison.csv (CPU vs GPU + speedup)
    # Each record: mode, task, model_name, article_id, elapsed_seconds
    timing_records = []

    NER_MODELS = [
        ("arabert", Config.ARABERT_NER_MODEL),
        ("camel", Config.CAMEL_NER_MODEL),
        ("mbert", Config.MBERT_NER_MODEL),
    ]
    SENT_MODELS = [
        ("arabert", Config.ARABERT_SENTIMENT_MODEL, probs_norm_prali22_4class),
        ("camel", Config.CAMEL_SENTIMENT_MODEL, probs_norm_camel_3class),
        ("mbert", Config.MBERT_SENTIMENT_MODEL, probs_norm_nlptown_to_3class),
    ]

    # =========================
    # RUN CONFIG
    # =========================
    run_config = {
        "run_id": run_dir.name,
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
                "scikit-learn": _pkg_version("scikit-learn"),
            },
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": getattr(torch.version, "cuda", None),
        },
        "benchmark": {
            "cpu_device": CPU_DEVICE,
            "gpu_device": GPU_DEVICE,
            "gpu_cooldown_sec": GPU_COOLDOWN_SEC,
            "timing_scope": "predict() only (excludes MySQL writes)",
            "cpu_strategy": "article-by-article with 3 models loaded (NER) + 3 models loaded (Sentiment)",
            "gpu_strategy": "model-by-model (one model loaded at a time) to avoid OOM",
            "mysql_writes": "GPU pass only for NER/Sentiment; Keywords written once on CPU",
        },
        "sampling": {
            "sample_size": sample_size,
            "filters": [
                "body IS NOT NULL",
                "LENGTH(body) > 50",
                "id_language = 2 (Arabic in DB)",
            ],
        },
        "preprocessing_presets": {
            "lang_detect": PREPROCESS_LANG_DETECT_PARAMS,
            "ner": PREPROCESS_NER_PARAMS,
            "sentiment": PREPROCESS_SENTIMENT_PARAMS,
            "keywords": PREPROCESS_KEYWORDS_PARAMS,
        },
        "keywords": {
            "flexible_extraction": {
                "min_k": KW_MIN_K,
                "max_k": KW_MAX_K,
                "rel_threshold": KW_REL_THRESHOLD,
                "coverage_target": KW_COVERAGE_TARGET,
            },
            "schema_note": "keyword_results includes keywords_count",
        },
    }

    logger.info("=== RUN CONFIG ===\n" + json.dumps(run_config, indent=2, ensure_ascii=False))
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
        SELECT id, body, id_language
        FROM {raw_table}
        WHERE body IS NOT NULL
          AND id_language = 2
        ORDER BY crawl_date DESC, id DESC
        LIMIT {sample_size}
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Fetched {len(df)} rows from {raw_table}")

    lang_ref = pd.read_sql("SELECT id, code FROM language", engine)
    lang_map = dict(zip(lang_ref["id"], lang_ref["code"]))
    df["expected_lang"] = df["id_language"].map(lang_map)

    preproc = ArabicPreprocessor(logger=logger)
    df["text_raw"] = df["body"].fillna("")
    df["text_langdetect"] = df["text_raw"].apply(preproc.preprocess_for_lang_detect)

    # Language detection -> DB (CPU)
    detector = FastTextLanguageDetector(
        model_path=Config.FASTTEXT_MODEL_PATH,
        logger=logger,
        preprocessor=None,
    )

    lang_rows = []
    for r in df.itertuples(index=False):
        res = detector.detect(r.text_langdetect)
        expected = r.expected_lang if pd.notna(r.expected_lang) else None
        is_correct = (expected == res.lang) if expected is not None else None

        db.save_lang_detection(
            article_id=int(r.id),
            predicted_lang=res.lang,
            confidence=round(res.score, 4),
            expected_lang=expected,
            is_correct=is_correct,
        )
        lang_rows.append({"article_id": int(r.id), "lang": res.lang, "score": res.score})

    lang_df = pd.DataFrame(lang_rows)

    merged = df.merge(
        lang_df[["article_id", "lang", "score"]],
        left_on="id",
        right_on="article_id",
        how="left",
    )
    arabic_df = merged[(merged["lang"] == "ar") & (merged["score"] >= LANG_THRESHOLD)].copy()
    logger.info(f"Arabic subset (lang='ar' & score>={LANG_THRESHOLD}): {len(arabic_df)} rows")

    if arabic_df.empty:
        logger.warning("No Arabic rows passed threshold; stopping.")
        db.close()
        return

    arabic_df["text_ner"] = arabic_df["text_raw"].apply(preproc.preprocess_for_ner)
    arabic_df["text_sentiment"] = arabic_df["text_raw"].apply(preproc.preprocess_for_sentiment)
    arabic_df["text_keywords"] = arabic_df["text_raw"].apply(preproc.preprocess_for_keywords)

    # Save preprocessed texts -> DB (once)
    for r in arabic_df.itertuples(index=False):
        aid = int(r.id)
        db.save_preprocess_ner(aid, r.text_ner)
        db.save_preprocess_sentiment(aid, r.text_sentiment)
        db.save_preprocess_keywords(aid, r.text_keywords)

    # =========================
    # KEYWORDS (CPU ONCE) -> DB
    # =========================
    logger.info("=== KEYWORD EXTRACTION (CPU) ===")
    kw_extractor = TFIDFKeywordExtractor(
        stopwords=Config.ARABIC_STOPWORDS,
        max_features=KW_MAX_FEATURES,
        logger=logger,
    )
    kw_extractor.fit(arabic_df["text_keywords"].tolist())

    for r in arabic_df.itertuples(index=False):
        aid = int(r.id)
        kws = kw_extractor.extract_flexible(
            r.text_keywords,
            min_k=KW_MIN_K,
            max_k=KW_MAX_K,
            rel_threshold=KW_REL_THRESHOLD,
            coverage_target=KW_COVERAGE_TARGET,
        )
        kw_str = ", ".join(f"{w} ({s:.4f})" for w, s in kws)
        db.save_keyword_results(article_id=aid, keywords=kw_str, keywords_count=len(kws))

    # =========================
    # CPU PASS (article-by-article, 3 models loaded) - TIMING ONLY
    # =========================
    logger.info("=== CPU PASS (TIMING ONLY, 3 MODELS LOADED) ===")

    with torch.inference_mode():
        ner_cpu_arabert = TransformersNER(model_name=Config.ARABERT_NER_MODEL, logger=logger, preprocessor=None, device=CPU_DEVICE, **NER_PARAMS)
        ner_cpu_camel = TransformersNER(model_name=Config.CAMEL_NER_MODEL, logger=logger, preprocessor=None, device=CPU_DEVICE, **NER_PARAMS)
        ner_cpu_mbert = TransformersNER(model_name=Config.MBERT_NER_MODEL, logger=logger, preprocessor=None, device=CPU_DEVICE, **NER_PARAMS)

        sent_cpu_arabert = TransformersSentiment(
            model_name=Config.ARABERT_SENTIMENT_MODEL,
            logger=logger,
            preprocessor=None,
            device=CPU_DEVICE,
            probs_normalizer=probs_norm_prali22_4class,
            **SENTIMENT_PARAMS,
        )
        sent_cpu_camel = TransformersSentiment(
            model_name=Config.CAMEL_SENTIMENT_MODEL,
            logger=logger,
            preprocessor=None,
            device=CPU_DEVICE,
            probs_normalizer=probs_norm_camel_3class,
            **SENTIMENT_PARAMS,
        )
        sent_cpu_mbert = TransformersSentiment(
            model_name=Config.MBERT_SENTIMENT_MODEL,
            logger=logger,
            preprocessor=None,
            device=CPU_DEVICE,
            probs_normalizer=probs_norm_nlptown_to_3class,
            **SENTIMENT_PARAMS,
        )

        for r in arabic_df.itertuples(index=False):
            aid = int(r.id)

            t0 = time.perf_counter()
            _ = ner_cpu_arabert.predict(r.text_ner)
            timing_records.append({"mode": "cpu", "task": "ner", "model_name": "arabert", "article_id": aid, "elapsed_seconds": round(time.perf_counter() - t0, 6)})

            t0 = time.perf_counter()
            _ = ner_cpu_camel.predict(r.text_ner)
            timing_records.append({"mode": "cpu", "task": "ner", "model_name": "camel", "article_id": aid, "elapsed_seconds": round(time.perf_counter() - t0, 6)})

            t0 = time.perf_counter()
            _ = ner_cpu_mbert.predict(r.text_ner)
            timing_records.append({"mode": "cpu", "task": "ner", "model_name": "mbert", "article_id": aid, "elapsed_seconds": round(time.perf_counter() - t0, 6)})

            t0 = time.perf_counter()
            _ = sent_cpu_arabert.predict(r.text_sentiment)
            timing_records.append({"mode": "cpu", "task": "sentiment", "model_name": "arabert", "article_id": aid, "elapsed_seconds": round(time.perf_counter() - t0, 6)})

            t0 = time.perf_counter()
            _ = sent_cpu_camel.predict(r.text_sentiment)
            timing_records.append({"mode": "cpu", "task": "sentiment", "model_name": "camel", "article_id": aid, "elapsed_seconds": round(time.perf_counter() - t0, 6)})

            t0 = time.perf_counter()
            _ = sent_cpu_mbert.predict(r.text_sentiment)
            timing_records.append({"mode": "cpu", "task": "sentiment", "model_name": "mbert", "article_id": aid, "elapsed_seconds": round(time.perf_counter() - t0, 6)})

        del ner_cpu_arabert, ner_cpu_camel, ner_cpu_mbert
        del sent_cpu_arabert, sent_cpu_camel, sent_cpu_mbert
        gc.collect()

    # =========================
    # GPU PASS (model-by-model) - WRITE RESULTS TO DB, TIME predict() ONLY
    # =========================
    if not torch.cuda.is_available():
        logger.warning("CUDA not available -> GPU pass skipped. (No NER/Sentiment DB results will be written.)")
    else:
        logger.info("=== GPU PASS (MODEL-BY-MODEL, WRITE RESULTS TO DB) ===")

        article_ids = [int(x) for x in arabic_df["id"].tolist()]

        # ---- NER GPU ----
        ner_out = {aid: {"arabert": "", "camel": "", "mbert": ""} for aid in article_ids}

        with torch.inference_mode():
            for model_key, model_name in NER_MODELS:
                logger.info(f"[GPU][NER] loading {model_key}: {model_name}")
                ner = TransformersNER(model_name=model_name, logger=logger, preprocessor=None, device=GPU_DEVICE, **NER_PARAMS)

                for r in arabic_df.itertuples(index=False):
                    aid = int(r.id)
                    t0 = time.perf_counter()
                    ents = ner.predict(r.text_ner)
                    _cuda_sync_if_needed(GPU_DEVICE)
                    dt = time.perf_counter() - t0

                    timing_records.append({
                        "mode": "gpu",
                        "task": "ner",
                        "model_name": model_key,
                        "article_id": aid,
                        "elapsed_seconds": round(dt, 6),
                    })
                    ner_out[aid][model_key] = _format_entities(ents)

                del ner
                _gpu_cleanup(logger, cooldown_sec=GPU_COOLDOWN_SEC)

        # write NER results to DB (outside timing)
        for aid in article_ids:
            db.save_ner_results(
                article_id=aid,
                arabert_entities=ner_out[aid]["arabert"],
                camel_entities=ner_out[aid]["camel"],
                mbert_entities=ner_out[aid]["mbert"],
            )

        # ---- Sentiment GPU ----
        sent_out = {
            aid: {
                "arabert_label": None, "arabert_score": None, "arabert_probs": None,
                "camel_label": None, "camel_score": None, "camel_probs": None,
                "mbert_label": None, "mbert_score": None, "mbert_probs": None,
            }
            for aid in article_ids
        }

        with torch.inference_mode():
            for model_key, model_name, normalizer in SENT_MODELS:
                logger.info(f"[GPU][SENT] loading {model_key}: {model_name}")
                sent = TransformersSentiment(
                    model_name=model_name,
                    logger=logger,
                    preprocessor=None,
                    device=GPU_DEVICE,
                    probs_normalizer=normalizer,
                    **SENTIMENT_PARAMS,
                )

                for r in arabic_df.itertuples(index=False):
                    aid = int(r.id)
                    t0 = time.perf_counter()
                    res = sent.predict(r.text_sentiment)
                    _cuda_sync_if_needed(GPU_DEVICE)
                    dt = time.perf_counter() - t0

                    timing_records.append({
                        "mode": "gpu",
                        "task": "sentiment",
                        "model_name": model_key,
                        "article_id": aid,
                        "elapsed_seconds": round(dt, 6),
                    })

                    sent_out[aid][f"{model_key}_label"] = res.label
                    sent_out[aid][f"{model_key}_score"] = round(res.score, 4)
                    sent_out[aid][f"{model_key}_probs"] = json.dumps(_round_probs(res.probs), ensure_ascii=False)

                del sent
                _gpu_cleanup(logger, cooldown_sec=GPU_COOLDOWN_SEC)

        # write sentiment results to DB (outside timing)
        for aid in article_ids:
            db.save_sentiment_results(
                article_id=aid,
                arabert_label=sent_out[aid]["arabert_label"],
                arabert_score=sent_out[aid]["arabert_score"],
                arabert_probs=sent_out[aid]["arabert_probs"],
                camel_label=sent_out[aid]["camel_label"],
                camel_score=sent_out[aid]["camel_score"],
                camel_probs=sent_out[aid]["camel_probs"],
                mbert_label=sent_out[aid]["mbert_label"],
                mbert_score=sent_out[aid]["mbert_score"],
                mbert_probs=sent_out[aid]["mbert_probs"],
            )

    # =========================
    # COMPARISON REPORT
    # =========================
    generate_comparison_report(run_dir=run_dir, timing_records=timing_records)

    total = time.perf_counter() - pipeline_t0
    logger.info(f"Pipeline complete in {total:.2f}s")
    db.close()


if __name__ == "__main__":
    main()