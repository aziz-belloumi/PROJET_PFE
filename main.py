from datetime import datetime
from pathlib import Path
import logging
import json
import sys
import platform
import importlib.metadata as md
import time

import pandas as pd

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

# Root-level comparison package (NLP_MENA/comparison)
from comparison.report_generator import generate_comparison_report


def setup_logger():
    # All outputs for this run go under results/<timestamp>/
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


def main():
    logger, run_dir = setup_logger()
    pipeline_t0 = time.perf_counter()

    # ============================================================
    # PARAMETERS
    # ============================================================
    sample_size = 100
    raw_table = getattr(Config, "RAW_TABLE", "article")
    LANG_THRESHOLD = 0.60

    NER_PARAMS = dict(DEFAULT_NER_PARAMS)
    SENTIMENT_PARAMS = dict(DEFAULT_SENTIMENT_PARAMS)

    # Keyword extraction parameters
    KW_MAX_FEATURES = 1000

    # Option D (flexible keywords)
    KW_MIN_K = 5
    KW_MAX_K = 25
    KW_REL_THRESHOLD = 0.30
    KW_COVERAGE_TARGET = 0.70

    # Timing records collected locally (NOT stored in MySQL)
    timing_records = []

    # ============================================================
    # RUN CONFIG (stored in results/<run_id>/run_config.json)
    # ============================================================
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
        },
        "database": {
            "host": Config.DB_HOST,
            "port": Config.DB_PORT,
            "db_name": Config.DB_NAME,
            "raw_table": raw_table,
        },
        "sampling": {
            "sample_size": sample_size,
            "filters": [
                "body IS NOT NULL",
                "LENGTH(body) > 50",
                "id_language = 2 (Arabic in DB)",
            ],
            "ordering": "ORDER BY crawl_date DESC, id DESC",
        },
        "preprocessing_presets": {
            "lang_detect": PREPROCESS_LANG_DETECT_PARAMS,
            "ner": PREPROCESS_NER_PARAMS,
            "sentiment": PREPROCESS_SENTIMENT_PARAMS,
            "keywords": PREPROCESS_KEYWORDS_PARAMS,
        },
        "language_detection": {
            "fasttext_model_path": Config.FASTTEXT_MODEL_PATH,
            "threshold": LANG_THRESHOLD,
            "storage": "MySQL table: lang_detection",
        },
        "ner": {
            "models": {
                "arabert": {"id_model": 0, "model_name": Config.ARABERT_NER_MODEL},
                "camel": {"id_model": 1, "model_name": Config.CAMEL_NER_MODEL},
                "mbert": {"id_model": 2, "model_name": Config.MBERT_NER_MODEL},
            },
            "params": NER_PARAMS,
            "label_unification": "all labels → first 3 letters uppercased",
            "storage": "MySQL table: ner_results (1 row/article)",
        },
        "sentiment": {
            "models": {
                "arabert": {"id_model": 0, "model_name": Config.ARABERT_SENTIMENT_MODEL},
                "camel": {"id_model": 1, "model_name": Config.CAMEL_SENTIMENT_MODEL},
                "mbert": {"id_model": 2, "model_name": Config.MBERT_SENTIMENT_MODEL},
            },
            "params": SENTIMENT_PARAMS,
            "storage": "MySQL table: sentiment_results (1 row/article)",
        },
        "keywords": {
            "engine": "TF-IDF (scikit-learn)",
            "max_features": KW_MAX_FEATURES,
            "vectorizer_params": {"max_df": 0.95, "min_df": 1},
            "flexible_extraction": {
                "min_k": KW_MIN_K,
                "max_k": KW_MAX_K,
                "rel_threshold": KW_REL_THRESHOLD,
                "coverage_target": KW_COVERAGE_TARGET,
            },
            "storage": "MySQL table: keyword_results (1 row/article)",
            "schema_note": "keyword_results includes keywords_count",
        },
        "mysql_result_tables": [
            "lang_detection",
            "preprocess_ner",
            "preprocess_sentiment",
            "preprocess_keywords",
            "ner_results",
            "sentiment_results",
            "keyword_results",
        ],
        "comparison_outputs": {
            "location": str(run_dir),
            "files": [
                "comparison.log",
                "ner_comparison.csv",
                "sentiment_comparison.csv",
                "keyword_comparison.csv",
                "timing_comparison.csv",
            ],
        },
    }

    logger.info("=== RUN CONFIG ===\n" + json.dumps(run_config, indent=2, ensure_ascii=False))
    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved run config JSON: {run_config_path}")

    # ============================================================
    # PIPELINE START
    # ============================================================
    logger.info(f"FASTTEXT_MODEL_PATH = {Config.FASTTEXT_MODEL_PATH}")
    logger.info(f"DB = {Config.DB_NAME}@{Config.DB_HOST}:{Config.DB_PORT}")

    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()

    # Drop previous results and create fresh tables
    db.init_result_tables()

    # 1) Fetch from raw table
    query = f"""
        SELECT id, body, id_language
        FROM {raw_table}
        WHERE body IS NOT NULL
          AND LENGTH(body) > 50
          AND id_language = 2
        ORDER BY crawl_date DESC, id DESC
        LIMIT {sample_size}
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Fetched {len(df)} rows from {raw_table}")

    # language mapping: id_language -> code
    lang_ref = pd.read_sql("SELECT id, code FROM language", engine)
    lang_map = dict(zip(lang_ref["id"], lang_ref["code"]))
    df["expected_lang"] = df["id_language"].map(lang_map)

    # 2) Preprocess for language detection
    preproc = ArabicPreprocessor(logger=logger)
    df["text_raw"] = df["body"].fillna("")
    df["text_langdetect"] = df["text_raw"].apply(preproc.preprocess_for_lang_detect)

    # 3) Language detection -> DB
    detector = FastTextLanguageDetector(
        model_path=Config.FASTTEXT_MODEL_PATH,
        logger=logger,
        preprocessor=None,
    )

    lang_results = []
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

        lang_results.append({"article_id": int(r.id), "lang": res.lang, "score": res.score})
        logger.info(f"[LANG {r.id}] expected={expected} pred={res.lang} correct={is_correct} score={res.score:.3f}")

    lang_df = pd.DataFrame(lang_results)
    logger.info(f"Language detection complete: {len(lang_df)} rows saved to DB")

    # ============================================================
    # Filter Arabic articles
    # ============================================================
    merged = df.merge(
        lang_df[["article_id", "lang", "score"]],
        left_on="id",
        right_on="article_id",
        how="left",
    )
    arabic_df = merged[(merged["lang"] == "ar") & (merged["score"] >= LANG_THRESHOLD)].copy()
    logger.info(f"Arabic subset (lang='ar' & score>={LANG_THRESHOLD}): {len(arabic_df)} rows")

    # Task-specific preprocessing
    arabic_df["text_ner"] = arabic_df["text_raw"].apply(preproc.preprocess_for_ner)
    arabic_df["text_sentiment"] = arabic_df["text_raw"].apply(preproc.preprocess_for_sentiment)
    arabic_df["text_keywords"] = arabic_df["text_raw"].apply(preproc.preprocess_for_keywords)

    # Save preprocessed texts to DB
    for r in arabic_df.itertuples(index=False):
        aid = int(r.id)
        db.save_preprocess_ner(aid, r.text_ner)
        db.save_preprocess_sentiment(aid, r.text_sentiment)
        db.save_preprocess_keywords(aid, r.text_keywords)
    logger.info(f"Saved preprocessed texts to DB for {len(arabic_df)} articles")

    # ============================================================
    # NER (3 models) -> DB (1 row/article)
    # ============================================================
    ner_arabert = TransformersNER(model_name=Config.ARABERT_NER_MODEL, logger=logger, preprocessor=None, **NER_PARAMS)
    ner_camel = TransformersNER(model_name=Config.CAMEL_NER_MODEL, logger=logger, preprocessor=None, **NER_PARAMS)
    ner_mbert = TransformersNER(model_name=Config.MBERT_NER_MODEL, logger=logger, preprocessor=None, **NER_PARAMS)

    for r in arabic_df.itertuples(index=False):
        text = r.text_ner
        article_id = int(r.id)

        t0 = time.perf_counter()
        try:
            ents_a = ner_arabert.predict(text)
        except Exception as e:
            logger.error(f"AraBERT NER failed for article {article_id}: {e}")
            ents_a = []
        timing_records.append({"article_id": article_id, "task": "ner", "model_name": "arabert",
                               "elapsed_seconds": round(time.perf_counter() - t0, 6)})

        t0 = time.perf_counter()
        try:
            ents_c = ner_camel.predict(text)
        except Exception as e:
            logger.error(f"CAMeL NER failed for article {article_id}: {e}")
            ents_c = []
        timing_records.append({"article_id": article_id, "task": "ner", "model_name": "camel",
                               "elapsed_seconds": round(time.perf_counter() - t0, 6)})

        t0 = time.perf_counter()
        try:
            ents_m = ner_mbert.predict(text)
        except Exception as e:
            logger.error(f"mBERT NER failed for article {article_id}: {e}")
            ents_m = []
        timing_records.append({"article_id": article_id, "task": "ner", "model_name": "mbert",
                               "elapsed_seconds": round(time.perf_counter() - t0, 6)})

        db.save_ner_results(
            article_id=article_id,
            arabert_entities=_format_entities(ents_a),
            camel_entities=_format_entities(ents_c),
            mbert_entities=_format_entities(ents_m),
        )
        logger.info(f"[NER {article_id}] arabert={len(ents_a)} camel={len(ents_c)} mbert={len(ents_m)}")

    logger.info("NER complete: results saved to DB")

    # ============================================================
    # Sentiment (3 models) -> DB (1 row/article)
    # ============================================================
    sent_arabert = TransformersSentiment(
        model_name=Config.ARABERT_SENTIMENT_MODEL,
        logger=logger,
        preprocessor=None,
        probs_normalizer=probs_norm_prali22_4class,
        **SENTIMENT_PARAMS,
    )
    sent_camel = TransformersSentiment(
        model_name=Config.CAMEL_SENTIMENT_MODEL,
        logger=logger,
        preprocessor=None,
        probs_normalizer=probs_norm_camel_3class,
        **SENTIMENT_PARAMS,
    )
    sent_mbert = TransformersSentiment(
        model_name=Config.MBERT_SENTIMENT_MODEL,
        logger=logger,
        preprocessor=None,
        probs_normalizer=probs_norm_nlptown_to_3class,
        **SENTIMENT_PARAMS,
    )

    for r in arabic_df.itertuples(index=False):
        article_id = int(r.id)
        text = r.text_sentiment

        t0 = time.perf_counter()
        try:
            ra = sent_arabert.predict(text)
        except Exception as e:
            logger.error(f"AraBERT sentiment failed for article {article_id}: {e}")
            ra = None
        timing_records.append({"article_id": article_id, "task": "sentiment", "model_name": "arabert",
                               "elapsed_seconds": round(time.perf_counter() - t0, 6)})

        t0 = time.perf_counter()
        try:
            rc = sent_camel.predict(text)
        except Exception as e:
            logger.error(f"CAMeL sentiment failed for article {article_id}: {e}")
            rc = None
        timing_records.append({"article_id": article_id, "task": "sentiment", "model_name": "camel",
                               "elapsed_seconds": round(time.perf_counter() - t0, 6)})

        t0 = time.perf_counter()
        try:
            rm = sent_mbert.predict(text)
        except Exception as e:
            logger.error(f"mBERT sentiment failed for article {article_id}: {e}")
            rm = None
        timing_records.append({"article_id": article_id, "task": "sentiment", "model_name": "mbert",
                               "elapsed_seconds": round(time.perf_counter() - t0, 6)})

        db.save_sentiment_results(
            article_id=article_id,
            arabert_label=ra.label if ra else None,
            arabert_score=round(ra.score, 4) if ra else None,
            arabert_probs=json.dumps(_round_probs(ra.probs), ensure_ascii=False) if ra else None,
            camel_label=rc.label if rc else None,
            camel_score=round(rc.score, 4) if rc else None,
            camel_probs=json.dumps(_round_probs(rc.probs), ensure_ascii=False) if rc else None,
            mbert_label=rm.label if rm else None,
            mbert_score=round(rm.score, 4) if rm else None,
            mbert_probs=json.dumps(_round_probs(rm.probs), ensure_ascii=False) if rm else None,
        )

        logger.info(
            f"[SENT {article_id}] "
            f"arabert={getattr(ra, 'label', None)} camel={getattr(rc, 'label', None)} mbert={getattr(rm, 'label', None)}"
        )

    logger.info("Sentiment complete: results saved to DB")

    # ============================================================
    # Keywords (TF-IDF) -> DB (1 row/article) [FLEXIBLE COUNT]
    # ============================================================
    logger.info("=== KEYWORD EXTRACTION ===")

    stopwords_list = Config.ARABIC_STOPWORDS
    logger.info(f"Loaded {len(stopwords_list)} stopwords from Config")

    kw_extractor = TFIDFKeywordExtractor(
        stopwords=stopwords_list,
        max_features=KW_MAX_FEATURES,
        logger=logger,
    )

    fit_t0 = time.perf_counter()
    corpus = arabic_df["text_keywords"].tolist()
    kw_extractor.fit(corpus)
    logger.info(f"[KW FIT] TF-IDF fit time: {time.perf_counter() - fit_t0:.4f}s on {len(corpus)} documents")

    for r in arabic_df.itertuples(index=False):
        article_id = int(r.id)
        text = r.text_keywords

        t0 = time.perf_counter()
        try:
            keywords = kw_extractor.extract_flexible(
                text,
                min_k=KW_MIN_K,
                max_k=KW_MAX_K,
                rel_threshold=KW_REL_THRESHOLD,
                coverage_target=KW_COVERAGE_TARGET,
            )
        except Exception as e:
            logger.error(f"Keyword extraction failed for article {article_id}: {e}")
            keywords = []

        timing_records.append({
            "article_id": article_id,
            "task": "keywords",
            "model_name": "tfidf",
            "elapsed_seconds": round(time.perf_counter() - t0, 6),
        })

        kw_str = ", ".join(f"{word} ({score:.4f})" for word, score in keywords)
        kw_count = len(keywords)

        # UPDATED: save keywords_count too
        db.save_keyword_results(
            article_id=article_id,
            keywords=kw_str,
            keywords_count=kw_count,
        )

        logger.info(f"[KW {article_id}] extracted {kw_count} keywords")

    logger.info("Keyword extraction complete: results saved to DB")

    # ============================================================
    # COMPARISON REPORT (local CSVs in the SAME run_dir)
    # ============================================================
    try:
        generate_comparison_report(run_dir=run_dir, timing_records=timing_records)
        logger.info("Comparison CSVs generated successfully.")
    except Exception as e:
        logger.error(f"Comparison report generation failed: {e}")

    # ============================================================
    # DONE
    # ============================================================
    pipeline_total = time.perf_counter() - pipeline_t0
    logger.info(f"Pipeline complete in {pipeline_total:.2f}s")

    db.close()


if __name__ == "__main__":
    main()