from datetime import datetime
from pathlib import Path
import logging
import json
import sys
import platform
import importlib.metadata as md

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
from src.ner_extraction import (
    TransformersNER,
    DEFAULT_NER_PARAMS,
)
from src.sentiment_analysis import (
    TransformersSentiment,
    DEFAULT_SENTIMENT_PARAMS,
    probs_norm_prali22_4class,
    probs_norm_camel_3class,
    probs_norm_nlptown_to_3class,
)
from src.keyword_extraction import TFIDFKeywordExtractor


def setup_logger():
    run_dir = Path("experiments") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    """Round all probability values to N decimals."""
    return {k: round(v, decimals) for k, v in probs.items()}


def _format_entities(entities, decimals: int = 2) -> str:
    """Format entity list into string: 'text (LBL, 0.94), text (LBL, 0.99), ...'"""
    if not entities:
        return ""
    return ", ".join(
        f"{e.text} ({e.label}, {e.score:.{decimals}f})" for e in entities
    )


def main():
    logger, run_dir = setup_logger()

    # ============================================================
    # EXPERIMENT PARAMETERS (tune here)
    # ============================================================
    sample_size = 2000
    raw_table = getattr(Config, "RAW_TABLE", "article")

    # Language detection parameters
    LANG_THRESHOLD = 0.60

    # NER parameters (start from module defaults, override as needed)
    NER_PARAMS = dict(DEFAULT_NER_PARAMS)

    # Sentiment parameters (start from module defaults, override as needed)
    SENTIMENT_PARAMS = dict(DEFAULT_SENTIMENT_PARAMS)

    # Keyword extraction parameters
    KW_MAX_FEATURES = 1000
    KW_TOP_N = 10

    # ============================================================
    # LOG RUN CONFIG (run.log + run_config.json)
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
                "id_language <> 0",
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
        },
        "ner": {
            "models": {
                "arabert": {"id_model": 0, "model_name": Config.ARABERT_NER_MODEL},
                "camel": {"id_model": 1, "model_name": Config.CAMEL_NER_MODEL},
                "mbert": {"id_model": 2, "model_name": Config.MBERT_NER_MODEL},
            },
            "params": NER_PARAMS,
            "label_unification": "all labels → first 3 letters uppercased (PER, LOC, ORG, EVE, MIS)",
            "storage": "1 row per article in ner_results table",
        },
        "sentiment": {
            "models": {
                "arabert": {"id_model": 0, "model_name": Config.ARABERT_SENTIMENT_MODEL, "labels": "POS/NEG/NEU/MIX"},
                "camel": {"id_model": 1, "model_name": Config.CAMEL_SENTIMENT_MODEL, "labels": "POS/NEG/NEU"},
                "mbert": {"id_model": 2, "model_name": Config.MBERT_SENTIMENT_MODEL, "labels": "POS/NEG/NEU (stars mapped)"},
            },
            "params": SENTIMENT_PARAMS,
            "storage": "1 row per article in sentiment_results table (3 models × label+score+probs)",
        },
        "keywords": {
            "engine": "TF-IDF (scikit-learn)",
            "max_features": KW_MAX_FEATURES,
            "top_n": KW_TOP_N,
            "vectorizer_params": {
                "max_df": 0.95,
                "min_df": 1,
            },
            "storage": "1 row per article in keyword_results table",
        },
        "result_tables": [
            "lang_detection",
            "preprocess_ner",
            "preprocess_sentiment",
            "preprocess_keywords",
            "ner_results",
            "sentiment_results",
            "keyword_results",
        ],
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

    # 1) Fetch directly from raw table
    query = f"""
        SELECT id, body, id_language
        FROM {raw_table}
        WHERE body IS NOT NULL
          AND LENGTH(body) > 50
          AND id_language <> 0
        ORDER BY crawl_date DESC, id DESC
        LIMIT {sample_size}
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Fetched {len(df)} rows from {raw_table}")

    # --- load language mapping (id_language -> code) and compute expected_lang ---
    lang_ref = pd.read_sql("SELECT id, code FROM language", engine)
    lang_map = dict(zip(lang_ref["id"], lang_ref["code"]))
    df["expected_lang"] = df["id_language"].map(lang_map)

    # 2) Preprocess for language detection
    preproc = ArabicPreprocessor(logger=logger)
    df["text_raw"] = df["body"].fillna("")
    df["text_langdetect"] = df["text_raw"].apply(preproc.preprocess_for_lang_detect)

    # 3) Language detection → save to lang_detection table
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

        lang_results.append({
            "article_id": int(r.id),
            "lang": res.lang,
            "score": res.score,
        })
        logger.info(f"[LANG {r.id}] expected={expected} pred={res.lang} correct={is_correct} score={res.score:.3f}")

    lang_df = pd.DataFrame(lang_results)
    logger.info(f"Language detection complete: {len(lang_df)} results saved to DB")

    # ============================================================
    # Filter Arabic articles for NER + Sentiment + Keywords
    # ============================================================
    merged = df.merge(
        lang_df[["article_id", "lang", "score"]],
        left_on="id",
        right_on="article_id",
        how="left",
    )

    arabic_df = merged[(merged["lang"] == "ar") & (merged["score"] >= LANG_THRESHOLD)].copy()
    logger.info(f"Arabic subset (lang='ar' & score>={LANG_THRESHOLD}): {len(arabic_df)} rows")

    # Build task-specific preprocessed text columns
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
    # NER (AraBERT vs CAMeL vs mBERT) → 1 row per article
    # ============================================================
    ner_arabert = TransformersNER(model_name=Config.ARABERT_NER_MODEL, logger=logger, preprocessor=None, **NER_PARAMS)
    ner_camel = TransformersNER(model_name=Config.CAMEL_NER_MODEL, logger=logger, preprocessor=None, **NER_PARAMS)
    ner_mbert = TransformersNER(model_name=Config.MBERT_NER_MODEL, logger=logger, preprocessor=None, **NER_PARAMS)

    for r in arabic_df.itertuples(index=False):
        text = r.text_ner
        article_id = int(r.id)

        try:
            ents_a = ner_arabert.predict(text)
        except Exception as e:
            logger.error(f"AraBERT NER failed for article {article_id}: {e}")
            ents_a = []

        try:
            ents_c = ner_camel.predict(text)
        except Exception as e:
            logger.error(f"CAMeL NER failed for article {article_id}: {e}")
            ents_c = []

        try:
            ents_m = ner_mbert.predict(text)
        except Exception as e:
            logger.error(f"mBERT NER failed for article {article_id}: {e}")
            ents_m = []

        db.save_ner_results(
            article_id=article_id,
            arabert_entities=_format_entities(ents_a),
            camel_entities=_format_entities(ents_c),
            mbert_entities=_format_entities(ents_m),
        )

        logger.info(f"[NER {article_id}] arabert={len(ents_a)} camel={len(ents_c)} mbert={len(ents_m)}")

    logger.info(f"NER complete: results saved to DB")

    # ============================================================
    # Sentiment → 1 row per article (3 models × label+score+probs)
    # ============================================================
    sent_arabert = TransformersSentiment(
        model_name=Config.ARABERT_SENTIMENT_MODEL,
        logger=logger,
        preprocessor=None,
        probs_normalizer=probs_norm_prali22_4class,
        **SENTIMENT_PARAMS
    )
    sent_camel = TransformersSentiment(
        model_name=Config.CAMEL_SENTIMENT_MODEL,
        logger=logger,
        preprocessor=None,
        probs_normalizer=probs_norm_camel_3class,
        **SENTIMENT_PARAMS
    )
    sent_mbert = TransformersSentiment(
        model_name=Config.MBERT_SENTIMENT_MODEL,
        logger=logger,
        preprocessor=None,
        probs_normalizer=probs_norm_nlptown_to_3class,
        **SENTIMENT_PARAMS
    )

    for r in arabic_df.itertuples(index=False):
        article_id = int(r.id)
        text = r.text_sentiment

        try:
            ra = sent_arabert.predict(text)
        except Exception as e:
            logger.error(f"AraBERT sentiment failed for article {article_id}: {e}")
            ra = None

        try:
            rc = sent_camel.predict(text)
        except Exception as e:
            logger.error(f"CAMeL sentiment failed for article {article_id}: {e}")
            rc = None

        try:
            rm = sent_mbert.predict(text)
        except Exception as e:
            logger.error(f"mBERT sentiment failed for article {article_id}: {e}")
            rm = None

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

    logger.info(f"Sentiment complete: results saved to DB")

    # ============================================================
    # Keyword Extraction (TF-IDF) → 1 row per article
    # ============================================================
    logger.info("=== KEYWORD EXTRACTION ===")

    stopwords_list = Config.ARABIC_STOPWORDS
    logger.info(f"Loaded {len(stopwords_list)} stopwords from Config")

    kw_extractor = TFIDFKeywordExtractor(
        stopwords=stopwords_list,
        max_features=KW_MAX_FEATURES,
        logger=logger,
    )

    corpus = arabic_df["text_keywords"].tolist()
    kw_extractor.fit(corpus)

    for r in arabic_df.itertuples(index=False):
        article_id = int(r.id)
        text = r.text_keywords

        try:
            keywords = kw_extractor.extract(text, top_n=KW_TOP_N)
        except Exception as e:
            logger.error(f"Keyword extraction failed for article {article_id}: {e}")
            keywords = []

        kw_str = ", ".join(f"{word} ({score:.4f})" for word, score in keywords)

        db.save_keyword_results(article_id=article_id, keywords=kw_str)

        logger.info(f"[KW {article_id}] extracted {len(keywords)} keywords")

    logger.info(f"Keyword extraction complete: results saved to DB")

    # ============================================================
    # DONE
    # ============================================================
    db.close()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()