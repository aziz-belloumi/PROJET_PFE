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
    PREPROCESS_KEYWORDS_PARAMS,
)
from src.language_detection import FastTextLanguageDetector
from src.ner_extraction import (
    TransformersNER,
    DEFAULT_NER_PARAMS,
    MBERT_NER_MODEL,
)
from src.sentiment_analysis import (
    TransformersSentiment,
    DEFAULT_SENTIMENT_PARAMS,
    ARABERT_SENTIMENT_MODEL,
    CAMEL_SENTIMENT_MODEL,
    MBERT_SENTIMENT_MODEL,
    normalize_3class_label,
    normalize_nlptown_stars,
    normalize_arabert_prali4,  # <-- IMPORTANT UPDATE
)


def setup_logger():
    run_dir = Path("experiments") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_test")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("test")
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


def main():
    logger, run_dir = setup_logger()

    # ============================================================
    # EXPERIMENT PARAMETERS (tune here)
    # ============================================================
    sample_size = 5
    raw_table = getattr(Config, "RAW_TABLE", "article")

    # Language detection parameters
    LANG_THRESHOLD = 0.60

    # NER parameters (start from module defaults, override as needed)
    NER_PARAMS = dict(DEFAULT_NER_PARAMS)

    camel_ner_model = getattr(Config, "CAMEL_NER_MODEL", None)
    if not camel_ner_model:
        raise AttributeError("Missing Config.CAMEL_NER_MODEL (e.g., CAMeL-Lab/bert-base-arabic-camelbert-msa-ner)")
    mbert_ner_model = getattr(Config, "MBERT_NER_MODEL", None) or MBERT_NER_MODEL

    # Sentiment parameters (start from module defaults, override as needed)
    SENTIMENT_PARAMS = dict(DEFAULT_SENTIMENT_PARAMS)

    # Sentiment model names
    arabert_sent_model = getattr(Config, "ARABERT_SENTIMENT_MODEL", None) or ARABERT_SENTIMENT_MODEL
    camel_sent_model = getattr(Config, "CAMEL_SENTIMENT_MODEL", None) or CAMEL_SENTIMENT_MODEL
    mbert_sent_model = getattr(Config, "MBERT_SENTIMENT_MODEL", None) or MBERT_SENTIMENT_MODEL

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
            "keywords": PREPROCESS_KEYWORDS_PARAMS,
            "ner": PREPROCESS_NER_PARAMS,
        },
        "language_detection": {
            "fasttext_model_path": Config.FASTTEXT_MODEL_PATH,
            "threshold": LANG_THRESHOLD,
        },
        "ner": {
            "models": {
                "arabert": {"id_model": 0, "model_name": Config.ARABERT_NER_MODEL},
                "camel": {"id_model": 1, "model_name": camel_ner_model},
                "mbert": {"id_model": 2, "model_name": mbert_ner_model},
            },
            "params": NER_PARAMS,
            "outputs": {
                "sample_ner_preprocessed.csv": "article_id + text_ner",
                "ner_entities.csv": "1 row per entity; id_model=0 arabert, id_model=1 camel, id_model=2 mbert",
            },
        },
        "sentiment": {
            "models": {
                "arabert": {"id_model": 0, "model_name": arabert_sent_model, "labels": "POS/NEG/NEU/MIX"},
                "camel": {"id_model": 1, "model_name": camel_sent_model, "labels": "POS/NEG/NEU (expected)"},
                "mbert": {"id_model": 2, "model_name": mbert_sent_model, "labels": "1-5 stars mapped to NEG/NEU/POS"},
            },
            "params": SENTIMENT_PARAMS,
            "outputs": {
                "sentiment_results.csv": "1 row per article per model",
            },
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

    # 1) Fetch directly from raw table (NO title)
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
    # --------------------------------------------------------------------------

    # 2) Save raw sample CSV
    sample_csv = run_dir / "sample_raw.csv"
    df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved raw sample CSV: {sample_csv}")

    # 3) Preprocess for language detection + save CSV
    preproc = ArabicPreprocessor(logger=logger)
    df["text_raw"] = df["body"].fillna("")
    df["text_langdetect"] = df["text_raw"].apply(preproc.preprocess_for_lang_detect)

    pre_csv = run_dir / "sample_lang_detect_preprocessed.csv"
    df[["id", "id_language", "expected_lang", "text_langdetect"]].to_csv(pre_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved lang-detect preprocessed CSV: {pre_csv}")

    # 4) Language detection
    detector = FastTextLanguageDetector(
        model_path=Config.FASTTEXT_MODEL_PATH,
        logger=logger,
        preprocessor=None,
    )

    out = []
    for r in df.itertuples(index=False):
        res = detector.detect(r.text_langdetect)
        expected = r.expected_lang if pd.notna(r.expected_lang) else None
        is_correct = (expected == res.lang) if expected is not None else None

        out.append({
            "article_id": int(r.id),
            "lang": res.lang,
            "score": res.score,
            "expected_lang": expected,
            "is_correct": is_correct,
        })
        logger.info(f"[{r.id}] expected={expected} pred={res.lang} correct={is_correct} score={res.score:.3f}")

    lang_csv = run_dir / "lang_detection.csv"
    lang_df = pd.DataFrame(out)
    lang_df.to_csv(lang_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved language detection CSV: {lang_csv}")

    # ============================================================
    # Filter Arabic articles for NER + Sentiment
    # ============================================================
    merged = df.merge(
        lang_df[["article_id", "lang", "score"]],
        left_on="id",
        right_on="article_id",
        how="left",
    )

    arabic_df = merged[(merged["lang"] == "ar") & (merged["score"] >= LANG_THRESHOLD)].copy()
    logger.info(f"Arabic subset (lang='ar' & score>={LANG_THRESHOLD}): {len(arabic_df)} rows")

    arabic_df["text_ner"] = arabic_df["text_raw"].apply(preproc.preprocess_for_ner)

    ner_pre_csv = run_dir / "sample_ner_preprocessed.csv"
    arabic_df[["id", "text_ner"]].to_csv(ner_pre_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved NER preprocessed CSV: {ner_pre_csv}")

    # ============================================================
    # NER (AraBERT vs CAMeL vs mBERT)
    # ============================================================
    ner_arabert = TransformersNER(model_name=Config.ARABERT_NER_MODEL, logger=logger, preprocessor=None, **NER_PARAMS)
    ner_camel = TransformersNER(model_name=camel_ner_model, logger=logger, preprocessor=None, **NER_PARAMS)
    ner_mbert = TransformersNER(model_name=mbert_ner_model, logger=logger, preprocessor=None, **NER_PARAMS)

    entity_rows = []
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

        for e in ents_a:
            entity_rows.append({"article_id": article_id, "id_model": 0, "text": e.text, "label": e.label, "start": e.start, "end": e.end, "score": e.score})
        for e in ents_c:
            entity_rows.append({"article_id": article_id, "id_model": 1, "text": e.text, "label": e.label, "start": e.start, "end": e.end, "score": e.score})
        for e in ents_m:
            entity_rows.append({"article_id": article_id, "id_model": 2, "text": e.text, "label": e.label, "start": e.start, "end": e.end, "score": e.score})

        logger.info(f"[NER {article_id}] arabert={len(ents_a)} camel={len(ents_c)} mbert={len(ents_m)}")

    ner_entities_csv = run_dir / "ner_entities.csv"
    pd.DataFrame(entity_rows).to_csv(ner_entities_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved NER entities CSV: {ner_entities_csv}")

    # ============================================================
    # Sentiment (1 row per article per model)
    # ============================================================
    sent_arabert = TransformersSentiment(
        model_name=arabert_sent_model,
        logger=logger,
        preprocessor=None,
        label_normalizer=normalize_arabert_prali4,  # <-- IMPORTANT UPDATE
        **SENTIMENT_PARAMS
    )
    sent_camel = TransformersSentiment(
        model_name=camel_sent_model,
        logger=logger,
        preprocessor=None,
        label_normalizer=normalize_3class_label,
        **SENTIMENT_PARAMS
    )
    sent_mbert = TransformersSentiment(
        model_name=mbert_sent_model,
        logger=logger,
        preprocessor=None,
        label_normalizer=normalize_nlptown_stars,
        **SENTIMENT_PARAMS
    )

    sentiment_rows = []
    for r in arabic_df.itertuples(index=False):
        article_id = int(r.id)
        text = r.text_ner

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

        if ra is not None:
            sentiment_rows.append({
                "article_id": article_id,
                "id_model": 0,
                "label": ra.label,
                "score": ra.score,
                "raw_best_label": ra.raw_best_label,
                "probs_json": json.dumps(ra.probs, ensure_ascii=False),
            })

        if rc is not None:
            sentiment_rows.append({
                "article_id": article_id,
                "id_model": 1,
                "label": rc.label,
                "score": rc.score,
                "raw_best_label": rc.raw_best_label,
                "probs_json": json.dumps(rc.probs, ensure_ascii=False),
            })

        if rm is not None:
            sentiment_rows.append({
                "article_id": article_id,
                "id_model": 2,
                "label": rm.label,
                "score": rm.score,
                "raw_best_label": rm.raw_best_label,
                "probs_json": json.dumps(rm.probs, ensure_ascii=False),
            })

        logger.info(
            f"[SENT {article_id}] "
            f"arabert={getattr(ra, 'label', None)} camel={getattr(rc, 'label', None)} mbert={getattr(rm, 'label', None)}"
        )

    sentiment_csv = run_dir / "sentiment_results.csv"
    pd.DataFrame(sentiment_rows).to_csv(sentiment_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved sentiment results CSV: {sentiment_csv}")

    db.close()
    logger.info("Test done.")


if __name__ == "__main__":
    main()