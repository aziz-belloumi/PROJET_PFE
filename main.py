from datetime import datetime
from pathlib import Path
import logging
import json
import pandas as pd

from src.config import Config
from src.db_config import DatabaseConnection
from src.preprocessing import ArabicPreprocessor
from src.language_detection import FastTextLanguageDetector
from src.ner_extraction import TransformersNER


def setup_logger():
    run_dir = Path("experiments") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_test")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("test")
    logger.setLevel(Config.LOG_LEVEL)
    logger.handlers.clear()

    fmt = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, run_dir


def main():
    logger, run_dir = setup_logger()

    logger.info(f"FASTTEXT_MODEL_PATH = {Config.FASTTEXT_MODEL_PATH}")
    logger.info(f"DB = {Config.DB_NAME}@{Config.DB_HOST}:{Config.DB_PORT}")

    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()  # if not available: engine = db.engine

    sample_size = 200
    raw_table = getattr(Config, "RAW_TABLE", "article")

    # 1) Fetch directly from raw table (NO title)
    query = f"""
        SELECT id, body,id_language
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

    # 2) Save raw sample CSV inside experiment folder
    sample_csv = run_dir / "sample_raw.csv"
    df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved raw sample CSV: {sample_csv}")

    # 3) Preprocess for language detection + save preprocessed CSV
    preproc = ArabicPreprocessor(logger=logger)

    df["text_raw"] = df["body"].fillna("")
    df["text_langdetect"] = df["text_raw"].apply(preproc.preprocess_for_lang_detect)

    pre_csv = run_dir / "sample_langdetect_preprocessed.csv"
    df[["id", "id_language", "expected_lang", "text_langdetect"]].to_csv(
        pre_csv, index=False, encoding="utf-8-sig"
    )
    logger.info(f"Saved lang-detect preprocessed CSV: {pre_csv}")

    # 4) Language detection (use the already-preprocessed text)
    detector = FastTextLanguageDetector(
        model_path=Config.FASTTEXT_MODEL_PATH,
        logger=logger,
        preprocessor=None
    )

    out = []
    for r in df.itertuples(index=False):
        res = detector.detect(r.text_langdetect)

        expected = r.expected_lang if pd.notna(r.expected_lang) else None
        is_correct = (expected == res.lang) if expected is not None else None

        out.append({
            "article_id": int(r.id),
            "lang": res.lang,              # predicted
            "score": res.score,
            "expected_lang": expected,     # from id_language mapping
            "is_correct": is_correct
        })

        logger.info(
            f"[{r.id}] expected={expected} pred={res.lang} correct={is_correct} score={res.score:.3f}"
        )

    # 5) Save detection results to CSV
    lang_csv = run_dir / "lang_detection.csv"
    lang_df = pd.DataFrame(out)
    lang_df.to_csv(lang_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved language detection CSV: {lang_csv}")


    # NER (AraBERT vs CAMeL) - 1 row per article (JSON columns)
    camel_ner_model = getattr(Config, "CAMEL_NER_MODEL", None)
    if not camel_ner_model:
        raise AttributeError("Missing Config.CAMEL_NER_MODEL (e.g., CAMeL-Lab/bert-base-arabic-camelbert-msa-ner)")

    merged = df.merge(
        lang_df[["article_id", "lang", "score"]],
        left_on="id",
        right_on="article_id",
        how="left"
    )

    LANG_THRESHOLD = 0.60
    arabic_df = merged[(merged["lang"] == "ar") & (merged["score"] >= LANG_THRESHOLD)].copy()
    logger.info(f"NER input after filtering Arabic (lang='ar' & score>={LANG_THRESHOLD}): {len(arabic_df)} rows")

    arabic_df["text_ner"] = arabic_df["text_raw"].apply(preproc.preprocess_for_ner)

    ner_pre_csv = run_dir / "sample_ner_preprocessed.csv"
    arabic_df[["id", "id_language", "expected_lang", "text_ner"]].to_csv(
        ner_pre_csv, index=False, encoding="utf-8-sig"
    )
    logger.info(f"Saved NER preprocessed CSV: {ner_pre_csv}")

    # Load both NER models once
    ner_arabert = TransformersNER(model_name=Config.ARABERT_NER_MODEL, logger=logger, preprocessor=None)
    ner_camel = TransformersNER(model_name=camel_ner_model, logger=logger, preprocessor=None)

    ner_rows = []
    for r in arabic_df.itertuples(index=False):
        text = r.text_ner

        try:
            ents_a = ner_arabert.predict(text)
        except Exception as e:
            logger.error(f"AraBERT NER failed for article {r.id}: {e}")
            ents_a = []

        try:
            ents_c = ner_camel.predict(text)
        except Exception as e:
            logger.error(f"CAMeL NER failed for article {r.id}: {e}")
            ents_c = []

        ents_a_list = [e.__dict__ for e in ents_a]
        ents_c_list = [e.__dict__ for e in ents_c]

        ner_rows.append({
            "article_id": int(r.id),
            "text_ner": text,

            "arabert_model": "ARABERT_NER_MODEL",
            "camel_model": "CAMEL_NER_MODEL",

            "arabert_count": len(ents_a_list),
            "camel_count": len(ents_c_list),

            # JSON in same row (alternative design)
            "arabert_entities_json": json.dumps(ents_a_list, ensure_ascii=False),
            "camel_entities_json": json.dumps(ents_c_list, ensure_ascii=False),
        })

        logger.info(f"[NER {r.id}] arabert={len(ents_a_list)} camel={len(ents_c_list)}")

    ner_csv = run_dir / "ner_comparison.csv"
    pd.DataFrame(ner_rows).to_csv(ner_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved NER comparison CSV: {ner_csv}")

    # =========================

    db.close()
    logger.info("Test done.")


if __name__ == "__main__":
    main()