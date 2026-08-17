"""
sample_extraction.py
--------------------
Extracts a stratified sample of annotated articles from the production database
and writes them to data/manual_eval_global.csv for manual evaluation.

NOTE: This script requires the main application project's database environment
      (src.db_config, src.preprocessing) to be available in PYTHONPATH.
      It is kept here for reference and is NOT run as part of the benchmark pipeline.

Usage:
    python annotation/sample_extraction.py --n_ar 150 --n_en 150 --n_fr 10
"""
from __future__ import annotations

import sys
import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import text, bindparam

# Ensure the main application project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# These imports require the external application environment
from src.db_config import DatabaseConnection  # noqa: E402
from src.preprocessing import PreprocessRouter
import logging

MODEL_VERSIONS = [0]


def _make_expanding_in_clause(base_sql: str, param_name: str = "ids"):
    return text(base_sql).bindparams(bindparam(param_name, expanding=True))


def _pick_language_per_article(sent_long: pd.DataFrame) -> pd.DataFrame:
    """
    Articles are in long format. Language should be same per article,
    but we pick the first non-null value.
    """
    if sent_long.empty or "language" not in sent_long.columns:
        return pd.DataFrame(columns=["article_id", "language"])

    tmp = sent_long.dropna(subset=["language"]).copy()
    tmp["language"] = tmp["language"].astype(str).str.lower().str.strip()
    tmp = tmp[tmp["language"] != ""]
    if tmp.empty:
        return pd.DataFrame(columns=["article_id", "language"])

    return (
        tmp.sort_values("article_id")
           .groupby("article_id", as_index=False)
           .first()[["article_id", "language"]]
    )


def main(n_ar: int = 150, n_en: int = 150, n_fr: int = 10, raw_table: str = "article") -> Path | list[Path]:
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db = DatabaseConnection()
    engine = db.get_engine()

    # -------------------------------------------------------------------------
    # 1) Sample article IDs (must have topic + sentiment + at least 1 entity)
    # -------------------------------------------------------------------------
    sample_sql = f"""
        (
            SELECT a.id AS article_id
            FROM {raw_table} a
            JOIN articles_enriched ae ON ae.article_id = a.id
              AND ae.language = 'ar'
              AND EXISTS (SELECT 1 FROM article_topics t WHERE t.article_id = a.id)
              AND EXISTS (SELECT 1 FROM article_entities ane WHERE ane.article_id = a.id)
            GROUP BY a.id
            ORDER BY RAND()
            LIMIT :n_ar
        )
        UNION ALL
        (
            SELECT a.id AS article_id
            FROM {raw_table} a
            JOIN articles_enriched ae ON ae.article_id = a.id
            WHERE a.body IS NOT NULL
              AND ae.language = 'en'
              AND EXISTS (SELECT 1 FROM article_topics t WHERE t.article_id = a.id)
              AND EXISTS (SELECT 1 FROM article_entities ane WHERE ane.article_id = a.id)
            GROUP BY a.id
            ORDER BY RAND()
            LIMIT :n_en
        )
        UNION ALL
        (
            SELECT a.id AS article_id
            FROM {raw_table} a
            JOIN articles_enriched ae ON ae.article_id = a.id
            WHERE a.body IS NOT NULL
              AND ae.language = 'fr'
              AND EXISTS (SELECT 1 FROM article_topics t WHERE t.article_id = a.id)
              AND EXISTS (SELECT 1 FROM article_entities ane WHERE ane.article_id = a.id)
            GROUP BY a.id
            ORDER BY RAND()
            LIMIT :n_fr
        )
    """
    sample_ids = pd.read_sql(text(sample_sql), engine, params={"n_ar": int(n_ar), "n_en": int(n_en), "n_fr": int(n_fr)})
    if sample_ids.empty:
        raise RuntimeError("No eligible articles found (need topic + entities).")

    ids = [int(x) for x in sample_ids["article_id"].tolist()]

    # -------------------------------------------------------------------------
    # 2) Load raw article fields
    # -------------------------------------------------------------------------
    articles_sql = _make_expanding_in_clause(f"""
        SELECT
            a.id AS article_id,
            a.body
        FROM {raw_table} a
        WHERE a.id IN :ids
    """)
    articles = pd.read_sql(articles_sql, engine, params={"ids": ids})
    articles["body"] = articles["body"].fillna("")

    # -------------------------------------------------------------------------
    # 3) Load topic predictions
    # -------------------------------------------------------------------------
    topics_sql = _make_expanding_in_clause("""
        SELECT
            article_id,
            topic_label
        FROM article_topics
        WHERE article_id IN :ids
    """)
    topics_long = pd.read_sql(topics_sql, engine, params={"ids": ids})

    if not topics_long.empty:
        topics_long["topic_label"] = topics_long["topic_label"].astype(str).str.strip()
        topic_label_first = topics_long.groupby("article_id")["topic_label"].first()
        topics_wide = pd.DataFrame({"thème prédit": topic_label_first}).reset_index()
    else:
        topics_wide = pd.DataFrame({"article_id": ids, "thème prédit": ""})

    # -------------------------------------------------------------------------
    # 4) Load sentiment (long format), then pivot wide
    # -------------------------------------------------------------------------
    sent_sql = _make_expanding_in_clause("""
        SELECT
            article_id,
            language,
            sentiment_label
        FROM articles_enriched
        WHERE article_id IN :ids
    """)
    sent_long = pd.read_sql(sent_sql, engine, params={"ids": ids})

    lang_per_article = _pick_language_per_article(sent_long)

    if not sent_long.empty:
        sent_long["sentiment_label"] = sent_long["sentiment_label"].astype(str).str.upper().str.strip()
        sent_label_first = sent_long.groupby("article_id")["sentiment_label"].first()
        sent_wide = pd.DataFrame({"sentiment prédit": sent_label_first}).reset_index()
    else:
        sent_wide = pd.DataFrame({"article_id": ids, "sentiment prédit": ""})

    # -------------------------------------------------------------------------
    # 5) Merge everything into one article-level dataset
    # -------------------------------------------------------------------------
    out = articles.merge(topics_wide, on="article_id", how="left")
    out = out.merge(lang_per_article, on="article_id", how="left")
    out = out.merge(sent_wide, on="article_id", how="left")

    out["language"]         = out["language"].fillna("")
    out["body"]             = out["body"].fillna("")
    out["thème prédit"]     = out["thème prédit"].fillna("")
    out["sentiment prédit"] = out["sentiment prédit"].fillna("")

    print("Preprocessing text for sentiment...")
    preproc = PreprocessRouter(logger=logger)
    out["texte"] = out.apply(lambda r: preproc.preprocess(r["body"], str(r["language"]), "sentiment"), axis=1)

    out["langue"]                    = out["language"]
    out["thème attendu"]             = ""
    out["sentiment attendu"]         = ""
    out["correct/incorrect theme"]   = ""
    out["correct/incorrect sentiment"] = ""

    final_cols = [
        "texte",
        "langue",
        "thème attendu",
        "thème prédit",
        "correct/incorrect theme",
        "sentiment attendu",
        "sentiment prédit",
        "correct/incorrect sentiment"
    ]

    out = out[final_cols].sort_values(by="langue", ascending=True)

    out_path = out_dir / "manual_eval_global.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    db.close()
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ar",       type=int, default=150, help="Number of Arabic articles to export")
    ap.add_argument("--n_en",       type=int, default=150, help="Number of English articles to export")
    ap.add_argument("--n_fr",       type=int, default=10,  help="Number of French articles to export")
    ap.add_argument("--raw_table",  type=str, default="article", help="Raw article table name")
    args = ap.parse_args()

    main(n_ar=args.n_ar, n_en=args.n_en, n_fr=args.n_fr, raw_table=args.raw_table)
