from __future__ import annotations

import sys
import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import text, bindparam

# Ensure imports work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db_config import DatabaseConnection  # noqa: E402


MODEL_VERSION_NAME = {
    0: "arabert",
    1: "camel",
    2: "en_bert",
    3: "fr_bert",
}




def _make_expanding_in_clause(base_sql: str, param_name: str = "ids"):
    return text(base_sql).bindparams(bindparam(param_name, expanding=True))




def _pick_language_per_article(sent_long: pd.DataFrame) -> pd.DataFrame:
    """
    articles_enriched is long format. Language should be same per article,
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
        tmp.sort_values(["article_id", "model_version"])
           .groupby("article_id", as_index=False)
           .first()[["article_id", "language"]]
    )


def main(n_ar: int = 200, n_en: int = 200, raw_table: str = "article") -> Path | list[Path]:
    out_dir = Path(__file__).resolve().parent

    db = DatabaseConnection()
    engine = db.get_engine()

    # ---------------------------------------------------------------------
    # 1) Sample article IDs (must have topic + sentiment + at least 1 entity)
    # ---------------------------------------------------------------------
    sample_sql = f"""
        (
            SELECT a.id AS article_id
            FROM {raw_table} a
            JOIN articles_enriched ae ON ae.article_id = a.id
              AND ae.language = 'ar'
              AND ae.sentiment_label IS NOT NULL
              AND CHAR_LENGTH(a.body) >= 150
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
              AND ae.sentiment_label IS NOT NULL
              AND CHAR_LENGTH(a.body) >= 150
              AND EXISTS (SELECT 1 FROM article_topics t WHERE t.article_id = a.id)
              AND EXISTS (SELECT 1 FROM article_entities ane WHERE ane.article_id = a.id)
            GROUP BY a.id
            ORDER BY RAND()
            LIMIT :n_en
        )
    """
    sample_ids = pd.read_sql(text(sample_sql), engine, params={"n_ar": int(n_ar), "n_en": int(n_en)})
    if sample_ids.empty:
        raise RuntimeError("No eligible articles found (need topic + sentiment + entities).")

    ids = [int(x) for x in sample_ids["article_id"].tolist()]

    # ---------------------------------------------------------------------
    # 2) Load raw article fields
    # ---------------------------------------------------------------------
    articles_sql = _make_expanding_in_clause(f"""
        SELECT
            a.id AS article_id,
            a.body
        FROM {raw_table} a
        WHERE a.id IN :ids
    """)
    articles = pd.read_sql(articles_sql, engine, params={"ids": ids})

    # Ensure body is a string
    articles["body"] = articles["body"].fillna("")

    # ---------------------------------------------------------------------
    # 3) Load topic predictions
    # ---------------------------------------------------------------------
    topics_sql = _make_expanding_in_clause("""
        SELECT
            article_id,
            topic_label,
            topic_score
        FROM article_topics
        WHERE article_id IN :ids
    """)
    topics = pd.read_sql(topics_sql, engine, params={"ids": ids})

    # ---------------------------------------------------------------------
    # 4) Load sentiment (long format), then pivot wide
    # ---------------------------------------------------------------------
    sent_sql = _make_expanding_in_clause("""
        SELECT
            article_id,
            language,
            model_version,
            sentiment_label,
            sentiment_score,
            cpu_time_ner,
            cpu_time_sentiment,
            cpu_time_topic,
            gpu_time_ner,
            gpu_time_sentiment,
            gpu_time_topic,
            dominant_topic
        FROM articles_enriched
        WHERE article_id IN :ids
    """)
    sent_long = pd.read_sql(sent_sql, engine, params={"ids": ids})

    lang_per_article = _pick_language_per_article(sent_long)

    if not sent_long.empty:
        sent_long["model_version"] = pd.to_numeric(sent_long["model_version"], errors="coerce")
        sent_long = sent_long.dropna(subset=["model_version"]).copy()
        sent_long["model_version"] = sent_long["model_version"].astype(int)

        sent_long["sentiment_label"] = sent_long["sentiment_label"].astype(str).str.upper().str.strip()
        sent_long["sentiment_score"] = pd.to_numeric(sent_long["sentiment_score"], errors="coerce")

        sent_label_wide = sent_long.pivot_table(
            index="article_id", columns="model_version", values="sentiment_label", aggfunc="first"
        )
        sent_score_wide = sent_long.pivot_table(
            index="article_id", columns="model_version", values="sentiment_score", aggfunc="first"
        )

        sent_label_wide.columns = [f"sentiment_label_mv{c}" for c in sent_label_wide.columns]
        sent_score_wide.columns = [f"sentiment_score_mv{c}" for c in sent_score_wide.columns]

        sent_wide = pd.concat([sent_label_wide, sent_score_wide], axis=1).reset_index()
    else:
        sent_wide = pd.DataFrame({"article_id": ids})

    # ---------------------------------------------------------------------
    # 6) Merge everything into one article-level dataset
    # ---------------------------------------------------------------------
    out = articles.merge(topics, on="article_id", how="left")
    out = out.merge(lang_per_article, on="article_id", how="left")
    out = out.merge(sent_wide, on="article_id", how="left")

    # Nice ordering
    front = [
        "article_id", "language",
        "topic_label", "topic_score",
        "body",
    ]
    front = [c for c in front if c in out.columns]
    remaining = [c for c in out.columns if c not in front]
    out = out[front + remaining]

    # Explicitly sort by language: 'ar' < 'en' so ascending puts ar first
    out = out.sort_values(by="language", ascending=True)

    out["topic_verdict"] = ""
    out["true_prediction"] = ""

    # ---------------------------------------------------------------------
    # 7) Save CSVs in scripts/ folder
    # ---------------------------------------------------------------------
    topic_cols = ["article_id", "language", "body", "topic_label", "topic_score", "topic_verdict", "true_prediction"]
    topic_cols = [c for c in topic_cols if c in out.columns]
    
    
    sent_cols = ["article_id", "language", "body", "true_prediction"] + [
        c for c in out.columns if c.startswith("sentiment")
    ]
    
    out_topic = out_dir / "manual_eval_sample_topic.csv"
    out_sent = out_dir / "manual_eval_sample_sentiment.csv"

    out[topic_cols].to_csv(out_topic, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    out[sent_cols].to_csv(out_sent, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    db.close()
    print(f"Saved: {out_topic}")
    print(f"Saved: {out_sent}")
    return out_topic


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ar", type=int, default=200, help="Number of Arabic articles to export")
    ap.add_argument("--n_en", type=int, default=200, help="Number of English articles to export")
    ap.add_argument("--raw_table", type=str, default="article", help="Raw article table name")
    args = ap.parse_args()

    main(n_ar=args.n_ar, n_en=args.n_en, raw_table=args.raw_table)