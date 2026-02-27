from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _to_year_month(crawl_date_val, year_val=None, month_val=None) -> Optional[pd.Timestamp]:
    """
    Primary: crawl_date -> YYYY-MM-01
    Fallback: (year, month) -> YYYY-MM-01 if crawl_date missing/invalid
    """
    cd = pd.to_datetime(crawl_date_val, errors="coerce")
    if not pd.isna(cd):
        try:
            return pd.Timestamp(year=int(cd.year), month=int(cd.month), day=1)
        except Exception:
            pass

    try:
        if pd.notna(year_val) and pd.notna(month_val):
            y = int(year_val)
            m = int(month_val)
            if 1 <= m <= 12 and 1900 <= y <= 2100:
                return pd.Timestamp(year=y, month=m, day=1)
    except Exception:
        pass

    return None


def topics_by_month(engine, raw_table: str = "article") -> pd.DataFrame:
    """
    Topic frequency per month (GLOBAL: no country, no language).

    Output (one row per month x topic):
      year_month, topic_label,
      articles_count, topic_share,
      mean_topic_score, median_topic_score,
      total_articles_in_month
    """

    query = f"""
        SELECT
            t.article_id,
            t.topic_label,
            t.topic_score,
            a.crawl_date,
            a.year,
            a.month
        FROM article_topics t
        JOIN {raw_table} a
          ON a.id = t.article_id
        WHERE t.topic_label IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"[topics_by_month] Loaded {len(df)} topic rows (joined with {raw_table})")
    if df.empty:
        return pd.DataFrame()

    df["topic_score"] = pd.to_numeric(df["topic_score"], errors="coerce")
    df["topic_label"] = df["topic_label"].astype(str).str.strip()

    df["year_month"] = df.apply(
        lambda r: _to_year_month(r.get("crawl_date"), r.get("year"), r.get("month")),
        axis=1,
    )
    df = df.dropna(subset=["year_month", "topic_label"]).copy()
    if df.empty:
        return pd.DataFrame()

    ym = pd.to_datetime(df["year_month"], errors="coerce").dropna()
    if not ym.empty:
        logger.info(f"[topics_by_month] available_date_range: {ym.min().date()} -> {ym.max().date()}")

    # Aggregate per (month, topic)
    agg = (
        df.groupby(["year_month", "topic_label"], as_index=False)
          .agg(
              articles_count=("article_id", "nunique"),
              mean_topic_score=("topic_score", "mean"),
              median_topic_score=("topic_score", "median"),
          )
    )

    agg["mean_topic_score"] = agg["mean_topic_score"].round(4)
    agg["median_topic_score"] = agg["median_topic_score"].round(4)

    # Total articles per month (for share)
    totals = (
        agg.groupby(["year_month"], as_index=False)
           .agg(total_articles_in_month=("articles_count", "sum"))
    )
    agg = agg.merge(totals, on="year_month", how="left")

    agg["topic_share"] = agg.apply(
        lambda r: (float(r["articles_count"]) / float(r["total_articles_in_month"]))
        if r["total_articles_in_month"] else 0.0,
        axis=1,
    )
    agg["topic_share"] = agg["topic_share"].round(4)

    agg = agg.sort_values(
        ["year_month", "articles_count", "mean_topic_score", "topic_label"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    return agg


def dominant_topic_by_month(engine, raw_table: str = "article") -> pd.DataFrame:
    """
    Dominant topic per month (GLOBAL).

    Output (one row per month):
      year_month,
      dominant_topic,
      dominant_articles_count,
      dominant_topic_share,
      dominant_mean_topic_score,
      dominant_median_topic_score,
      total_articles_in_month
    """
    dist = topics_by_month(engine=engine, raw_table=raw_table)
    if dist.empty:
        return pd.DataFrame()

    # pick top topic per month (tie-breakers already ensured by sort order)
    top = (
        dist.sort_values(
            ["year_month", "articles_count", "mean_topic_score", "topic_label"],
            ascending=[True, False, False, True],
        )
        .groupby("year_month", as_index=False)
        .first()
    )

    top = top.rename(columns={
        "topic_label": "dominant_topic",
        "articles_count": "dominant_articles_count",
        "topic_share": "dominant_topic_share",
        "mean_topic_score": "dominant_mean_topic_score",
        "median_topic_score": "dominant_median_topic_score",
    })

    cols = [
        "year_month",
        "dominant_topic",
        "dominant_articles_count",
        "dominant_topic_share",
        "dominant_mean_topic_score",
        "dominant_median_topic_score",
        "total_articles_in_month",
    ]
    return top[cols].sort_values("year_month").reset_index(drop=True)