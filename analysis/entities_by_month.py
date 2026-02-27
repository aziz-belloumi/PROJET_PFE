from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

YEAR_MONTH_FMT = "%Y-%m"  # display format (no day)


def _to_year_month(crawl_date_val, year_val=None, month_val=None) -> Optional[pd.Timestamp]:
    """
    Internal month bucket.

    We keep a real datetime internally (Timestamp), so we use day=1 as a standard
    month-start anchor. We later FORMAT it to "YYYY-MM" before returning/exporting.
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


def entities_by_month(
    engine,
    raw_table: str = "article",
    top_k_per_month: int = 50,
) -> pd.DataFrame:
    """
    Frequency of dominant entities by month (GLOBAL: no country, no language).

    Output is TOP-K per (year_month) to keep the CSV usable.

    Ranking metric:
      1) distinct_articles_count (DESC)
      2) mentions_count (DESC)
      3) mean_confidence (DESC)

    Output year_month is formatted as "YYYY-MM" (no day).
    """

    query = f"""
        SELECT
            ae.article_id,
            ae.model_version,
            ae.confidence_score,
            e.entity_type,
            e.normalized_name,
            a.crawl_date,
            a.year,
            a.month
        FROM article_entities ae
        JOIN entities e
          ON e.entity_id = ae.entity_id
        JOIN {raw_table} a
          ON a.id = ae.article_id
        WHERE e.normalized_name IS NOT NULL
          AND e.entity_type IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"[entities_by_month] Loaded {len(df)} entity mention rows")
    if df.empty:
        return pd.DataFrame()

    # Clean
    df["entity_type"] = df["entity_type"].astype(str).str.upper().str.strip()
    df["normalized_name"] = df["normalized_name"].astype(str).str.strip()
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")

    df["normalized_name"] = df["normalized_name"].replace({"nan": "", "none": ""})
    df = df[(df["normalized_name"] != "") & (df["entity_type"] != "")].copy()
    if df.empty:
        return pd.DataFrame()

    # Compute year_month (internal month-start Timestamp)
    df["year_month"] = df.apply(
        lambda r: _to_year_month(r.get("crawl_date"), r.get("year"), r.get("month")),
        axis=1,
    )
    df = df.dropna(subset=["year_month"]).copy()
    if df.empty:
        return pd.DataFrame()

    # Log available date range (no filtering) — formatted as YYYY-MM
    ym = pd.to_datetime(df["year_month"], errors="coerce").dropna()
    if not ym.empty:
        logger.info(
            f"[entities_by_month] available_date_range: "
            f"{ym.min().strftime(YEAR_MONTH_FMT)} -> {ym.max().strftime(YEAR_MONTH_FMT)}"
        )

    # mentions_count (raw) per month/entity key
    raw_mentions = (
        df.groupby(["year_month", "entity_type", "normalized_name"], as_index=False)
          .agg(mentions_count=("article_id", "size"))
    )

    # deduplicate per article entity key (across models)
    per_article = (
        df.groupby(
            ["year_month", "article_id", "entity_type", "normalized_name"],
            as_index=False,
        )
        .agg(max_confidence=("confidence_score", "max"))
    )

    # aggregate distinct-article frequency per month/entity key
    agg = (
        per_article.groupby(["year_month", "entity_type", "normalized_name"], as_index=False)
        .agg(
            distinct_articles_count=("article_id", "nunique"),
            mean_confidence=("max_confidence", "mean"),
        )
    )
    agg["mean_confidence"] = agg["mean_confidence"].round(4)

    # join mentions_count
    agg = agg.merge(
        raw_mentions,
        on=["year_month", "entity_type", "normalized_name"],
        how="left",
    )

    # rank TOP-K per month
    agg = agg.sort_values(
        ["year_month", "distinct_articles_count", "mentions_count", "mean_confidence", "normalized_name"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    agg["rank_in_month"] = agg.groupby(["year_month"]).cumcount() + 1
    result = agg[agg["rank_in_month"] <= int(top_k_per_month)].copy()

    # FINAL DISPLAY: remove day notation
    result["year_month"] = pd.to_datetime(result["year_month"], errors="coerce").dt.strftime(YEAR_MONTH_FMT)

    cols = [
        "year_month",
        "rank_in_month",
        "entity_type",
        "normalized_name",
        "distinct_articles_count",
        "mentions_count",
        "mean_confidence",
    ]
    return result[cols]