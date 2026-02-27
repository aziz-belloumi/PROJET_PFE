from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

YEAR_MONTH_FMT = "%Y-%m"  # display format (no day)


def parse_id_countries(val) -> List[int]:
    if val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() in {"null", "none", "nan"}:
        return []
    out: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out


def _to_effective_date(crawl_date_val, year_val=None, month_val=None) -> Optional[pd.Timestamp]:
    """
    Primary: crawl_date
    Fallback: year/month -> YYYY-MM-01 (month anchor)
    Used only to log min/max available dates (no filtering).
    """
    cd = pd.to_datetime(crawl_date_val, errors="coerce")
    if not pd.isna(cd):
        return cd

    try:
        if pd.notna(year_val) and pd.notna(month_val):
            y = int(year_val)
            m = int(month_val)
            if 1 <= m <= 12 and 1900 <= y <= 2100:
                return pd.Timestamp(year=y, month=m, day=1)
    except Exception:
        pass

    return None


def top_entities_by_country(
    engine,
    raw_table: str = "article",
    top_k: int = 50,
    default_country_id: int = 8,
) -> pd.DataFrame:
    """
    Top entities per country (using article.id_countries), regardless of language.

    Frequency definition:
      - distinct_articles_count = number of DISTINCT articles mentioning the entity
      - mentions_count = total extracted mentions (raw rows) across articles/models

    Notes:
    - Deduplicates per article: (article_id, entity_type, normalized_name) across models,
      keeping max confidence per article for that entity key.
    - Uses crawl_date (fallback year/month) only to log the available date range; no date filtering.
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
            a.month,
            a.id_countries
        FROM article_entities ae
        JOIN entities e
          ON e.entity_id = ae.entity_id
        JOIN {raw_table} a
          ON a.id = ae.article_id
        WHERE e.normalized_name IS NOT NULL
          AND e.entity_type IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    logger.info(f"[top_entities_by_country] Loaded {len(df)} entity mention rows")
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

    # Log available date range (no filtering) — formatted as YYYY-MM
    df["effective_date"] = df.apply(
        lambda r: _to_effective_date(r.get("crawl_date"), r.get("year"), r.get("month")),
        axis=1,
    )
    eff = pd.to_datetime(df["effective_date"], errors="coerce").dropna()
    if not eff.empty:
        logger.info(
            f"[top_entities_by_country] available_date_range: "
            f"{eff.min().strftime(YEAR_MONTH_FMT)} -> {eff.max().strftime(YEAR_MONTH_FMT)}"
        )
    df = df.drop(columns=["effective_date"], errors="ignore")

    # Parse / explode id_countries
    df["country_ids"] = df["id_countries"].apply(parse_id_countries)
    missing_mask = df["country_ids"].apply(len) == 0
    if missing_mask.any():
        n_missing = int(missing_mask.sum())
        logger.info(
            f"[top_entities_by_country] id_countries missing for {n_missing} rows -> defaulting to {default_country_id}"
        )
        df.loc[missing_mask, "country_ids"] = [[int(default_country_id)]] * n_missing

    df = df.explode("country_ids").rename(columns={"country_ids": "country_id"})
    df["country_id"] = pd.to_numeric(df["country_id"], errors="coerce")
    df = df.dropna(subset=["country_id"]).copy()
    df["country_id"] = df["country_id"].astype(int)

    # mentions_count (raw)
    raw_mentions = (
        df.groupby(["country_id", "entity_type", "normalized_name"], as_index=False)
          .agg(mentions_count=("article_id", "size"))
    )

    # Deduplicate per-article entity key (across models)
    per_article = (
        df.groupby(
            ["country_id", "article_id", "entity_type", "normalized_name"],
            as_index=False,
        )
        .agg(max_confidence=("confidence_score", "max"))
    )

    # Aggregate frequency by DISTINCT articles
    agg = (
        per_article.groupby(["country_id", "entity_type", "normalized_name"], as_index=False)
        .agg(
            distinct_articles_count=("article_id", "nunique"),
            mean_confidence=("max_confidence", "mean"),
        )
    )
    agg["mean_confidence"] = agg["mean_confidence"].round(4)

    # Join raw mention counts
    agg = agg.merge(raw_mentions, on=["country_id", "entity_type", "normalized_name"], how="left")

    # Country labels
    countries = pd.read_sql(
        "SELECT id AS country_id, label_en, label_fr, label_ar FROM country",
        engine,
    )
    agg = agg.merge(countries, on="country_id", how="left")

    # Rank Top-K per country
    agg = agg.sort_values(
        ["country_id", "distinct_articles_count", "mentions_count", "mean_confidence", "normalized_name"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    agg["rank_in_country"] = agg.groupby(["country_id"]).cumcount() + 1
    result = agg[agg["rank_in_country"] <= int(top_k)].copy()

    cols = [
        "country_id", "label_en",
        "rank_in_country",
        "entity_type", "normalized_name",
        "distinct_articles_count", "mentions_count", "mean_confidence",
    ]
    return result[cols]