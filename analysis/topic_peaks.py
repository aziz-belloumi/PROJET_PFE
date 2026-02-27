from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def topic_peaks(
    topics_by_month_df: pd.DataFrame,
    window: int = 6,
    z_threshold: float = 2.5,
    min_periods: int = 3,
    min_articles_in_month: int = 5,
) -> pd.DataFrame:
    """
    Detect abnormal peaks in GLOBAL topic frequency time series (no country, no language).

    Expected input columns (from topics_by_month()):
      - year_month (datetime-like or str)
      - topic_label
      - articles_count
      - total_articles_in_month
      - topic_share

    Method:
      - Build a complete monthly timeline between min and max available months
      - For each topic: fill missing months with 0
      - Compute rolling mean/std on topic_share
      - z = (share - mean) / std
      - Flag peaks when z >= z_threshold and total_articles_in_month >= min_articles_in_month
    """

    df = topics_by_month_df.copy()
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"year_month", "topic_label", "articles_count", "total_articles_in_month", "topic_share"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(f"[topic_peaks] Missing required columns: {sorted(missing)}")
        return pd.DataFrame()

    # Normalize
    df["topic_label"] = df["topic_label"].astype(str).str.strip()
    df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
    df["articles_count"] = pd.to_numeric(df["articles_count"], errors="coerce").fillna(0).astype(int)
    df["total_articles_in_month"] = pd.to_numeric(df["total_articles_in_month"], errors="coerce").fillna(0).astype(int)
    df["topic_share"] = pd.to_numeric(df["topic_share"], errors="coerce").fillna(0.0).astype(float)

    df = df.dropna(subset=["year_month", "topic_label"]).copy()
    if df.empty:
        return pd.DataFrame()

    # Ensure month-start timestamps
    df["year_month"] = df["year_month"].dt.to_period("M").dt.to_timestamp()

    # Log available range
    all_months = df["year_month"].dropna()
    logger.info(f"[topic_peaks] available_date_range: {all_months.min().date()} -> {all_months.max().date()}")

    # ---------------------------------------------------------
    # Step 1: create full monthly calendar
    # ---------------------------------------------------------
    min_m = df["year_month"].min()
    max_m = df["year_month"].max()
    if pd.isna(min_m) or pd.isna(max_m):
        return pd.DataFrame()

    calendar = pd.DataFrame({"year_month": pd.date_range(min_m, max_m, freq="MS")})

    # totals per month (from df)
    totals = (
        df.groupby("year_month", as_index=False)
          .agg(total_articles_in_month=("total_articles_in_month", "max"))
    )
    # Merge totals onto calendar, fill missing with 0
    totals_full = calendar.merge(totals, on="year_month", how="left")
    totals_full["total_articles_in_month"] = (
        pd.to_numeric(totals_full["total_articles_in_month"], errors="coerce")
          .fillna(0)
          .astype(int)
    )

    # ---------------------------------------------------------
    # Step 2: build topic-month series on full calendar
    # ---------------------------------------------------------
    topic_keys = df[["topic_label"]].drop_duplicates()
    topic_calendar = topic_keys.merge(calendar, how="cross")  # cartesian: all topics x all months

    topic_full = topic_calendar.merge(
        df[["year_month", "topic_label", "articles_count"]],
        on=["year_month", "topic_label"],
        how="left",
    )
    topic_full["articles_count"] = pd.to_numeric(topic_full["articles_count"], errors="coerce").fillna(0).astype(int)

    # Attach totals
    topic_full = topic_full.merge(totals_full, on="year_month", how="left")

    # Recompute share safely
    topic_full["topic_share"] = topic_full.apply(
        lambda r: (float(r["articles_count"]) / float(r["total_articles_in_month"]))
        if r["total_articles_in_month"] else 0.0,
        axis=1,
    )

    # ---------------------------------------------------------
    # Step 3: rolling z-score per topic
    # ---------------------------------------------------------
    topic_full = topic_full.sort_values(["topic_label", "year_month"]).reset_index(drop=True)

    def _compute(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("year_month").copy()
        g["roll_mean"] = g["topic_share"].rolling(window=window, min_periods=min_periods).mean()
        g["roll_std"] = g["topic_share"].rolling(window=window, min_periods=min_periods).std(ddof=1)

        g["z_score"] = (g["topic_share"] - g["roll_mean"]) / g["roll_std"]
        g.loc[g["roll_std"].isna() | (g["roll_std"] == 0) | (g["total_articles_in_month"] == 0), "z_score"] = 0.0
        return g

    topic_full = topic_full.groupby("topic_label", group_keys=False).apply(_compute)

    peaks = topic_full[
        (topic_full["total_articles_in_month"] >= int(min_articles_in_month)) &
        (topic_full["z_score"] >= float(z_threshold))
    ].copy()

    peaks["topic_share"] = peaks["topic_share"].round(4)
    peaks["roll_mean"] = peaks["roll_mean"].round(4)
    peaks["roll_std"] = peaks["roll_std"].round(6)
    peaks["z_score"] = peaks["z_score"].round(3)

    cols = [
        "year_month",
        "topic_label",
        "articles_count",
        "total_articles_in_month",
        "topic_share",
        "roll_mean",
        "roll_std",
        "z_score",
    ]
    peaks = peaks[cols].sort_values(["z_score", "year_month"], ascending=[False, True]).reset_index(drop=True)

    logger.info(f"[topic_peaks] Peaks found: {len(peaks)}")
    return peaks