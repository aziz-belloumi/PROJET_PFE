from __future__ import annotations

from pathlib import Path
import logging
from typing import Optional, Union

import pandas as pd

from src.config import Config
from src.config.db_config import DatabaseConnection

from analysis.topics_by_month import topics_by_month, dominant_topic_by_month
from analysis.top_entities_by_country import top_entities_by_country
from analysis.entities_by_month import entities_by_month
from analysis.topic_peaks import topic_peaks

logger = logging.getLogger(__name__)


def build_analytics_summary(
    topics_month_df: Optional[pd.DataFrame] = None,
    dom_topics_df: Optional[pd.DataFrame] = None,
    peaks_df: Optional[pd.DataFrame] = None,
    entities_month_df: Optional[pd.DataFrame] = None,
    top_entities_df: Optional[pd.DataFrame] = None,
    top_n_per_category: int = 5,
) -> pd.DataFrame:
    """
    Builds a single consolidated summary DataFrame synthesizing insights from all analytics tables in the DB:
      1. Executive Overview / Volume & KPI Metrics
      2. Dominant Topics by Month
      3. Significant Topic Peaks & Surges (Z >= 2.5)
      4. Top Mentioned Entities by Month (Top-N per month)
      5. Top Entities by Country (Top-N per country)
    """
    records = []

    # -------------------------------------------------------------
    # 1. Executive Overview / High-Level KPIs
    # -------------------------------------------------------------
    total_months = 0
    date_range_str = "N/A"
    total_articles = 0
    unique_topics_count = 0

    if topics_month_df is not None and not topics_month_df.empty:
        if "year_month" in topics_month_df.columns:
            months = topics_month_df["year_month"].dropna().unique()
            total_months = len(months)
            if len(months) > 0:
                date_range_str = f"{min(months)} -> {max(months)}"

        if "total_articles_in_month" in topics_month_df.columns:
            total_articles = int(
                topics_month_df.groupby("year_month")["total_articles_in_month"].first().sum()
            )
        elif "articles_count" in topics_month_df.columns:
            total_articles = int(topics_month_df["articles_count"].sum())

        if "topic_label" in topics_month_df.columns:
            unique_topics_count = int(topics_month_df["topic_label"].nunique())

    peaks_count = len(peaks_df) if peaks_df is not None and not peaks_df.empty else 0
    unique_entities_count = 0
    if entities_month_df is not None and not entities_month_df.empty and "normalized_name" in entities_month_df.columns:
        unique_entities_count = int(entities_month_df["normalized_name"].nunique())

    records.extend([
        {
            "section": "OVERVIEW_METRICS",
            "period_or_scope": date_range_str,
            "category": "KPI",
            "item_name": "Active Date Range (Months)",
            "metric_count": total_months,
            "metric_share_or_score": None,
            "details": f"Covered time span: {date_range_str} ({total_months} monthly intervals)",
        },
        {
            "section": "OVERVIEW_METRICS",
            "period_or_scope": "All Available",
            "category": "KPI",
            "item_name": "Total Processed Articles in Analytics",
            "metric_count": total_articles,
            "metric_share_or_score": None,
            "details": f"Cumulative volume across analyzed monthly buckets: {total_articles:,} articles",
        },
        {
            "section": "OVERVIEW_METRICS",
            "period_or_scope": "All Available",
            "category": "KPI",
            "item_name": "Distinct Topic Categories",
            "metric_count": unique_topics_count,
            "metric_share_or_score": None,
            "details": f"Total distinct topic labels classified: {unique_topics_count}",
        },
        {
            "section": "OVERVIEW_METRICS",
            "period_or_scope": "All Available",
            "category": "KPI",
            "item_name": "Topic Surges Detected (Z >= 2.5)",
            "metric_count": peaks_count,
            "metric_share_or_score": None,
            "details": f"Detected anomaly spikes in topic share: {peaks_count} occurrences",
        },
        {
            "section": "OVERVIEW_METRICS",
            "period_or_scope": "All Available",
            "category": "KPI",
            "item_name": "Top Unique Named Entities",
            "metric_count": unique_entities_count,
            "metric_share_or_score": None,
            "details": f"Total distinct top entity entries tracked across months: {unique_entities_count}",
        },
    ])

    # -------------------------------------------------------------
    # 2. Dominant Topics by Month
    # -------------------------------------------------------------
    if dom_topics_df is not None and not dom_topics_df.empty:
        for _, row in dom_topics_df.iterrows():
            ym = str(row.get("year_month", ""))
            top_topic = str(row.get("dominant_topic", ""))
            count = int(row.get("dominant_articles_count", 0)) if pd.notna(row.get("dominant_articles_count")) else 0
            share = float(row.get("dominant_topic_share", 0.0)) if pd.notna(row.get("dominant_topic_share")) else 0.0
            tot = int(row.get("total_articles_in_month", 0)) if pd.notna(row.get("total_articles_in_month")) else 0
            records.append({
                "section": "DOMINANT_TOPICS_BY_MONTH",
                "period_or_scope": ym,
                "category": "TOPIC",
                "item_name": top_topic,
                "metric_count": count,
                "metric_share_or_score": round(share, 4),
                "details": f"Dominant topic in {ym} ({count}/{tot} articles, {share:.1%} share)",
            })

    # -------------------------------------------------------------
    # 3. Topic Peaks & Surge Events
    # -------------------------------------------------------------
    if peaks_df is not None and not peaks_df.empty:
        for _, row in peaks_df.iterrows():
            ym = str(row.get("year_month", ""))
            topic = str(row.get("topic_label", ""))
            count = int(row.get("articles_count", 0)) if pd.notna(row.get("articles_count")) else 0
            z_score = float(row.get("z_score", 0.0)) if pd.notna(row.get("z_score")) else 0.0
            share = float(row.get("topic_share", 0.0)) if pd.notna(row.get("topic_share")) else 0.0
            roll_m = float(row.get("roll_mean", 0.0)) if pd.notna(row.get("roll_mean")) else 0.0
            records.append({
                "section": "TOPIC_SURGES_PEAKS",
                "period_or_scope": ym,
                "category": "TOPIC_PEAK",
                "item_name": topic,
                "metric_count": count,
                "metric_share_or_score": round(z_score, 3),
                "details": f"Spike Z={z_score:.2f} (Share: {share:.1%} vs rolling avg: {roll_m:.1%}, Count: {count})",
            })

    # -------------------------------------------------------------
    # 4. Top Entities by Month (Top-N per month)
    # -------------------------------------------------------------
    if entities_month_df is not None and not entities_month_df.empty:
        df_sub = entities_month_df[entities_month_df["rank_in_month"] <= int(top_n_per_category)]
        for _, row in df_sub.iterrows():
            ym = str(row.get("year_month", ""))
            rank = int(row.get("rank_in_month", 0))
            etype = str(row.get("entity_type", ""))
            ename = str(row.get("normalized_name", ""))
            distinct_cnt = int(row.get("distinct_articles_count", 0)) if pd.notna(row.get("distinct_articles_count")) else 0
            mentions = int(row.get("mentions_count", 0)) if pd.notna(row.get("mentions_count")) else 0
            conf = float(row.get("mean_confidence", 0.0)) if pd.notna(row.get("mean_confidence")) else 0.0
            records.append({
                "section": "TOP_ENTITIES_BY_MONTH",
                "period_or_scope": ym,
                "category": f"ENTITY_{etype}",
                "item_name": ename,
                "metric_count": distinct_cnt,
                "metric_share_or_score": round(conf, 4),
                "details": f"Rank #{rank} in {ym} ({distinct_cnt} articles, {mentions} mentions, conf: {conf:.3f})",
            })

    # -------------------------------------------------------------
    # 5. Top Entities by Country (Top-N per country)
    # -------------------------------------------------------------
    if top_entities_df is not None and not top_entities_df.empty:
        df_sub = top_entities_df[top_entities_df["rank_in_country"] <= int(top_n_per_category)]
        for _, row in df_sub.iterrows():
            lbl = row.get("label_en")
            cid = row.get("country_id")
            if pd.notna(lbl) and str(lbl).strip().lower() not in {"nan", "none", ""}:
                country = str(lbl).strip()
            else:
                country = f"Country ID {cid}"
            rank = int(row.get("rank_in_country", 0))
            etype = str(row.get("entity_type", ""))
            ename = str(row.get("normalized_name", ""))
            distinct_cnt = int(row.get("distinct_articles_count", 0)) if pd.notna(row.get("distinct_articles_count")) else 0
            mentions = int(row.get("mentions_count", 0)) if pd.notna(row.get("mentions_count")) else 0
            conf = float(row.get("mean_confidence", 0.0)) if pd.notna(row.get("mean_confidence")) else 0.0
            records.append({
                "section": "TOP_ENTITIES_BY_COUNTRY",
                "period_or_scope": country,
                "category": f"ENTITY_{etype}",
                "item_name": ename,
                "metric_count": distinct_cnt,
                "metric_share_or_score": round(conf, 4),
                "details": f"Rank #{rank} in {country} ({distinct_cnt} articles, {mentions} mentions, conf: {conf:.3f})",
            })

    cols = [
        "section",
        "period_or_scope",
        "category",
        "item_name",
        "metric_count",
        "metric_share_or_score",
        "details",
    ]
    return pd.DataFrame(records, columns=cols)


def generate_analytics_reports(
    run_dir: Union[str, Path] = "analysis/reports",
    raw_table: str = "article",
    top_entities_k: int = 50,
    top_entities_by_month_k: int = 50,
    peaks_window: int = 6,
    peaks_z_threshold: float = 2.5,
    default_country_id: int = 8,
    export_summary_csv: bool = True,
) -> Path:
    """
    Computes all analytics and stores the results directly in dedicated MySQL database tables.
    Optionally exports a single consolidated summary CSV file (`analytics_summary.csv`)
    summarizing all DB analytics tables.

    Database tables written:
      - analytics_topics_by_month
      - analytics_topic_peaks
      - analytics_entities_by_month
      - analytics_top_entities_by_country

    CSV Output (optional, when export_summary_csv=True):
      - analytics_summary.csv
    """
    run_dir = Path(run_dir)
    if export_summary_csv:
        run_dir.mkdir(parents=True, exist_ok=True)

    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()

    topics_month_df = pd.DataFrame()
    dom_topics_df = pd.DataFrame()
    peaks_df = pd.DataFrame()
    entities_month_df = pd.DataFrame()
    top_entities_df = pd.DataFrame()

    try:
        # 1) Topics distribution by month (GLOBAL)
        topics_month_df = topics_by_month(engine=engine, raw_table=raw_table)
        if topics_month_df.empty:
            logger.info("topics_by_month returned no rows (no dated topic data yet) — skipping topic analytics.")
        else:
            db.save_analytics_dataframe(topics_month_df, "analytics_topics_by_month")
            logger.info(f"Updated DB table: analytics_topics_by_month | rows={len(topics_month_df)}")

            # 2) Dominant topic by month (GLOBAL) — computed in-memory for the summary report
            dom_topics_df = dominant_topic_by_month(engine=engine, raw_table=raw_table)

            # 3) Topic peaks (GLOBAL) — computed from topics_by_month dataframe
            peaks_df = topic_peaks(
                topics_by_month_df=topics_month_df,
                window=peaks_window,
                z_threshold=peaks_z_threshold,
                min_periods=3,
                min_articles_in_month=5,
            )
            if not peaks_df.empty:
                db.save_analytics_dataframe(peaks_df, "analytics_topic_peaks")
                logger.info(f"Updated DB table: analytics_topic_peaks | rows={len(peaks_df)}")

        # 4) Entities by month (GLOBAL: no country, no language)
        entities_month_df = entities_by_month(
            engine=engine,
            raw_table=raw_table,
            top_k_per_month=top_entities_by_month_k,
        )
        if not entities_month_df.empty:
            db.save_analytics_dataframe(entities_month_df, "analytics_entities_by_month")
            logger.info(f"Updated DB table: analytics_entities_by_month | rows={len(entities_month_df)}")

        # 5) Top entities by country (no language split)
        top_entities_df = top_entities_by_country(
            engine=engine,
            raw_table=raw_table,
            top_k=top_entities_k,
            default_country_id=default_country_id,
        )
        if not top_entities_df.empty:
            db.save_analytics_dataframe(top_entities_df, "analytics_top_entities_by_country")
            logger.info(f"Updated DB table: analytics_top_entities_by_country | rows={len(top_entities_df)}")

        # 6) Optional Consolidated Summary CSV
        if export_summary_csv:
            summary_df = build_analytics_summary(
                topics_month_df=topics_month_df,
                dom_topics_df=dom_topics_df,
                peaks_df=peaks_df,
                entities_month_df=entities_month_df,
                top_entities_df=top_entities_df,
                top_n_per_category=5,
            )
            if not summary_df.empty:
                summary_path = run_dir / "analytics_summary.csv"
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
                logger.info(f"Exported summary CSV: {summary_path} | rows={len(summary_df)}")

    finally:
        db.close()

    return run_dir


if __name__ == "__main__":
    generate_analytics_reports(
        run_dir="analysis/reports",
        export_summary_csv=True,
    )