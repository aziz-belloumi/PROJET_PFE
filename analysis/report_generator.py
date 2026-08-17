from __future__ import annotations

from pathlib import Path
import logging
from typing import Union

import pandas as pd

from src.config import Config
from src.db_config import DatabaseConnection

from analysis.topics_by_month import topics_by_month, dominant_topic_by_month
from analysis.top_entities_by_country import top_entities_by_country
from analysis.entities_by_month import entities_by_month
from analysis.topic_peaks import topic_peaks

logger = logging.getLogger(__name__)


def generate_analytics_reports(
    run_dir: Union[str, Path],
    raw_table: str = "article",
    top_entities_k: int = 50,
    top_entities_by_month_k: int = 50,
    peaks_window: int = 6,
    peaks_z_threshold: float = 2.5,
    default_country_id: int = 8,
) -> Path:
    """
    Generates analytics CSVs (post-pipeline) by reading from MySQL tables and writing into run_dir.

    No fixed dates:
    - The scripts use crawl_date (fallback year/month) and work on the available date range in the DB.

    Outputs (in run_dir):
      - topics_by_month.csv                 (distribution: month x topic)
      - dominant_topic_by_month.csv         (one dominant topic per month)
      - topic_peaks.csv                     (anomalies on topic_share)
      - entities_by_month.csv               (top entities per month, global)
      - top_entities_by_country.csv         (top entities per country, no language split)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()

    try:
        # 1) Topics distribution by month (GLOBAL)
        topics_month_df = topics_by_month(engine=engine, raw_table=raw_table)
        if topics_month_df.empty:
            logger.info("topics_by_month returned no rows (no dated topic data yet) — skipping topic CSVs.")
        else:
            topics_path = run_dir / "topics_by_month.csv"
            topics_month_df.to_csv(topics_path, index=False, encoding="utf-8-sig")
            logger.info(f"Saved: {topics_path} | rows={len(topics_month_df)}")

            # 2) Dominant topic by month (GLOBAL)
            dom_topics_df = dominant_topic_by_month(engine=engine, raw_table=raw_table)
            if not dom_topics_df.empty:
                dom_topics_path = run_dir / "dominant_topic_by_month.csv"
                dom_topics_df.to_csv(dom_topics_path, index=False, encoding="utf-8-sig")
                logger.info(f"Saved: {dom_topics_path} | rows={len(dom_topics_df)}")

            # 3) Topic peaks (GLOBAL) — computed from topics_by_month dataframe
            peaks_df = topic_peaks(
                topics_by_month_df=topics_month_df,
                window=peaks_window,
                z_threshold=peaks_z_threshold,
                min_periods=3,
                min_articles_in_month=5,
            )
            if not peaks_df.empty:
                peaks_path = run_dir / "topic_peaks.csv"
                peaks_df.to_csv(peaks_path, index=False, encoding="utf-8-sig")
                logger.info(f"Saved: {peaks_path} | rows={len(peaks_df)}")

        # 4) Entities by month (GLOBAL: no country, no language)
        entities_month_df = entities_by_month(
            engine=engine,
            raw_table=raw_table,
            top_k_per_month=top_entities_by_month_k,
        )
        entities_month_path = run_dir / "entities_by_month.csv"
        entities_month_df.to_csv(entities_month_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {entities_month_path} | rows={len(entities_month_df)}")

        # 5) Top entities by country (no language split)
        top_entities_df = top_entities_by_country(
            engine=engine,
            raw_table=raw_table,
            top_k=top_entities_k,
            default_country_id=default_country_id,
        )
        top_entities_path = run_dir / "top_entities_by_country.csv"
        top_entities_df.to_csv(top_entities_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {top_entities_path} | rows={len(top_entities_df)}")

    finally:
        db.close()

    return run_dir


if __name__ == "__main__":
    generate_analytics_reports("analysis/reports")