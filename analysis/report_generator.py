from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Union

import pandas as pd

from src.config import Config
from src.db_config import DatabaseConnection

from analysis.topics_by_month import topics_by_month, dominant_topic_by_month
from analysis.top_entities_by_country import top_entities_by_country
from analysis.entities_by_month import entities_by_month
from analysis.topic_peaks import topic_peaks

from analysis.ner_comparison import run_ner_comparison
from analysis.sentiment_comparison import run_sentiment_comparison
from analysis.timing_comparison import run_timing_comparison

logger = logging.getLogger(__name__)


def _setup_comparison_logger(run_dir: Path) -> logging.Logger:
    comp_logger = logging.getLogger("comparison")

    level = getattr(Config, "LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)

    comp_logger.setLevel(level)

    # Reset handlers to avoid duplicate logs when called multiple times
    comp_logger.handlers.clear()
    comp_logger.propagate = False

    fmt = logging.Formatter(
        getattr(Config, "LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
        datefmt=getattr(Config, "LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
    )

    fh = logging.FileHandler(run_dir / "comparison.log", encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    comp_logger.addHandler(fh)
    comp_logger.addHandler(ch)
    return comp_logger


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
        topics_path = run_dir / "topics_by_month.csv"
        topics_month_df.to_csv(topics_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {topics_path} | rows={len(topics_month_df)}")

        # 2) Dominant topic by month (GLOBAL) — what supervisor asked for
        dom_topics_df = dominant_topic_by_month(engine=engine, raw_table=raw_table)
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


def generate_comparison_report(
    run_dir: Optional[Union[str, Path]] = None,
    results_root: Union[str, Path] = "results",
) -> Path:
    """
    Generates model comparison reports (NER, Sentiment, and Timing) and concatenates them vertically.
    """
    if run_dir is None:
        results_root = Path(results_root)
        run_dir = results_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    comp_logger = _setup_comparison_logger(run_dir)
    comp_logger.info(f"Comparison output directory: {run_dir.resolve()}")

    def _concatenate_reports(ner_df: pd.DataFrame, sent_df: pd.DataFrame, time_df: pd.DataFrame, run_dir: Path):
        try:
            # Add a clear column to identify which comparison the row belongs to
            if not ner_df.empty:
                ner_df.insert(0, "Report_Type", "NER")
            if not sent_df.empty:
                sent_df.insert(0, "Report_Type", "SENTIMENT")
            if not time_df.empty:
                time_df.insert(0, "Report_Type", "TIMING")

            # Concatenate them vertically (axis=0) preserving all structure and rows
            dfs_to_concat = [df for df in [ner_df, sent_df, time_df] if not df.empty]
            if dfs_to_concat:
                merged = pd.concat(dfs_to_concat, axis=0, ignore_index=True)
                out_path = run_dir / "merged_comparison.csv"
                merged.to_csv(out_path, index=False, encoding="utf-8-sig")
                comp_logger.info(f"Successfully concatenated comparison reports vertically into {out_path.name}")
            else:
                comp_logger.warning("No reports to concatenate (all DataFrames are empty).")
        except Exception as e:
            comp_logger.exception(f"Failed to concatenate comparison reports vertically: {e}")

    db = DatabaseConnection(logger=comp_logger)
    engine = db.get_engine()

    ner_df = pd.DataFrame()
    sent_df = pd.DataFrame()
    time_df = pd.DataFrame()

    # ---- NER ----
    try:
        comp_logger.info("Running NER comparison...")
        ner_df = run_ner_comparison(engine)
    except Exception as e:
        comp_logger.exception(f"NER comparison failed: {e}")

    # ---- Sentiment ----
    try:
        comp_logger.info("Running sentiment comparison...")
        sent_df = run_sentiment_comparison(engine)
    except Exception as e:
        comp_logger.exception(f"Sentiment comparison failed: {e}")

    # ---- Timing (reads from articles_enriched, cumulative across all runs) ----
    try:
        comp_logger.info("Running timing comparison...")
        time_df = run_timing_comparison(engine)
    except Exception as e:
        comp_logger.exception(f"Timing comparison failed: {e}")

    db.close()

    # ---- Concatenate all three reports vertically ----
    comp_logger.info("Concatenating individual comparison reports vertically (in memory only)...")
    _concatenate_reports(ner_df, sent_df, time_df, run_dir)
    
    comp_logger.info("Comparison report generation finished.")
    return run_dir


if __name__ == "__main__":
    generate_comparison_report()