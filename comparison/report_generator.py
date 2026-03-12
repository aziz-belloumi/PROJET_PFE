from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Union

from src.config import Config
from src.db_config import DatabaseConnection

from comparison.ner_comparison import run_ner_comparison
from comparison.sentiment_comparison import run_sentiment_comparison
from comparison.timing_comparison import run_timing_comparison
import pandas as pd


def _setup_comparison_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("comparison")

    # LOG_LEVEL might be "INFO" etc. logging.setLevel accepts string in recent Python,
    # but to be safe we convert if needed.
    level = getattr(Config, "LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)

    logger.setLevel(level)

    # Reset handlers to avoid duplicate logs when called multiple times
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        getattr(Config, "LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
        datefmt=getattr(Config, "LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
    )

    fh = logging.FileHandler(run_dir / "comparison.log", encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def generate_comparison_report(
    run_dir: Optional[Union[str, Path]] = None,
    results_root: Union[str, Path] = "results",
) -> Path:

    if run_dir is None:
        results_root = Path(results_root)
        run_dir = results_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_comparison_logger(run_dir)
    logger.info(f"Comparison output directory: {run_dir.resolve()}")

    def _concatenate_reports(ner_df: pd.DataFrame, sent_df: pd.DataFrame, time_df: pd.DataFrame, run_dir: Path):
        try:

            # Add a clear column to identify which comparison the row belongs to
            ner_df.insert(0, "Report_Type", "NER")
            sent_df.insert(0, "Report_Type", "SENTIMENT")
            time_df.insert(0, "Report_Type", "TIMING")

            # Concatenate them vertically (axis=0) preserving all structure and rows
            merged = pd.concat([ner_df, sent_df, time_df], axis=0, ignore_index=True)

            out_path = run_dir / "merged_comparison.csv"
            merged.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"Successfully concatenated comparison reports vertically into {out_path.name}")
        except Exception as e:
            logger.exception(f"Failed to concatenate comparison reports vertically: {e}")


    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()

    ner_df = pd.DataFrame()
    sent_df = pd.DataFrame()
    time_df = pd.DataFrame()

    # ---- NER ----
    try:
        logger.info("Running NER comparison...")
        ner_df = run_ner_comparison(engine)
    except Exception as e:
        logger.exception(f"NER comparison failed: {e}")

    # ---- Sentiment ----
    try:
        logger.info("Running sentiment comparison...")
        sent_df = run_sentiment_comparison(engine)
    except Exception as e:
        logger.exception(f"Sentiment comparison failed: {e}")

    # ---- Timing (reads from articles_enriched, cumulative across all runs) ----
    try:
        logger.info("Running timing comparison...")
        time_df = run_timing_comparison(engine)
    except Exception as e:
        logger.exception(f"Timing comparison failed: {e}")

    db.close()

    # ---- Concatenate all three reports vertically ----
    logger.info("Concatenating individual comparison reports vertically (in memory only)...")
    _concatenate_reports(ner_df, sent_df, time_df, run_dir)
    
    logger.info("Comparison report generation finished.")
    return run_dir


if __name__ == "__main__":
    generate_comparison_report()