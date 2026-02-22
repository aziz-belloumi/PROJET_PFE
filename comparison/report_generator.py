# comparison/report_generator.py
"""
Generate model comparison outputs as CSV files.

Outputs (in one folder):
- comparison.log
- ner_comparison.csv
- sentiment_comparison.csv
- timing_comparison.csv          (only if timing_records provided)

IMPORTANT:
- If run_dir is provided, results are written into that existing folder.
- If run_dir is None, a new folder is created under ./results/<timestamp>/ (standalone usage).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Union

from src.config import Config
from src.db_config import DatabaseConnection

from comparison.ner_comparison import run_ner_comparison
from comparison.sentiment_comparison import run_sentiment_comparison
from comparison.timing_comparison import run_timing_comparison


def _setup_comparison_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("comparison")
    logger.setLevel(Config.LOG_LEVEL)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)

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
    timing_records: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Generate comparison CSVs by reading pipeline outputs from MySQL tables.

    Args:
        run_dir:
            - If provided: write all comparison outputs inside this folder.
            - If None: create a new timestamped folder under results_root.
        results_root: base folder used only when run_dir is None.
        timing_records: list of timing dicts collected during pipeline execution.

    Returns:
        Path to the folder that contains the comparison outputs.
    """
    if run_dir is None:
        results_root = Path(results_root)
        run_dir = results_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_comparison_logger(run_dir)
    logger.info(f"Comparison output directory: {run_dir.resolve()}")

    # DB connection (read-only usage here)
    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()

    # ---- NER ----
    try:
        logger.info("Running NER comparison...")
        ner_df = run_ner_comparison(engine)
        ner_path = run_dir / "ner_comparison.csv"
        ner_df.to_csv(ner_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {ner_path}")
    except Exception as e:
        logger.exception(f"NER comparison failed: {e}")

    # ---- Sentiment ----
    try:
        logger.info("Running sentiment comparison...")
        sent_df = run_sentiment_comparison(engine)
        sent_path = run_dir / "sentiment_comparison.csv"
        sent_df.to_csv(sent_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {sent_path}")
    except Exception as e:
        logger.exception(f"Sentiment comparison failed: {e}")

    # ---- Timing (local only) ----
    if timing_records is not None:
        try:
            logger.info("Running timing comparison...")
            timing_df = run_timing_comparison(timing_records)

            # Keep only the relevant tasks (ner, sentiment, topic)
            if not timing_df.empty and "task" in timing_df.columns:
                timing_df = timing_df[timing_df["task"].isin(["ner", "sentiment", "topic"])].copy()

            timing_path = run_dir / "timing_comparison.csv"
            timing_df.to_csv(timing_path, index=False, encoding="utf-8-sig")
            logger.info(f"Saved: {timing_path}")
        except Exception as e:
            logger.exception(f"Timing comparison failed: {e}")
    else:
        logger.warning("No timing_records provided -> timing_comparison.csv will not be generated.")

    db.close()
    logger.info("Comparison report generation finished.")
    return run_dir


if __name__ == "__main__":
    # Standalone usage
    generate_comparison_report()