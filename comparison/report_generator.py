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

    db = DatabaseConnection(logger=logger)
    engine = db.get_engine()

    # ---- NER ----
    try:
        logger.info("Running NER comparison...")
        ner_df = run_ner_comparison(engine)
        ner_path = run_dir / "ner_comparison.csv"
        ner_df.to_csv(ner_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {ner_path} | rows={len(ner_df)}")
    except Exception as e:
        logger.exception(f"NER comparison failed: {e}")

    # ---- Sentiment ----
    try:
        logger.info("Running sentiment comparison...")
        sent_df = run_sentiment_comparison(engine)
        sent_path = run_dir / "sentiment_comparison.csv"
        sent_df.to_csv(sent_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {sent_path} | rows={len(sent_df)}")
    except Exception as e:
        logger.exception(f"Sentiment comparison failed: {e}")

    # ---- Timing (reads from articles_enriched, cumulative across all runs) ----
    try:
        logger.info("Running timing comparison...")
        timing_df = run_timing_comparison(engine)
        timing_path = run_dir / "timing_comparison.csv"
        timing_df.to_csv(timing_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {timing_path} | rows={len(timing_df)}")
    except Exception as e:
        logger.exception(f"Timing comparison failed: {e}")

    db.close()
    logger.info("Comparison report generation finished.")
    return run_dir


if __name__ == "__main__":
    generate_comparison_report()