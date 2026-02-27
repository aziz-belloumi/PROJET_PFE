from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging

from src.config import Config


@dataclass
class RunContext:
    logger: logging.Logger
    run_dir: Path
    run_id: str


def setup_run(
    logger_name: str = "nlp_pipeline",
    results_root: str | Path = "results",
    run_id: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
    save_run_config: bool = True,
) -> RunContext:
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- logger ----
    logger = logging.getLogger(logger_name)
    logger.setLevel(Config.LOG_LEVEL)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # ---- optional: save run_config ----
    if save_run_config and run_config is not None:
        cfg_path = run_dir / "run_config.json"
        cfg_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved run config JSON: {cfg_path}")

    return RunContext(logger=logger, run_dir=run_dir, run_id=run_id)