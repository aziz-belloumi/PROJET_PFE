from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from src.config import Config


@dataclass
class RunContext:
    logger: logging.Logger
    run_id: str

    def __iter__(self):
        yield self.logger
        yield self.run_id


def setup_run(
    logger_name: str = "nlp_pipeline",
    run_id: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> RunContext:
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ---- logger ----
    logger = logging.getLogger(logger_name)
    logger.setLevel(Config.LOG_LEVEL)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return RunContext(logger=logger, run_id=run_id)