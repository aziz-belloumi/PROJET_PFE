from datetime import datetime
import logging
import sys
import platform
import os
import importlib.metadata as md
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import psutil
import torch

from src.text_utils import init_console_encoding
init_console_encoding()

from src.config import Config
from src.db_config import DatabaseConnection
from src.ner_extraction import GLiNERNER, TransformersNER
from src.sentiment_extraction import LLMSentiment
from src.topic_extraction import LLMTopic

from pipeline.sampler  import run_sampling
from pipeline.gpu_pass import run_gpu_pass

from analysis.report_generator import generate_analytics_reports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pipeline_t0 = time.perf_counter()

    sample_size = Config.SAMPLE_SIZE
    raw_table = Config.RAW_TABLE

    # Active model selections
    ner_models_by_lang = dict(Config.NER_MODELS_BY_LANG) if Config.RUN_NER else {}
    sent_models_by_lang = dict(Config.SENT_MODELS_BY_LANG) if Config.RUN_SENTIMENT else {}
    topic_extractors = list(Config.TOPIC_EXTRACTORS) if Config.RUN_TOPIC else []

    # ---- Build run_config (for logging / reproducibility) ----
    run_config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python":       sys.version.split()[0],
            "platform":     platform.platform(),
            "packages": {
                "pandas":        _pkg_version("pandas"),
                "sqlalchemy":    _pkg_version("SQLAlchemy"),
                "transformers":  _pkg_version("transformers"),
                "torch":         _pkg_version("torch"),
                "fast-langdetect": _pkg_version("fast-langdetect"),
            },
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "benchmark": {
            "cpu_device":      Config.CPU_DEVICE,
            "gpu_device":      Config.GPU_DEVICE,
            "gpu_cooldown_sec": Config.GPU_COOLDOWN_SEC,
            "timing_scope":    "predict() only (excludes MySQL writes)",
            "cpu_strategy":    "article-by-article with all models loaded (per-language)",
            "gpu_strategy":    "model-by-model to avoid OOM",
        },
        "sampling": {
            "sample_size": sample_size,
            "lang_threshold": Config.LANG_THRESHOLD,
            "filters": ["body IS NOT NULL", "no language restriction (fastText decides)"],
        },
        "preprocessing_presets": {
            "lang_detect": {"ar": Config.PREPROCESS_LANG_DETECT_PARAMS, "latin": Config.LATIN_LANG_DETECT_PARAMS},
            "ner":         {"ar": Config.PREPROCESS_NER_PARAMS,         "latin": Config.LATIN_NER_PARAMS},
            "sentiment":   {"ar": Config.PREPROCESS_SENTIMENT_PARAMS,   "latin": Config.LATIN_SENTIMENT_PARAMS},
            "topic":       {"ar": Config.PREPROCESS_SENTIMENT_PARAMS,   "latin": Config.LATIN_TOPIC_PARAMS},
        },
        "models": {
            "ner": [
                {
                    "language": lang,
                    "model_version": mv,
                    "model_name": name,
                    "type": "gliner" if "gliner" in name.lower() else "transformers",
                }
                for lang, models in ner_models_by_lang.items()
                for mv, name in models
            ],
            "sentiment": [
                {
                    "language": lang,
                    "model_version": mv,
                    "model_name": name,
                }
                for lang, models in sent_models_by_lang.items()
                for mv, name in models
            ],
            "topic": [
                {
                    "language": lang,
                    "model_version": mv,
                    "model_type": extractor_type,
                }
                for mv, extractor_type, lang in topic_extractors
            ],
        },
        "db": {
            "host": Config.DB_HOST,
            "port": Config.DB_PORT,
            "db":   Config.DB_NAME,
            "raw_table": raw_table,
        },
    }

    # ---- Setup logger ----
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("nlp_pipeline")
    logger.setLevel(logging.INFO)

    # ---- System info banner ----
    vram_mb = (
        f"{torch.cuda.get_device_properties(Config.GPU_DEVICE).total_memory / (1024**2):.0f}MB"
        if torch.cuda.is_available()
        else "N/A"
    )

    # ---- Database init ----
    db = DatabaseConnection(logger=logger)
    db.init_result_tables()

    # ================================================================
    # STAGE 1 — Sampling & Language Detection (Single Pass)
    # ================================================================
    work_df, skipped_df = run_sampling(
        db=db,
        logger=logger,
        sample_size=sample_size,
        lang_threshold=Config.LANG_THRESHOLD,
        ner_models_by_lang=ner_models_by_lang,
        raw_table=raw_table,
    )

    if work_df.empty:
        logger.warning("No rows passed language filter; stopping.")
        db.close()
        return

    # ================================================================
    # STAGE 2 — GPU Model Inference (NER, Sentiment, Topic)
    # ================================================================
    gpu_time_ner_ms, gpu_time_sentiment_ms, gpu_time_topic_ms = {}, {}, {}

    if not (Config.RUN_NER or Config.RUN_SENTIMENT or Config.RUN_TOPIC):
        logger.info("BERT models disabled (RUN_NER=False, RUN_SENTIMENT=False, RUN_TOPIC=False) → GPU pass skipped.")
        sent_results_buffer  = {}
        topic_results_buffer = {}
    elif not torch.cuda.is_available():
        logger.warning("CUDA not available → GPU pass skipped.")
        sent_results_buffer  = {}
        topic_results_buffer = {}
    else:
        (
            sent_results_buffer,
            topic_results_buffer,
            _,
            _,
            _,
            ner_results_buffer,
        ) = run_gpu_pass(
            work_df=work_df,
            db=db,
            ner_models_by_lang=ner_models_by_lang,
            sent_models_by_lang=sent_models_by_lang,
            topic_extractors=topic_extractors,
            ner_params=dict(Config.DEFAULT_NER_PARAMS),
            sentiment_params=dict(Config.DEFAULT_SENTIMENT_PARAMS),
            cpu_time_ner_ms=gpu_time_ner_ms,
            cpu_time_sentiment_ms=gpu_time_sentiment_ms,
            cpu_time_topic_ms=gpu_time_topic_ms,
            logger=logger,
        )

    # Update global entity mention counts in entities dictionary
    try:
        db.update_entity_frequencies()
        logger.info("Global entity frequencies updated.")
    except Exception as e:
        logger.error(f"Failed to update entity frequencies: {e}")

    # ================================================================
    # STAGE 3 — Comparative Qwen 2.5 LLM Pass (Optional)
    # ================================================================
    if Config.RUN_QWEN:
        try:
            from src.qwen.run_qwen_pass import run_qwen_pass
            run_qwen_pass(work_df, db, logger, skipped_df=skipped_df)
        except Exception as e:
            logger.error(f"Failed to run Qwen pass: {e}")

    # ================================================================
    # STAGE 4 — Analytics DB Tables & Optional Summary CSV
    # ================================================================
    reports_dir = Path("analysis/reports")
    try:
        generate_analytics_reports(
            run_dir=reports_dir,
            raw_table=raw_table,
            top_entities_k=50,
            top_entities_by_month_k=50,
            peaks_window=6,
            peaks_z_threshold=2.5,
            default_country_id=8,
            export_summary_csv=Config.EXPORT_ANALYTICS_CSV,
        )
        logger.info("Analytics DB tables and summary report updated.")
    except Exception as e:
        logger.error(f"Analytics report failed: {e}")

    logger.info(f"Pipeline complete in {time.perf_counter() - pipeline_t0:.2f}s")
    db.close()


if __name__ == "__main__":
    main()