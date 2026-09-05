import logging
import os
import sys
import time
import warnings
from pathlib import Path

import torch

# Configure console encoding for UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.config import Config
from src.config.db_config import DatabaseConnection

from pipeline.cpu_pass import run_cpu_pass
from pipeline.gpu_pass import run_gpu_pass

from analysis.report_generator import generate_analytics_reports


def main():
    pipeline_t0 = time.perf_counter()

    # ---- Setup logger ----
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("nlp_pipeline")
    logger.setLevel(logging.INFO)

    sample_size = Config.SAMPLE_SIZE
    raw_table = Config.RAW_TABLE

    # Active model selections
    ner_models_by_lang = dict(Config.NER_MODELS_BY_LANG) if Config.RUN_NER else {}
    sent_models_by_lang = dict(Config.SENT_MODELS_BY_LANG) if Config.RUN_SENTIMENT else {}
    topic_extractors = list(Config.TOPIC_EXTRACTORS) if Config.RUN_TOPIC else []

    # ---- Database init ----
    db = DatabaseConnection(logger=logger)
    db.init_result_tables()

    # ================================================================
    # STAGE 1 — CPU Pass: Sampling, Language Detection & Preprocessing
    # ================================================================
    work_df, skipped_df = run_cpu_pass(
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