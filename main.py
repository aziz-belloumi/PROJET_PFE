from datetime import datetime
from pathlib import Path
import logging
import json
import sys
import platform
import importlib.metadata as md
import time

import torch

from src.config import Config
from src.logger import setup_run
from src.db_config import DatabaseConnection
from src.ner_extraction import DEFAULT_NER_PARAMS
from src.sentiment_analysis import (
    DEFAULT_SENTIMENT_PARAMS,
)
from src.preprocessing.router import (
    PREPROCESS_LANG_DETECT_PARAMS,
    PREPROCESS_NER_PARAMS,
    PREPROCESS_SENTIMENT_PARAMS,
    LATIN_LANG_DETECT_PARAMS,
    LATIN_NER_PARAMS,
    LATIN_SENTIMENT_PARAMS,
    LATIN_TOPIC_PARAMS,
)

from pipeline.sampler  import run_sampling
from pipeline.cpu_pass import run_cpu_pass
from pipeline.gpu_pass import run_gpu_pass
from pipeline.exporter import export_lang_samples_csv

from analysis.report_generator   import generate_analytics_reports, generate_comparison_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

NER_MODELS_BY_LANG = {
    # "ar": [
    #     (0, Config.ARABERT_NER_MODEL),
    #     (1, Config.CAMEL_NER_MODEL),
    # ],
    # "en": [(2, Config.EN_NER_MODEL)],
    # "fr": [(3, Config.FR_NER_MODEL)],
    "ar": [(4, "urchade/gliner_multi-v2.1")],
    "en": [(4, "urchade/gliner_multi-v2.1")],
    "fr": [(4, "urchade/gliner_multi-v2.1")],
}

SENT_MODELS_BY_LANG = {
    "ar": [(0, Config.LLM_MODEL)],
    "en": [(0, Config.LLM_MODEL)],
    "fr": [(0, Config.LLM_MODEL)],
}

# Topic extractors: (model_version, extractor_type)
# extractor_type must be 'llm'
TOPIC_EXTRACTORS = [
    (0, "llm"),
]

NER_PARAMS       = dict(DEFAULT_NER_PARAMS)
SENTIMENT_PARAMS = dict(DEFAULT_SENTIMENT_PARAMS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pipeline_t0 = time.perf_counter()

    sample_size = 1000
    raw_table   = getattr(Config, "RAW_TABLE", "article")

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
                "fasttext-wheel": _pkg_version("fasttext-wheel"),
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
            "lang_detect": {"ar": PREPROCESS_LANG_DETECT_PARAMS, "latin": LATIN_LANG_DETECT_PARAMS},
            "ner":         {"ar": PREPROCESS_NER_PARAMS,         "latin": LATIN_NER_PARAMS},
            "sentiment":   {"ar": PREPROCESS_SENTIMENT_PARAMS,   "latin": LATIN_SENTIMENT_PARAMS},
            "topic":       {"ar": PREPROCESS_SENTIMENT_PARAMS,   "latin": LATIN_TOPIC_PARAMS},
        },
        "topic_extraction": {
            "extractors": [
                {
                    "model_version": mv,
                    "type": etype,
                    "model": Config.LLM_MODEL,
                }
                for mv, etype in TOPIC_EXTRACTORS
            ],
        },
        "models": {
            "ner":       NER_MODELS_BY_LANG,
            "sentiment": {k: [(mv, name) for mv, name in v] for k, v in SENT_MODELS_BY_LANG.items()},
            "topic":     TOPIC_EXTRACTORS,
        },
    }

    ctx    = setup_run(run_config=run_config, save_run_config=True)
    logger = ctx.logger
    run_dir = ctx.run_dir
    run_config["run_id"] = run_dir.name
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("=== RUN CONFIG ===\n" + json.dumps(run_config, indent=2, ensure_ascii=False))

    # ---- DB ----
    db = DatabaseConnection(logger=logger)
    db.init_result_tables()

    # ================================================================
    # STAGE 1 — Sample + Preprocess
    # ================================================================
    work_df, _ = run_sampling(
        db=db,
        sample_size=sample_size,
        ner_models_by_lang=NER_MODELS_BY_LANG,
        logger=logger,
        raw_table=raw_table,
    )

    if work_df.empty:
        logger.warning("No rows passed language filter; stopping.")
        db.close()
        return

    # ================================================================
    # STAGE 2 — CPU Timing Pass (Temporarily Commented)
    # ================================================================
    cpu_time_ner_ms, cpu_time_sentiment_ms, cpu_time_topic_ms = {}, {}, {}
    # cpu_time_ner_ms, cpu_time_sentiment_ms, cpu_time_topic_ms = run_cpu_pass(
    #     work_df=work_df,
    #     ner_models_by_lang=NER_MODELS_BY_LANG,
    #     sent_models_by_lang=SENT_MODELS_BY_LANG,
    #     ner_params=NER_PARAMS,
    #     sentiment_params=SENTIMENT_PARAMS,
    #     logger=logger,
    # )

    # ================================================================
    # STAGE 3 — GPU Inference + DB Writes
    # ================================================================
    if not torch.cuda.is_available():
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
            ner_models_by_lang=NER_MODELS_BY_LANG,
            sent_models_by_lang=SENT_MODELS_BY_LANG,
            topic_extractors=TOPIC_EXTRACTORS,
            ner_params=NER_PARAMS,
            sentiment_params=SENTIMENT_PARAMS,
            cpu_time_ner_ms=cpu_time_ner_ms,
            cpu_time_sentiment_ms=cpu_time_sentiment_ms,
            cpu_time_topic_ms=cpu_time_topic_ms,
            logger=logger,
        )

        # ================================================================
        # STAGE 4 — Export EN/FR review CSVs
        # ================================================================
        export_lang_samples_csv(
            run_dir=run_dir,
            work_df=work_df,
            sent_results_buffer=sent_results_buffer,
            topic_results_buffer=topic_results_buffer,
            logger=logger,
        )

        # Export GLiNER results locally
        if ner_results_buffer:
            import pandas as pd
            gliner_df = pd.DataFrame([
                {"article_id": aid, "preprocessed_text": data["text_ner"], "entities": ", ".join(data["entities"])}
                for aid, data in ner_results_buffer.items()
            ])
            gliner_csv_path = run_dir / "gliner_ner_results.csv"
            gliner_df.to_csv(gliner_csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"GLiNER results saved to {gliner_csv_path}")

        # Export Sentiment results locally
        if sent_results_buffer:
            import pandas as pd
            # Map sentiment results back to their preprocessed text
            sentiment_data = []
            for r in work_df.itertuples(index=False):
                aid = int(r.id)
                res = sent_results_buffer.get((aid, 0))
                if res:
                    sentiment_data.append({
                        "article_id": aid,
                        "preprocessed_text": r.text_sentiment,
                        "sentiment_label": res["label"],
                        "sentiment_score": res["score"]
                    })
            
            if sentiment_data:
                sentiment_df = pd.DataFrame(sentiment_data)
                sentiment_csv_path = run_dir / "sentiment_results.csv"
                sentiment_df.to_csv(sentiment_csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"Sentiment results saved to {sentiment_csv_path}")

        # Export Topic & Sentiment results locally (Combined)
        if topic_results_buffer:
            import pandas as pd
            ts_data = []
            for r in work_df.itertuples(index=False):
                aid = int(r.id)
                t_res = topic_results_buffer.get((aid, 0))
                s_res = sent_results_buffer.get((aid, 0))
                if t_res:
                    ts_data.append({
                        "article_id": aid,
                        "preprocessed_text_topic": r.text_topic,
                        "topic_label": t_res["label"],
                        "sentiment_label": s_res.get("label") if s_res else None,
                    })
            
            if ts_data:
                ts_df = pd.DataFrame(ts_data)
                ts_csv_path = run_dir / "topic_sentiment_results.csv"
                ts_df.to_csv(ts_csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"Topic & Sentiment results saved to {ts_csv_path}")

    # ================================================================
    # STAGE 5 — Analytics & Comparison Reports
    # ================================================================
    try:
        generate_analytics_reports(
            run_dir=run_dir,
            raw_table=raw_table,
            top_entities_k=50,
            top_entities_by_month_k=50,
            peaks_window=6,
            peaks_z_threshold=2.5,
            default_country_id=8,
        )
        logger.info("Analytics CSVs generated.")
    except Exception as e:
        logger.error(f"Analytics report failed: {e}")

    try:
        generate_comparison_report(
            run_dir=run_dir,
        )
        logger.info("Comparison reports generated.")
    except Exception as e:
        logger.error(f"Comparison report failed: {e}")

    # ================================================================
    # STAGE 6 — Global Entity Frequencies
    # ================================================================
    try:
        db.update_entity_frequencies()
        logger.info("Global entity frequencies updated.")
    except Exception as e:
        logger.error(f"Failed to update entity frequencies: {e}")

    logger.info(f"Pipeline complete in {time.perf_counter() - pipeline_t0:.2f}s")
    db.close()


if __name__ == "__main__":
    main()