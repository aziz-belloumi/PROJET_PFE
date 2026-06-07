import os
import sys
import threading
import logging
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Resolve project root and append to sys.path so we can import original src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topic_generation import LLMTopic
from src.sentiment_analysis import LLMSentiment
from src.preprocessing import PreprocessRouter

# ==========================================
# CONFIGURATION
# ========================================== 
PC_ID = 5          # Partition identifier: 1, 2, 3, or 4
NUM_WORKERS = 4   # Number of concurrent workers calling Ollama in parallel
SAVE_INTERVAL = 100  # Autosave every N articles (INCREASED to reduce save frequency)

# ==========================================
# FILE SYSTEM PATHS
# ==========================================
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"
INPUT_FILE = CLEANED_DATA_DIR / f"global_data_pc{PC_ID}.csv"
ENRICHED_OUTPUT_FILE = CLEANED_DATA_DIR / f"articles_enriched_pc{PC_ID}.csv"
TOPICS_OUTPUT_FILE = CLEANED_DATA_DIR / f"article_topics_pc{PC_ID}.csv"

# ==========================================
# DIALECT LANGUAGE MAPPING (ISO-639-1 standardizer)
# ==========================================
DIALECT_TO_ISO = {
    "msa": "ar",
    "egy": "ar",
    "lev": "ar",
    "glf": "ar",
    "mgr": "ar"
}

def map_language_code(lang: str) -> str:
    if not isinstance(lang, str):
        return "en"
    clean_lang = lang.strip().lower()
    return DIALECT_TO_ISO.get(clean_lang, clean_lang)

def custom_preprocess(preproc, text: str, lang: str, task: str) -> str:
    """
    Custom preprocessing routing logic to satisfy user specifications:
    - Sentiment (all languages): remove @user and http entirely (standardize_social=False), leave repetition intact (reduce_repetitions/remove_repeated=False).
    - Topic (all languages): remove @user and http entirely (standardize_social=False), remove repeated letters (reduce_repetitions/remove_repeated=True).
    """
    lang = (lang or "").lower().strip()
    task = (task or "").lower().strip()

    if lang == "ar":
        from src.preprocessing.router import PREPROCESS_SENTIMENT_PARAMS
        params = PREPROCESS_SENTIMENT_PARAMS.copy()
        if task == "topic":
            params["remove_repeated"] = True
        elif task == "sentiment":
            params["remove_repeated"] = False
        return preproc.ar.preprocess(text, **params)

    # Latin languages (en, fr, etc.)
    if task == "sentiment":
        from src.preprocessing.router import LATIN_SENTIMENT_PARAMS
        params = LATIN_SENTIMENT_PARAMS.copy()
        params["standardize_social"] = False
        params["reduce_repetitions"] = False
        return preproc.lat.preprocess(text, **params)

    if task == "topic":
        from src.preprocessing.router import LATIN_TOPIC_PARAMS
        params = LATIN_TOPIC_PARAMS.copy()
        params["standardize_social"] = False
        params["reduce_repetitions"] = True
        return preproc.lat.preprocess(text, **params)

    return preproc.preprocess(text, lang, task)


def run_phase(phase_name, records, needs_ids, data_store, process_fn, save_fn,
              save_lock, save_interval, logger, num_workers):
    """Run a full annotation phase (topic or sentiment) over all pending records with separate locks for data and save."""
    if not records:
        logger.info(f"[{phase_name}] No articles to process — all already annotated.")
        return

    processed_counter = 0
    shutdown_requested = False
    progress_bar = tqdm(total=len(records), desc=f"{phase_name} Pass", unit="art")
    
    # Separate locks: one for data updates, one for save operations
    data_lock = threading.Lock()
    
    # Timing tracking
    timings = {
        "total": []
    }
    
    phase_start_time = time.time()

    def worker(row):
        nonlocal processed_counter, shutdown_requested
        if shutdown_requested:
            return

        aid = int(row["id"])
        text = str(row["text"])
        orig_lang = str(row["language"])
        mapped_lang = map_language_code(orig_lang)

        if not text.strip():
            text = "empty"

        if aid in needs_ids:
            t_total_start = time.time()
            result = process_fn(aid, text, mapped_lang, orig_lang)
            t_total_end = time.time()
            timings["total"].append(t_total_end - t_total_start)
        else:
            result = None  # already in checkpoint, nothing to do

        if result is not None:
            # Quick lock just for updating the dict
            with data_lock:
                data_store[aid] = result
                processed_counter += 1
                should_save = (processed_counter % save_interval == 0)
            
            # SEPARATE: Do save operation OUTSIDE the data lock
            if should_save:
                with save_lock:
                    t_save_before = time.time()
                    save_fn()
                    t_save_after = time.time()
                    save_time = t_save_after - t_save_before
                    if save_time > 2:
                        print(f"\n⚠️  [{phase_name}] SAVE OPERATION TOOK {save_time:.2f}s (articles in memory: {len(data_store)})")

        progress_bar.update(1)
        
        # Print diagnostic every 20 articles
        if processed_counter > 0 and processed_counter % 20 == 0:
            avg_total = sum(timings["total"][-20:]) / min(20, len(timings["total"]))
            elapsed = time.time() - phase_start_time
            rate = processed_counter / elapsed if elapsed > 0 else 0
            print(f"\n[{phase_name}] Processed {processed_counter:,} articles | Avg time: {avg_total:.2f}s/art | Rate: {rate:.2f} art/s")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(worker, r): r for r in records}
            for future in as_completed(futures):
                if shutdown_requested:
                    break
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[{phase_name}] Worker exception: {e}")

    except KeyboardInterrupt:
        print("\n" + "!" * 60)
        print(f" [WARNING] Interrupted during {phase_name} phase!")
        print(" Saving progress and shutting down safely...")
        print("!" * 60 + "\n")
        shutdown_requested = True
        with save_lock:
            save_fn()
        progress_bar.close()
        print("[SUCCESS] Progress saved — you can resume anytime.")
        sys.exit(0)

    progress_bar.close()
    
    # Final save
    with save_lock:
        save_fn()
    
    # Print final diagnostics
    total_phase_time = time.time() - phase_start_time
    if timings["total"]:
        avg_total = sum(timings["total"]) / len(timings["total"])
        print(f"\n{'='*60}")
        print(f"[{phase_name}] PHASE DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"Total articles processed: {processed_counter:,}")
        print(f"Total phase time: {total_phase_time:.2f}s ({total_phase_time/60:.2f} minutes)")
        print(f"Average time per article: {avg_total:.2f}s")
        print(f"Throughput: {processed_counter / total_phase_time:.2f} articles/second")
        print(f"{'='*60}\n")
    
    logger.info(f"[{phase_name}] Phase complete. {processed_counter:,} articles annotated this session.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("libelisation")

    print("\n" + "=" * 60)
    print(f"      QWEN PARALLEL ANNOTATOR (PC {PC_ID}) — GPU PASS ONLY")
    print("=" * 60)
    print(f"Configured Workers  : {NUM_WORKERS} threads")
    print(f"Save Interval       : Every {SAVE_INTERVAL} articles")
    print(f"Input File         : {INPUT_FILE}")
    print(f"Enriched Output    : {ENRICHED_OUTPUT_FILE}")
    print(f"Topics Output      : {TOPICS_OUTPUT_FILE}")
    print("=" * 60 + "\n")

    # [1] Input validation
    if not INPUT_FILE.exists():
        logger.error(f"Partition input file does not exist: {INPUT_FILE}")
        sys.exit(1)

    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # [2] Load partitioned input
    logger.info(f"Loading partitioned dataset: {INPUT_FILE.name}...")
    input_df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(input_df):,} articles for PC {PC_ID}.")

    if "id" not in input_df.columns or "text" not in input_df.columns:
        logger.error("Input CSV must contain 'id' and 'text' columns.")
        sys.exit(1)
    if "language" not in input_df.columns:
        input_df["language"] = "en"

    # [3] Load existing checkpoints
    enriched_data = {}
    topic_data = {}

    if ENRICHED_OUTPUT_FILE.exists():
        logger.info(f"Loading existing enriched checkpoint: {ENRICHED_OUTPUT_FILE.name}")
        try:
            temp_df = pd.read_csv(ENRICHED_OUTPUT_FILE)
            for r in temp_df.to_dict("records"):
                aid = int(r["article_id"])
                enriched_data[aid] = {
                    "article_id": aid,
                    "language": r.get("language"),
                    "sentiment_label": r.get("sentiment_label")
                }
        except Exception as e:
            logger.warning(f"Failed to read existing enriched output: {e}. Starting fresh.")

    if TOPICS_OUTPUT_FILE.exists():
        logger.info(f"Loading existing topics checkpoint: {TOPICS_OUTPUT_FILE.name}")
        try:
            temp_df = pd.read_csv(TOPICS_OUTPUT_FILE)
            for r in temp_df.to_dict("records"):
                aid = int(r["article_id"])
                topic_data[aid] = {
                    "article_id": aid,
                    "topic_label": r.get("topic_label")
                }
        except Exception as e:
            logger.warning(f"Failed to read existing topics output: {e}. Starting fresh.")

    save_lock = threading.Lock()

    def save_topics():
        if topic_data:
            df = pd.DataFrame(list(topic_data.values()))
            cols = ["article_id", "topic_label"]
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(TOPICS_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    def save_enriched():
        if enriched_data:
            df = pd.DataFrame(list(enriched_data.values()))
            cols = ["article_id", "language", "sentiment_label"]
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(ENRICHED_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # [4] Identify what still needs processing
    needs_sentiment_ids = set()
    needs_topic_ids = set()

    for row in input_df.itertuples(index=False):
        aid = int(row.id)

        sent_info = enriched_data.get(aid)
        if not sent_info or pd.isna(sent_info.get("sentiment_label")) or str(sent_info.get("sentiment_label")).strip() == "":
            needs_sentiment_ids.add(aid)

        topic_info = topic_data.get(aid)
        if not topic_info or pd.isna(topic_info.get("topic_label")) or str(topic_info.get("topic_label")).strip() == "":
            needs_topic_ids.add(aid)

    logger.info(f"Dynamic state analysis complete:")
    logger.info(f"  - Total articles in partition    : {len(input_df):,}")
    logger.info(f"  - Needing topic prediction       : {len(needs_topic_ids):,}")
    logger.info(f"  - Needing sentiment prediction   : {len(needs_sentiment_ids):,}")

    if not needs_topic_ids and not needs_sentiment_ids:
        print("\n[SUCCESS] All articles in this partition are already fully annotated!")
        return

    # [5] Initialize models
    logger.info("Initializing Preprocessor Router and Qwen LLM Analyzers...")
    preproc = PreprocessRouter(logger=logger)
    topic_model = LLMTopic(logger=logger)
    sentiment_model = LLMSentiment(logger=logger)

    # =====================================================================
    # PHASE 1 — TOPIC
    # =====================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1 / 2 — TOPIC ANNOTATION")
    print("=" * 60)

    topic_records = input_df[input_df["id"].isin(needs_topic_ids)].to_dict("records")

    def process_topic(aid, text, mapped_lang, _orig_lang):
        try:
            preprocessed = custom_preprocess(preproc, text, mapped_lang, "topic")
            res = topic_model.predict(preprocessed, mapped_lang)
            label = res.label
        except Exception as e:
            logger.warning(f"[Topic] Failed article_id={aid}: {e}")
            from src.topic_generation import CATEGORY_DISPLAY
            label = CATEGORY_DISPLAY[17].get(mapped_lang, "General")
        return {"article_id": aid, "topic_label": label}

    run_phase(
        phase_name="Topic",
        records=topic_records,
        needs_ids=needs_topic_ids,
        data_store=topic_data,
        process_fn=process_topic,
        save_fn=save_topics,
        save_lock=save_lock,
        save_interval=SAVE_INTERVAL,
        logger=logger,
        num_workers=NUM_WORKERS,
    )

    # =====================================================================
    # PHASE 2 — SENTIMENT
    # =====================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2 / 2 — SENTIMENT ANNOTATION")
    print("=" * 60)

    sentiment_records = input_df[input_df["id"].isin(needs_sentiment_ids)].to_dict("records")

    def process_sentiment(aid, text, mapped_lang, orig_lang):
        try:
            preprocessed = custom_preprocess(preproc, text, mapped_lang, "sentiment")
            res = sentiment_model.predict(preprocessed, mapped_lang)
            label = res.label
        except Exception as e:
            logger.warning(f"[Sentiment] Failed article_id={aid}: {e}")
            label = "UNK"
        return {"article_id": aid, "language": orig_lang, "sentiment_label": label}

    run_phase(
        phase_name="Sentiment",
        records=sentiment_records,
        needs_ids=needs_sentiment_ids,
        data_store=enriched_data,
        process_fn=process_sentiment,
        save_fn=save_enriched,
        save_lock=save_lock,
        save_interval=SAVE_INTERVAL,
        logger=logger,
        num_workers=NUM_WORKERS,
    )

    # =====================================================================
    # DONE
    # =====================================================================
    print("\n" + "=" * 60)
    print("      ANNOTATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Topic results exported to      : {TOPICS_OUTPUT_FILE.name}")
    print(f"Enriched results exported to   : {ENRICHED_OUTPUT_FILE.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()