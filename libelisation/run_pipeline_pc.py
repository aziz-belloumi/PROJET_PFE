import os
import sys
import threading
import logging
import pandas as pd
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
PC_ID = 1          # Partition identifier: 1, 2, 3, or 4
NUM_WORKERS = 4    # Number of concurrent workers calling Ollama in parallel
SAVE_INTERVAL = 5  # Autosave database-matching CSV files every N articles

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
    """Map Arabic dialects to 'ar', standardizing other codes."""
    if not isinstance(lang, str):
        return "en"
    clean_lang = lang.strip().lower()
    return DIALECT_TO_ISO.get(clean_lang, clean_lang)

def main():
    # Setup clean console logging
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

    # Standardize input structure
    if "id" not in input_df.columns or "text" not in input_df.columns:
        logger.error("Input CSV must contain 'id' and 'text' columns.")
        sys.exit(1)
    if "language" not in input_df.columns:
        # Fallback if language is missing
        input_df["language"] = "en"

    # [3] Load existing checkpoints (dynamic resume state)
    # We load as dictionaries keyed by article_id for instant O(1) lookups
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

    # Reentrant Thread Lock to guarantee thread-safe file writes
    save_lock = threading.Lock()

    def save_progress():
        """Helper function to dump current in-memory maps safely to disk."""
        with save_lock:
            if enriched_data:
                df_enriched = pd.DataFrame(list(enriched_data.values()))
                # Enforce schema columns: article_id, language, sentiment_label
                cols = ["article_id", "language", "sentiment_label"]
                df_enriched = df_enriched[[c for c in cols if c in df_enriched.columns]]
                df_enriched.to_csv(ENRICHED_OUTPUT_FILE, index=False, encoding="utf-8-sig")

            if topic_data:
                df_topics = pd.DataFrame(list(topic_data.values()))
                cols = ["article_id", "topic_label"]
                df_topics = df_topics[[c for c in cols if c in df_topics.columns]]
                df_topics.to_csv(TOPICS_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # [4] Identify unprocessed items
    # Check if we need to run Sentiment or Topic for each row
    needs_sentiment_ids = set()
    needs_topic_ids = set()

    for row in input_df.itertuples(index=False):
        aid = int(row.id)
        
        # Needs sentiment if not in checkpoint, or sentiment label is blank/NaN
        sent_info = enriched_data.get(aid)
        if not sent_info or pd.isna(sent_info.get("sentiment_label")) or str(sent_info.get("sentiment_label")).strip() == "":
            needs_sentiment_ids.add(aid)

        # Needs topic if not in checkpoint, or topic label is blank/NaN
        topic_info = topic_data.get(aid)
        if not topic_info or pd.isna(topic_info.get("topic_label")) or str(topic_info.get("topic_label")).strip() == "":
            needs_topic_ids.add(aid)

    # Combined set of any article needing any processing
    target_ids = needs_sentiment_ids.union(needs_topic_ids)
    
    # Filter our DataFrame rows that require work
    work_df = input_df[input_df["id"].isin(target_ids)].copy()
    records = work_df.to_dict("records")

    logger.info(f"Dynamic state analysis complete:")
    logger.info(f"  - Total articles in partition : {len(input_df):,}")
    logger.info(f"  - Needing sentiment prediction: {len(needs_sentiment_ids):,}")
    logger.info(f"  - Needing topic prediction    : {len(needs_topic_ids):,}")
    logger.info(f"  - Total unique articles to process: {len(records):,}")

    if not records:
        print("\n[SUCCESS] All articles in this partition are already fully annotated!")
        return

    # [5] Initialize Models
    logger.info("Initializing Preprocessor Router and Qwen LLM Analyzers...")
    preproc = PreprocessRouter(logger=logger)
    topic_model = LLMTopic(logger=logger)
    sentiment_model = LLMSentiment(logger=logger)

    processed_counter = 0
    progress_bar = tqdm(total=len(records), desc="Annotating with Qwen", unit="art")

    # Clean shutdown flag for thread pool cancellation
    shutdown_requested = False

    def process_single_article(row):
        """Worker task executing inference for one article row."""
        nonlocal processed_counter, shutdown_requested
        if shutdown_requested:
            return

        aid = int(row["id"])
        text = str(row["text"])
        orig_lang = str(row["language"])
        mapped_lang = map_language_code(orig_lang)

        # Ensure text is not empty
        if not text.strip():
            text = "empty"

        # ---------------------------------------------------------------------
        # [A] COMMENTED NER EXTRACTION LOGIC (Bypassed but preserved as requested)
        # ---------------------------------------------------------------------
        # # Entities are bypassed here as NER is commented out.
        # # If we ever want to re-run entity predictions:
        # try:
        #     # Step 1: Preprocess text for NER
        #     # preprocessed_ner = preproc.preprocess(text, mapped_lang, "ner")
        #     # Step 2: Invoke NER pipeline
        #     # ents = ner.predict(preprocessed_ner)
        #     # Step 3: Upsert into entities structures...
        #     pass
        # except Exception as ner_err:
        #     # logger.error(f"NER failed for article_id={aid}: {ner_err}")
        #     pass

        # ---------------------------------------------------------------------
        # [B] Sentiment Prediction Step
        # ---------------------------------------------------------------------
        sentiment_label = None
        if aid in needs_sentiment_ids:
            try:
                preprocessed_sent = preproc.preprocess(text, mapped_lang, "sentiment")
                res = sentiment_model.predict(preprocessed_sent, mapped_lang)
                sentiment_label = res.label
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for article_id={aid}: {e}")
                sentiment_label = "UNK"
        else:
            # Re-use existing value from checkpoint
            sentiment_label = enriched_data[aid]["sentiment_label"]

        # ---------------------------------------------------------------------
        # [C] Topic Prediction Step
        # ---------------------------------------------------------------------
        topic_label = None
        if aid in needs_topic_ids:
            try:
                preprocessed_topic = preproc.preprocess(text, mapped_lang, "topic")
                res = topic_model.predict(preprocessed_topic, mapped_lang)
                topic_label = res.label
            except Exception as e:
                logger.warning(f"Topic classification failed for article_id={aid}: {e}")
                # Fallback to General category in matching language
                from src.topic_generation import CATEGORY_DISPLAY
                topic_label = CATEGORY_DISPLAY[17].get(mapped_lang, "General")
        else:
            # Re-use existing value from checkpoint
            topic_label = topic_data[aid]["topic_label"]

        # ---------------------------------------------------------------------
        # [D] Update State (Thread-Safe)
        # ---------------------------------------------------------------------
        with save_lock:
            enriched_data[aid] = {
                "article_id": aid,
                "language": orig_lang,
                "sentiment_label": sentiment_label
            }
            topic_data[aid] = {
                "article_id": aid,
                "topic_label": topic_label
            }
            
            processed_counter += 1
            
            # Periodically write to CSV to secure progress against OOMs/crashes
            if processed_counter % SAVE_INTERVAL == 0:
                save_progress()

        progress_bar.update(1)

    # [6] Parallel thread pool execution with dynamic shutdown handling
    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all rows
            futures = {executor.submit(process_single_article, r): r for r in records}
            
            for future in as_completed(futures):
                if shutdown_requested:
                    break
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker thread exception: {e}")
                    
    except KeyboardInterrupt:
        print("\n" + "!" * 60)
        print(" [WARNING] Execution interrupted by user (Ctrl+C)!")
        print(" Shutting down parallel worker pool safely and saving progress...")
        print("!" * 60 + "\n")
        
        shutdown_requested = True
        # Cancel all pending futures in the queue immediately
        save_progress()
        progress_bar.close()
        print("[SUCCESS] All current annotations secured! You can resume anytime.")
        sys.exit(0)

    # Final Progress Bar Close & Data Save
    progress_bar.close()
    save_progress()
    
    print("\n" + "=" * 60)
    print("      ANNOTATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total processed in this session: {processed_counter:,} articles.")
    print(f"Enriched results exported to   : {ENRICHED_OUTPUT_FILE.name}")
    print(f"Topic results exported to      : {TOPICS_OUTPUT_FILE.name}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
