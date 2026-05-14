import sys
import os
import pandas as pd
import logging
import json
import time
import requests
from pathlib import Path
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force stdout to UTF-8 for Windows compatibility with Arabic characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
MAX_RETRIES = 3
MAX_WORKERS = 6  # Reduced for 7B on 6GB VRAM

DATA_DIR = PROJECT_ROOT / "data_seperation"
OUTPUT_DIR = DATA_DIR / "check_dialect"
PARALLEL_DIR = OUTPUT_DIR / "parallel_data"

# PC Identifier - CHANGE THIS TO 'pc2' on the other machine
PC_ID = "pc1"

def call_qwen(text):
    """Call Qwen to classify the text using MARBERTv2 labels."""
    prompt = f"""You are an expert in Arabic linguistics.
Task: Classify the following Arabic text into one of these exact categories:
- MSA (Modern Standard Arabic)
- EGY (Egyptian dialect)
- LEV (Levantine dialect: Syrian, Lebanese, Palestinian, Jordanian)
- GLF (Gulf dialect: Saudi, Emirati, Kuwaiti, Qatari, etc.)
- MGR (Maghrebi dialect: Moroccan, Algerian, Tunisian, Libyan)

Text:
{text[:2000]}

Instructions:
1. Respond with only the CODE: 'MSA', 'EGY', 'LEV', 'GLF', or 'MGR'.
2. No explanation, no other text.

Classification:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 2048,
            "temperature": 0,
            "num_predict": 10
        }
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()["response"].strip().upper()
                for code in ['MSA', 'EGY', 'LEV', 'GLF', 'MGR']:
                    if code in result: return code
                return result
        except Exception as e:
            time.sleep(1)
    
    return "ERROR"

def process_row(row, abs_idx, source_type):
    """Helper to process a single row for the thread pool."""
    article_text = str(row.get('text', ''))
    if not article_text.strip():
        return None

    verdict = call_qwen(article_text)
    original_region = str(row.get('dialect_region', 'MSA')).upper()
    is_confused = (verdict != original_region) and (verdict != "ERROR")
    
    return {
        "id": row.get('id', abs_idx),
        "text_snippet": article_text[:200],
        "original_source": source_type,
        "original_region": original_region,
        "qwen_verdict": verdict,
        "is_confused": is_confused,
        "abs_idx": abs_idx
    }

def main():
    # PC-specific files
    results_file = OUTPUT_DIR / f"dialect_check_results_{PC_ID}.csv"
    progress_file = OUTPUT_DIR / f"annotation_progress_{PC_ID}.json"
    log_file = OUTPUT_DIR / f"qwen_annotator_{PC_ID}.log"

    # PC-specific sources from the parallel_data folder
    sources = {
        "MSA": PARALLEL_DIR / f"arabic_msa_{PC_ID}.csv",
        "Dialectal": PARALLEL_DIR / f"arabic_dialectal_{PC_ID}.csv"
    }

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Setup Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"--- {PC_ID.upper()} starting (MARBERTv2 labels) ---")

    # Load progress
    progress = {"MSA": 0, "Dialectal": 0}
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        except:
            pass
    
    # Initialize or load results
    if results_file.exists():
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=["id", "text_snippet", "original_source", "original_region", "qwen_verdict", "is_confused"])

    for source_type, source_path in sources.items():
        if not source_path.exists():
            logger.warning(f"Source file not found: {source_path}. Skipping.")
            continue

        logger.info(f"Processing {source_type} articles starting from index {progress[source_type]}...")
        
        try:
            skip = progress[source_type]
            batch_size = 500
            
            while True:
                try:
                    df_chunk = pd.read_csv(source_path, skiprows=range(1, skip + 1), nrows=batch_size)
                    if df_chunk.empty:
                        logger.info(f"Finished processing {source_type}.")
                        break
                    
                    if 'text' not in df_chunk.columns and 'texte' in df_chunk.columns:
                        df_chunk.rename(columns={'texte': 'text'}, inplace=True)

                    rows_to_process = []
                    for idx, row in df_chunk.iterrows():
                        rows_to_process.append((row, skip + idx))

                    new_rows = []
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        future_to_row = {
                            executor.submit(process_row, row, abs_idx, source_type): abs_idx 
                            for row, abs_idx in rows_to_process
                        }
                        
                        for future in as_completed(future_to_row):
                            res = future.result()
                            if res:
                                abs_idx = res.pop('abs_idx')
                                print(f"\r[{PC_ID.upper()}] {source_type} {abs_idx}...", end="", flush=True)
                                
                                if res['is_confused']:
                                    logger.info(f"[CONFUSED] ID {res['id']} at idx {abs_idx} ({res['original_region']} vs Qwen {res['qwen_verdict']})")
                                
                                new_rows.append(res)
                                
                                # Update progress and save periodically
                                if len(new_rows) >= 10:
                                    temp_df = pd.DataFrame(new_rows)
                                    results_df = pd.concat([results_df, temp_df], ignore_index=True)
                                    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
                                    # Use the latest finished index for progress
                                    progress[source_type] = max(progress[source_type], abs_idx + 1)
                                    with open(progress_file, 'w') as f: json.dump(progress, f)
                                    new_rows = []

                    if new_rows:
                        temp_df = pd.DataFrame(new_rows)
                        results_df = pd.concat([results_df, temp_df], ignore_index=True)
                        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
                        progress[source_type] = skip + len(df_chunk)
                        with open(progress_file, 'w') as f: json.dump(progress, f)
                    
                    skip += len(df_chunk)
                    
                except pd.errors.EmptyDataError:
                    break
                except Exception as e:
                    logger.error(f"Error in batch: {e}")
                    time.sleep(5)
                    break

        except KeyboardInterrupt:
            logger.info("\nInterrupted. Progress saved.")
            return
        except Exception as e:
            logger.error(f"Critical error processing {source_type}: {e}")

    logger.info(f"Annotation complete for {PC_ID}.")

if __name__ == "__main__":
    main()
