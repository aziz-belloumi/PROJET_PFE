import pandas as pd
from pathlib import Path
import sys
import logging
import io

# Force stdout to UTF-8 for Windows compatibility with Arabic/French characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.topic_generation import LLMTopic
from src.sentiment_analysis import LLMSentiment

# Target CSV filename
CSV_NAME = "manual_eval_sample_topic.csv"  # The script will also check for 'manual_eval_topic.csv'
CSV_PATH = PROJECT_ROOT / "scripts" / CSV_NAME

# Alternative path if the user meant a specific file
ALT_CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_topic.csv"

def main():
    target_path = ALT_CSV_PATH if ALT_CSV_PATH.exists() else CSV_PATH
    
    if not target_path.exists():
        print(f"Error: Neither {CSV_PATH} nor {ALT_CSV_PATH} was found.")
        return

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print(f"Loading CSV: {target_path}")
    df = pd.read_csv(target_path)
    
    # Standardize column types
    df['langue'] = df['langue'].astype(str).str.lower()
    
    # Filter for French rows that don't have predictions yet
    # The user mentioned "new translated French texts" which usually have empty prediction columns
    fr_mask = (df['langue'] == 'fr')
    target_indices = df.index[fr_mask].tolist()
    
    if not target_indices:
        print("No French articles found in the CSV.")
        return

    print(f"Found {len(target_indices)} French articles to process with Qwen.")

    # Initialize models
    # Note: These use qwen2.5:7b via Ollama as configured in the source files
    topic_model = LLMTopic(logger=logger)
    sentiment_model = LLMSentiment(logger=logger)

    # Process rows
    processed_count = 0
    try:
        for i, idx in enumerate(target_indices):
            text = str(df.loc[idx, 'texte'])
            
            # Skip if already has predictions (optional, but safer if script is re-run)
            if pd.notna(df.loc[idx, 'thème prédit']) and pd.notna(df.loc[idx, 'sentiment prédit']):
                continue

            print(f"[{i+1}/{len(target_indices)}] Processing article at index {idx}...")
            
            try:
                # Topic prediction
                topic_res = topic_model.predict(text, lang='fr')
                df.loc[idx, 'thème prédit'] = topic_res.label
                
                # Sentiment prediction
                sentiment_res = sentiment_model.predict(text, lang='fr')
                df.loc[idx, 'sentiment prédit'] = sentiment_res.label
                
                processed_count += 1
                
                # Save periodically to avoid data loss
                if processed_count % 5 == 0:
                    df.to_csv(target_path, index=False, encoding='utf-8-sig')
                    print(f" > Progress saved ({processed_count} articles processed).")
                    
            except Exception as e:
                print(f"    Error processing article at index {idx}: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    finally:
        # Final save
        df.to_csv(target_path, index=False, encoding='utf-8-sig')
        print(f"\nProcessing complete. Updated {processed_count} French articles in {target_path.name}")

if __name__ == "__main__":
    main()
