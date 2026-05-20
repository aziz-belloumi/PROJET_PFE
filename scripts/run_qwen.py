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

CSV_NAME = "manual_eval_global.csv"
CSV_PATH = PROJECT_ROOT / "scripts" / CSV_NAME

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} was not found.")
        return

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Standardize column types
    df['langue'] = df['langue'].astype(str).str.lower().str.strip()
    
    # Ensure prediction columns exist in case they are missing
    for col in ['thème prédit', 'sentiment prédit']:
        if col not in df.columns:
            df[col] = ""
            
    # Filter for rows that are missing predictions (NaN, None, or empty string)
    target_indices = df.index[
        df['thème prédit'].isna() | (df['thème prédit'].astype(str).str.strip() == "") |
        df['sentiment prédit'].isna() | (df['sentiment prédit'].astype(str).str.strip() == "")
    ].tolist()
    
    if not target_indices:
        print("All rows in the evaluation set already have Qwen predictions! Nothing to process.")
        return

    print(f"Found {len(target_indices)} unprocessed articles to annotate with Qwen.")

    # Initialize models (Uses qwen2.5:7b via local Ollama connection)
    topic_model = LLMTopic(logger=logger)
    sentiment_model = LLMSentiment(logger=logger)

    processed_count = 0
    try:
        for i, idx in enumerate(target_indices):
            text = str(df.loc[idx, 'texte'])
            row_lang = str(df.loc[idx, 'langue']).lower().strip()
            if row_lang not in ['ar', 'en', 'fr']:
                row_lang = 'en'  # Fallback

            # Skip if already has both predictions (optional, double-check to be safe)
            t_pred = str(df.loc[idx, 'thème prédit']).strip()
            s_pred = str(df.loc[idx, 'sentiment prédit']).strip()
            if (pd.notna(df.loc[idx, 'thème prédit']) and t_pred != "" and t_pred.lower() != 'nan') and \
               (pd.notna(df.loc[idx, 'sentiment prédit']) and s_pred != "" and s_pred.lower() != 'nan'):
                continue

            print(f"\n[{i+1}/{len(target_indices)}] Processing article at index {idx} (LANG: {row_lang.upper()})...")
            
            try:
                # Topic prediction
                topic_res = topic_model.predict(text, lang=row_lang)
                df.loc[idx, 'thème prédit'] = topic_res.label
                print(f"  -> Predicted Theme    : {topic_res.label}")
                
                # Sentiment prediction
                sentiment_res = sentiment_model.predict(text, lang=row_lang)
                df.loc[idx, 'sentiment prédit'] = sentiment_res.label
                print(f"  -> Predicted Sentiment: {sentiment_res.label}")
                
                processed_count += 1
                
                # Save periodically to avoid data loss
                if processed_count % 5 == 0:
                    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
                    print(f" > Progress saved ({processed_count} articles processed).")
                    
            except Exception as e:
                print(f"    Error processing article at index {idx}: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    finally:
        # Final save
        df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\nProcessing complete. Updated {processed_count} articles in {CSV_PATH.name}")

if __name__ == "__main__":
    main()
