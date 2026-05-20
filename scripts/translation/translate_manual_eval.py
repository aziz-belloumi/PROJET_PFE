import pandas as pd
from pathlib import Path
import sys
import os

# Updated path resolution for being in scripts/translation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from translation import translate_en_to_fr

CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_global.csv"

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    print("Loading CSV data...")
    df = pd.read_csv(CSV_PATH)
    
    # Remove any existing French rows to start fresh
    df = df[df['langue'] != 'fr'].copy()
    
    en_df = df[df['langue'] == 'en'].copy()
    print(f"Found {len(en_df)} English articles to translate to French.")

    fr_rows = []
    for idx, row in en_df.iterrows():
        print(f"Translating article {len(fr_rows)+1}/{len(en_df)}...")
        
        try:
            # Only translate the text
            translated_text = translate_en_to_fr(str(row['texte']))
            
            new_row = row.copy()
            new_row['texte'] = translated_text
            new_row['langue'] = 'fr'
            
            # Clear all label and evaluation columns as requested.
            # These will be added manually later.
            cols_to_clear = [
                'thème attendu', 'thème prédit', 
                'correct/incorrect theme', 'sentiment attendu', 
                'sentiment prédit', 'correct/incorrect sentiment'
            ]
            for col in cols_to_clear:
                new_row[col] = None
            
            fr_rows.append(new_row)
        except Exception as e:
            print(f"Failed to translate article at index {idx}: {e}")

    if fr_rows:
        new_fr_df = pd.DataFrame(fr_rows)
        # Append the new French rows
        updated_df = pd.concat([df, new_fr_df], ignore_index=True)
        updated_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\nDone! Successfully appended {len(fr_rows)} translated French articles to {CSV_PATH}")
    else:
        print("\nNo articles were translated.")

if __name__ == "__main__":
    main()
