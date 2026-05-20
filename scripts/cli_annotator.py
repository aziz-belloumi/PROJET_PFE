import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import io

# Force stdout to UTF-8 for Arabic rendering in Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.config import Config
from src.topic_generation import CATEGORY_DISPLAY

CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_global.csv"

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    BIDI_AVAILABLE = True
except ImportError:
    BIDI_AVAILABLE = False

def format_text_for_terminal(text, lang):
    text = str(text)
    if lang == 'ar' and BIDI_AVAILABLE:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text

def display_themes(lang):
    print("\n" + "="*30)
    print("      AVAILABLE THEMES")
    print("="*30)
    taxonomy = CATEGORY_DISPLAY
    
    # Split into two columns for better readability if necessary, or just single column
    for idx, translations in taxonomy.items():
        theme_name = translations.get(lang, translations["en"])
        display_name = format_text_for_terminal(theme_name, lang)
        print(f"[{idx:2d}] {display_name}")
    print("="*30 + "\n")

def main():
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please run sample_extraction.py first to generate the file.")
        return

    # Read CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure required columns exist
    for col in ["thème attendu", "sentiment attendu", "correct/incorrect theme", "correct/incorrect sentiment"]:
        if col not in df.columns:
            df[col] = ""

    # Replace NaNs with empty string
    df = df.fillna("")

    total_rows = len(df)
    
    # Find how many are already annotated
    annotated_count = df["thème attendu"].astype(str).str.strip().astype(bool).sum()
    print(f"Starting annotation CLI. {annotated_count}/{total_rows} articles already annotated.")
    
    for index, row in df.iterrows():
        theme_attendu = str(row.get("thème attendu", "")).strip()
        sentiment_attendu = str(row.get("sentiment attendu", "")).strip()
        correct_theme = str(row.get("correct/incorrect theme", "")).strip()
        correct_sent = str(row.get("correct/incorrect sentiment", "")).strip()
        
        # If the row is not completely filled, prompt the user
        if not theme_attendu or not sentiment_attendu or not correct_theme or not correct_sent:
            lang = str(row['langue']).lower()
            if lang not in ['ar', 'en', 'fr']:
                lang = 'en'
                
            print("\n" + "#"*80)
            print(f"ARTICLE {index + 1}/{total_rows} | LANG: {lang.upper()}")
            print("#"*80)
            
            display_texte = format_text_for_terminal(row['texte'], lang)
            print(f"\n{display_texte}\n")
            print("-" * 80)
            
            display_predicted_theme = format_text_for_terminal(row['thème prédit'], lang)
            print(f"PREDICTED THEME    : {display_predicted_theme}")
            print(f"PREDICTED SENTIMENT: {row['sentiment prédit']}")
            print("-" * 80)
            
            display_themes(lang)
            
            print("--- Manual Annotation ---")
            print("(Type 'q' to save and quit at any prompt)\n")
            
            # 1. Expected Theme
            while True:
                display_pred_prompt = format_text_for_terminal(row['thème prédit'], lang)
                theme_expected = input(f"[1] Expected Theme (0-17) [Press Enter to keep '{display_pred_prompt}']: ").strip()
                if theme_expected.lower() in ['q', 'quit']:
                    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                    print("\nProgress saved. Exiting.")
                    return
                
                if theme_expected == "":
                    theme_expected = str(row['thème prédit'])
                    break
                elif theme_expected.isdigit() and int(theme_expected) in CATEGORY_DISPLAY:
                    idx_val = int(theme_expected)
                    theme_expected = CATEGORY_DISPLAY[idx_val].get(lang, CATEGORY_DISPLAY[idx_val]["en"])
                    break
                elif not theme_expected.isdigit():
                    # Accept whatever string they typed if they want to override
                    break
                else:
                    print("Invalid ID. Please enter a number between 0 and 17.")
                    
            # 2. Expected Sentiment
            while True:
                sentiment_expected = input(f"[2] Expected Sentiment (P=Positive, N=Negative, U=Neutral) [Press Enter to keep '{row['sentiment prédit']}']: ").strip()
                if sentiment_expected.lower() in ['q', 'quit']:
                    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                    print("\nProgress saved. Exiting.")
                    return
                    
                if sentiment_expected == "":
                    sentiment_expected = str(row['sentiment prédit'])
                    break
                else:
                    mapper = {"p": "POSITIVE", "n": "NEGATIVE", "u": "NEUTRAL"}
                    char_key = sentiment_expected.lower()[0] if sentiment_expected else ""
                    if char_key in mapper:
                        sentiment_expected = mapper[char_key]
                        break
                    else:
                        # Accept typed string
                        break
            
            # 3. Correct / Incorrect Theme
            while True:
                correct_t_input = input(f"[3] Is the THEME prediction correct? (y=Yes/Correct, n=No/Incorrect): ").strip()
                if correct_t_input.lower() in ['q', 'quit']:
                    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                    print("\nProgress saved. Exiting.")
                    return
                    
                if correct_t_input.lower() in ['y', 'yes', '1', 'c', 'correct']:
                    final_correct_theme = "correct"
                    break
                elif correct_t_input.lower() in ['n', 'no', '0', 'i', 'incorrect']:
                    final_correct_theme = "incorrect"
                    break
                elif correct_t_input == "":
                    print("This field is required. Please enter y or n.")
                else:
                    print("Invalid input. Please enter y or n.")

            # 4. Correct / Incorrect Sentiment
            while True:
                correct_s_input = input(f"[4] Is the SENTIMENT prediction correct? (y=Yes/Correct, n=No/Incorrect): ").strip()
                if correct_s_input.lower() in ['q', 'quit']:
                    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                    print("\nProgress saved. Exiting.")
                    return
                    
                if correct_s_input.lower() in ['y', 'yes', '1', 'c', 'correct']:
                    final_correct_sent = "correct"
                    break
                elif correct_s_input.lower() in ['n', 'no', '0', 'i', 'incorrect']:
                    final_correct_sent = "incorrect"
                    break
                elif correct_s_input == "":
                    print("This field is required. Please enter y or n.")
                else:
                    print("Invalid input. Please enter y or n.")
            
            # Update dataframe and save immediately
            df.at[index, "thème attendu"] = theme_expected
            df.at[index, "sentiment attendu"] = sentiment_expected
            df.at[index, "correct/incorrect theme"] = final_correct_theme
            df.at[index, "correct/incorrect sentiment"] = final_correct_sent
            
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"--> Saved Article {index + 1}.")

    print("\nAll articles have been annotated!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting gracefully without data loss...")
