import pandas as pd
from pathlib import Path
import sys
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Path to the CSV file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_sample_topic.csv"

def check_inconsistencies():
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check if required columns exist
    required_cols = ["thème prédit", "thème attendu", "sentiment prédit", "sentiment attendu", 
                     "correct/incorrect theme", "correct/incorrect sentiment"]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        return

    inconsistencies_found = False

    for index, row in df.iterrows():
        # Line number in a typical CSV editor (header is line 1, first data row is line 2)
        line_num = index + 2

        # 1. Theme Checks
        theme_predit = str(row.get("thème prédit", "")).strip()
        theme_attendu = str(row.get("thème attendu", "")).strip()
        correct_theme = str(row.get("correct/incorrect theme", "")).strip().lower()

        if correct_theme == "incorrect" and theme_predit == theme_attendu and theme_attendu != "":
            print(f"Line {line_num}: Theme marked as 'incorrect' but 'thème attendu' matches 'thème prédit' ({theme_predit}).")
            inconsistencies_found = True
        
        if correct_theme == "correct" and theme_predit != theme_attendu and theme_attendu != "":
            print(f"Line {line_num}: Theme marked as 'correct' but 'thème attendu' ({theme_attendu}) is different from 'thème prédit' ({theme_predit}).")
            inconsistencies_found = True

        # 2. Sentiment Checks
        sentiment_predit = str(row.get("sentiment prédit", "")).strip()
        sentiment_attendu = str(row.get("sentiment attendu", "")).strip()
        correct_sent = str(row.get("correct/incorrect sentiment", "")).strip().lower()

        if correct_sent == "incorrect" and sentiment_predit == sentiment_attendu and sentiment_attendu != "":
            print(f"Line {line_num}: Sentiment marked as 'incorrect' but 'sentiment attendu' matches 'sentiment prédit' ({sentiment_predit}).")
            inconsistencies_found = True
        
        if correct_sent == "correct" and sentiment_predit != sentiment_attendu and sentiment_attendu != "":
            print(f"Line {line_num}: Sentiment marked as 'correct' but 'sentiment attendu' ({sentiment_attendu}) is different from 'sentiment prédit' ({sentiment_predit}).")
            inconsistencies_found = True

    if not inconsistencies_found:
        print("No inconsistencies found! All annotations look logical.")

if __name__ == "__main__":
    print("Checking for annotation inconsistencies...")
    check_inconsistencies()
