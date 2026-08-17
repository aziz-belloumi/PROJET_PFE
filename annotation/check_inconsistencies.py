import pandas as pd
from pathlib import Path
import sys
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformers.shared import find_column, normalize_topic_label

CSV_PATH = PROJECT_ROOT / "data" / "manual_eval_global.csv"


def check_inconsistencies():
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    theme_att_col = find_column(df, ["thème attendu", "theme attendu", "topic attendu"])
    sent_att_col = find_column(df, ["sentiment attendu", "sentiment"])
    theme_pred_col = find_column(df, ["thème prédit", "theme predit", "topic prédit"])
    sent_pred_col = find_column(df, ["sentiment prédit", "sentiment predit"])
    correct_theme_col = find_column(df, ["correct/incorrect theme", "theme correct"])
    correct_sent_col = find_column(df, ["correct/incorrect sentiment", "sentiment correct"])

    inconsistencies_found = False

    # Check theme consistency if both predicted and correct flags exist
    if theme_att_col and theme_pred_col and correct_theme_col:
        for index, row in df.iterrows():
            line_num = index + 2
            t_pred = normalize_topic_label(str(row.get(theme_pred_col, "")))
            t_att = normalize_topic_label(str(row.get(theme_att_col, "")))
            c_theme = str(row.get(correct_theme_col, "")).strip().lower()

            if c_theme == "incorrect" and t_pred == t_att and t_att != "":
                print(f"Line {line_num}: Theme marked as 'incorrect' but 'thème attendu' matches 'thème prédit' ({t_pred}).")
                inconsistencies_found = True

            if c_theme == "correct" and t_pred != t_att and t_att != "":
                print(f"Line {line_num}: Theme marked as 'correct' but 'thème attendu' ({t_att}) differs from 'thème prédit' ({t_pred}).")
                inconsistencies_found = True

    # Check sentiment consistency if both predicted and correct flags exist
    if sent_att_col and sent_pred_col and correct_sent_col:
        for index, row in df.iterrows():
            line_num = index + 2
            s_pred = str(row.get(sent_pred_col, "")).strip().upper()
            s_att = str(row.get(sent_att_col, "")).strip().upper()
            c_sent = str(row.get(correct_sent_col, "")).strip().lower()

            if c_sent == "incorrect" and s_pred == s_att and s_att != "":
                print(f"Line {line_num}: Sentiment marked as 'incorrect' but 'sentiment attendu' matches 'sentiment prédit' ({s_pred}).")
                inconsistencies_found = True

            if c_sent == "correct" and s_pred != s_att and s_att != "":
                print(f"Line {line_num}: Sentiment marked as 'correct' but 'sentiment attendu' ({s_att}) differs from 'sentiment prédit' ({s_pred}).")
                inconsistencies_found = True

    # Check valid ground truth values
    if sent_att_col:
        invalid_sent = df[~df[sent_att_col].astype(str).str.strip().str.upper().isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL', '', 'NAN'])]
        if not invalid_sent.empty:
            print(f"Found {len(invalid_sent)} rows with unexpected ground truth sentiment values:")
            for idx, r in invalid_sent.iterrows():
                print(f"  Line {idx + 2}: {repr(r[sent_att_col])}")
            inconsistencies_found = True

    if not inconsistencies_found:
        print("✅ No inconsistencies found! Ground truth and annotations are consistent.")


if __name__ == "__main__":
    print("Checking for annotation inconsistencies...")
    check_inconsistencies()
