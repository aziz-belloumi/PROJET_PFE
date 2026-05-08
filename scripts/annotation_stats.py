import pandas as pd
from pathlib import Path
import sys
import io
from sklearn.metrics import classification_report

# Force stdout to UTF-8 for Arabic rendering in Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_sample_topic.csv"
MODELS_CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_sample_topic_with_models.csv"
ALL_MODELS_CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_sample_all_models.csv"

SENT_LANG_MAP = {
    'bert-base-arabic-camelbert-msa-sentiment_pred': 'AR',
    'bert-base-arabic-camelbert-da-sentiment_pred': 'AR',
    'bert-base-arabic-camelbert-mix-sentiment_pred': 'AR',
    'AraBert-Arabic-Sentiment-Analysis_pred': 'AR',
    'twitter-roberta-base-sentiment-latest_pred': 'EN',
    'sentiment-roberta-large-english-3-classes_pred': 'EN',
    'bertweet-base-sentiment-analysis_pred': 'EN',
    'distilcamembert-base-sentiment_pred': 'FR',
    'bert-base-multilingual-uncased-sentiment_pred': 'FR',
    'distilcamembert-base-nli_pred': 'FR',
    'twitter-xlm-roberta-base-sentiment_pred': 'MULTI',
    'distilbert-base-multilingual-cased-sentiments-student_pred': 'MULTI',
}

TOPIC_LANG_MAP = {
    'DeBERTa-v3-large-mnli-fever-anli-ling-wanli_topic_pred': 'EN',
    'roberta-large-mnli_topic_pred': 'EN',
    'bart-large-mnli_topic_pred': 'EN',
    'nli-deberta-v3-large_topic_pred': 'EN',
    'french_xlm_xnli_topic_pred': 'FR',
    'distilcamembert-base-nli_topic_pred': 'FR',
    'mDeBERTa-v3-base-xnli-multilingual-nli-2mil7_topic_pred': 'MULTI',
    'mDeBERTa-v3-base-mnli-xnli_topic_pred': 'MULTI',
    'xlm-roberta-large-xnli_topic_pred': 'MULTI',
}


def print_classification_metrics(y_true, y_pred, model_name, indent="    "):
    """Print accuracy, precision, recall, F1 for a model."""
    if len(y_true) == 0:
        print(f"{indent}{model_name}: N/A (no valid predictions)")
        return
    try:
        report = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0
        )
        acc = (pd.Series(y_true).values == pd.Series(y_pred).values).mean() * 100
        macro = report.get('macro avg', {})
        print(f"{indent}{model_name}:")
        print(f"{indent}  Accuracy : {acc:.1f}%")
        print(f"{indent}  Precision: {macro.get('precision', 0)*100:.1f}%")
        print(f"{indent}  Recall   : {macro.get('recall', 0)*100:.1f}%")
        print(f"{indent}  F1 Macro : {macro.get('f1-score', 0)*100:.1f}%")
        print(f"{indent}  Samples  : {len(y_true)}")
    except Exception as e:
        print(f"{indent}{model_name}: Error computing metrics — {e}")


def print_stats():
    if ALL_MODELS_CSV_PATH.exists():
        active_csv = ALL_MODELS_CSV_PATH
    elif MODELS_CSV_PATH.exists():
        active_csv = MODELS_CSV_PATH
    else:
        active_csv = CSV_PATH

    if not active_csv.exists():
        print(f"Error: CSV file not found at {active_csv}")
        return

    try:
        df = pd.read_csv(active_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("=" * 70)
    print("      MANUAL ANNOTATION & MODEL STATISTICS")
    print(f"      Source: {active_csv.name}")
    print("=" * 70)

    total_rows = len(df)
    print(f"Total samples: {total_rows}")

    req_cols = ["thème attendu", "sentiment attendu", "correct/incorrect theme", "correct/incorrect sentiment"]
    for col in req_cols:
        if col not in df.columns:
            print(f"\nMissing column '{col}'. Cannot compute full stats.")
            return

    df['is_annotated'] = df['thème attendu'].astype(str).str.strip() != ''
    annotated_count = df['is_annotated'].sum()

    print(f"Annotated samples: {annotated_count} ({annotated_count/total_rows*100:.1f}%)")
    print(f"Unannotated samples: {total_rows - annotated_count} ({(total_rows - annotated_count)/total_rows*100:.1f}%)")

    if annotated_count == 0:
        print("\nNo samples have been annotated yet.")
        return

    df_ann = df[df['is_annotated']].copy()

    # --- Language Breakdown ---
    if 'langue' in df_ann.columns:
        print("\n--- Language Breakdown ---")
        lang_counts = df_ann['langue'].value_counts()
        for lang, count in lang_counts.items():
            print(f"  {str(lang).upper()}: {count} ({count/annotated_count*100:.1f}%)")

    # --- Original Pipeline Theme Accuracy ---
    print("\n--- Original Pipeline Theme Prediction Accuracy ---")
    df_ann['correct/incorrect theme'] = df_ann['correct/incorrect theme'].astype(str).str.strip().str.lower()
    theme_correct = (df_ann['correct/incorrect theme'] == 'correct').sum()
    theme_incorrect = (df_ann['correct/incorrect theme'] == 'incorrect').sum()
    total_theme_eval = theme_correct + theme_incorrect
    if total_theme_eval > 0:
        print(f"  Correct: {theme_correct} ({theme_correct/total_theme_eval*100:.1f}%)")
        print(f"  Incorrect: {theme_incorrect} ({theme_incorrect/total_theme_eval*100:.1f}%)")
    else:
        print("  No theme correctness evaluations found.")

    # --- Original Pipeline Sentiment Accuracy ---
    print("\n--- Original Pipeline Sentiment Prediction Accuracy ---")
    df_ann['correct/incorrect sentiment'] = df_ann['correct/incorrect sentiment'].astype(str).str.strip().str.lower()
    sent_correct = (df_ann['correct/incorrect sentiment'] == 'correct').sum()
    sent_incorrect = (df_ann['correct/incorrect sentiment'] == 'incorrect').sum()
    total_sent_eval = sent_correct + sent_incorrect
    if total_sent_eval > 0:
        print(f"  Correct: {sent_correct} ({sent_correct/total_sent_eval*100:.1f}%)")
        print(f"  Incorrect: {sent_incorrect} ({sent_incorrect/total_sent_eval*100:.1f}%)")
    else:
        print("  No sentiment correctness evaluations found.")

    # --- Expected Theme Distribution ---
    print("\n--- Expected Theme Distribution (Top 5) ---")
    theme_counts = df_ann['thème attendu'].astype(str).str.strip().value_counts().head(5)
    for theme, count in theme_counts.items():
        if theme:
            print(f"  {theme}: {count} ({count/annotated_count*100:.1f}%)")

    # --- Expected Sentiment Distribution ---
    print("\n--- Expected Sentiment Distribution ---")
    sent_counts = df_ann['sentiment attendu'].astype(str).str.strip().str.upper().value_counts()
    for sent, count in sent_counts.items():
        if sent:
            print(f"  {sent}: {count} ({count/annotated_count*100:.1f}%)")

    # -------------------------------------------------------
    # --- Sentiment Models: Accuracy + Precision/Recall/F1 ---
    # -------------------------------------------------------
    print("\n--- Sentiment Models Performance (Accuracy + Precision + Recall + F1) ---")

    sent_model_cols = [
        c for c in df_ann.columns
        if c.endswith('_pred')
        and not c.endswith('_topic_pred')
        and not c.endswith('_ner_pred')
        and c not in ['thème prédit', 'sentiment prédit']
    ]

    if not sent_model_cols:
        print("  No new sentiment model predictions found in the CSV.")
    else:
        expected_sent = df_ann['sentiment attendu'].astype(str).str.strip().str.upper()
        valid_sent_mask = expected_sent.isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])

        if not valid_sent_mask.any():
            print("  No valid sentiment ground truth values found.")
        else:
            lang_groups = {}
            for col in sent_model_cols:
                lang = SENT_LANG_MAP.get(col, 'OTHER')
                lang_groups.setdefault(lang, []).append(col)

            for lang_label in ['AR', 'EN', 'FR', 'MULTI', 'OTHER']:
                cols = lang_groups.get(lang_label, [])
                if not cols:
                    continue
                print(f"\n  [{lang_label}]")
                for col in cols:
                    model_preds = df_ann[col].astype(str).str.strip().str.upper()
                    eval_mask = (
                        valid_sent_mask &
                        (model_preds != 'NAN') &
                        (model_preds != '') &
                        model_preds.notna()
                    )
                    y_true = expected_sent[eval_mask].tolist()
                    y_pred = model_preds[eval_mask].tolist()
                    model_name = col.replace('_pred', '')
                    print_classification_metrics(y_true, y_pred, model_name)

    # -------------------------------------------------------
    # --- Topic Models: Accuracy + Precision/Recall/F1 ---
    # -------------------------------------------------------
    print("\n--- Topic Models Performance (Accuracy + Precision + Recall + F1) ---")

    topic_model_cols = [c for c in df_ann.columns if c.endswith('_topic_pred')]

    if not topic_model_cols:
        print("  No new topic model predictions found in the CSV.")
    else:
        expected_topic = df_ann['thème attendu'].astype(str).str.strip()
        valid_topic_mask = (
            (expected_topic != '') &
            (expected_topic != 'nan') &
            expected_topic.notna()
        )

        if not valid_topic_mask.any():
            print("  No valid topic ground truth values found.")
        else:
            lang_groups = {}
            for col in topic_model_cols:
                lang = TOPIC_LANG_MAP.get(col, 'OTHER')
                lang_groups.setdefault(lang, []).append(col)

            for lang_label in ['AR', 'EN', 'FR', 'MULTI', 'OTHER']:
                cols = lang_groups.get(lang_label, [])
                if not cols:
                    continue
                print(f"\n  [{lang_label}]")
                for col in cols:
                    model_preds = df_ann[col].astype(str).str.strip()
                    eval_mask = (
                        valid_topic_mask &
                        (model_preds != 'nan') &
                        (model_preds != '') &
                        model_preds.notna()
                    )
                    y_true = expected_topic[eval_mask].tolist()
                    y_pred = model_preds[eval_mask].tolist()
                    model_name = col.replace('_topic_pred', '')
                    print_classification_metrics(y_true, y_pred, model_name)

    # -------------------------------------------------------
    # --- NER Models Statistics ---
    # -------------------------------------------------------
    print("\n--- New NER Models Statistics ---")
    ner_model_cols = [c for c in df_ann.columns if c.endswith('_ner_pred')]

    if not ner_model_cols:
        print("  No NER model predictions found in the CSV.")
    else:
        for col in ner_model_cols:
            model_preds = df_ann[col].astype(str).str.strip()
            valid_preds = model_preds[
                (model_preds != 'nan') & (model_preds != '') & model_preds.notna()
            ]
            total_processed = len(valid_preds)
            if total_processed > 0:
                with_entities = valid_preds[valid_preds != 'None'].count()
                total_entities = valid_preds[valid_preds != 'None'].apply(
                    lambda x: len(x.split(' | '))
                ).sum()
                avg_entities = total_entities / total_processed
                pct_with_entities = (with_entities / total_processed) * 100
                print(f"  {col.replace('_ner_pred', '')}:")
                print(f"    - Extracted entities from {with_entities}/{total_processed} texts ({pct_with_entities:.1f}%)")
                print(f"    - Total entities extracted: {total_entities}")
                print(f"    - Average entities per text: {avg_entities:.2f}")
            else:
                print(f"  {col.replace('_ner_pred', '')}: N/A (no valid predictions found)")

    # -------------------------------------------------------
    # --- NER Comparison (GLiNER vs Language-Specific) ---
    # -------------------------------------------------------
    print("\n--- NER Model Comparison (GLiNER vs Language-Specific) ---")
    if 'langue' in df_ann.columns:
        for lang in df_ann['langue'].unique():
            lang_df = df_ann[df_ann['langue'] == lang]
            if lang_df.empty:
                continue
            print(f"\n  Language: {str(lang).upper()} ({len(lang_df)} texts)")

            gliner_col = 'gliner_multi-v2.1_ner_pred'
            other_ner_cols = [
                c for c in lang_df.columns
                if c.endswith('_ner_pred') and c != gliner_col
            ]

            if gliner_col in lang_df.columns:
                g_valid = lang_df[gliner_col].astype(str).str.strip()
                g_non_none = g_valid[
                    (g_valid != 'None') & (g_valid != 'nan') & (g_valid != '')
                ]
                g_ents = g_non_none.apply(lambda x: len(str(x).split(' | '))).sum()
                print(f"    - GLiNER: {g_ents} entities (Avg: {g_ents/len(lang_df):.2f})")

            for col in other_ner_cols:
                m_valid = lang_df[col].astype(str).str.strip()
                lang_valid_mask = (m_valid != 'nan') & (m_valid != '') & m_valid.notna()
                if lang_valid_mask.any():
                    m_non_none = m_valid[lang_valid_mask & (m_valid != 'None')]
                    m_ents = m_non_none.apply(lambda x: len(x.split(' | '))).sum()
                    print(f"    - {col.replace('_ner_pred', '')}: {m_ents} entities (Avg: {m_ents/len(lang_df):.2f})")

    print("=" * 70)


if __name__ == "__main__":
    print_stats()