import pandas as pd
import torch
from transformers import pipeline
from pathlib import Path
import sys
import io
import contextlib
from sklearn.metrics import classification_report
import unicodedata

# Force stdout to UTF-8
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.config import Config
from src.ner_extraction import TransformersNER, GLiNERNER

CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_global.csv"

# --- CATEGORY MAPPING (For Topic Models) ---
CATEGORY_DISPLAY = {
    0: {"ar": "السياسة",       "fr": "Politique",      "en": "Politics"},
    1: {"ar": "الاقتصاد",      "fr": "Économie",       "en": "Economy"},
    2: {"ar": "الأمن",         "fr": "Sécurité",       "en": "Security"},
    3: {"ar": "الطاقة",        "fr": "Énergie",        "en": "Energy"},
    4: {"ar": "النزاع",        "fr": "Conflit",        "en": "Conflict"},
    5: {"ar": "الانتخابات",    "fr": "Élections",      "en": "Elections"},
    6: {"ar": "العدالة",       "fr": "Justice",        "en": "Justice"},
    7: {"ar": "الصحة",         "fr": "Santé",          "en": "Health"},
    8: {"ar": "الطقس",         "fr": "Météo",          "en": "Weather"},
    9: {"ar": "الرياضة",       "fr": "Sport",          "en": "Sports"},
    10: {"ar": "الثقافة",      "fr": "Culture",        "en": "Culture"},
    11: {"ar": "التعليم",      "fr": "Éducation",      "en": "Education"},
    12: {"ar": "التكنولوجيا",  "fr": "Technologie",    "en": "Technology"},
    13: {"ar": "البيئة",       "fr": "Environnement",  "en": "Environment"},
    14: {"ar": "الدبلوماسية",  "fr": "Diplomatie",     "en": "Diplomacy"},
    15: {"ar": "الدين",        "fr": "Religion",       "en": "Religion"},
    16: {"ar": "الهجرة",       "fr": "Migration",      "en": "Migration"},
    17: {"ar": "عام",         "fr": "Général",          "en": "General"},
}

def get_topic_labels(lang):
    return [cats[lang] for cats in CATEGORY_DISPLAY.values()]

# --- CANONICAL TOPIC MAPPING HELPERS ---
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii.strip().lower()

# Build the pre-built canonical mapping
TOPIC_CANONICAL_MAP = {}
for canonical_id, lang_dict in CATEGORY_DISPLAY.items():
    for lang, val in lang_dict.items():
        TOPIC_CANONICAL_MAP[normalize_text(val)] = canonical_id

# Synonym/custom overrides
TOPIC_SYNONYMS = {
    normalize_text("العدل"): 6,
    normalize_text("العدالة"): 6,
}
for k, v in TOPIC_SYNONYMS.items():
    TOPIC_CANONICAL_MAP[k] = v

def get_canonical_topic_id(label):
    if not isinstance(label, str):
        return -1
    norm = normalize_text(label)
    return TOPIC_CANONICAL_MAP.get(norm, -1)

# --- SENTIMENT HELPER ---
MODEL_LABEL_MAP = {
    'camelbert-msa-sentiment': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL'},
    'camelbert-da-sentiment': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL'},
    'camelbert-mix-sentiment': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL'},
    'AraBert-Arabic-Sentiment-Analysis': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL', 'LABEL_3': 'NEUTRAL'},
    'twitter-roberta-base-sentiment-latest': {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'},
    'sentiment-roberta-large-english-3-classes': {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'},
    'bertweet-base-sentiment-analysis': {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'},
    'distilcamembert-base-sentiment': {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL', 'LABEL_3': 'POSITIVE', 'LABEL_4': 'POSITIVE'},
    'bert-base-multilingual-uncased-sentiment': {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL', 'LABEL_3': 'POSITIVE', 'LABEL_4': 'POSITIVE'},
    'distilcamembert-base-nli': {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'POSITIVE', 'LABEL_2': 'NEUTRAL'},
    'twitter-xlm-roberta-base-sentiment': {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'},
    'distilbert-base-multilingual-cased-sentiments-student': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'NEGATIVE'},
}

def standardize_sentiment(label, model_name=None):
    if not isinstance(label, str): return "NEUTRAL"
    l_upper = label.upper().strip()
    
    # Model-specific mapping for generic sequence labels or custom mappings
    if l_upper.startswith("LABEL_") and model_name is not None:
        for pattern, mapping in MODEL_LABEL_MAP.items():
            if pattern.lower() in model_name.lower():
                return mapping.get(l_upper, "NEUTRAL")
                
    # Specific NLI / custom mappings if raw strings were returned instead of LABEL_x
    if model_name is not None and 'distilcamembert-base-nli' in model_name.lower():
        if l_upper == 'CONTRADICTION': return 'NEGATIVE'
        if l_upper == 'ENTAILMENT': return 'POSITIVE'
        if l_upper == 'NEUTRAL': return 'NEUTRAL'
        
    l = label.lower()
    if 'star' in l:
        if '1' in l or '2' in l: return 'NEGATIVE'
        if '3' in l: return 'NEUTRAL'
        return 'POSITIVE'
    if 'pos' in l: return 'POSITIVE'
    if 'neg' in l: return 'NEGATIVE'
    if 'neu' in l or 'mixed' in l: return 'NEUTRAL'
    return "NEUTRAL"

# --- NER HELPERS ---
def format_entities(entities) -> str:
    """Convert a List[NEREntity] from ner_extraction into the CSV string format."""
    if not entities:
        return "None"
    parts = [f"{e.text} ({e.label})" for e in entities]
    return " | ".join(parts)

# --- STATISTICS & EVALUATION REPORTING ---
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

def generate_report_content(df):
    print("=" * 70)
    print("      MANUAL ANNOTATION & MODEL STATISTICS")
    print("=" * 70)

    total_rows = len(df)
    print(f"Total samples: {total_rows}")

    req_cols = ["thème attendu", "sentiment attendu", "correct/incorrect theme", "correct/incorrect sentiment"]
    for col in req_cols:
        if col not in df.columns:
            print(f"\nMissing column '{col}'. Cannot compute full stats.")
            return

    df['is_annotated'] = df['thème attendu'].astype(str).str.strip() != ''
    df['is_annotated'] = df['is_annotated'] & df['thème attendu'].notna() & (df['thème attendu'].astype(str).str.lower() != 'nan')
    annotated_count = df['is_annotated'].sum()

    print(f"Annotated samples: {annotated_count} ({annotated_count/total_rows*100:.1f}%)")
    print(f"Unannotated samples: {total_rows - annotated_count} ({(total_rows - annotated_count)/total_rows*100:.1f}%)")

    if annotated_count == 0:
        print("\nNo samples have been annotated yet.")
        return

    df_ann = df[df['is_annotated']].copy()

    # --- Arabic Subsets Identification ---
    ar_df = df_ann[df_ann['langue'] == 'ar']
    msa_df = ar_df.iloc[:150]
    dial_df = ar_df.iloc[150:]

    # Helper function to get model language group based on its column name
    def get_model_lang(col_name):
        c_low = col_name.lower()
        if any(w in c_low for w in ['arabic', 'arabert', 'camelbert', 'ar_ner', 'camel']):
            return 'AR'
        if any(w in c_low for w in ['french', 'camembert', 'distilcamembert']):
            return 'FR'
        if any(w in c_low for w in ['english', 'bertweet', 'roberta-large-ner', 'base-ner', 'large-ner']):
            return 'EN'
        if any(w in c_low for w in ['multi', 'wikineural', 'gliner', 'multilingual']):
            return 'MULTI'
        return 'OTHER'

    # --- Language Breakdown ---
    if 'langue' in df_ann.columns:
        print("\n--- Language Breakdown ---")
        lang_counts = df_ann['langue'].value_counts()
        for lang, count in lang_counts.items():
            if lang == 'ar':
                print(f"  AR (MSA): {len(msa_df)} ({len(msa_df)/annotated_count*100:.1f}%)")
                print(f"  AR (Dialectal): {len(dial_df)} ({len(dial_df)/annotated_count*100:.1f}%)")
                print(f"  AR (Total): {count} ({count/annotated_count*100:.1f}%)")
            else:
                print(f"  {str(lang).upper()}: {count} ({count/annotated_count*100:.1f}%)")

    # --- Original Pipeline Theme Accuracy ---
    print("\n--- Original Pipeline Theme Prediction Accuracy ---")
    eval_groups = [
        ("Global", df_ann),
        ("AR (MSA)", msa_df),
        ("AR (Dialectal)", dial_df),
        ("AR (Total)", ar_df),
        ("EN", df_ann[df_ann['langue'] == 'en']),
        ("FR", df_ann[df_ann['langue'] == 'fr'])
    ]
    for name, sub in eval_groups:
        if len(sub) == 0:
            continue
        correct = (sub['correct/incorrect theme'].astype(str).str.strip().str.lower() == 'correct').sum()
        incorrect = (sub['correct/incorrect theme'].astype(str).str.strip().str.lower() == 'incorrect').sum()
        total = correct + incorrect
        if total > 0:
            print(f"  [{name}]:")
            print(f"    Correct: {correct} ({correct/total*100:.1f}%)")
            print(f"    Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")

    # --- Original Pipeline Sentiment Accuracy ---
    print("\n--- Original Pipeline Sentiment Prediction Accuracy ---")
    for name, sub in eval_groups:
        if len(sub) == 0:
            continue
        correct = (sub['correct/incorrect sentiment'].astype(str).str.strip().str.lower() == 'correct').sum()
        incorrect = (sub['correct/incorrect sentiment'].astype(str).str.strip().str.lower() == 'incorrect').sum()
        total = correct + incorrect
        if total > 0:
            print(f"  [{name}]:")
            print(f"    Correct: {correct} ({correct/total*100:.1f}%)")
            print(f"    Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")

    # --- Expected Theme Distribution ---
    print("\n--- Expected Theme Distribution (Top 5) ---")
    for name, sub in [("Global", df_ann), ("AR (MSA)", msa_df), ("AR (Dialectal)", dial_df), ("AR (Total)", ar_df), ("EN", df_ann[df_ann['langue'] == 'en']), ("FR", df_ann[df_ann['langue'] == 'fr'])]:
        if len(sub) == 0:
            continue
        print(f"  [{name}]:")
        theme_counts = sub['thème attendu'].astype(str).str.strip().value_counts().head(5)
        for theme, count in theme_counts.items():
            if theme and theme.lower() != 'nan':
                print(f"    {theme}: {count} ({count/len(sub)*100:.1f}%)")

    # --- Expected Sentiment Distribution ---
    print("\n--- Expected Sentiment Distribution ---")
    for name, sub in [("Global", df_ann), ("AR (MSA)", msa_df), ("AR (Dialectal)", dial_df), ("AR (Total)", ar_df), ("EN", df_ann[df_ann['langue'] == 'en']), ("FR", df_ann[df_ann['langue'] == 'fr'])]:
        if len(sub) == 0:
            continue
        print(f"  [{name}]:")
        sent_counts = sub['sentiment attendu'].astype(str).str.strip().str.upper().value_counts()
        for sent, count in sent_counts.items():
            if sent and sent.lower() != 'nan':
                print(f"    {sent}: {count} ({count/len(sub)*100:.1f}%)")

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
                lang = SENT_LANG_MAP.get(col, get_model_lang(col))
                lang_groups.setdefault(lang, []).append(col)

            for lang_label in ['AR', 'EN', 'FR', 'MULTI', 'OTHER']:
                cols = lang_groups.get(lang_label, [])
                if not cols:
                    continue
                
                if lang_label == 'AR':
                    for sub_name, sub_df in [("AR (MSA)", msa_df), ("AR (Dialectal)", dial_df), ("AR (Total)", ar_df)]:
                        if len(sub_df) == 0:
                            continue
                        print(f"\n  [{sub_name}]")
                        for col in cols:
                            sub_expected = sub_df['sentiment attendu'].astype(str).str.strip().str.upper()
                            sub_valid = sub_expected.isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
                            model_preds = sub_df[col].astype(str).str.strip().str.upper()
                            eval_mask = (
                                sub_valid &
                                (model_preds != 'NAN') &
                                (model_preds != '') &
                                model_preds.notna()
                            )
                            y_true = sub_expected[eval_mask].tolist()
                            y_pred = model_preds[eval_mask].tolist()
                            model_name = col.replace('_pred', '')
                            print_classification_metrics(y_true, y_pred, model_name)
                else:
                    print(f"\n  [{lang_label}]")
                    for col in cols:
                        model_preds = df_ann[col].astype(str).str.strip().str.upper()
                        eval_mask = (
                            valid_sent_mask &
                            (model_preds != 'NAN') &
                            (model_preds != '') &
                            model_preds.notna()
                        )
                        # Filter by language group
                        if lang_label in ['EN', 'FR']:
                            eval_mask = eval_mask & (df_ann['langue'] == lang_label.lower())
                        
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
        topic_groups = [
            ("AR (MSA)", msa_df),
            ("AR (Dialectal)", dial_df),
            ("AR (Total)", ar_df),
            ("EN", df_ann[df_ann['langue'] == 'en']),
            ("FR", df_ann[df_ann['langue'] == 'fr'])
        ]
        
        for group_name, group_df in topic_groups:
            if len(group_df) == 0:
                continue
            
            # Determine which models should run on this language group
            applicable_cols = []
            for col in topic_model_cols:
                model_target = TOPIC_LANG_MAP.get(col, get_model_lang(col))
                if "AR" in group_name and model_target == 'MULTI':
                    applicable_cols.append(col)
                elif group_name == 'EN' and model_target in ['EN', 'MULTI']:
                    applicable_cols.append(col)
                elif group_name == 'FR' and model_target in ['FR', 'MULTI']:
                    applicable_cols.append(col)
            
            if not applicable_cols:
                continue
                
            print(f"\n  [{group_name}]")
            for col in applicable_cols:
                expected_topic = group_df['thème attendu'].astype(str).str.strip()
                valid_topic_mask = (
                    (expected_topic != '') &
                    (expected_topic != 'nan') &
                    expected_topic.notna()
                )
                model_preds = group_df[col].astype(str).str.strip()
                eval_mask = (
                    valid_topic_mask &
                    (model_preds != 'nan') &
                    (model_preds != '') &
                    model_preds.notna()
                )
                y_true_raw = expected_topic[eval_mask].tolist()
                y_pred_raw = model_preds[eval_mask].tolist()
                
                # Map to canonical topic IDs (0-17)
                y_true = [get_canonical_topic_id(yt) for yt in y_true_raw]
                y_pred = [get_canonical_topic_id(yp) for yp in y_pred_raw]
                
                # Warning debug logs for unmapped values
                unmapped_true = [yt for yt in y_true_raw if get_canonical_topic_id(yt) == -1]
                unmapped_pred = [yp for yp in y_pred_raw if get_canonical_topic_id(yp) == -1]
                if unmapped_true:
                    print(f"    WARNING: Unmapped expected labels: {set(unmapped_true)}")
                if unmapped_pred:
                    print(f"    WARNING: Unmapped predicted labels: {set(unmapped_pred)}")
                
                # Filter out unmapped pairs
                valid_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != -1 and yp != -1]
                if valid_pairs:
                    y_true, y_pred = zip(*valid_pairs)
                    y_true, y_pred = list(y_true), list(y_pred)
                else:
                    y_true, y_pred = [], []
                
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
            model_lang = get_model_lang(col)
            
            subsets_to_eval = []
            if model_lang in ['AR', 'MULTI']:
                subsets_to_eval.extend([
                    ("AR (MSA)", msa_df),
                    ("AR (Dialectal)", dial_df),
                    ("AR (Total)", ar_df)
                ])
            if model_lang in ['EN', 'MULTI']:
                subsets_to_eval.append(("EN", df_ann[df_ann['langue'] == 'en']))
            if model_lang in ['FR', 'MULTI']:
                subsets_to_eval.append(("FR", df_ann[df_ann['langue'] == 'fr']))
                
            print(f"  {col.replace('_ner_pred', '')}:")
            for sub_name, sub_df in subsets_to_eval:
                if len(sub_df) == 0:
                    continue
                model_preds = sub_df[col].astype(str).str.strip()
                valid_preds = model_preds[
                    (model_preds != 'nan') & (model_preds != '') & model_preds.notna()
                ]
                total_processed = len(valid_preds)
                if total_processed > 0:
                    with_entities = valid_preds[valid_preds != 'None'].count()
                    total_entities = valid_preds[valid_preds != 'None'].apply(
                        lambda x: len(str(x).split(' | '))
                    ).sum()
                    avg_entities = total_entities / total_processed
                    pct_with_entities = (with_entities / total_processed) * 100
                    print(f"    [{sub_name}]:")
                    print(f"      - Extracted entities from {with_entities}/{total_processed} texts ({pct_with_entities:.1f}%)")
                    print(f"      - Total entities extracted: {total_entities}")
                    print(f"      - Average entities per text: {avg_entities:.2f}")
                else:
                    print(f"    [{sub_name}]: N/A (no predictions)")

    # -------------------------------------------------------
    # --- NER Comparison (GLiNER vs Language-Specific) ---
    # -------------------------------------------------------
    print("\n--- NER Model Comparison (GLiNER vs Language-Specific) ---")
    if 'langue' in df_ann.columns:
        comp_groups = [
            ("AR (MSA)", msa_df),
            ("AR (Dialectal)", dial_df),
            ("AR (Total)", ar_df),
            ("EN", df_ann[df_ann['langue'] == 'en']),
            ("FR", df_ann[df_ann['langue'] == 'fr'])
        ]
        
        for group_name, group_df in comp_groups:
            if len(group_df) == 0:
                continue
            print(f"\n  Language: {group_name} ({len(group_df)} texts)")

            gliner_col = 'gliner_multi-v2.1_ner_pred'
            other_ner_cols = [
                c for c in group_df.columns
                if c.endswith('_ner_pred') and c != gliner_col
            ]
            
            # Filter other models appropriate for this group's language
            if "AR" in group_name:
                other_ner_cols = [c for c in other_ner_cols if get_model_lang(c) in ['AR', 'MULTI']]
            elif group_name == 'EN':
                other_ner_cols = [c for c in other_ner_cols if get_model_lang(c) in ['EN', 'MULTI']]
            elif group_name == 'FR':
                other_ner_cols = [c for c in other_ner_cols if get_model_lang(c) in ['FR', 'MULTI']]

            if gliner_col in group_df.columns:
                g_valid = group_df[gliner_col].astype(str).str.strip()
                g_non_none = g_valid[
                    (g_valid != 'None') & (g_valid != 'nan') & (g_valid != '')
                ]
                g_ents = g_non_none.apply(lambda x: len(str(x).split(' | '))).sum()
                print(f"    - GLiNER: {g_ents} entities (Avg: {g_ents/len(group_df):.2f})")

            for col in other_ner_cols:
                m_valid = group_df[col].astype(str).str.strip()
                lang_valid_mask = (m_valid != 'nan') & (m_valid != '') & m_valid.notna()
                if lang_valid_mask.any():
                    m_non_none = m_valid[lang_valid_mask & (m_valid != 'None')]
                    m_ents = m_non_none.apply(lambda x: len(str(x).split(' | '))).sum()
                    print(f"    - {col.replace('_ner_pred', '')}: {m_ents} entities (Avg: {m_ents/len(group_df):.2f})")

    # -------------------------------------------------------
    # --- Resource Usage & Timing Benchmarks ---
    # -------------------------------------------------------
    benchmark_csv = PROJECT_ROOT / "scripts" / "ressources" / "resource_usage_report.csv"
    if benchmark_csv.exists():
        print("\n" + "=" * 70)
        print("      RESOURCE USAGE & TIMING BENCHMARKS")
        print("=" * 70)
        try:
            bench_df = pd.read_csv(benchmark_csv)
            for device in ['CPU', 'GPU']:
                device_df = bench_df[bench_df['device'] == device]
                if device_df.empty:
                    continue
                print(f"\n--- {device} Execution Performance & Memory Footprint ---")
                
                for task in ['topic', 'sentiment', 'ner']:
                    task_df = device_df[device_df['task'] == task]
                    if task_df.empty:
                        continue
                    print(f"\n  [{task.upper()} Models]")
                    print(f"    {'Model':<48} | {'Lang':<4} | {'Load Time':<10} | {'Avg/Doc':<10} | {'Peak CPU':<10} | {'Peak GPU':<10}")
                    print(f"    {'-'*48}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
                    for _, row in task_df.iterrows():
                        model_short = str(row['model']).replace('MoritzLaurer/', '').replace('facebook/', '').replace('morit/', '').replace('cardiffnlp/', '').replace('cmarkea/', '').replace('dslim/', '').replace('Jean-Baptiste/', '').replace('urchade/', '').replace('hatmimoha/', '')
                        load_time = f"{row['load_time_sec']:.2f}s"
                        avg_inf = f"{row['avg_ms_per_doc']:.1f}ms"
                        cpu_mem = f"{row['peak_cpu_mb']:.1f}MB"
                        gpu_mem = f"{row['peak_gpu_mb']:.1f}MB"
                        print(f"    {model_short:<48} | {str(row['lang']).upper():<4} | {load_time:<10} | {avg_inf:<10} | {cpu_mem:<10} | {gpu_mem:<10}")

            # Summary of CPU vs GPU Speedup if both are available
            cpu_inf = bench_df[bench_df['device'] == 'CPU']
            gpu_inf = bench_df[bench_df['device'] == 'GPU']
            if not cpu_inf.empty and not gpu_inf.empty:
                print("\n--- CPU vs GPU Hardware Acceleration Analysis ---")
                # Merge on task, model, lang
                merged = pd.merge(cpu_inf, gpu_inf, on=['task', 'model', 'lang'], suffixes=('_cpu', '_gpu'))
                for _, row in merged.iterrows():
                    model_short = str(row['model']).replace('MoritzLaurer/', '').replace('facebook/', '').replace('morit/', '').replace('cardiffnlp/', '').replace('cmarkea/', '').replace('dslim/', '').replace('Jean-Baptiste/', '').replace('urchade/', '').replace('hatmimoha/', '')
                    speedup = row['avg_ms_per_doc_cpu'] / row['avg_ms_per_doc_gpu'] if row['avg_ms_per_doc_gpu'] > 0 else 0
                    print(f"  * {row['task'].upper()} | {model_short} ({str(row['lang']).upper()}): GPU is {speedup:.1f}x faster than CPU")
        except Exception as e:
            print(f"Error reading resource usage benchmarks: {e}")

    print("=" * 70)

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    print(f"Loading CSV: {CSV_PATH.name}")
    df = pd.read_csv(CSV_PATH)
    df['langue'] = df['langue'].astype(str).str.lower()
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # --- MODEL CONFIGURATION ---
    # Organized by task -> language -> models
    tasks = {
        'topic': {
            'en': [
                'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', 
                'roberta-large-mnli',
                'facebook/bart-large-mnli',
                'cross-encoder/nli-deberta-v3-large'
            ],
            'fr': ['morit/french_xlm_xnli', 'cmarkea/distilcamembert-base-nli'],
            'multi': [
                'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7', 
                'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
                'joeddav/xlm-roberta-large-xnli'
            ]
        },
        'sentiment': {
            'ar': [
                'CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment', 
                'CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment',
                'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment',
                'PRAli22/AraBert-Arabic-Sentiment-Analysis'
            ],
            'en': [
                'cardiffnlp/twitter-roberta-base-sentiment-latest', 
                'j-hartmann/sentiment-roberta-large-english-3-classes',
                'finiteautomata/bertweet-base-sentiment-analysis'
            ],
            'fr': [
                'cmarkea/distilcamembert-base-sentiment', 
                'nlptown/bert-base-multilingual-uncased-sentiment',
                'cmarkea/distilcamembert-base-nli'
                ],
            'multi': [
                'cardiffnlp/twitter-xlm-roberta-base-sentiment', 
                'lxyuan/distilbert-base-multilingual-cased-sentiments-student'
            ]
        },
        'ner': {
            'ar': [
                Config.ARABERT_NER_MODEL, 
                Config.CAMEL_NER_MODEL,
                'CAMeL-Lab/bert-base-arabic-camelbert-mix-ner',
                'MostafaAhmed98/AraBert-Arabic-NER-CoNLLpp'
            ],
            'en': [
                Config.EN_NER_MODEL,
                'dslim/bert-large-NER',
                'Jean-Baptiste/roberta-large-ner-english'
            ],
            'fr': [
                Config.FR_NER_MODEL,
                'cmarkea/distilcamembert-base-ner'
            ],
            'multi': [
                Config.GLINER_MODEL,
                'Davlan/bert-base-multilingual-cased-ner-hrl',
                'Babelscape/wikineural-multilingual-ner'
            ]
        }
    }

    # Set these to True if you want to force-recompute/overwrite already processed columns in the CSV
    FORCE_RERUN_TOPIC = False
    FORCE_RERUN_SENTIMENT = False
    FORCE_RERUN_NER = True

    # --- HYPOTHESIS TEMPLATES FOR TOPIC MODELING ---
    HYPOTHESIS_TEMPLATES = {
        'en': "This news article is about {}.",
        'fr': "Cet article de presse concerne {}.",
        'ar': "هذا المقال الإخباري يتحدث عن {}."
    }

    # --- EXECUTION ENGINE ---
    for task_name, lang_dict in tasks.items():
        print(f"\n{'='*20} TASK: {task_name.upper()} {'='*20}")
        
        for lang, model_list in lang_dict.items():
            for model_name in model_list:
                suffix = "topic_pred" if task_name == 'topic' else ("pred" if task_name == 'sentiment' else "ner_pred")
                col_name = f"{model_name.split('/')[-1]}_{suffix}"
                
                # Check which rows need processing for THIS model
                if col_name not in df.columns:
                    df[col_name] = None
                
                # Determine which indices need to be processed
                if (task_name == 'topic' and FORCE_RERUN_TOPIC) or \
                   (task_name == 'sentiment' and FORCE_RERUN_SENTIMENT) or \
                   (task_name == 'ner' and FORCE_RERUN_NER):
                    is_missing = pd.Series(True, index=df.index)
                else:
                    is_missing = (
                        df[col_name].isna() |
                        (df[col_name].astype(str).str.strip() == "") |
                        (df[col_name].astype(str).str.strip().str.lower() == "nan")
                    )
                    
                    # Reprocess sentiment: also process rows where the value is 'NEUTRAL' to fix previous mapping errors
                    if task_name == 'sentiment':
                        is_missing = is_missing | (df[col_name].astype(str).str.strip().str.upper() == "NEUTRAL")
                
                target_indices = df.index[is_missing].tolist()
                
                # If lang is specific (ar, en, fr), filter by that language
                if lang != 'multi':
                    target_indices = [i for i in target_indices if df.loc[i, 'langue'] == lang]
                
                if not target_indices:
                    print(f" -> [{lang.upper()}] Skipping {model_name} (all rows already processed).")
                    continue
                
                print(f" -> [{lang.upper()}] Running {model_name} on {len(target_indices)} rows...")
                
                try:
                    if task_name == 'topic':
                        pipe = pipeline("zero-shot-classification", model=model_name, device=device)
                        for idx in target_indices:
                            try:
                                row_lang = df.loc[idx, 'langue']
                                labels = get_topic_labels(row_lang if row_lang in ['ar', 'en', 'fr'] else 'en')
                                hypothesis_lang = row_lang if row_lang in ['ar', 'en', 'fr'] else 'en'
                                template = HYPOTHESIS_TEMPLATES[hypothesis_lang]
                                res = pipe(str(df.loc[idx, 'texte'])[:1500], candidate_labels=labels, hypothesis_template=template)
                                df.loc[idx, col_name] = res['labels'][0]
                            except Exception as e:
                                print(f"      Row {idx} failed: {e}")
                            
                    elif task_name == 'sentiment':
                        m_len = 128 if 'bertweet' in model_name.lower() else 512
                        pipe = pipeline("sentiment-analysis", model=model_name, device=device, truncation=True, max_length=m_len)
                        for idx in target_indices:
                            try:
                                res = pipe(str(df.loc[idx, 'texte']))[0]
                                df.loc[idx, col_name] = standardize_sentiment(res['label'], model_name)
                            except Exception as e:
                                print(f"      Row {idx} failed: {e}")
                            
                    elif task_name == 'ner':
                        if 'gliner' in model_name.lower():
                            ner_model = GLiNERNER(model_name=model_name, device=device)
                        else:
                            ner_model = TransformersNER(model_name=model_name, device=device)
                        for idx in target_indices:
                            try:
                                row_lang = df.loc[idx, 'langue']
                                text = str(df.loc[idx, 'texte'])
                                if isinstance(ner_model, GLiNERNER):
                                    entities = ner_model.predict(text, language=row_lang)
                                else:
                                    entities = ner_model.predict(text)
                                df.loc[idx, col_name] = format_entities(entities)
                            except Exception as e:
                                print(f"      Row {idx} failed: {e}")
                    
                    # Save periodically to avoid losing progress
                    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
                    
                except Exception as e:
                    print(f"    Failed: {e}")

    print(f"\nAll models completed. Final results saved to {CSV_PATH}")

    # --- REPORT GENERATION HOOK ---
    try:
        print("\nGenerating comprehensive report.txt...")
        import io
        
        report_path = PROJECT_ROOT / "scripts" / "report.txt"
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            generate_report_content(df)
        report_content = f.getvalue()
        
        with open(report_path, "w", encoding="utf-8") as rep_f:
            rep_f.write(report_content)
        
        print(f"Report successfully generated and saved to {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
