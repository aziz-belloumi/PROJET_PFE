import pandas as pd
import torch
from transformers import pipeline
from pathlib import Path
import sys
import io

# Force stdout to UTF-8
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.config import Config
from src.ner_extraction import TransformersNER, GLiNERNER

CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_sample_topic.csv"

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

# --- SENTIMENT HELPER ---
def standardize_sentiment(label):
    if not isinstance(label, str): return "NEUTRAL"
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
                
                # Condition: only process rows where col_name is NaN. 
                # Focusing on 'fr' as requested, but logic applies to all if NaN.
                target_indices = df.index[df[col_name].isna()].tolist()
                
                # If lang is specific (ar, en, fr), filter by that language
                if lang != 'multi':
                    target_indices = [i for i in target_indices if df.loc[i, 'langue'] == lang]
                
                if not target_indices:
                    continue
                
                print(f" -> [{lang.upper()}] Running {model_name} on {len(target_indices)} rows...")
                
                try:
                    if task_name == 'topic':
                        pipe = pipeline("zero-shot-classification", model=model_name, device=device)
                        for idx in target_indices:
                            try:
                                row_lang = df.loc[idx, 'langue']
                                labels = get_topic_labels(row_lang if row_lang in ['ar', 'en', 'fr'] else 'en')
                                res = pipe(str(df.loc[idx, 'texte'])[:1500], candidate_labels=labels)
                                df.loc[idx, col_name] = res['labels'][0]
                            except Exception as e:
                                print(f"      Row {idx} failed: {e}")
                            
                    elif task_name == 'sentiment':
                        m_len = 128 if 'bertweet' in model_name.lower() else 512
                        pipe = pipeline("sentiment-analysis", model=model_name, device=device, truncation=True, max_length=m_len)
                        for idx in target_indices:
                            try:
                                res = pipe(str(df.loc[idx, 'texte']))[0]
                                df.loc[idx, col_name] = standardize_sentiment(res['label'])
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

if __name__ == "__main__":
    main()
