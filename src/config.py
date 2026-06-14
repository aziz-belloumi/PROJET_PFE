# NOTE: No static LOG_DIR / EXPORT_DIR creation here (experiments handle output paths dynamically).

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


# --- Resolve project root and load .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../NLP_MENA
if (PROJECT_ROOT / ".env.local").exists():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env.local")
else:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def _getenv(key: str, default=None, cast=None):
    val = os.getenv(key, default)
    if val is None:
        return None
    return cast(val) if cast else val


class Config:
    # ========================================
    # Project Paths (base only)
    # ========================================
    PROJECT_ROOT = str(PROJECT_ROOT)
    EXPERIMENTS_DIR = _getenv("EXPERIMENTS_DIR", "experiments")  # Experiment folders go here

    # ========================================
    # Database
    # ========================================
    DB_HOST = _getenv("DB_HOST")
    DB_PORT = _getenv("DB_PORT")
    DB_USER = _getenv("DB_USER")
    DB_PASSWORD = _getenv("DB_PASSWORD")
    DB_NAME = _getenv("DB_NAME")

    # Safer for special characters in passwords
    _DB_PASSWORD_ESCAPED = quote_plus(DB_PASSWORD) if DB_PASSWORD else None

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{_DB_PASSWORD_ESCAPED}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    )

    # ========================================
    # Processing Parameters
    # ========================================
    SAMPLE_SIZE = _getenv("SAMPLE_SIZE", 10_000, int)
    BATCH_SIZE = _getenv("BATCH_SIZE", 100, int)

    # models path
    FASTTEXT_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "lid.176.ftz")# for language detection

    # ========================================
    # NLP Models Configuration
    # ========================================
    # Arabic (keep)
    ARABERT_NER_MODEL = "hatmimoha/arabic-ner"
    CAMEL_NER_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"

    # Unified LLM model config name for Hugging Face multi-model sentiment and topic analysis
    LLM_MODEL = "HF-Transformers-Multilingual"

    SENTIMENT_MODELS = {
        "ar": "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment",
        "en": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "fr": "cmarkea/distilcamembert-base-sentiment",
    }

    TOPIC_MODELS = {
        "ar": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "en": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "fr": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    }

    # English 
    EN_NER_MODEL = "dslim/bert-base-NER"

    # French 
    FR_NER_MODEL = "Jean-Baptiste/camembert-ner"

    # Multilingual NER
    GLINER_MODEL = "urchade/gliner_multi-v2.1"


    # ========================================
    # Topic Extraction
    # ========================================

    # Multilingual topic category taxonomy (index → {lang: label})
    CATEGORY_DISPLAY = {
        0:  {"ar": "السياسة",       "fr": "Politique",      "en": "Politics"},
        1:  {"ar": "الاقتصاد",      "fr": "Économie",       "en": "Economy"},
        2:  {"ar": "الأمن",         "fr": "Sécurité",       "en": "Security"},
        3:  {"ar": "الطاقة",        "fr": "Énergie",        "en": "Energy"},
        4:  {"ar": "النزاع",        "fr": "Conflit",        "en": "Conflict"},
        5:  {"ar": "الانتخابات",    "fr": "Élections",      "en": "Elections"},
        6:  {"ar": "العدالة",       "fr": "Justice",        "en": "Justice"},
        7:  {"ar": "الصحة",         "fr": "Santé",          "en": "Health"},
        8:  {"ar": "الطقس",         "fr": "Météo",          "en": "Weather"},
        9:  {"ar": "الرياضة",       "fr": "Sport",          "en": "Sports"},
        10: {"ar": "الثقافة",      "fr": "Culture",        "en": "Culture"},
        11: {"ar": "التعليم",      "fr": "Éducation",      "en": "Education"},
        12: {"ar": "التكنولوجيا",  "fr": "Technologie",    "en": "Technology"},
        13: {"ar": "البيئة",       "fr": "Environnement",  "en": "Environment"},
        14: {"ar": "الدبلوماسية",  "fr": "Diplomatie",     "en": "Diplomacy"},
        15: {"ar": "الدين",        "fr": "Religion",       "en": "Religion"},
        16: {"ar": "الهجرة",       "fr": "Migration",      "en": "Migration"},
        17: {"ar": "عام",          "fr": "Général",        "en": "General"},
    }

    # Zero-shot NLI hypothesis templates per language
    HYPOTHESIS_TEMPLATES = {
        "en": "This news article is about {}.",
        "fr": "Cet article de presse concerne {}.",
        "ar": "هذا المقال الإخباري يتحدث عن {}.",
    }


    # ========================================
    # Pipeline Runtime Parameters
    # ========================================
    LANG_THRESHOLD    = _getenv("LANG_THRESHOLD",    0.51, float)
    CPU_DEVICE        = _getenv("CPU_DEVICE",        -1,   int)
    GPU_DEVICE        = _getenv("GPU_DEVICE",         0,   int)
    GPU_COOLDOWN_SEC  = _getenv("GPU_COOLDOWN_SEC",  1.0,  float)

    # ========================================
    # Text Chunking
    # ========================================
    CHUNK_SIZE        = _getenv("CHUNK_SIZE",         3000, int)
    CHUNK_OVERLAP     = _getenv("CHUNK_OVERLAP",       400, int)

    # ========================================
    # Topic Quality Gate
    # ========================================
    TOPIC_MIN_TEXT_CHARS = _getenv("TOPIC_MIN_TEXT_CHARS", 30,  int)
    TOPIC_MIN_TEXT_WORDS = _getenv("TOPIC_MIN_TEXT_WORDS",  5,  int)


    # ========================================
    # Logging defaults (used by Experiment)
    # ========================================
    LOG_LEVEL = _getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = _getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
    LOG_DATE_FORMAT = _getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")