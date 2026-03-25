# NOTE: No static LOG_DIR / EXPORT_DIR creation here (experiments handle output paths dynamically).

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


# --- Resolve project root and load .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../NLP_MENA
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

    ARABERT_SENTIMENT_MODEL = "PRAli22/AraBert-Arabic-Sentiment-Analysis"
    CAMEL_SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"

    # English 
    EN_NER_MODEL = "dslim/bert-base-NER"
    # EN_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EN_SENTIMENT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    # French 
    FR_NER_MODEL = "Jean-Baptiste/camembert-ner"
    FR_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"



    # Topic Modeling
    TOPIC_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ========================================
    # Pipeline Runtime Parameters
    # ========================================
    LANG_THRESHOLD    = _getenv("LANG_THRESHOLD",    0.51, float)
    CPU_DEVICE        = _getenv("CPU_DEVICE",        -1,   int)
    GPU_DEVICE        = _getenv("GPU_DEVICE",         0,   int)
    GPU_COOLDOWN_SEC  = _getenv("GPU_COOLDOWN_SEC",  1.0,  float)



    # ========================================
    # Logging defaults (used by Experiment)
    # ========================================
    LOG_LEVEL = _getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = _getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
    LOG_DATE_FORMAT = _getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")