# Contains all configuration parameters for the NLP pipeline.
# Loads environment variables from .env file.



# src/config.py
# Centralized configuration for the project.
# Loads environment variables from the .env located at the project root.
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
    ARABERT_NER_MODEL = "hatmimoha/arabic-ner"
    CAMEL_NER_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"


    CAMEL_SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"


    # ========================================
    # Logging defaults (used by Experiment)
    # ========================================
    LOG_LEVEL = _getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = _getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
    LOG_DATE_FORMAT = _getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

    # ========================================
    # Arabic Stopwords (keywords)
    # ========================================
    ARABIC_STOPWORDS = [
        "في", "من", "إلى", "على", "هذا", "هذه", "ذلك", "تلك",
        "الذي", "التي", "الذين", "اللتان", "اللواتي",
        "كان", "كانت", "كانوا", "يكون", "تكون",
        "أن", "إن", "لكن", "لأن", "كما", "عند", "لدى",
        "بعد", "قبل", "حول", "خلال", "ضد", "مع", "بين",
        "عن", "منذ", "حتى", "لو", "لولا", "أم", "أو",
        "هل", "ما", "لا", "لم", "لن", "ليس", "ليست",
        "قد", "قال", "قالت", "كل", "بعض", "أي", "أية",
        "هنا", "هناك", "حيث", "عندما", "كيف", "لماذا",
        "نحن", "أنتم", "هم", "هن", "أنا", "أنت", "هو", "هي",
    ]