# src/config/db.py
"""
Database connection configuration.
"""

from __future__ import annotations

from urllib.parse import quote_plus
from .base import getenv


class DatabaseConfig:
    DB_HOST = getenv("DB_HOST", "localhost")
    DB_PORT = getenv("DB_PORT", "3306")
    DB_USER = getenv("DB_USER", "root")
    DB_PASSWORD = getenv("DB_PASSWORD", "")
    DB_NAME = getenv("DB_NAME", "nlp_mena")

    _DB_PASSWORD_ESCAPED = quote_plus(DB_PASSWORD) if DB_PASSWORD else None

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{_DB_PASSWORD_ESCAPED}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    )

    DB_POOL_SIZE = getenv("DB_POOL_SIZE", 10, int)
    DB_MAX_OVERFLOW = getenv("DB_MAX_OVERFLOW", 20, int)
    DB_POOL_RECYCLE = getenv("DB_POOL_RECYCLE", 3600, int)
    RAW_TABLE = getenv("RAW_TABLE", "article")
