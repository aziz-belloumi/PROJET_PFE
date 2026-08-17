# src/config/heuristics.py
"""
Sentiment domain rules, cues, question starters, and topic taxonomies.
"""

from __future__ import annotations

from typing import Dict, List, Set


class HeuristicsConfig:
    SENTIMENTS: List[str] = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

    AMBIGUOUS_SHORT_TEXT_MAX_WORDS: int = 8
    AMBIGUOUS_SHORT_TEXT_MAX_CHARS: int = 60

    NEGATIVE_DOMAIN_CUES: Dict[str, List[str]] = {
        "ar": ["تقشف", "البطالة", "بطالة", "عقوبات", "انهيار", "ركود", "خسائر", "ضحايا", "كارثة"],
        "en": ["austerity", "unemployment", "sanctions", "collapse", "recession", "losses", "casualties", "victims", "disaster"],
        "fr": ["austérité", "chômage", "sanctions", "effondrement", "récession", "pertes", "victimes", "catastrophe"],
    }

    QUESTION_STARTERS: Dict[str, Set[str]] = {
        "ar": {"من", "ما", "ماذا", "هل", "لماذا", "كيف", "أين", "متى", "أي", "كم"},
        "en": {"who", "what", "why", "how", "where", "when", "which", "is", "are", "do", "does", "did", "can", "could", "will", "would"},
        "fr": {"qui", "que", "quoi", "pourquoi", "comment", "où", "ou", "quand", "quel", "quelle", "quels", "quelles", "est-ce", "peut", "doit"},
    }

    # Multilingual topic category taxonomy (index → {lang: label})
    CATEGORY_DISPLAY: Dict[int, Dict[str, str]] = {
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
