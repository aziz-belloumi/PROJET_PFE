from __future__ import annotations

import logging
import re
import requests
import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TopicResult:
    """Holds the topic prediction for a single article."""
    label: str
    score: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400
MAX_RETRIES = 3

# Expanded multilingual topic label mapping
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
    17: {"ar": "عام",          "fr": "Général",        "en": "General"},
}

# Group allowed labels strictly by language to prevent cross-contamination leaks
ALLOWED_LABELS_BY_LANG = {
    "ar": {row["ar"] for row in CATEGORY_DISPLAY.values()},
    "fr": {row["fr"] for row in CATEGORY_DISPLAY.values()},
    "en": {row["en"] for row in CATEGORY_DISPLAY.values()},
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def chunk_article(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long article into overlapping chunks using character offsets."""
    chunks = []
    start = 0
    if not text:
        return chunks

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


def build_topic_prompt(language, article):
    """Build the prompt with strict structural rules and exclusive options."""
    table_lines = ["Arabic | French | English"]
    for row in CATEGORY_DISPLAY.values():
        table_lines.append(f"{row['ar']} | {row['fr']} | {row['en']}")
    label_table = "\n".join(table_lines)

    return f"""You are evaluating topic classification for news articles.

The article is written in: {language}

Article:
{article}

**Instructions:**
1. Choose the MOST appropriate general category from the table below.
2. You MUST select ONLY from the provided 18 categories. Do NOT suggest or invent new categories.
3. Return the label in {language} (matching the language of the article).
4. The category must match EXACTLY one of the labels from the table below.

**Available Categories (18 options only):**
{label_table}

**Output format (One single line only, no explanations):**
true_prediction: <exact label from table>"""


def call_llm(prompt):
    """Call local Ollama instance with fallback error safety hooks."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 2048,
            "temperature": 0,
            "num_predict": 30
        }
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=20
            )
            if response.status_code == 200:
                return response.json()["response"]
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("LLM request failed after retries")


def parse_llm_response(response_text):
    """Parses LLM block to safely extract only 'true_prediction'."""
    prediction = ""

    # Clean capture of prediction line irrespective of leading/trailing tags
    match_p = re.search(r"true_prediction:\s*(.*)", response_text, re.IGNORECASE)
    if match_p:
        prediction = match_p.group(1).strip()

    return prediction


def validate_prediction(prediction: str, logger: logging.Logger, lang: str = "en") -> tuple[str, float]:
    """Validate prediction strictly against valid items matching the exact source language."""
    valid_set = ALLOWED_LABELS_BY_LANG.get(lang, ALLOWED_LABELS_BY_LANG["en"])
    
    if prediction in valid_set:
        return prediction, 1.0
    
    # Language fallback fallback initialization
    fallback_label = CATEGORY_DISPLAY[17].get(lang, "General")
    logger.warning(f"[LLMTopic] Invalid or language-leaked prediction: '{prediction}' for lang '{lang}' - using fallback: '{fallback_label}'")
    return fallback_label, 0.0


# ---------------------------------------------------------------------------
# LLM Topic Extractor
# ---------------------------------------------------------------------------

class LLMTopic:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, text: str, lang: str) -> TopicResult:
        if not text or not text.strip():
            self.logger.warning("[LLMTopic] Received empty text; returning fallback.")
            fallback = CATEGORY_DISPLAY[17].get(lang, "General")
            return TopicResult(label=fallback, score=0.0)

        chunks = chunk_article(text)
        if not chunks:
            chunks = [text]

        lang_map = {"ar": "Arabic", "en": "English", "fr": "French"}
        language = lang_map.get(lang, "English")

        self.logger.info(
            f"[LLMTopic] Processing {len(chunks)} chunk(s) "
            f"(text length={len(text)} chars, lang={lang})"
        )

        label_scores: dict[str, float] = {}
        label_counts: dict[str, int]   = {}

        for idx, chunk in enumerate(chunks):
            prompt = build_topic_prompt(language, chunk)
            try:
                response = call_llm(prompt)
                prediction = parse_llm_response(response)
                
                # Validate against target isolated map
                validated_label, score = validate_prediction(prediction, self.logger, lang)

                label_scores[validated_label] = label_scores.get(validated_label, 0.0) + score
                label_counts[validated_label] = label_counts.get(validated_label, 0) + 1

                self.logger.debug(
                    f"[LLMTopic] chunk {idx + 1}/{len(chunks)} → "
                    f"label='{validated_label}' score={score:.3f}"
                )
            except Exception as exc:
                self.logger.warning(
                    f"[LLMTopic] chunk {idx + 1}/{len(chunks)} failed: {exc}"
                )

        if not label_scores:
            self.logger.error("[LLMTopic] All chunks failed; returning fallback.")
            fallback = CATEGORY_DISPLAY[17].get(lang, "General")
            return TopicResult(label=fallback, score=0.0)

        # Vote aggregator execution
        best_label = max(label_counts, key=lambda lbl: (label_counts[lbl], label_scores[lbl]))
        mean_score = label_scores[best_label] / label_counts[best_label]

        self.logger.info(
            f"[LLMTopic] Aggregated result → label='{best_label}' "
            f"(votes={label_counts[best_label]}/{len(chunks)}, "
            f"mean_score={mean_score:.3f})"
        )
        return TopicResult(label=best_label, score=mean_score)

    def unload(self) -> None:
        """No-op execution target wrapper."""
        pass