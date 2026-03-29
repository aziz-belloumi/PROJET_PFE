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
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400
MAX_RETRIES = 3

# Multilingual topic label mapping — matches the database CATEGORY_DISPLAY exactly
CATEGORY_DISPLAY = {
    0: {"ar": "السياسة",    "fr": "Politique",  "en": "Politics"},
    1: {"ar": "الاقتصاد",  "fr": "Économie",   "en": "Economy"},
    2: {"ar": "الأمن",      "fr": "Sécurité",   "en": "Security"},
    3: {"ar": "الطاقة",    "fr": "Énergie",    "en": "Energy"},
    4: {"ar": "النزاع",    "fr": "Conflit",    "en": "Conflict"},
    5: {"ar": "الانتخابات","fr": "Élections",  "en": "Elections"},
    6: {"ar": "العدالة",   "fr": "Justice",    "en": "Justice"},
    7: {"ar": "الصحة",     "fr": "Santé",      "en": "Health"},
    8: {"ar": "الطقس",     "fr": "Météo",      "en": "Weather"},
    9: {"ar": "الرياضة",    "fr": "Sport",    "en": "Sports"},
    10: {"ar": "الثقافة",    "fr": "Culture",  "en": "Culture"},
}

# Flat set of every valid label across all languages (for validation)
TOPIC_ALLOWED_LABELS = {v for row in CATEGORY_DISPLAY.values() for v in row.values()}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def chunk_article(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long article into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


def build_topic_prompt(language, article):
    # Build the multilingual label table for the prompt
    table_lines = ["Arabic | French | English"]
    for row in CATEGORY_DISPLAY.values():
        table_lines.append(f"{row['ar']} | {row['fr']} | {row['en']}")
    label_table = "\n".join(table_lines)

    return f"""You are evaluating topic classification for news articles.

The article may be written in Arabic, English, or French.
Article language: {language}

Article:
{article}

You MUST choose the topic from the following table and return it EXACTLY as written.
Do not translate, paraphrase, or invent new labels.

{label_table}

Rules:
- If the article is in Arabic, return the Arabic label.
- If the article is in English, return the English label.
- If the article is in French, return the French label.
- Respond ONLY with the two lines below. No explanations. No additional text.

true_prediction: <exact label from the table above>
true_language: <Arabic | English | French>"""


def call_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "temperature": 0,
            "num_predict": 100
        }
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=300
            )

            if response.status_code == 200:
                return response.json()["response"]

        except Exception:
            pass

        time.sleep(2)

    raise RuntimeError("LLM request failed after retries")


def parse_llm_response(response_text):
    """
    Parses the LLM output to extract 'true_prediction' and 'true_language'.
    """
    prediction = ""
    language = ""

    # Extract prediction
    match_p = re.search(r"true_prediction:\s*(.*)", response_text, re.IGNORECASE)
    if match_p:
        prediction = match_p.group(1).strip()

    # Extract language
    match_l = re.search(r"true_language:\s*(.*)", response_text, re.IGNORECASE)
    if match_l:
        language = match_l.group(1).strip().lower()
        # Map back to codes
        if "arabic" in language: language = "ar"
        elif "english" in language: language = "en"
        elif "french" in language: language = "fr"

    return prediction, language


# ---------------------------------------------------------------------------
# LLM Topic Extractor
# ---------------------------------------------------------------------------

class LLMTopic:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, text: str, lang: str) -> TopicResult:
        if not text or not text.strip():
            self.logger.warning("[LLMTopic] Received empty text; returning fallback.")
            return TopicResult(label="Unknown", score=0.0)

        # Use the first chunk if text is too long
        chunks = chunk_article(text)
        article_text = chunks[0] if chunks else text

        # Map language code to full name
        lang_map = {"ar": "Arabic", "en": "English", "fr": "French"}
        language = lang_map.get(lang, "English")

        prompt = build_topic_prompt(language, article_text)

        try:
            response = call_llm(prompt)
            prediction, detected_lang = parse_llm_response(response)

            # Validate prediction
            if prediction in TOPIC_ALLOWED_LABELS:
                return TopicResult(label=prediction, score=1.0)
            else:
                self.logger.warning(f"[LLMTopic] Invalid prediction: {prediction}")
                return TopicResult(label="Unknown", score=0.0)

        except Exception as exc:
            self.logger.exception(f"[LLMTopic] Inference failed: {exc}")
            return TopicResult(label="Unknown", score=0.0)

    def unload(self) -> None:
        """No-op for LLM-based extractor."""
        pass