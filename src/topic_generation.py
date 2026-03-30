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
    17: {"ar": "أخرى",         "fr": "Autre",          "en": "Other"},
}

# Flat set of every valid label across all languages (for known category validation)
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
    """
    Build the prompt with the expanded category table and instructions
    to allow the LLM to suggest new general categories if needed.
    """
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

**Instructions:**
1. Choose the MOST appropriate general category from the table below.
2. If NONE of the categories fit well, you may suggest ONE new general category at the same abstraction level.
   Examples of valid new categories: "Infrastructure", "Transportation", "Corruption", "Human Rights"
3. Return the label in the SAME language as the article.
4. Do NOT use specific terms like "Oil Prices", "COVID-19", or "Gaza War" — only high-level themes.
5. The category must be 1-3 words maximum.

**Available Categories:**
{label_table}

**Output format (two lines only, no explanations):**

true_prediction: <exact label from table OR new general category>
true_language: <Arabic | English | French>"""


def call_llm(prompt):
    """Call the Ollama LLM with retry logic."""
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
        if "arabic" in language: 
            language = "ar"
        elif "english" in language: 
            language = "en"
        elif "french" in language: 
            language = "fr"

    return prediction, language


def validate_prediction(prediction: str, logger: logging.Logger) -> tuple[str, float]:
    """
    Validate the LLM's prediction.
    
    Returns:
        tuple: (validated_label, confidence_score)
    """
    # Check if it's a known category
    if prediction in TOPIC_ALLOWED_LABELS:
        return prediction, 1.0
    
    # If not in the predefined list, validate it's a reasonable general term
    if prediction:
        # Remove common punctuation
        cleaned = prediction.strip().strip('.,!?;:')
        word_count = len(cleaned.split())
        
        # Accept if:
        # 1. Not empty
        # 2. 1-3 words
        # 3. Not "Unknown" or similar fallback terms
        if (word_count >= 1 and 
            word_count <= 3 and 
            cleaned.lower() not in ['unknown', 'other', 'none', 'n/a', 'unclear']):
            
            logger.info(f"[LLMTopic] New category suggested: '{cleaned}'")
            return cleaned, 0.8  # Lower confidence for emergent categories
    
    # Fallback
    logger.warning(f"[LLMTopic] Invalid prediction: '{prediction}' - using 'Other'")
    return "Other", 0.0


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
        """
        Predict the topic category for a given article.
        
        Args:
            text: The article body
            lang: Language code (ar/en/fr)
            
        Returns:
            TopicResult with label and confidence score
        """
        if not text or not text.strip():
            self.logger.warning("[LLMTopic] Received empty text; returning fallback.")
            return TopicResult(label="Other", score=0.0)

        # Use the first chunk if text is too long
        chunks = chunk_article(text)
        article_text = chunks[0] if chunks else text

        # Map language code to full name
        lang_map = {"ar": "Arabic", "en": "English", "fr": "French"}
        language = lang_map.get(lang, "English")

        # Build prompt
        prompt = build_topic_prompt(language, article_text)

        try:
            # Call LLM
            response = call_llm(prompt)
            prediction, detected_lang = parse_llm_response(response)

            # Validate and return
            validated_label, score = validate_prediction(prediction, self.logger)
            return TopicResult(label=validated_label, score=score)

        except Exception as exc:
            self.logger.exception(f"[LLMTopic] Inference failed: {exc}")
            return TopicResult(label="Other", score=0.0)

    def unload(self) -> None:
        """No-op for LLM-based extractor (no model loaded in memory)."""
        pass


# ---------------------------------------------------------------------------
# Utility: Get emergent categories from database
# ---------------------------------------------------------------------------

def get_emergent_categories(session):
    """
    Query to find all topic labels that aren't in the original CATEGORY_DISPLAY.
    Use this in your analytics to track new categories suggested by the LLM.
    
    Args:
        session: SQLAlchemy session
        
    Returns:
        List of tuples: [(category_name, frequency), ...]
    """
    # Build the list of known labels for the WHERE clause
    known_labels = "', '".join(TOPIC_ALLOWED_LABELS)
    
    query = f"""
    SELECT DISTINCT topic_label, COUNT(*) as frequency
    FROM article_topics
    WHERE topic_label NOT IN ('{known_labels}')
    GROUP BY topic_label
    ORDER BY frequency DESC;
    """
    
    result = session.execute(query)
    return result.fetchall()