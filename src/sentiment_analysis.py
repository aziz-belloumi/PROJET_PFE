# src/sentiment_analysis.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple, Callable
import logging
import requests
import re
import time


DEFAULT_SENTIMENT_PARAMS: Dict[str, Any] = {
    "max_chunk_tokens": 450,
    "overlap_tokens": 50,
    "aggregation": "mean_probs",  # supported: mean_probs, max_chunk
}


@dataclass
class SentimentResult:
    label: str
    score: float
    probs: Dict[str, float]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400
MAX_RETRIES = 3

# Sentiment normalization map
SENTIMENT_NORM = {
    "POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL",
    "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL",
}

# Sentiments allowed
SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
SENTIMENT_LIST_TEXT = "\n".join(SENTIMENTS)


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


def build_sentiment_prompt(language, article):
    return f"""
You are evaluating sentiment classification for news articles.

The article may be written in Arabic, English, or French.

Article language: {language}

Article:
{article}

Possible sentiments:
{SENTIMENT_LIST_TEXT}

Tasks:
1. Determine the TRUE sentiment of the article.

Output STRICTLY in this format:

true_prediction: one of the sentiment names
true_language: one of [Arabic, English, French]
"""


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
# LLM Sentiment Analyzer
# ---------------------------------------------------------------------------

class LLMSentiment:

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = preprocessor

    def _preprocess_text(self, text: str) -> str:
        if self.preprocessor is None:
            return text
        if callable(self.preprocessor):
            return self.preprocessor(text)
        if hasattr(self.preprocessor, "preprocess_for_sentiment"):
            return self.preprocessor.preprocess_for_sentiment(text)
        if hasattr(self.preprocessor, "preprocess_for_ner"):
            return self.preprocessor.preprocess_for_ner(text)
        return self.preprocessor.preprocess(text)

    def predict(self, text: str, lang: str = "en") -> SentimentResult:
        if not text or not isinstance(text, str):
            return SentimentResult(label="UNK", score=0.0, probs={})

        text = self._preprocess_text(text)

        # Use the first chunk if text is too long
        chunks = chunk_article(text)
        article_text = chunks[0] if chunks else text

        # Map language code to full name
        lang_map = {"ar": "Arabic", "en": "English", "fr": "French"}
        language = lang_map.get(lang, "English")

        prompt = build_sentiment_prompt(language, article_text)

        try:
            response = call_llm(prompt)
            prediction, detected_lang = parse_llm_response(response)

            # Normalize prediction
            norm_prediction = SENTIMENT_NORM.get(prediction.upper(), prediction.upper())

            if norm_prediction in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                # Create probabilities - give 1.0 to predicted, 0.0 to others
                probs = {
                    "POSITIVE": 1.0 if norm_prediction == "POSITIVE" else 0.0,
                    "NEGATIVE": 1.0 if norm_prediction == "NEGATIVE" else 0.0,
                    "NEUTRAL": 1.0 if norm_prediction == "NEUTRAL" else 0.0,
                }
                return SentimentResult(label=norm_prediction, score=1.0, probs=probs)
            else:
                self.logger.warning(f"[LLMSentiment] Invalid prediction: {prediction}")
                return SentimentResult(label="UNK", score=0.0, probs={})

        except Exception as exc:
            self.logger.exception(f"[LLMSentiment] Inference failed: {exc}")
            return SentimentResult(label="UNK", score=0.0, probs={})