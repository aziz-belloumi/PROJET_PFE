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
You are an expert news analyst evaluating the true public sentiment and impact of an article.

Article language: {language}

Article:
{article}

Possible sentiments:
{SENTIMENT_LIST_TEXT}

Tasks & Guidelines:
1. Determine the TRUE contextual public impact and consequences of the events described, rather than just looking at surface-level structural words (like "announced" or "witnessed").
2. Context matters: Scenarios like "unemployment rising" or "economic slowdown" evaluate as NEGATIVE even if framed gently without explicitly negative adjectives.
3. Domain-specific terms: Words like "austerity", "sanctions", "conflict", and "crisis" inherently describe NEGATIVE events.
4. Language nuances: Account for formal or diplomatic news registers, especially in Arabic, that may use detached, neutral-sounding objective language to report strictly negative events (e.g., casualties, disasters, economic decline).
5. Label POSITIVE for events bringing true societal or economic benefit (e.g., peace, growth). Label NEUTRAL only for purely informational or mixed scenarios with no clear directional impact.

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

        chunks = chunk_article(text)
        if not chunks:
            chunks = [text]

        # Map language code to full name
        lang_map = {"ar": "Arabic", "en": "English", "fr": "French"}
        language = lang_map.get(lang, "English")

        self.logger.info(
            f"[LLMSentiment] Processing {len(chunks)} chunk(s) "
            f"(text length={len(text)} chars, lang={lang})"
        )

        label_scores: dict[str, float] = {}
        label_counts: dict[str, int]   = {}

        for idx, chunk in enumerate(chunks):
            prompt = build_sentiment_prompt(language, chunk)
            try:
                response = call_llm(prompt)
                prediction, detected_lang = parse_llm_response(response)

                # Normalize prediction
                norm_prediction = SENTIMENT_NORM.get(prediction.upper(), prediction.upper())

                if norm_prediction in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                    validated_label = norm_prediction
                    score = 1.0
                else:
                    self.logger.warning(f"[LLMSentiment] Invalid prediction: {prediction} - using 'UNK'")
                    validated_label = "UNK"
                    score = 0.0

                label_scores[validated_label] = label_scores.get(validated_label, 0.0) + score
                label_counts[validated_label] = label_counts.get(validated_label, 0) + 1

                self.logger.debug(
                    f"[LLMSentiment] chunk {idx + 1}/{len(chunks)} → "
                    f"label='{validated_label}' score={score:.3f}"
                )

            except Exception as exc:
                self.logger.warning(
                    f"[LLMSentiment] chunk {idx + 1}/{len(chunks)} failed: {exc}"
                )

        if not label_scores:
            self.logger.error("[LLMSentiment] All chunks failed; returning fallback.")
            return SentimentResult(label="UNK", score=0.0, probs={})

        # Winner = majority vote (highest count), with cumulative score as tiebreaker
        best_label = max(label_counts, key=lambda lbl: (label_counts[lbl], label_scores[lbl]))
        mean_score = label_scores[best_label] / label_counts[best_label]

        self.logger.info(
            f"[LLMSentiment] Aggregated result → label='{best_label}' "
            f"(votes={label_counts[best_label]}/{len(chunks)}, "
            f"mean_score={mean_score:.3f})"
        )

        probs = {
            "POSITIVE": 1.0 if best_label == "POSITIVE" else 0.0,
            "NEGATIVE": 1.0 if best_label == "NEGATIVE" else 0.0,
            "NEUTRAL": 1.0 if best_label == "NEUTRAL" else 0.0,
        }
        return SentimentResult(label=best_label, score=mean_score, probs=probs)