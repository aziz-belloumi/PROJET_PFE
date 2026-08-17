from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List
import logging
import requests
import re
import time

from src.config import Config

DEFAULT_SENTIMENT_PARAMS: Dict[str, Any] = Config.DEFAULT_SENTIMENT_PARAMS

@dataclass
class SentimentResult:
    label: str
    score: float
    probs: Dict[str, float]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = Config.OLLAMA_URL
MODEL_NAME = Config.QWEN_MODEL_NAME
CHUNK_SIZE = Config.CHUNK_SIZE
CHUNK_OVERLAP = Config.CHUNK_OVERLAP
MAX_RETRIES = Config.QWEN_MAX_RETRIES

SENTIMENT_NORM = {
    "POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL",
    "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL",
}

SENTIMENTS = Config.SENTIMENTS
SENTIMENT_LIST_TEXT = "\n".join(SENTIMENTS)

AMBIGUOUS_SHORT_TEXT_MAX_WORDS = Config.AMBIGUOUS_SHORT_TEXT_MAX_WORDS
AMBIGUOUS_SHORT_TEXT_MAX_CHARS = Config.AMBIGUOUS_SHORT_TEXT_MAX_CHARS

NEGATIVE_DOMAIN_CUES = Config.NEGATIVE_DOMAIN_CUES

# Compiled Regex Patterns
RE_PREDICTION = re.compile(r"true_prediction:\s*(\w+)", re.IGNORECASE)
RE_TOKENS = re.compile(r"\S+")

ASSERTIVE_NEGATIVE_PATTERNS = [
    r"\bقتل\s+\d+", r"\bمقتل\b", r"\bقتلى\b", r"\bوفاة\b", r"\bانفجار\b", r"\bهجوم\b",
    r"\bاشتباكات\b", r"\bحريق\b", r"\bأزمة\b", r"\bتراجع\b", r"\bانهيار\b", r"\bبطالة\b", r"\bضحايا\b",
    r"\bkilled\s+\d+", r"\bdeath\b", r"\bexplosion\b", r"\battack\b", r"\bcrisis\b",
    r"\bdecline\b", r"\bunemployment\b", r"\bcasualties\b", r"\bvictims\b",
    r"\btué\b", r"\bmort\b", r"\bexplosion\b", r"\battaque\b", r"\bcrise\b", r"\bbaisse\b", r"\bchômage\b", r"\bvictimes\b"
]

ASSERTIVE_POSITIVE_PATTERNS = [
    r"\bفاز\b", r"\bنجاح\b", r"\bتحسن\b", r"\bنمو\b", r"\bسلام\b", r"\bاتفاق\b", r"\bاستقرار\b",
    r"\bانتعاش\b", r"\bتقدم\b", r"\bwon\b", r"\bsuccess\b", r"\bimprovement\b", r"\bgrowth\b",
    r"\bpeace\b", r"\bagreement\b", r"\bstability\b", r"\brecovery\b", r"\bprogress\b",
    r"\bvictoire\b", r"\bsuccès\b", r"\bamélioration\b", r"\bcroissance\b", r"\bpaix\b", r"\baccord\b",
    r"\bstabilité\b", r"\bprogrès\b"
]

RE_ASSERTIVE = re.compile("|".join(ASSERTIVE_NEGATIVE_PATTERNS + ASSERTIVE_POSITIVE_PATTERNS), re.IGNORECASE)

AR_STARTERS = {"من", "ما", "ماذا", "هل", "لماذا", "كيف", "أين", "متى", "أي", "كم"}
EN_STARTERS = {"who", "what", "why", "how", "where", "when", "which", "is", "are", "do", "does", "did", "can", "could", "will", "would"}
FR_STARTERS = {"qui", "que", "quoi", "pourquoi", "comment", "où", "ou", "quand", "quel", "quelle", "quels", "quelles", "est-ce", "peut", "doit"}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def chunk_article(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text) or size <= overlap:
            break
    return chunks


def build_sentiment_prompt(language: str, article: str) -> str:
    return f"""You are an expert news analyst evaluating the true public sentiment and impact of a news text.

Text language: {language}

Text:
{article}

Possible sentiments:
{SENTIMENT_LIST_TEXT}

Tasks & Guidelines:
1. Determine the TRUE contextual public impact and consequences of the events described, rather than only reacting to isolated words.
2. Context matters: scenarios like unemployment rising, economic slowdown, sanctions, conflict, casualties, disasters, crisis, or austerity measures are usually NEGATIVE when the harmful impact is clearly described.
3. Label POSITIVE only when the text clearly describes beneficial outcomes such as peace, recovery, growth, aid, improvement, stability, or success.
4. Label NEUTRAL whenever the text is mainly informational, ambiguous, incomplete, interrogative, mixed, or lacks enough context to support a clear POSITIVE or NEGATIVE judgment.
5. Questions are not automatically NEGATIVE.
6. Do not infer hidden intent from isolated words.
7. Economic crisis terms should usually be NEGATIVE when describing real developments.
8. If not enough evidence, choose NEUTRAL.

Output STRICTLY in this format:

true_prediction: one of the sentiment names
"""


def call_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": Config.QWEN_NUM_CTX,
            "temperature": Config.QWEN_TEMPERATURE,
            "num_predict": Config.QWEN_NUM_PREDICT,
        }
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=Config.QWEN_TIMEOUT_SEC)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("LLM request failed after retries")


def parse_llm_response(response_text: str) -> str:
    match_p = RE_PREDICTION.search(response_text)
    if match_p:
        return match_p.group(1).strip()
    return ""


def _tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    return RE_TOKENS.findall(text.strip())


def _is_question_like(text: str, lang: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if "?" in stripped or "؟" in stripped:
        return True
    words = stripped.split()
    if not words:
        return False
    first_word = words[0].lower()
    if lang == "ar":
        return first_word in AR_STARTERS
    if lang == "fr":
        return first_word in FR_STARTERS
    return first_word in EN_STARTERS


def _is_ambiguous_short_question(text: str, lang: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    words = _tokenize_words(stripped)
    if not words:
        return False
    short_enough = (
        len(words) <= AMBIGUOUS_SHORT_TEXT_MAX_WORDS
        or len(stripped) <= AMBIGUOUS_SHORT_TEXT_MAX_CHARS
    )
    if not short_enough:
        return False
    return _is_question_like(stripped, lang)


def _contains_negative_domain_cue(text: str, lang: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    cues = NEGATIVE_DOMAIN_CUES.get(lang, NEGATIVE_DOMAIN_CUES["en"])
    return any(cue in lowered for cue in cues)


def _apply_safe_post_rules(text: str, lang: str, label: str, score: float) -> Tuple[str, float]:
    if label == "NEUTRAL" and _contains_negative_domain_cue(text, lang):
        return "NEGATIVE", 0.75
    return label, score


# ---------------------------------------------------------------------------
# LLM Sentiment Analyzer (Qwen/Ollama)
# ---------------------------------------------------------------------------

class LLMSentiment:

    def __init__(self, logger: Optional[logging.Logger] = None, preprocessor: Any = None):
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = preprocessor

    def _preprocess_text(self, text: str) -> str:
        if self.preprocessor is None:
            return text
        if callable(self.preprocessor):
            return self.preprocessor(text)
        for attr in ["preprocess_for_sentiment", "preprocess_for_ner", "preprocess"]:
            if hasattr(self.preprocessor, attr):
                return getattr(self.preprocessor, attr)(text)
        return text

    def predict(self, text: str, lang: str = "en") -> SentimentResult:
        if not text or not isinstance(text, str):
            return SentimentResult(label="UNK", score=0.0,
                                   probs={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0})

        text = self._preprocess_text(text)

        if _is_ambiguous_short_question(text, lang):
            return SentimentResult(
                label="NEUTRAL",
                score=0.6,
                probs={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 1.0},
            )

        chunks = chunk_article(text)
        if not chunks:
            chunks = [text]

        lang_map = {"ar": "Arabic", "en": "English", "fr": "French"}
        language = lang_map.get(lang, "English")

        label_scores: Dict[str, float] = {k: 0.0 for k in SENTIMENTS}
        label_counts = {k: 0 for k in SENTIMENTS}
        valid_chunks_processed = 0

        for chunk in chunks:
            prompt = build_sentiment_prompt(language, chunk)
            try:
                response = call_llm(prompt)
                prediction = parse_llm_response(response)
                norm_prediction = SENTIMENT_NORM.get(prediction.upper(), "UNK")

                if norm_prediction in label_scores:
                    label_scores[norm_prediction] += 1.0
                    label_counts[norm_prediction] += 1
                    valid_chunks_processed += 1

            except Exception:
                pass

        if valid_chunks_processed == 0:
            return SentimentResult(label="UNK", score=0.0,
                                   probs={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0})

        best_label = max(label_counts, key=lambda lbl: (label_counts[lbl], label_scores[lbl]))
        mean_score = label_counts[best_label] / valid_chunks_processed

        best_label, mean_score = _apply_safe_post_rules(text, lang, best_label, mean_score)

        probs = {k: label_counts[k] / valid_chunks_processed for k in SENTIMENTS}

        return SentimentResult(label=best_label, score=mean_score, probs=probs)
