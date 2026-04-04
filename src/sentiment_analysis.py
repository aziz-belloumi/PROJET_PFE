# src/sentiment_analysis.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
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

# Heuristic thresholds
AMBIGUOUS_SHORT_TEXT_MAX_WORDS = 8
AMBIGUOUS_SHORT_TEXT_MAX_CHARS = 60

# Safe domain overrides: only applied when LLM returns NEUTRAL
# and the text contains strong, high-precision negative policy/economic cues.
NEGATIVE_DOMAIN_CUES = {
    "ar": [
        "تقشف",
        "البطالة",
        "بطالة",
        "عقوبات",
        "انهيار",
        "ركود",
        "خسائر",
        "ضحايا",
        "كارثة",
    ],
    "en": [
        "austerity",
        "unemployment",
        "sanctions",
        "collapse",
        "recession",
        "losses",
        "casualties",
        "victims",
        "disaster",
    ],
    "fr": [
        "austérité",
        "chômage",
        "sanctions",
        "effondrement",
        "récession",
        "pertes",
        "victimes",
        "catastrophe",
    ],
}


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
You are an expert news analyst evaluating the true public sentiment and impact of a news text.

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
5. Questions are not automatically NEGATIVE. Headlines, fragments, slogans, short phrases, or context-poor statements must be labeled NEUTRAL unless the sentiment is clearly supported by the text itself.
6. Do not infer hidden intent, moral meaning, or implied emotion from a single word alone. References to violence, crime, death, or conflict are NEGATIVE only if the text clearly conveys harmful event impact or consequences.
7. Economic or policy terms such as austerity, unemployment, sanctions, collapse, or recession should usually be considered NEGATIVE when presented as real developments or measures.
8. If the text does not provide enough evidence for a confident polarity decision, choose NEUTRAL.

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

    match_p = re.search(r"true_prediction:\s*(.*)", response_text, re.IGNORECASE)
    if match_p:
        prediction = match_p.group(1).strip()

    match_l = re.search(r"true_language:\s*(.*)", response_text, re.IGNORECASE)
    if match_l:
        language = match_l.group(1).strip().lower()
        if "arabic" in language:
            language = "ar"
        elif "english" in language:
            language = "en"
        elif "french" in language:
            language = "fr"

    return prediction, language


def _tokenize_words(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\S+", text.strip())


def _is_question_like(text: str, lang: str) -> bool:
    if not text:
        return False

    stripped = text.strip()
    lowered = stripped.lower()

    if "?" in stripped or "؟" in stripped:
        return True

    ar_starters = (
        "من", "ما", "ماذا", "هل", "لماذا", "كيف", "أين", "متى", "أي", "كم"
    )
    en_starters = (
        "who", "what", "why", "how", "where", "when", "which", "is", "are",
        "do", "does", "did", "can", "could", "will", "would"
    )
    fr_starters = (
        "qui", "que", "quoi", "pourquoi", "comment", "où", "ou", "quand",
        "quel", "quelle", "quels", "quelles", "est-ce", "peut", "doit"
    )

    first_word = lowered.split()[0] if lowered.split() else ""

    if lang == "ar":
        return first_word in ar_starters
    if lang == "fr":
        return first_word in fr_starters
    return first_word in en_starters


def _looks_like_assertive_short_event(text: str, lang: str) -> bool:
    """
    Identify short texts that still clearly assert an event or outcome.
    These should not be auto-neutralized.

    Examples:
    - "قتل 20 مدنياً"
    - "ارتفعت البطالة"
    - "فاز المنتخب"
    - "تحسن الاقتصاد"
    """
    lowered = (text or "").strip().lower()

    assertive_negative_patterns = [
        r"\bقتل\s+\d+",
        r"\bمقتل\b",
        r"\bقتلى\b",
        r"\bوفاة\b",
        r"\bانفجار\b",
        r"\bهجوم\b",
        r"\bاشتباكات\b",
        r"\bحريق\b",
        r"\bأزمة\b",
        r"\bتراجع\b",
        r"\bانهيار\b",
        r"\bبطالة\b",
        r"\bضحايا\b",
        r"\bkilled\s+\d+",
        r"\bdeath\b",
        r"\bexplosion\b",
        r"\battack\b",
        r"\bcrisis\b",
        r"\bdecline\b",
        r"\bunemployment\b",
        r"\bcasualties\b",
        r"\bvictims\b",
        r"\btué\b",
        r"\bmort\b",
        r"\bexplosion\b",
        r"\battaque\b",
        r"\bcrise\b",
        r"\bbaisse\b",
        r"\bchômage\b",
        r"\bvictimes\b",
    ]

    assertive_positive_patterns = [
        r"\bفاز\b",
        r"\bنجاح\b",
        r"\bتحسن\b",
        r"\bنمو\b",
        r"\bسلام\b",
        r"\bاتفاق\b",
        r"\bاستقرار\b",
        r"\bانتعاش\b",
        r"\bتقدم\b",
        r"\bwon\b",
        r"\bsuccess\b",
        r"\bimprovement\b",
        r"\bgrowth\b",
        r"\bpeace\b",
        r"\bagreement\b",
        r"\bstability\b",
        r"\brecovery\b",
        r"\bprogress\b",
        r"\bvictoire\b",
        r"\bsuccès\b",
        r"\bamélioration\b",
        r"\bcroissance\b",
        r"\bpaix\b",
        r"\baccord\b",
        r"\bstabilité\b",
        r"\bprogrès\b",
    ]

    return any(re.search(pat, lowered) for pat in assertive_negative_patterns + assertive_positive_patterns)


def _is_ambiguous_short_question(text: str, lang: str) -> bool:
    """
    Return True for short, interrogative, low-context texts that should default
    to NEUTRAL instead of being over-interpreted by the LLM.

    Important refinement:
    - If the text is question-like and short, it is treated as ambiguous by default.
    - Only clearly assertive short event statements bypass this safeguard.
    """
    if not text or not isinstance(text, str):
        return False

    stripped = text.strip()
    if not stripped:
        return False

    words = _tokenize_words(stripped)
    if not words:
        return False

    short_enough = (
        len(words) <= AMBIGUOUS_SHORT_TEXT_MAX_WORDS
        or len(stripped) <= AMBIGUOUS_SHORT_TEXT_MAX_CHARS
    )

    if not short_enough:
        return False

    if not _is_question_like(stripped, lang):
        return False

    # For short questions, ambiguity dominates.
    # We do not want single negative cue words like "killed"/"tué"/"قتل"
    # to force a non-neutral interpretation in context-poor interrogative text.
    return True


def _contains_negative_domain_cue(text: str, lang: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    cues = NEGATIVE_DOMAIN_CUES.get(lang, NEGATIVE_DOMAIN_CUES["en"])
    return any(cue in lowered for cue in cues)


def _apply_safe_post_rules(text: str, lang: str, label: str, score: float) -> tuple[str, float]:
    """
    Apply narrow, low-risk post-processing rules.

    Current safe rule:
    - If LLM says NEUTRAL but the text contains a strong negative policy/economic cue,
      shift to NEGATIVE with moderate confidence.
    """
    if label == "NEUTRAL" and _contains_negative_domain_cue(text, lang):
        return "NEGATIVE", 0.75

    return label, score


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

        # Refined narrow heuristic:
        # short + interrogative + low-context => NEUTRAL
        if _is_ambiguous_short_question(text, lang):
            self.logger.info(
                "[LLMSentiment] Ambiguous short question detected; "
                "returning NEUTRAL without LLM inference."
            )
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

        self.logger.info(
            f"[LLMSentiment] Processing {len(chunks)} chunk(s) "
            f"(text length={len(text)} chars, lang={lang})"
        )

        label_scores: dict[str, float] = {}
        label_counts: dict[str, int] = {}

        for idx, chunk in enumerate(chunks):
            prompt = build_sentiment_prompt(language, chunk)
            try:
                response = call_llm(prompt)
                prediction, detected_lang = parse_llm_response(response)

                norm_prediction = SENTIMENT_NORM.get(prediction.upper(), prediction.upper())

                if norm_prediction in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                    validated_label = norm_prediction
                    score = 1.0
                else:
                    self.logger.warning(
                        f"[LLMSentiment] Invalid prediction: {prediction} - using 'UNK'"
                    )
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

        best_label = max(label_counts, key=lambda lbl: (label_counts[lbl], label_scores[lbl]))
        mean_score = label_scores[best_label] / label_counts[best_label]

        best_label, mean_score = _apply_safe_post_rules(text, lang, best_label, mean_score)

        self.logger.info(
            f"[LLMSentiment] Aggregated result → label='{best_label}' "
            f"(votes={label_counts.get(best_label, 0)}/{len(chunks)}, "
            f"mean_score={mean_score:.3f})"
        )

        probs = {
            "POSITIVE": 1.0 if best_label == "POSITIVE" else 0.0,
            "NEGATIVE": 1.0 if best_label == "NEGATIVE" else 0.0,
            "NEUTRAL": 1.0 if best_label == "NEUTRAL" else 0.0,
        }
        return SentimentResult(label=best_label, score=mean_score, probs=probs)