# src/ner/gliner_wrapper.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

from gliner import GLiNER

# Import NEREntity for compatibility
from ..ner_extraction import NEREntity


# GLiNER label mapping to project's canonical labels
GLINER_LABEL_MAPPING = {
    'person name': 'PER',
    'organization or institution or government body': 'ORG',
    'geographic location or city or country': 'LOC',
    'date or time expression': 'DAT',
    'named event or armed conflict or political crisis': 'EVE',
    'commercial product or brand name': 'PRO',
    'sports competition or league or tournament': 'COM',
    # Legacy mappings for backward compatibility
    'person': 'PER',
    'organization': 'ORG',
    'company': 'ORG',
    'institution': 'ORG',
    'agency': 'ORG',
    'university': 'ORG',
    'ministry': 'ORG',
    'government': 'ORG',
    'location': 'LOC',
    'city': 'LOC',
    'country': 'LOC',
    'region': 'LOC',
    'state': 'LOC',
    'province': 'LOC',
    'facility': 'LOC',
    'gpe': 'LOC',
    'date': 'DAT',
    'time': 'DAT',
    'datetime': 'DAT',
    'event': 'EVE',
    'misc': 'MIS',
    'miscellaneous': 'MIS',
    'other': 'MIS',
    'product': 'PRO',
    'brand': 'PRO',
    'app': 'PRO',
    'software': 'PRO',
    'platform': 'PRO',
    'competition': 'COM',
    'league': 'COM',
    'tournament': 'COM',
    'championship': 'COM',
}

# Minimum word count per label to filter out generic single-word false positives
MIN_WORD_COUNT_PER_LABEL = {
    'PER': 1,
    'ORG': 1,
    'LOC': 1,
    'DAT': 1,
    'EVE': 2,
    'PRO': 2,
    'COM': 1,
    'MIS': 2,
}

# Language-aware thresholds — Arabic needs slightly lower threshold
# because names often appear without surrounding context in short texts
LANGUAGE_THRESHOLDS = {
    'ar': 0.6,
    'en': 0.60,
    'fr': 0.60,
}

# Chunk configuration for long texts
CHUNK_SIZE = 450        # words per chunk
CHUNK_OVERLAP = 100      # overlapping words between chunks to avoid boundary loss


class GLiNERWrapper:
    """
    Standalone GLiNER wrapper for entity extraction.
    Designed for future integration into the project's NER pipeline.
    """

    DEFAULT_MODEL = 'urchade/gliner_multi-v2.1'
    DEFAULT_THRESHOLD = 0.60
    DEFAULT_LABELS = [
        'person name',
        'organization or institution or government body',
        'geographic location or city or country',
        'date or time expression',
        'named event or armed conflict or political crisis',
        'commercial product or brand name',
        'sports competition or league or tournament',
    ]

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        labels: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        device: int = -1,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.threshold = threshold
        self.labels = labels or self.DEFAULT_LABELS.copy()
        self.device = device

        self.logger.info(f"Loading GLiNER model: {self.model_name}")
        try:
            self.model = GLiNER.from_pretrained(self.model_name)
            if self.device >= 0:
                self.model = self.model.to(f"cuda:{self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load GLiNER model {self.model_name}: {e}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split long text into overlapping word-based chunks.
        This ensures GLiNER processes the full article and doesn't
        miss entities that appear beyond its token limit.
        """
        words = text.split()
        if len(words) <= CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

        self.logger.debug(f"Text split into {len(chunks)} chunks ({len(words)} words total)")
        return chunks

    def _filter_entities(self, entities: List[NEREntity]) -> List[NEREntity]:
        """
        Post-processing filter:
        1. Reject entities that are too short
        2. Reject entities below minimum word count for their label
        3. Deduplicate by (normalized text, label) pair
        """
        filtered = []
        for e in entities:
            text_stripped = e.text.strip()
            word_count = len(text_stripped.split())
            min_words = MIN_WORD_COUNT_PER_LABEL.get(e.label, 1)

            if len(text_stripped) < 2:
                self.logger.debug(f"Filtered (too short): '{e.text}' ({e.label})")
                continue

            if word_count < min_words:
                self.logger.debug(f"Filtered (word count {word_count} < {min_words}): '{e.text}' ({e.label})")
                continue

            filtered.append(e)

        # Deduplicate by (normalized lowercase text, label)
        seen = set()
        deduped = []
        for e in filtered:
            key = (e.text.strip().lower(), e.label)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
            else:
                self.logger.debug(f"Deduplicated: '{e.text}' ({e.label})")

        return deduped

    def predict(
        self,
        text: str,
        language: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> List[NEREntity]:
        
        if not text.strip():
            return []

        use_labels = labels or self.labels

        # Apply language-aware threshold
        effective_threshold = LANGUAGE_THRESHOLDS.get(language, self.threshold) if language else self.threshold
        self.logger.debug(f"Using threshold {effective_threshold} for language '{language}'")

        # Split into chunks for long articles
        chunks = self._chunk_text(text)

        all_entities: List[NEREntity] = []

        for idx, chunk in enumerate(chunks):
            self.logger.debug(f"Processing chunk {idx + 1}/{len(chunks)}")
            try:
                raw_entities = self.model.predict_entities(chunk, use_labels, threshold=effective_threshold)
            except Exception as e:
                self.logger.error(f"GLiNER prediction failed on chunk {idx + 1}: {e}")
                continue

            for entity in raw_entities:
                original_label = entity['label'].lower()
                normalized_label = GLINER_LABEL_MAPPING.get(original_label, 'MIS')

                ner_entity = NEREntity(
                    text=entity['text'],
                    label=normalized_label,
                    start=entity['start'],
                    end=entity['end'],
                    score=entity['score']
                )
                all_entities.append(ner_entity)

        self.logger.debug(f"Extracted {len(all_entities)} entities before filtering")

        # Apply post-processing filter + deduplication
        final_entities = self._filter_entities(all_entities)

        self.logger.debug(f"Returning {len(final_entities)} entities after filtering")
        return final_entities

    def unload(self) -> None:
        """Release model from memory if needed."""
        del self.model
        self.logger.info("GLiNER model unloaded.")


def test_gliner():
    """
    Standalone test function for GLiNER wrapper.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    samples = {
        ('English', 'en'): "Barack Obama was the 44th President of the United States and visited Paris in 2015.",
        ('French', 'fr'): "Emmanuel Macron est le président de la France et a rencontré Angela Merkel à Berlin.",
        ('Arabic', 'ar'): "الرئيس الأمريكي باراك أوباما زار باريس في عام 2015.",
        ('Arabic_News', 'ar'): "المستشارة الأممية ستيفاني وليامز تعرب عن قلقها إزاء إغلاق حقول النفط وتعليق بعض الرحلات المدنية.",
        ('Arabic_Politics', 'ar'): "93 نائبا يؤكدون أن مبادرة ويليامز هي مسار موازي ومفاجئ لاتفاقهم مع الأعلى للدولة.",
        ('English_Conflict', 'en'): "Evacuation route out of Mariupol was mined, Red Cross says.",
        ('English_War', 'en'): "There's a supposed phone call between Putin and Shoigu doing the rounds that sounds fake as hell.",
    }

    wrapper = GLiNERWrapper(logger=logger)

    for (lang_name, lang_code), text in samples.items():
        print(f"\n--- {lang_name} ---")
        print(f"Text: {text}")
        entities = wrapper.predict(text, language=lang_code)
        if entities:
            for ent in entities:
                print(f"  {ent.label}: '{ent.text}' (score: {ent.score:.2f}, pos: {ent.start}-{ent.end})")
        else:
            print("  No entities extracted.")


if __name__ == "__main__":
    test_gliner()