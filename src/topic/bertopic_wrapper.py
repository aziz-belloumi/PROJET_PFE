from typing import List, Tuple, Optional
import logging
import re

try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None


class BERTopicWrapper:
    """
    Wrapper for BERTopic to handle batch-level topic discovery and assignment.

    Design goals:
    - Fit on a single-language batch of preprocessed topic texts.
    - Produce one deterministic label and one score per input document.
    - Improve topic quality with stronger vectorization and topic-word cleaning.
    - Handle small / weak batches gracefully with explicit fallback outputs.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        min_topic_size: int = 5,
        nr_topics: Optional[object] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_model_name = embedding_model
        self.base_min_topic_size = max(2, int(min_topic_size))
        self.nr_topics = nr_topics
        self.kwargs = kwargs

        if BERTopic is None:
            self.logger.error("BERTopic is not installed. Please install it to use BERTopicWrapper.")
            raise ImportError("BERTopic not installed.")

        # Domain / news / boilerplate stopwords across EN / FR / AR
        # Kept conservative to avoid over-pruning meaningful topic words.
        self.domain_stopwords = {
            # English
            "said", "says", "according", "government", "minister", "ministry",
            "president", "official", "officials", "country", "state", "states",
            "today", "yesterday", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday", "year", "years", "new", "last",
            "first", "former", "would", "could", "also", "one", "two",
            # French
            "selon", "gouvernement", "ministre", "ministère", "président",
            "officiel", "officiels", "pays", "etat", "état", "aujourdhui",
            "aujourd", "hier", "année", "années", "nouveau", "nouvelle",
            "dernier", "dernière", "également", "aussi",
            # Arabic
            "قال", "وأضاف", "اكد", "أكد", "ذكرت", "ذكر", "أعلنت", "اعلنت",
            "الحكومة", "وزير", "وزارة", "رئيس", "مسؤول", "مسؤولون", "الدولة",
            "البلاد", "اليوم", "أمس", "عام", "أعوام", "الجديد", "الجديدة",
            "الأول", "الأولى", "كما", "ايضا", "أيضا",
        }

        self._compiled_invalid_token_patterns = [
            re.compile(r"^\d+$"),                  # digits only
            re.compile(r"^[\W_]+$", re.UNICODE),   # punctuation only
        ]

    # ------------------------------------------------------------------ utils

    def _load_vectorizer(self):
        """
        Build a stronger vectorizer for multilingual news corpora.
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            import nltk
            from nltk.corpus import stopwords

            try:
                stops_en = stopwords.words("english")
            except LookupError:
                nltk.download("stopwords", quiet=True)
                stops_en = stopwords.words("english")

            try:
                stops_ar = stopwords.words("arabic")
            except LookupError:
                nltk.download("stopwords", quiet=True)
                stops_ar = stopwords.words("arabic")

            try:
                stops_fr = stopwords.words("french")
            except LookupError:
                nltk.download("stopwords", quiet=True)
                stops_fr = stopwords.words("french")

            stop_words = list(set(stops_en + stops_ar + stops_fr + list(self.domain_stopwords)))

            # Important:
            # - ngram_range=(1, 2) helps produce more meaningful news phrases
            # - min_df/max_df reduce very rare noise and overly common boilerplate
            # - token_pattern is relaxed enough for multilingual word-like tokens
            vectorizer_model = CountVectorizer(
                stop_words=stop_words,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.85,
                token_pattern=r"(?u)\b\w[\w\-]+\b",
            )
            return vectorizer_model

        except Exception as e:
            self.logger.warning(
                f"Failed to initialize enhanced CountVectorizer: {e}. "
                "Falling back to BERTopic defaults."
            )
            return None

    def _build_representation_model(self):
        try:
            from bertopic.representation import KeyBERTInspired
            return KeyBERTInspired()
        except Exception as e:
            self.logger.warning(f"Could not initialize KeyBERTInspired representation: {e}")
            return None

    def _adaptive_min_topic_size(self, doc_count: int) -> int:
        """
        Adaptive min_topic_size based on batch size.
        Conservative defaults to avoid both overfragmentation and giant vague topics.
        """
        if doc_count < 20:
            return max(2, min(self.base_min_topic_size, 3))
        if doc_count < 50:
            return max(3, min(self.base_min_topic_size, 5))
        if doc_count < 120:
            return max(4, self.base_min_topic_size)
        return max(5, self.base_min_topic_size)

    def _adaptive_nr_topics(self, doc_count: int):
        """
        Avoid blind topic reduction on smaller corpora.
        """
        if self.nr_topics is not None:
            return self.nr_topics
        if doc_count < 40:
            return None
        return "auto"

    def _sanitize_term(self, term: str) -> Optional[str]:
        if term is None:
            return None

        term = str(term).strip().lower()
        term = re.sub(r"\s+", "_", term)

        if not term:
            return None
        if len(term) <= 2:
            return None
        if term in self.domain_stopwords:
            return None

        for pattern in self._compiled_invalid_token_patterns:
            if pattern.match(term):
                return None

        # Remove repeated underscores and trim noise
        term = re.sub(r"_+", "_", term).strip("_")
        if not term:
            return None

        return term

    def _build_topic_label(self, topic_id: int) -> str:
        """
        Deterministic topic label from top representative words.
        """
        try:
            words_info = self.model.get_topic(topic_id)
        except Exception:
            words_info = None

        cleaned_terms = []
        seen = set()

        if words_info:
            for word_info in words_info:
                raw_word = word_info[0] if isinstance(word_info, (list, tuple)) and len(word_info) > 0 else None
                clean_word = self._sanitize_term(raw_word)
                if clean_word and clean_word not in seen:
                    seen.add(clean_word)
                    cleaned_terms.append(clean_word)
                if len(cleaned_terms) >= 4:
                    break

        if not cleaned_terms:
            return f"Topic_{topic_id}: unknown"

        return f"Topic_{topic_id}: {'_'.join(cleaned_terms[:4])}"

    def _safe_assignment_score(self, topic_id: int, prob_obj) -> float:
        """
        Extract a stable per-document score.

        BERTopic probabilities can vary by version / config.
        We use a cautious strategy:
        - scalar => direct float
        - iterable => max value as assignment strength fallback
        - otherwise => 0.0

        This avoids unsafe assumptions about direct topic_id -> array index mapping.
        """
        try:
            if prob_obj is None:
                return 0.0

            if isinstance(prob_obj, (float, int)):
                return float(prob_obj)

            if hasattr(prob_obj, "tolist"):
                prob_obj = prob_obj.tolist()

            if hasattr(prob_obj, "__iter__"):
                vals = []
                for x in prob_obj:
                    try:
                        vals.append(float(x))
                    except Exception:
                        continue
                if vals:
                    return float(max(vals))

            return 0.0
        except Exception:
            return 0.0

    def _fallback_outputs(self, batch_texts: List[str], reason: str) -> Tuple[List[str], List[float]]:
        label = f"Topic_Fallback:{reason}"
        return [label] * len(batch_texts), [0.0] * len(batch_texts)

    # -------------------------------------------------------------- main API

    def fit_predict(self, batch_texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Fit BERTopic on the provided batch and assign one topic per document.

        Returns:
            labels: deterministic topic labels aligned with input order
            scores: assignment-strength scores aligned with input order
        """
        if not batch_texts:
            return [], []

        # Defensive filtering expectation:
        # gpu_pass should already filter invalid / too-short texts,
        # but we still guard against pathological inputs here.
        usable_texts = [str(t).strip() if t is not None else "" for t in batch_texts]
        if not any(usable_texts):
            return self._fallback_outputs(batch_texts, "empty_batch")

        doc_count = len(usable_texts)
        min_topic_size = self._adaptive_min_topic_size(doc_count)
        nr_topics = self._adaptive_nr_topics(doc_count)

        vectorizer_model = self._load_vectorizer()
        representation_model = self._build_representation_model()

        # Optional tuning backends
        umap_model = None
        hdbscan_model = None

        try:
            from umap import UMAP
            n_neighbors = min(15, max(3, doc_count // 6))
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize UMAP backend: {e}")

        try:
            import hdbscan
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=max(1, min_topic_size // 2),
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize HDBSCAN backend: {e}")

        self.logger.info(
            f"Fitting BERTopic on {doc_count} docs "
            f"(embedding={self.embedding_model_name}, min_topic_size={min_topic_size}, nr_topics={nr_topics})"
        )

        try:
            self.model = BERTopic(
                embedding_model=self.embedding_model_name,
                min_topic_size=min_topic_size,
                nr_topics=nr_topics,
                calculate_probabilities=True,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=False,
                **self.kwargs,
            )

            topics, probabilities = self.model.fit_transform(usable_texts)

            labels_out: List[str] = []
            scores_out: List[float] = []

            for topic_id, prob_obj in zip(topics, probabilities):
                if topic_id == -1:
                    labels_out.append("Topic_Outlier: uncategorized")
                    scores_out.append(self._safe_assignment_score(topic_id, prob_obj))
                    continue

                label = self._build_topic_label(int(topic_id))
                score = self._safe_assignment_score(int(topic_id), prob_obj)

                labels_out.append(label)
                scores_out.append(score)

            if len(labels_out) != len(batch_texts) or len(scores_out) != len(batch_texts):
                self.logger.error(
                    "BERTopic output length mismatch. Falling back for entire batch."
                )
                return self._fallback_outputs(batch_texts, "length_mismatch")

            return labels_out, scores_out

        except Exception as e:
            self.logger.exception(f"BERTopic fitting failed: {e}")
            return self._fallback_outputs(batch_texts, "fit_failed")