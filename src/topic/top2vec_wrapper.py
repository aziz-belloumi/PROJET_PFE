from typing import List, Tuple, Optional
import logging
import re

try:
    from top2vec import Top2Vec
except ImportError:
    Top2Vec = None


class Top2VecWrapper:
    """
    Wrapper for Top2Vec to handle batch-level topic discovery and assignment.

    Design goals:
    - Fit on a single-language batch of preprocessed topic texts.
    - Produce one deterministic label and one score per input document.
    - Improve topic label quality with keyword cleaning.
    - Handle weak / small batches gracefully with explicit fallback outputs.
    """

    def __init__(
        self,
        embedding_model: str = "distiluse-base-multilingual-cased",
        speed: str = "learn",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_model = embedding_model
        self.speed = speed
        self.kwargs = kwargs

        if Top2Vec is None:
            self.logger.error("Top2Vec is not installed. Please install it to use Top2VecWrapper.")
            raise ImportError("Top2Vec not installed.")

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
            re.compile(r"^\d+$"),
            re.compile(r"^[\W_]+$", re.UNICODE),
        ]

    # ------------------------------------------------------------------ utils

    def _adaptive_min_count(self, doc_count: int) -> int:
        """
        Keep vocabulary filtering conservative for sampled news corpora.
        """
        if doc_count < 25:
            return 1
        if doc_count < 80:
            return 2
        if doc_count < 200:
            return 3
        return 5

    def _adaptive_umap_args(self, doc_count: int) -> dict:
        n_neighbors = min(15, max(3, doc_count // 6))
        return {
            "n_neighbors": n_neighbors,
            "n_components": 5,
            "metric": "cosine",
            "random_state": 42,
        }

    def _adaptive_hdbscan_args(self, doc_count: int) -> dict:
        if doc_count < 25:
            min_cluster_size = 2
        elif doc_count < 60:
            min_cluster_size = 3
        elif doc_count < 120:
            min_cluster_size = 5
        else:
            min_cluster_size = 8

        return {
            "min_cluster_size": min_cluster_size,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        }

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

        term = re.sub(r"_+", "_", term).strip("_")
        if not term:
            return None

        return term

    def _build_topic_label(self, topic_id: int, topic_words_row) -> str:
        """
        Deterministic topic label from cleaned Top2Vec topic words.
        """
        cleaned_terms = []
        seen = set()

        try:
            iterable_words = list(topic_words_row) if topic_words_row is not None else []
        except Exception:
            iterable_words = []

        for raw_word in iterable_words:
            clean_word = self._sanitize_term(raw_word)
            if clean_word and clean_word not in seen:
                seen.add(clean_word)
                cleaned_terms.append(clean_word)
            if len(cleaned_terms) >= 4:
                break

        if not cleaned_terms:
            return f"Topic_{topic_id}: unknown"

        return f"Topic_{topic_id}: {'_'.join(cleaned_terms[:4])}"

    def _safe_score(self, score_obj) -> float:
        """
        Top2Vec returns assignment strength / similarity-like values.
        We store them as topic_score without claiming calibrated probability.
        """
        try:
            if score_obj is None:
                return 0.0
            return float(score_obj)
        except Exception:
            return 0.0

    def _fallback_outputs(self, batch_texts: List[str], reason: str) -> Tuple[List[str], List[float]]:
        label = f"Topic_Fallback:{reason}"
        return [label] * len(batch_texts), [0.0] * len(batch_texts)

    # -------------------------------------------------------------- main API

    def fit_predict(self, batch_texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Fit Top2Vec on the provided batch and assign one topic per document.

        Returns:
            labels: deterministic topic labels aligned with input order
            scores: assignment-strength scores aligned with input order
        """
        if not batch_texts:
            return [], []

        usable_texts = [str(t).strip() if t is not None else "" for t in batch_texts]
        if not any(usable_texts):
            return self._fallback_outputs(batch_texts, "empty_batch")

        doc_count = len(usable_texts)
        min_count = self._adaptive_min_count(doc_count)
        umap_args = self._adaptive_umap_args(doc_count)
        hdbscan_args = self._adaptive_hdbscan_args(doc_count)

        self.logger.info(
            f"Fitting Top2Vec on {doc_count} docs "
            f"(embedding={self.embedding_model}, speed={self.speed}, "
            f"min_count={min_count}, umap_neighbors={umap_args['n_neighbors']}, "
            f"min_cluster_size={hdbscan_args['min_cluster_size']})"
        )

        try:
            model = Top2Vec(
                documents=usable_texts,
                speed=self.speed,
                embedding_model=self.embedding_model,
                min_count=min_count,
                umap_args=umap_args,
                hdbscan_args=hdbscan_args,
                **self.kwargs
            )

            topic_nums, topic_scores, topic_words, word_scores = model.get_documents_topics(
                doc_ids=list(range(doc_count))
            )

            labels_out: List[str] = []
            scores_out: List[float] = []

            for topic_id, score_obj, topic_words_row in zip(topic_nums, topic_scores, topic_words):
                label = self._build_topic_label(int(topic_id), topic_words_row)
                score = self._safe_score(score_obj)
                labels_out.append(label)
                scores_out.append(score)

            if len(labels_out) != len(batch_texts) or len(scores_out) != len(batch_texts):
                self.logger.error(
                    "Top2Vec output length mismatch. Falling back for entire batch."
                )
                return self._fallback_outputs(batch_texts, "length_mismatch")

            return labels_out, scores_out

        except ValueError as e:
            if "invalid embedding model" in str(e).lower():
                self.logger.exception(
                    f"Top2Vec fitting failed: invalid embedding model '{self.embedding_model}'. "
                    "Use a Top2Vec-supported embedding identifier such as "
                    "'distiluse-base-multilingual-cased' or another supported built-in option."
                )
            else:
                self.logger.exception(f"Top2Vec fitting failed: {e}")
            return self._fallback_outputs(batch_texts, "fit_failed")

        except Exception as e:
            self.logger.exception(f"Top2Vec fitting failed: {e}")
            return self._fallback_outputs(batch_texts, "fit_failed")