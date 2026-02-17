from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


class TFIDFKeywordExtractor:
    def __init__(self, stopwords: List[str], max_features: int = 1000, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            max_features=max_features,
            max_df=0.95,  # ignore words appearing in >95% of docs
            min_df=1      # keep words that appear in at least 1 doc (useful for small samples)
        )
        self.fitted = False
        self.feature_names = []

    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF model on the corpus."""
        if not texts:
            self.logger.warning("No texts provided for keyword extraction fit.")
            return

        self.logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.fitted = True
        self.logger.info("TF-IDF fitted successfully.")

    def extract(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Extract top N keywords for a single text (fixed-size output).
        Kept for backward compatibility.
        """
        if not self.fitted:
            raise ValueError("TF-IDF model not fitted. Call fit() first.")

        if not text or not text.strip():
            return []

        tfidf_matrix = self.vectorizer.transform([text])
        feature_index = tfidf_matrix.indices
        tfidf_scores = tfidf_matrix.data

        sorted_items = sorted(
            zip(feature_index, tfidf_scores),
            key=lambda x: x[1],
            reverse=True
        )

        keywords: List[Tuple[str, float]] = []
        for idx, score in sorted_items[: max(0, int(top_n))]:
            keywords.append((self.feature_names[idx], float(score)))

        return keywords

    def extract_flexible(
        self,
        text: str,
        *,
        min_k: int = 5,
        max_k: int = 25,
        rel_threshold: float = 0.30,
        coverage_target: float = 0.70,
    ) -> List[Tuple[str, float]]:
        """
        Flexible keyword extraction (Option D - hybrid rule):

        - Always return at least min_k keywords (if available).
        - Then keep adding keywords while:
            * score >= rel_threshold * best_score
              OR
            * cumulative_score / total_score < coverage_target
        - Never exceed max_k.

        This allows long/information-dense articles to output more keywords
        while keeping short articles compact.

        Notes:
        - rel_threshold should be in [0, 1]. Typical: 0.2 to 0.4
        - coverage_target should be in (0, 1]. Typical: 0.6 to 0.85
        """
        if not self.fitted:
            raise ValueError("TF-IDF model not fitted. Call fit() first.")

        if not text or not text.strip():
            return []

        min_k = max(0, int(min_k))
        max_k = max(min_k, int(max_k))

        tfidf_matrix = self.vectorizer.transform([text])
        feature_index = tfidf_matrix.indices
        tfidf_scores = tfidf_matrix.data

        if feature_index is None or len(feature_index) == 0:
            return []

        sorted_items = sorted(
            zip(feature_index, tfidf_scores),
            key=lambda x: x[1],
            reverse=True
        )

        best_score = float(sorted_items[0][1]) if sorted_items else 0.0
        total_score = float(tfidf_scores.sum()) if tfidf_scores is not None and len(tfidf_scores) else 0.0

        keywords: List[Tuple[str, float]] = []
        cumulative = 0.0

        for idx, score_np in sorted_items:
            if len(keywords) >= max_k:
                break

            score = float(score_np)

            # Always satisfy minimum
            if len(keywords) < min_k:
                keywords.append((self.feature_names[idx], score))
                cumulative += score
                continue

            # After min_k: apply hybrid rule
            keep_by_relative = False
            if rel_threshold is not None and best_score > 0:
                keep_by_relative = score >= float(rel_threshold) * best_score

            keep_by_coverage = False
            if coverage_target is not None and total_score > 0:
                keep_by_coverage = (cumulative / total_score) < float(coverage_target)

            if keep_by_relative or keep_by_coverage:
                keywords.append((self.feature_names[idx], score))
                cumulative += score
            else:
                # Stop once terms are no longer strong and coverage is reached
                break

        return keywords