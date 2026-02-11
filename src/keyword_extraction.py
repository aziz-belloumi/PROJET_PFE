from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import logging

class TFIDFKeywordExtractor:
    def __init__(self, stopwords: List[str], max_features: int = 1000, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            max_features=max_features,
            max_df=0.95,  # ignore words appearing in >95% of docs
            min_df=1      # ignore words appearing in <2 docs
        )
        self.fitted = False
        self.feature_names = []

    def fit(self, texts: List[str]):
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
        """Extract top N keywords for a single text."""
        if not self.fitted:
            raise ValueError("TF-IDF model not fitted. Call fit() first.")
        
        if not text.strip():
            return []

        # Transform single document
        tfidf_matrix = self.vectorizer.transform([text])
        feature_index = tfidf_matrix.indices
        tfidf_scores = tfidf_matrix.data

        # Sort by score descending
        sorted_items = sorted(zip(feature_index, tfidf_scores), key=lambda x: x[1], reverse=True)

        # Get top N words
        keywords = []
        for idx, score in sorted_items[:top_n]:
            keywords.append((self.feature_names[idx], float(score)))

        return keywords