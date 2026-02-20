from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import re


class TFIDFKeywordExtractor:
    def __init__(
        self,
        stopwords: List[str],
        max_features: int = 1000,
        logger=None,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        split_prefixes: bool = False,          # keep prefixes attached by default
        strip_definite_article: bool = False,  # keep "ال" by default
        remove_redundancy: bool = True,        # remove unigram redundancy when bigrams exist
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.split_prefixes = bool(split_prefixes)
        self.strip_definite_article = bool(strip_definite_article)
        self.remove_redundancy = bool(remove_redundancy)

        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            max_features=max_features,
            max_df=0.95,
            min_df=1,
            ngram_range=ngram_range,
        )

        self.fitted = False
        self.feature_names = []

        self._re_ar_word = re.compile(r"^[\u0600-\u06FF]{2,}$")
        self._re_ll_prefix = re.compile(r"\bلل(?=[\u0600-\u06FF]{2,})")
        self._re_single_prefix = re.compile(r"\b([وفبلك])(?=[\u0600-\u06FF]{2,})")

        # used for ordering keywords by first appearance
        self._boundary_chars = r"\s\.,،؛:!?()\[\]{}\"'«»ـ\-–—/\\"

    def _normalize_for_tfidf(self, text: str) -> str:
        if not text:
            return ""

        t = text

        if self.split_prefixes:
            t = self._re_ll_prefix.sub("ل ", t)
            t = self._re_single_prefix.sub(r"\1 ", t)

        if self.strip_definite_article:
            tokens = t.split()
            out = []
            for tok in tokens:
                if tok == "الله":
                    out.append(tok)
                    continue
                if tok.startswith("ال") and len(tok) > 3:
                    rest = tok[2:]
                    if self._re_ar_word.match(rest):
                        out.append(rest)
                    else:
                        out.append(tok)
                else:
                    out.append(tok)
            t = " ".join(out)

        return t

    @staticmethod
    def _norm_for_redundancy(tok: str) -> str:
        """
        Normalization used ONLY for redundancy detection.
        Output keywords remain unchanged.

        الهدف: اعتبار (بالناتو/والناتو/الناتو) نفس الشيء لغرض حذف التكرار
        عند وجود bigram مثل "حلف الناتو".
        """
        t = (tok or "").strip()
        if not t:
            return ""

        if len(t) >= 4 and t[0] in {"و", "ف", "ب", "ل", "ك"}:
            t = t[1:]

        if t.startswith("ال") and len(t) >= 4:
            t = t[2:]

        return t.strip()

    @classmethod
    def _reduce_redundancy(cls, keywords: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Remove unigrams that are covered by selected multi-word terms (bigrams),
        using robust matching via _norm_for_redundancy().
        """
        if not keywords:
            return keywords

        multi_terms = [t.strip() for (t, _) in keywords if " " in (t or "").strip()]
        if not multi_terms:
            return keywords

        covered = set()
        for term in multi_terms:
            parts = [p.strip() for p in term.split() if p.strip()]
            for p in parts:
                covered.add(p)
                covered.add(cls._norm_for_redundancy(p))

        filtered: List[Tuple[str, float]] = []
        for term, score in keywords:
            term_str = (term or "").strip()

            if " " not in term_str:
                if term_str in covered or cls._norm_for_redundancy(term_str) in covered:
                    continue

            filtered.append((term_str, score))

        return filtered

    def _first_pos(self, text: str, term: str) -> int:
        """
        First occurrence position of `term` in `text`.
        If not found, returns a large number so it goes to the end.
        """
        if not text or not term:
            return 10**12

        t = term.strip()
        if not t:
            return 10**12

        # boundary-aware regex (start/end OR whitespace/punct around term)
        pre = rf"(?:^|[{self._boundary_chars}])"
        post = rf"(?:$|[{self._boundary_chars}])"
        pat = re.escape(t)
        rgx = re.compile(pre + r"(" + pat + r")" + post)

        m = rgx.search(text)
        if m:
            return m.start(1)

        # fallback: simple find
        idx = text.find(t)
        return idx if idx >= 0 else 10**12

    def _order_by_text(self, text: str, keywords: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Reorder selected keywords to follow their first appearance in the article text.
        """
        if not keywords:
            return keywords
        return sorted(keywords, key=lambda kv: self._first_pos(text, kv[0]))

    def fit(self, texts: List[str]) -> None:
        if not texts:
            self.logger.warning("No texts provided for keyword extraction fit.")
            return

        self.logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        norm_texts = [self._normalize_for_tfidf(x) for x in texts]

        self.vectorizer.fit(norm_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.fitted = True
        self.logger.info("TF-IDF fitted successfully.")

    def extract(
        self,
        text: str,
        top_n: int = 5,
        *,
        preserve_text_order: bool = True,
    ) -> List[Tuple[str, float]]:
        """Fixed-size extraction (backward compatibility)."""
        if not self.fitted:
            raise ValueError("TF-IDF model not fitted. Call fit() first.")
        if not text or not text.strip():
            return []

        norm_text = self._normalize_for_tfidf(text)

        tfidf_matrix = self.vectorizer.transform([norm_text])
        feature_index = tfidf_matrix.indices
        tfidf_scores = tfidf_matrix.data

        sorted_items = sorted(zip(feature_index, tfidf_scores), key=lambda x: x[1], reverse=True)

        out: List[Tuple[str, float]] = []
        for idx, score in sorted_items[: max(0, int(top_n))]:
            out.append((self.feature_names[idx], float(score)))

        if self.remove_redundancy:
            out = self._reduce_redundancy(out)

        if preserve_text_order:
            out = self._order_by_text(norm_text, out)

        return out

    def extract_flexible(
        self,
        text: str,
        *,
        min_k: int = 5,
        max_k: int = 15,
        rel_threshold: float = 0.20,
        target_energy: float = 0.75,
        coverage_target: Optional[float] = None,  # alias
        preserve_text_order: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Minimal-words extraction with TF-IDF energy coverage:
            coverage = sum(score^2 of selected) / sum(score^2 of all terms)

        - Always take at least min_k (if available).
        - Then add the smallest number of additional terms until coverage >= target_energy,
          capped by max_k.
        - Soft stop: if score is very small and we're already very close to target.
        """
        if not self.fitted:
            raise ValueError("TF-IDF model not fitted. Call fit() first.")
        if not text or not text.strip():
            return []

        if coverage_target is not None:
            target_energy = float(coverage_target)

        min_k = max(0, int(min_k))
        max_k = max(min_k, int(max_k))
        target_energy = max(0.0, min(1.0, float(target_energy)))
        rel_threshold = max(0.0, float(rel_threshold))

        norm_text = self._normalize_for_tfidf(text)

        tfidf_matrix = self.vectorizer.transform([norm_text])
        feature_index = tfidf_matrix.indices
        tfidf_scores = tfidf_matrix.data

        if feature_index is None or len(feature_index) == 0:
            return []

        sorted_items = sorted(zip(feature_index, tfidf_scores), key=lambda x: x[1], reverse=True)

        best_score = float(sorted_items[0][1]) if sorted_items else 0.0
        if best_score <= 0.0:
            return []

        total_energy = float((tfidf_scores ** 2).sum()) if tfidf_scores is not None and len(tfidf_scores) else 0.0
        if total_energy <= 0.0:
            return []

        out: List[Tuple[str, float]] = []
        cum_energy = 0.0

        for idx, score_np in sorted_items:
            if len(out) >= max_k:
                break

            score = float(score_np)
            energy = score * score

            if len(out) < min_k:
                out.append((self.feature_names[idx], score))
                cum_energy += energy
                continue

            coverage = cum_energy / total_energy
            if coverage >= target_energy:
                break

            if score < rel_threshold * best_score and coverage >= (target_energy * 0.98):
                break

            out.append((self.feature_names[idx], score))
            cum_energy += energy

        if self.remove_redundancy:
            out = self._reduce_redundancy(out)

        if preserve_text_order:
            out = self._order_by_text(norm_text, out)

        return out