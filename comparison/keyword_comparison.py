# comparison/keyword_comparison.py
import re
import logging
from collections import Counter

import pandas as pd


logger = logging.getLogger(__name__)

KEYWORD_PATTERN = re.compile(r"([^\(,]+?)\s*\(([\d.]+)\)")


def _parse_keywords(kw_string):
    """Parse 'word (0.4134), word (0.2756)' into list of dicts."""
    if not kw_string or pd.isna(kw_string):
        return []
    results = []
    for match in KEYWORD_PATTERN.finditer(kw_string):
        results.append({
            "word": match.group(1).strip(),
            "score": float(match.group(2)),
        })
    return results


def run_keyword_comparison(engine, top_n=30):

    df = pd.read_sql("SELECT * FROM keyword_results", engine)
    logger.info(f"Loaded {len(df)} rows from keyword_results")

    counts = []
    scores = []
    word_counts = Counter()
    word_scores = {}

    for val in df["keywords"]:
        kws = _parse_keywords(val)
        counts.append(len(kws))
        for kw in kws:
            w = kw["word"]
            s = kw["score"]
            scores.append(s)
            word_counts[w] += 1
            if w not in word_scores:
                word_scores[w] = []
            word_scores[w].append(s)

    count_series = pd.Series(counts) if counts else pd.Series([0])
    score_series = pd.Series(scores) if scores else pd.Series([0.0])

    rows = [
        {"metric": "total_articles", "value": str(len(df))},
        {"metric": "avg_keywords_per_article", "value": str(round(count_series.mean(), 2))},
        {"metric": "std_keywords_per_article", "value": str(round(count_series.std(), 2))},
        {"metric": "avg_tfidf_score", "value": str(round(score_series.mean(), 4))},
        {"metric": "total_unique_keywords", "value": str(len(set(word_counts.keys())))},
        {"metric": "---", "value": "--- TOP KEYWORDS ---"},
    ]

    # Top keywords
    for word, freq in word_counts.most_common(top_n):
        avg = round(pd.Series(word_scores[word]).mean(), 4)
        rows.append({
            "metric": word,
            "value": f"freq={freq}, avg_score={avg}",
        })

    result = pd.DataFrame(rows)
    logger.info(f"\n=== KEYWORD COMPARISON ===\n{result.to_string(index=False)}")
    return result