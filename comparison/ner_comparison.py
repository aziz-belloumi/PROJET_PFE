import re
import logging
from collections import Counter

import pandas as pd


logger = logging.getLogger(__name__)

ENTITY_PATTERN = re.compile(r"([^,]+?)\s*\(([A-Z]{3}),\s*([\d.]+)\)")

MODEL_COLUMNS = {
    "arabert": "arabert_entities",
    "camel": "camel_entities",
    "mbert": "mbert_entities",
}

ALL_LABELS = ["PER", "LOC", "ORG", "EVE", "MIS"]


def _parse_entities(entity_string):
    """Parse 'text (LBL, 0.94), text (LBL, 0.87)' into list of dicts."""
    if not entity_string or pd.isna(entity_string):
        return []
    results = []
    for match in ENTITY_PATTERN.finditer(entity_string):
        results.append({
            "text": match.group(1).strip(),
            "label": match.group(2),
            "score": float(match.group(3)),
        })
    return results


def run_ner_comparison(engine):

    df = pd.read_sql("SELECT * FROM ner_results", engine)
    logger.info(f"Loaded {len(df)} rows from ner_results")

    model_names = list(MODEL_COLUMNS.keys())

    # Pre-parse all entities
    parsed = {}
    for model, col in MODEL_COLUMNS.items():
        parsed[model] = [_parse_entities(val) for val in df[col]]

    rows = []
    for model in model_names:
        ents_per_article = parsed[model]

        # Counts
        counts = [len(e) for e in ents_per_article]
        count_series = pd.Series(counts) if counts else pd.Series([0])

        # All scores and labels flattened
        all_scores = [ent["score"] for article in ents_per_article for ent in article]
        all_labels = [ent["label"] for article in ents_per_article for ent in article]
        score_series = pd.Series(all_scores) if all_scores else pd.Series([0.0])
        label_counts = Counter(all_labels)

        # Agreement: % of my entities also found by at least one other model
        others = [m for m in model_names if m != model]
        total_mine = 0
        total_agreed = 0
        for i in range(len(df)):
            my_set = {(e["text"], e["label"]) for e in ents_per_article[i]}
            others_set = set()
            for o in others:
                others_set |= {(e["text"], e["label"]) for e in parsed[o][i]}
            total_mine += len(my_set)
            total_agreed += len(my_set & others_set)
        agreement_pct = round(100 * total_agreed / total_mine, 2) if total_mine > 0 else 0

        # Unique: entities found only by this model
        total_unique = 0
        for i in range(len(df)):
            my_set = {(e["text"], e["label"]) for e in ents_per_article[i]}
            others_set = set()
            for o in others:
                others_set |= {(e["text"], e["label"]) for e in parsed[o][i]}
            total_unique += len(my_set - others_set)

        # Build row
        row = {
            "model": model,
            "total_entities": int(sum(counts)),
            "mean_per_article": round(count_series.mean(), 2),
            "std_per_article": round(count_series.std(), 2),
        }
        for lbl in ALL_LABELS:
            row[f"{lbl}_count"] = label_counts.get(lbl, 0)
        row.update({
            "mean_confidence": round(score_series.mean(), 4),
            "std_confidence": round(score_series.std(), 4),
            "min_confidence": round(score_series.min(), 4),
            "median_confidence": round(score_series.median(), 4),
            "agreement_pct": agreement_pct,
            "unique_entities": total_unique,
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info(f"\n=== NER COMPARISON ===\n{result.to_string(index=False)}")
    return result