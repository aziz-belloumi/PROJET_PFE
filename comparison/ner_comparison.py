import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Your DB labels are already unified in src/ner_extraction.py
ALL_LABELS = ["PER", "ORG", "LOC", "DAT", "EVE", "MIS", "PRO", "COM"]

MODEL_NAME_MAP = {
    0: "arabert",
    1: "camel",
    2: "mbert",
}


def run_ner_comparison(engine):
    query = """
        SELECT
            ae.article_id,
            ae.model_version,
            ae.confidence_score,
            e.entity_type,
            e.normalized_name
        FROM article_entities ae
        JOIN entities e ON ae.entity_id = e.entity_id
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} entity records from DB")

    if df.empty:
        logger.warning("No NER entities found in DB.")
        return pd.DataFrame()

    # Normalize
    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce")
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["entity_type"] = df["entity_type"].astype(str).str.upper().str.strip()
    df["normalized_name"] = df["normalized_name"].astype(str).str.strip()

    df = df.dropna(subset=["article_id", "model_version", "entity_type", "normalized_name"])
    if df.empty:
        logger.warning("NER records empty after cleaning.")
        return pd.DataFrame()

    df["model_version"] = df["model_version"].astype(int)
    df["model"] = df["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # Pre-aggregate sets for agreement/uniqueness calculation
    # dict: mv -> set of (article_id, normalized_name, entity_type)
    model_sets: dict[int, set[tuple]] = {}
    for mv in sorted(df["model_version"].unique()):
        subset = df[df["model_version"] == mv][["article_id", "normalized_name", "entity_type"]]
        model_sets[int(mv)] = set(subset.itertuples(index=False, name=None))

    rows = []

    for mv in sorted(df["model_version"].unique()):
        subset = df[df["model_version"] == mv]
        n_entities = len(subset)

        model_name = MODEL_NAME_MAP.get(int(mv), "unknown")

        if n_entities == 0:
            rows.append({"model_version": int(mv), "model": model_name, "total_entities": 0})
            continue

        # Per-article stats
        counts_per_article = subset.groupby("article_id").size()
        mean_per = float(counts_per_article.mean()) if not counts_per_article.empty else 0.0
        std_per = float(counts_per_article.std()) if len(counts_per_article) > 1 else 0.0

        # Label counts
        label_counts = subset["entity_type"].value_counts()

        # Confidence stats
        scores = subset["confidence_score"].dropna()

        # Agreement / unique
        my_set = model_sets.get(int(mv), set())
        others_set = set()
        for other_mv, s in model_sets.items():
            if other_mv != int(mv):
                others_set.update(s)

        intersection = my_set.intersection(others_set)
        unique_to_me = my_set - others_set

        agreement_pct = (len(intersection) / len(my_set)) * 100 if len(my_set) > 0 else 0.0

        row = {
            "model_version": int(mv),
            "model": model_name,
            "total_entities": int(n_entities),
            "mean_per_article": round(mean_per, 2),
            "std_per_article": round(std_per, 2),
        }

        for lbl in ALL_LABELS:
            row[f"{lbl}_count"] = int(label_counts.get(lbl, 0))

        row.update({
            "mean_confidence": round(float(scores.mean()), 4) if not scores.empty else 0.0,
            "std_confidence": round(float(scores.std()), 4) if len(scores) > 1 else 0.0,
            "min_confidence": round(float(scores.min()), 4) if not scores.empty else 0.0,
            "median_confidence": round(float(scores.median()), 4) if not scores.empty else 0.0,
            "max_confidence": round(float(scores.max()), 4) if not scores.empty else 0.0,
            "agreement_pct": round(float(agreement_pct), 2),
            "unique_entities": int(len(unique_to_me)),
        })

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("model_version").reset_index(drop=True)
    logger.info(f"\n=== NER COMPARISON (LONG FORMAT) ===\n{result.to_string(index=False)}")
    return result