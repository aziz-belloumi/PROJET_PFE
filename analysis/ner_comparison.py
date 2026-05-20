import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Your DB labels are already unified in src/ner_extraction.py
ALL_LABELS = ["PER", "ORG", "LOC", "DAT", "EVE", "MIS", "PRO", "COM"]

# CURRENT model versions in your pipeline
MODEL_NAME_MAP = {
    0: "arabert",
    1: "camel",
    2: "en_bert",
    3: "fr_bert",
}


def run_ner_comparison(engine):
    """
    Updated for your current multilingual setup.

    Key fix vs old code:
    - Adds language by joining articles_enriched on (article_id, model_version)
    - Computes agreement ONLY within the same language (otherwise meaningless)
    - Deduplicates entity occurrences per article using (article_id, entity_type, normalized_name)
      and keeps max confidence for that key.
    - Output is one row per (language, model_version)
    """

    query = """
        SELECT
            ae.article_id,
            ae.model_version,
            ae.confidence_score,
            e.entity_type,
            e.normalized_name,
            a.language
        FROM article_entities ae
        JOIN entities e
          ON ae.entity_id = e.entity_id
        LEFT JOIN articles_enriched a
          ON a.article_id = ae.article_id
         AND a.model_version = ae.model_version
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} entity records from DB")

    if df.empty:
        logger.warning("No NER entities found in DB.")
        return pd.DataFrame()

    # -----------------------
    # Normalize / clean
    # -----------------------
    df["model_version"] = pd.to_numeric(df["model_version"], errors="coerce")
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["entity_type"] = df["entity_type"].astype(str).str.upper().str.strip()
    df["normalized_name"] = df["normalized_name"].astype(str).str.strip()
    df["language"] = df.get("language", "").astype(str).str.lower().str.strip()

    df = df.dropna(subset=["article_id", "model_version", "entity_type", "normalized_name"]).copy()
    df = df[df["entity_type"] != ""].copy()

    # Remove useless normalized_name artifacts
    df["normalized_name"] = df["normalized_name"].replace({"nan": "", "none": ""})
    df = df[df["normalized_name"] != ""].copy()

    if df.empty:
        logger.warning("NER records empty after cleaning.")
        return pd.DataFrame()

    df["model_version"] = df["model_version"].astype(int)
    df["model"] = df["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # If language missing, infer from model_version (fallback)
    missing_lang = df["language"].isin(["", "xx", "nan", "none"])
    if missing_lang.any():
        mv_to_lang = {0: "ar", 1: "ar", 2: "en", 3: "fr"}
        df.loc[missing_lang, "language"] = df.loc[missing_lang, "model_version"].map(mv_to_lang).fillna("xx")

    # Keep only supported langs
    df = df[df["language"].isin(["ar", "en", "fr"])].copy()
    if df.empty:
        logger.warning("NER records empty after language filtering.")
        return pd.DataFrame()

    # -----------------------
    # Deduplicate at the right level:
    # one entity occurrence per article, per model, per (entity_type, normalized_name)
    # keep max confidence as representative
    # -----------------------
    df_u = (
        df.groupby(
            ["language", "model_version", "article_id", "entity_type", "normalized_name"],
            as_index=False,
        )["confidence_score"]
        .max()
    )
    df_u["model"] = df_u["model_version"].map(MODEL_NAME_MAP).fillna("unknown")

    # -----------------------
    # Build sets for agreement within each language
    # set key = (article_id, entity_type, normalized_name)
    # -----------------------
    model_sets_by_lang: dict[str, dict[int, set[tuple]]] = {}
    for (lang, mv), g in df_u.groupby(["language", "model_version"]):
        keyset = set(g[["article_id", "entity_type", "normalized_name"]].itertuples(index=False, name=None))
        model_sets_by_lang.setdefault(str(lang), {})[int(mv)] = keyset

    # -----------------------
    # Per-(language, model_version) stats
    # -----------------------
    rows = []
    for (lang, mv), g in df_u.groupby(["language", "model_version"]):
        lang = str(lang)
        mv = int(mv)
        model_name = MODEL_NAME_MAP.get(mv, "unknown")

        # total unique entity-occurrences (after dedup)
        total_entities = int(len(g))

        # per-article stats
        counts_per_article = g.groupby("article_id").size()
        mean_per = float(counts_per_article.mean()) if not counts_per_article.empty else 0.0
        std_per = float(counts_per_article.std()) if len(counts_per_article) > 1 else 0.0

        # label counts
        label_counts = g["entity_type"].value_counts()

        # confidence stats
        scores = g["confidence_score"].dropna()

        row = {
            "language": lang,
            "model_version": mv,
            "model": model_name,
            "total_entities": total_entities,
            "mean_per_article": round(mean_per, 2),
            "std_per_article": round(std_per, 2),
            # extra useful metric: distinct entity strings (type+name) for that model+lang
            "unique_entity_strings": int(g[["entity_type", "normalized_name"]].drop_duplicates().shape[0]),
        }

        for lbl in ALL_LABELS:
            row[f"{lbl}_count"] = int(label_counts.get(lbl, 0))

        row.update({
            "mean_confidence": round(float(scores.mean()), 4) if not scores.empty else None,
            "std_confidence": round(float(scores.std()), 4) if len(scores) > 1 else (0.0 if len(scores) == 1 else None),
            "min_confidence": round(float(scores.min()), 4) if not scores.empty else None,
            "median_confidence": round(float(scores.median()), 4) if not scores.empty else None,
            "max_confidence": round(float(scores.max()), 4) if not scores.empty else None,
        })

        # Agreement / unique (ONLY meaningful if there is >1 model for this language)
        my_set = model_sets_by_lang.get(lang, {}).get(mv, set())
        other_models = {k: v for k, v in model_sets_by_lang.get(lang, {}).items() if k != mv}

        if not other_models:
            row["agreement_pct"] = None
            row["unique_entities"] = None
        else:
            others_union = set().union(*other_models.values()) if other_models else set()
            intersection = my_set.intersection(others_union)
            unique_to_me = my_set - others_union
            agreement_pct = (len(intersection) / len(my_set)) * 100.0 if len(my_set) > 0 else 0.0

            row["agreement_pct"] = round(float(agreement_pct), 2)
            row["unique_entities"] = int(len(unique_to_me))

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sort nicely: ar then en then fr
    lang_order = {"ar": 0, "en": 1, "fr": 2}
    result["_lang_order"] = result["language"].map(lang_order).fillna(99)
    result = result.sort_values(["_lang_order", "model_version"]).drop(columns=["_lang_order"]).reset_index(drop=True)

    logger.info(f"\n=== NER COMPARISON (LONG FORMAT) ===\n{result.to_string(index=False)}")
    return result
