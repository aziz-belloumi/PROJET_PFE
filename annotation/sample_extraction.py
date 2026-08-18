"""
sample_extraction.py
--------------------
Extracts a stratified sample of annotated articles from the Data_Exploring project
(data_libelisation/global_data_libelised.csv) and prepares data/manual_eval_global.csv
for manual evaluation and benchmark evaluation.

Traceability & Language Partition:
  - 'ar' : Modern Standard Arabic (MSA)
  - 'da' : Dialectal Arabic (DA)
  - 'en' : English
  - 'fr' : French

Usage:
    python annotation/sample_extraction.py --n_ar 150 --n_da 150 --n_en 150 --n_fr 150
"""
from __future__ import annotations

import sys
import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_CSV = PROJECT_ROOT.parent / "Data_Exploring" / "data_libelisation" / "global_data_libelised.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "manual_eval_global.csv"


def extract_samples(
    source_csv: Path = DEFAULT_SOURCE_CSV,
    out_csv: Path = DEFAULT_OUTPUT_CSV,
    n_ar: int = 150,
    n_da: int = 150,
    n_en: int = 150,
    n_fr: int = 150,
    random_seed: int = 42,
) -> Path:
    """
    Extracts stratified samples from global_data_libelised.csv and saves to manual_eval_global.csv.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not source_csv.exists():
        raise FileNotFoundError(
            f"Source dataset not found at '{source_csv}'. "
            f"Please ensure Data_Exploring project has generated 'global_data_libelised.csv'."
        )

    logger.info(f"Loading source dataset from: {source_csv}")
    df_raw = pd.read_csv(source_csv, low_memory=False)

    # Standardize column names
    col_map = {c.lower().strip(): c for c in df_raw.columns}
    text_col = col_map.get("text", col_map.get("texte", "text"))
    lang_col = col_map.get("language", col_map.get("langue", "language"))
    sent_col = col_map.get("sentiment", "sentiment")
    topic_col = col_map.get("topic", col_map.get("thème", "topic"))

    df_raw["clean_lang"] = df_raw[lang_col].astype(str).str.lower().str.strip()

    samples = []
    sampling_plan = [
        ("ar", n_ar, "Modern Standard Arabic (MSA)"),
        ("da", n_da, "Dialectal Arabic (DA)"),
        ("en", n_en, "English"),
        ("fr", n_fr, "French"),
    ]

    for lang_code, n_target, lang_label in sampling_plan:
        if n_target <= 0:
            continue
        sub = df_raw[df_raw["clean_lang"] == lang_code]
        avail = len(sub)
        logger.info(f"Language '{lang_code}' ({lang_label}): {avail} available articles in source.")

        if avail == 0:
            logger.warning(f"No articles found for language '{lang_code}' in source dataset.")
            continue

        n_sample = min(n_target, avail)
        sampled_sub = sub.sample(n=n_sample, random_state=random_seed)
        samples.append(sampled_sub)
        logger.info(f"Sampled {n_sample}/{n_target} articles for '{lang_code}'.")

    if not samples:
        raise RuntimeError("No samples could be extracted from the source dataset.")

    df_sample = pd.concat(samples, ignore_index=True)

    # Build standard manual evaluation columns
    out = pd.DataFrame()
    out["texte"] = df_sample[text_col].fillna("").astype(str).str.strip()
    out["langue"] = df_sample["clean_lang"]
    out["thème attendu"] = df_sample[topic_col].fillna("").astype(str).str.strip() if topic_col in df_sample.columns else ""
    out["thème prédit"] = ""
    out["correct/incorrect theme"] = ""
    out["sentiment attendu"] = df_sample[sent_col].fillna("").astype(str).str.strip().str.upper() if sent_col in df_sample.columns else ""
    out["sentiment prédit"] = ""
    out["correct/incorrect sentiment"] = ""

    final_cols = [
        "texte",
        "langue",
        "thème attendu",
        "thème prédit",
        "correct/incorrect theme",
        "sentiment attendu",
        "sentiment prédit",
        "correct/incorrect sentiment",
    ]
    out = out[final_cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    logger.info(f"Successfully exported {len(out)} sample articles to: {out_csv}")
    print(f"Extraction complete! Saved: {out_csv} ({len(out)} articles total)")
    return out_csv


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract stratified samples from Data_Exploring dataset.")
    ap.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV, help="Path to global_data_libelised.csv")
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to output manual_eval_global.csv")
    ap.add_argument("--n_ar", type=int, default=150, help="Number of MSA Arabic articles (langue='ar')")
    ap.add_argument("--n_da", type=int, default=150, help="Number of Dialectal Arabic articles (langue='da')")
    ap.add_argument("--n_en", type=int, default=150, help="Number of English articles (langue='en')")
    ap.add_argument("--n_fr", type=int, default=150, help="Number of French articles (langue='fr')")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = ap.parse_args()

    extract_samples(
        source_csv=args.source_csv,
        out_csv=args.out_csv,
        n_ar=args.n_ar,
        n_da=args.n_da,
        n_en=args.n_en,
        n_fr=args.n_fr,
        random_seed=args.seed,
    )
