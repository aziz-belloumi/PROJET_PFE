"""
Aggregate original source CSV files and generate language-specific fine-tuning data.
This script lives inside original_data/ and processes CSV files present in that folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR.parent / "fine_tune_data"
BY_LANGUAGE_DIR = OUTPUT_DIR / "by_language"
CSV_EXT = ".csv"


def detect_file_type(path: Path) -> str:
    name = path.name.lower()
    if "global_data" in name or name.startswith("global"):
        return "global"
    if "articles_enriched" in name or "sentiment" in name:
        return "sentiment"
    if "article_topics" in name or "topic" in name:
        return "topic"
    return "other"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())
    if "article_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"article_id": "id"})
    df["source_file"] = path.name
    return df


def collect_source_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob(f"*{CSV_EXT}") if p.is_file() and p.name != Path(__file__).name])


def merge_all_files(input_dir: Path) -> pd.DataFrame:
    """Load all CSV files from input_dir, merge global/sentiment/topic data, and return one DataFrame."""
    source_files = collect_source_files(input_dir)
    if not source_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    global_parts: List[pd.DataFrame] = []
    sentiment_parts: List[pd.DataFrame] = []
    topic_parts: List[pd.DataFrame] = []

    for path in source_files:
        file_type = detect_file_type(path)
        df = load_csv(path)
        if file_type == "global":
            global_parts.append(df)
        elif file_type == "sentiment":
            sentiment_parts.append(df)
        elif file_type == "topic":
            topic_parts.append(df)
        elif {"text", "language"}.issubset(df.columns):
            global_parts.append(df)

    if not global_parts:
        raise ValueError("No global data files found.")

    global_df = pd.concat(global_parts, ignore_index=True)
    sentiment_df = pd.concat(sentiment_parts, ignore_index=True) if sentiment_parts else pd.DataFrame()
    topic_df = pd.concat(topic_parts, ignore_index=True) if topic_parts else pd.DataFrame()

    if "topic_label" not in topic_df.columns and "topic" in topic_df.columns:
        topic_df = topic_df.rename(columns={"topic": "topic_label"})
    if "sentiment_label" not in sentiment_df.columns and "sentiment" in sentiment_df.columns:
        sentiment_df = sentiment_df.rename(columns={"sentiment": "sentiment_label"})

    merged = global_df.loc[:, [c for c in ["id", "text", "language"] if c in global_df.columns]].copy()
    if "id" not in merged.columns:
        raise ValueError("Global data files must contain an 'id' column.")

    if not sentiment_df.empty and "id" in sentiment_df.columns:
        merged = merged.merge(
            sentiment_df.loc[:, [c for c in ["id", "sentiment_label"] if c in sentiment_df.columns]],
            on="id",
            how="left",
        )
    else:
        merged["sentiment_label"] = pd.NA

    if not topic_df.empty and "id" in topic_df.columns:
        merged = merged.merge(
            topic_df.loc[:, [c for c in ["id", "topic_label"] if c in topic_df.columns]],
            on="id",
            how="left",
        )
    else:
        merged["topic_label"] = pd.NA

    merged = merged.rename(columns={"sentiment_label": "sentiment", "topic_label": "topic"})
    merged["language"] = merged["language"].fillna("unknown")

    return merged[[c for c in ["id", "text", "language", "sentiment", "topic"] if c in merged.columns]]


def build_stats(df: pd.DataFrame) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "total_rows": int(len(df)),
        "unique_ids": int(df["id"].nunique()) if "id" in df.columns else 0,
        "language_counts": df["language"].fillna("unknown").value_counts().to_dict() if "language" in df.columns else {},
        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in ["text", "language", "sentiment", "topic"]
            if col in df.columns
        },
    }
    if "sentiment" in df.columns:
        stats["sentiment_counts"] = df["sentiment"].fillna("unknown").value_counts().to_dict()
    if "topic" in df.columns:
        stats["topic_counts"] = df["topic"].fillna("unknown").value_counts().to_dict()
    return stats


def format_stats_text(stats: Dict[str, object]) -> str:
    lines: List[str] = ["Fine-tuning data stats", "=" * 32, ""]
    lines.append(f"Total rows: {stats.get('total_rows')}")
    lines.append(f"Unique IDs: {stats.get('unique_ids')}")
    lines.append("")
    lines.append("Language counts:")
    for language, count in sorted(stats.get("language_counts", {}).items(), key=lambda x: str(x[0])):
        lines.append(f"  {language}: {count}")
    if stats.get("sentiment_counts"):
        lines.append("")
        lines.append("Sentiment counts:")
        for label, count in sorted(stats.get("sentiment_counts", {}).items(), key=lambda x: str(x[0])):
            lines.append(f"  {label}: {count}")
    if stats.get("topic_counts"):
        lines.append("")
        lines.append("Topic counts:")
        for label, count in sorted(stats.get("topic_counts", {}).items(), key=lambda x: str(x[0])):
            lines.append(f"  {label}: {count}")
    if stats.get("missing_counts"):
        lines.append("")
        lines.append("Missing values:")
        for col, count in sorted(stats["missing_counts"].items()):
            lines.append(f"  {col}: {count}")
    return "\n".join(lines)


def save_by_language(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for language, group in df.groupby(df["language"].fillna("unknown")):
        language_name = str(language).strip().lower().replace(" ", "_") or "unknown"
        group.to_csv(output_dir / f"global_data_{language_name}.csv", index=False)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate original CSV files and generate language-specific fine-tuning data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output folder for merged and language-specific files.",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    merged_data = merge_all_files(ROOT_DIR)
    args.output.mkdir(parents=True, exist_ok=True)
    merged_file = args.output / "global_data_merged.csv"
    merged_data.to_csv(merged_file, index=False)
    save_by_language(merged_data, BY_LANGUAGE_DIR)

    stats = build_stats(merged_data)
    stats_json = args.output / "fine_tuning_stats.json"
    stats_txt = args.output / "fine_tuning_stats.txt"
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_txt.write_text(format_stats_text(stats), encoding="utf-8")

    print(f"Merged global data saved to: {merged_file}")
    print(f"By-language files saved to: {BY_LANGUAGE_DIR}")
    print(f"Fine-tuning stats saved to: {stats_json} and {stats_txt}")


if __name__ == "__main__":
    main()
