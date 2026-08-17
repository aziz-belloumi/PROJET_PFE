#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Articles from MySQL Database & Preprocess into global_data.csv (Parallelised)
====================================================================================
Connects to MySQL database, extracts raw news articles in parallel batches of
100,000 rows (sub-batches of 10,000 across CPU cores), classifies each article
by language, writes valid articles into `global_data.csv` (id, text, language),
and outputs extraction metrics to `processing_stats.txt`.

Output CSV format:
  id, text, language
"""

import os
import sys
import csv
import argparse
import tempfile
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is in sys.path
CURRENT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db_config import DatabaseConnection
from src.language_detection import ArticleClassifier, FastTextLanguageDetector, MIN_CONFIDENCE_THRESHOLD, MIN_ALPHA_RATIO
from src.text_utils import init_console_encoding

# Initialize console encoding (prevents Windows UTF-8 crash on Arabic output)
init_console_encoding()

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CHUNK_SIZE     = 100_000
SUB_BATCH_SIZE = 10_000
NUM_WORKERS    = max(1, (os.cpu_count() or 4) - 1)

OUTPUT_CSV    = "global_data.csv"
HEADER_GLOBAL = ["id", "text", "language"]


# ------------------------------------------------------------------
# Worker function — runs in a child process (must be picklable)
# ------------------------------------------------------------------
def process_sub_batch(rows: list) -> list:
    classifier = ArticleClassifier()
    return [classifier.classify(article_id, body) for article_id, body in rows]


# ------------------------------------------------------------------
# CSV Writer Helper
# ------------------------------------------------------------------
def open_csv_writer(path: Path, header: list):
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(header)
    return fh, writer


def get_total_processed_count(csv_path: Path) -> int:
    """Returns the number of articles already written to global_data.csv."""
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    return 0


# ------------------------------------------------------------------
# Flush Batch Results into CSV + Counts
# ------------------------------------------------------------------
def flush_results(results: list, writer: csv.writer, counts: dict) -> None:
    samples = counts.setdefault("samples", {})
    for item in results:
        dest = item["dest"]
        counts["total"] += 1
        aid = item.get("id", item.get("row", [0])[0])
        sample_text = item.get("sample", "").replace("\n", " ").strip()

        if dest in ("ar", "en", "fr"):
            writer.writerow(item["row"])
            counts[dest] += 1
            counts[f"{dest}_words"]    += item.get("words", 0)
            counts[f"{dest}_conf_sum"] += item.get("conf", 0.0)
            if dest not in samples:
                samples[dest] = []
            if len(samples[dest]) < 5 and sample_text:
                samples[dest].append((aid, sample_text))
        elif dest == "rejected":
            counts["rejected"] += 1
            reason = item.get("reason", "rej_unsupported_lang")
            counts[reason] = counts.get(reason, 0) + 1
            if reason not in samples:
                samples[reason] = []
            if len(samples[reason]) < 5:
                samples[reason].append((aid, sample_text or "<empty>"))


# ------------------------------------------------------------------
# Processing Report
# ------------------------------------------------------------------
def save_processing_report(output_dir: Path, counts: dict) -> None:
    total    = counts["total"]
    accepted = counts["ar"] + counts["en"] + counts["fr"]
    rejected = counts["rejected"]
    acc_rate = (accepted / total * 100) if total else 0
    rej_rate = (rejected / total * 100) if total else 0
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def pct(n, d=total):
        return f"{n:>10,}  ({n/d*100:5.1f}%)" if d else f"{n:>10,}  (  N/A )"

    def format_samples(key: str) -> list:
        s_list = counts.get("samples", {}).get(key, [])
        if not s_list:
            return []
        out = [f"      -> Examples ({len(s_list)}):"]
        for idx, (aid, txt) in enumerate(s_list, 1):
            snippet = txt[:95] + ("..." if len(txt) > 95 else "")
            out.append(f"         [{idx}] ID {aid:<7d} : {snippet}")
        return out

    sep  = "=" * 60
    sep2 = "-" * 60

    lines = [
        sep,
        "  DATABASE ARTICLE EXTRACTION & PREPROCESSING REPORT",
        f"  Run date     : {run_time}",
        f"  Workers      : {NUM_WORKERS}   Sub-batch : {SUB_BATCH_SIZE:,}",
        f"  Total        : {total:,} articles",
        sep,
        "",
        f"  -- ACCEPTED  {accepted:,} / {total:,}  ({acc_rate:.1f}%) --",
        sep2,
        f"    Arabic   (ar)                    : {pct(counts['ar'])}",
        f"      -> Total words                 : {counts.get('ar_words', 0):,}",
        f"      -> Avg words/article           : {counts.get('ar_words', 0) / max(1, counts['ar']):.1f}",
        f"      -> Avg confidence              : {counts.get('ar_conf_sum', 0) / max(1, counts['ar']):.3f}",
        *format_samples("ar"),
        "",
        f"    English  (en)                    : {pct(counts['en'])}",
        f"      -> Total words                 : {counts.get('en_words', 0):,}",
        f"      -> Avg words/article           : {counts.get('en_words', 0) / max(1, counts['en']):.1f}",
        f"      -> Avg confidence              : {counts.get('en_conf_sum', 0) / max(1, counts['en']):.3f}",
        *format_samples("en"),
        "",
        f"    French   (fr)                    : {pct(counts['fr'])}",
        f"      -> Total words                 : {counts.get('fr_words', 0):,}",
        f"      -> Avg words/article           : {counts.get('fr_words', 0) / max(1, counts['fr']):.1f}",
        f"      -> Avg confidence              : {counts.get('fr_conf_sum', 0) / max(1, counts['fr']):.3f}",
        *format_samples("fr"),
        "",
        f"  -- REJECTED  {rejected:,} / {total:,}  ({rej_rate:.1f}%) --",
        sep2,
        f"    Too short     (<= 2 words)       : {pct(counts.get('rej_too_short', 0))}",
        *format_samples("rej_too_short"),
        "",
        f"    Low confidence (< {MIN_CONFIDENCE_THRESHOLD})          : {pct(counts.get('rej_low_confidence', 0))}",
        *format_samples("rej_low_confidence"),
        "",
        f"    Low alphabetic ratio (< {MIN_ALPHA_RATIO})      : {pct(counts.get('rej_low_alpha_ratio', 0))}",
        *format_samples("rej_low_alpha_ratio"),
        "",
        f"    Unsupported language             : {pct(counts.get('rej_unsupported_lang', 0))}",
        *format_samples("rej_unsupported_lang"),
        "",
        sep,
        f"  Output dataset: {output_dir / OUTPUT_CSV}",
        sep,
    ]

    report = "\n".join(lines)
    print(f"\n{report}")

    stats_path = output_dir / "processing_stats.txt"
    with open(stats_path, mode="w", encoding="utf-8") as f:
        f.write(report + "\n")

    try:
        (PROJECT_ROOT / "processing_stats.txt").write_text(report + "\n", encoding="utf-8")
    except Exception:
        pass

    print(f"\n  ✓ Stats saved to: {stats_path}")


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
def main():
    multiprocessing.freeze_support()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Fetch and preprocess articles from MySQL to global_data.csv")
    parser.add_argument("--fresh", "--reset", action="store_true", help="Delete existing global_data.csv and restart from offset 0")
    args, _ = parser.parse_known_args()

    output_dir = CURRENT_DIR
    csv_file   = output_dir / OUTPUT_CSV

    # Setup DB
    try:
        db = DatabaseConnection()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    # Pre-warm FastText model (downloads lid.176.bin on first run)
    print("\n[!] Pre-warming fastText language detector...")
    model_path = Path(tempfile.gettempdir()) / "fasttext-langdetect" / "lid.176.bin"
    if model_path.exists():
        try:
            if abs(model_path.stat().st_size - 131_272_845) > 1024 * 1024:
                print(f"[!] Corrupted model detected ({model_path.stat().st_size} bytes). Deleting...")
                model_path.unlink()
        except Exception:
            pass

    try:
        dummy = FastTextLanguageDetector(model="auto")
        dummy.detect("test")
        print("[!] Language detector initialized and ready.")
    except Exception as e:
        print(f"[!] Warning during pre-warm: {e}")
        if "vector too long" in str(e) and model_path.exists():
            model_path.unlink(missing_ok=True)
            dummy = FastTextLanguageDetector(model="auto")
            dummy.detect("test")

    # --- Resume / Fresh logic ---
    if args.fresh:
        if csv_file.exists():
            csv_file.unlink()
            print(f"\n[!] --fresh specified: Reset existing {OUTPUT_CSV}")
        current_offset = 0
    else:
        current_offset = get_total_processed_count(csv_file)

    if current_offset > 0:
        print(f"\n[!] Detected {current_offset:,} articles in {OUTPUT_CSV}. Resuming from offset...")
    else:
        if csv_file.exists():
            csv_file.unlink()
        print(f"Starting extraction from offset 0 | workers={NUM_WORKERS}  batch_size={CHUNK_SIZE:,}  sub_batch={SUB_BATCH_SIZE:,}")

    counts = {
        "ar": 0, "ar_words": 0, "ar_conf_sum": 0.0,
        "en": 0, "en_words": 0, "en_conf_sum": 0.0,
        "fr": 0, "fr_words": 0, "fr_conf_sum": 0.0,
        "rejected": 0,
        "rej_too_short": 0,
        "rej_low_confidence": 0,
        "rej_low_alpha_ratio": 0,
        "rej_unsupported_lang": 0,
        "total": 0,
        "samples": {"ar": [], "en": [], "fr": [],
                    "rej_too_short": [], "rej_low_confidence": [],
                    "rej_low_alpha_ratio": [], "rej_unsupported_lang": []},
    }

    fh, writer = open_csv_writer(csv_file, HEADER_GLOBAL)
    offset = current_offset
    chunk_index = 0

    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                sql = (
                    f"SELECT id, body FROM article "
                    f"WHERE body IS NOT NULL AND body != '' "
                    f"ORDER BY id ASC LIMIT {CHUNK_SIZE} OFFSET {offset}"
                )
                rows = db.fetch_all(sql)

                if not rows:
                    print("No more articles to process in database.")
                    break

                chunk_index += 1
                print(f"\n[Chunk {chunk_index}] Fetching {len(rows):,} articles (Offset {offset:,})...")

                sub_batches = [rows[i : i + SUB_BATCH_SIZE] for i in range(0, len(rows), SUB_BATCH_SIZE)]
                futures = {executor.submit(process_sub_batch, list(sb)): idx for idx, sb in enumerate(sub_batches)}

                for future in as_completed(futures):
                    try:
                        flush_results(future.result(), writer, counts)
                    except Exception as e:
                        print(f"  [WORKER ERROR] {e}")

                fh.flush()
                offset += CHUNK_SIZE

    finally:
        fh.close()
        db.close()

    save_processing_report(output_dir, counts)
    print(f"\n[✔] Extraction complete! All valid articles saved to: {csv_file}")


if __name__ == "__main__":
    main()
