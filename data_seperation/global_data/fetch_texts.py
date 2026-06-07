import sys
import csv
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.db_config import DatabaseConnection
from src.language_detection import FastTextLanguageDetector
from src.preprocessing.router import PreprocessRouter
from src.config import Config

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
ARABIC_RATIO_THRESHOLD   = 0.80
MIN_CONFIDENCE_THRESHOLD = 0.50
CHUNK_SIZE               = 100000
NUM_WORKERS              = max(1, os.cpu_count() - 1)
SUB_BATCH_SIZE           = 10000
MAX_ARTICLES_PER_RUN     = None  # Process everything in the database

# CSV filenames
CSV_ARABIC     = "arabic_texts.csv"
CSV_ARABIC_MSA = "arabic_msa.csv"
CSV_ARABIC_DIA = "arabic_dialectal.csv"
CSV_ENGLISH    = "english_texts.csv"
CSV_FRENCH     = "french_texts.csv"
CSV_MIXED      = "mixed_texts.csv"
CSV_REJECTED   = "rejected_texts.csv"

# (Dialect classification moved to check_dialect/ensemble_annotator.py)

# Headers
HEADER_ARABIC      = ["id", "text", "arabic_ratio"]
HEADER_ARABIC_DIAL = ["id", "text", "arabic_ratio", "dialect_label", "dialect_region"]
HEADER_LATIN       = ["id", "text", "detected_lang", "confidence"]
HEADER_MIXED       = ["id", "text", "detected_lang", "arabic_ratio", "confidence"]
HEADER_REJECTED    = ["id", "text", "reason"]


# ------------------------------------------------------------------
# Helper class (must be importable at top-level for pickling)
# ------------------------------------------------------------------
class LangDetectPreprocessor:
    def __init__(self):
        self.router = PreprocessRouter()

    def preprocess_for_lang_detect(self, text: str) -> str:
        return self.router.preprocess(text, lang="", task="lang_detect")

    def arabic_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        arabic_chars = sum(
            1 for c in text
            if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
        )
        total_chars = len(text.replace(" ", ""))
        return arabic_chars / total_chars if total_chars > 0 else 0.0


# ------------------------------------------------------------------
# Worker function — runs in a child process
# ------------------------------------------------------------------
def process_sub_batch(rows: list) -> list:
    preprocessor  = LangDetectPreprocessor()
    lang_detector = FastTextLanguageDetector(preprocessor=preprocessor, model="auto")

    results = []
    for article_id, body in rows:
        try:
            res = lang_detector.detect(body)
            detected_lang, confidence = res.lang, res.score
            cleaned_text = preprocessor.preprocess_for_lang_detect(body)
            words = cleaned_text.split()

            # 1. Too short
            if len(words) <= 2:
                results.append({
                    "dest": "rejected",
                    "reason": "rej_too_short",
                    "row": [article_id, body[:300], "too_short (<=2 words)"],
                })
                continue

            # 2. Low confidence
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                results.append({
                    "dest": "rejected",
                    "reason": "rej_low_confidence",
                    "row": [article_id, body[:300],
                            f"low_confidence ({detected_lang}={confidence:.3f}<{MIN_CONFIDENCE_THRESHOLD})"],
                })
                continue

            # 3. Quality check — Latin scripts only
            if detected_lang != "ar":
                if not preprocessor.router.lat.is_valid(
                    cleaned_text, min_tokens=3, min_alpha_ratio=0.65
                ):
                    results.append({
                        "dest": "rejected",
                        "reason": "rej_low_quality",
                        "row": [article_id, body[:300],
                                "low_quality (alpha_ratio<0.65 or <3 tokens)"],
                    })
                    continue

            ratio = preprocessor.arabic_ratio(cleaned_text)

            if detected_lang == "ar":
                if ratio >= ARABIC_RATIO_THRESHOLD:
                    results.append({
                        "dest": "ar",
                        "row": [article_id, cleaned_text, round(ratio, 3)],
                    })
                else:
                    results.append({
                        "dest": "mixed",
                        "row": [article_id, cleaned_text, detected_lang,
                                round(ratio, 3), round(confidence, 3)],
                    })

            elif detected_lang == "en":
                results.append({
                    "dest": "en",
                    "row": [article_id, cleaned_text, detected_lang, round(confidence, 3)],
                })

            elif detected_lang == "fr":
                results.append({
                    "dest": "fr",
                    "row": [article_id, cleaned_text, detected_lang, round(confidence, 3)],
                })

            else:
                results.append({
                    "dest": "rejected",
                    "reason": "rej_unsupported_lang",
                    "row": [article_id, body[:300],
                            f"unsupported_lang ({detected_lang}, conf={confidence:.3f})"],
                })

        except Exception as e:
            results.append({
                "dest": "rejected",
                "reason": "rej_unsupported_lang",
                "row": [article_id, str(body)[:300], f"processing_error: {e}"],
            })

    return results


def get_total_processed_count(output_dir: Path) -> int:
    """Calculates total articles already separated across all output CSVs."""
    total = 0
    files = [CSV_ARABIC, CSV_ENGLISH, CSV_FRENCH, CSV_MIXED, CSV_REJECTED]
    for fname in files:
        fpath = output_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    # Count lines minus header
                    count = sum(1 for _ in f) - 1
                    if count > 0:
                        total += count
            except Exception:
                pass
    return total


def load_comprehensive_stats(output_dir: Path) -> dict:
    """Reads all output CSV files to calculate cumulative statistics."""
    stats = {
        "ar": 0, "ar_words": 0, "ar_msa": 0, "ar_dia": 0,
        "ar_egy": 0, "ar_lev": 0, "ar_glf": 0, "ar_mgr": 0,
        "en": 0, "en_words": 0, "en_conf_sum": 0.0,
        "fr": 0, "fr_words": 0, "fr_conf_sum": 0.0,
        "mixed": 0, "mixed_words": 0, "mixed_conf_sum": 0.0,
        "rejected": 0,
        "rej_too_short": 0, "rej_low_confidence": 0,
        "rej_low_quality": 0, "rej_unsupported_lang": 0,
        "total": 0,
    }

    # 1. Arabic Master
    ar_path = output_dir / CSV_ARABIC
    if ar_path.exists() and ar_path.stat().st_size > 0:
        with open(ar_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or len(row) < 2: continue
                stats["ar"] += 1
                stats["ar_words"] += len(row[1].split())

    # 2. English & French (Latin)
    for fname, key in [(CSV_ENGLISH, "en"), (CSV_FRENCH, "fr")]:
        fpath = output_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if not row or len(row) < 4: continue
                    stats[key] += 1
                    stats[f"{key}_words"] += len(row[1].split())
                    try:
                        stats[f"{key}_conf_sum"] += float(row[3])
                    except: pass

    # 3. Mixed
    mx_path = output_dir / CSV_MIXED
    if mx_path.exists() and mx_path.stat().st_size > 0:
        with open(mx_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or len(row) < 5: continue
                stats["mixed"] += 1
                stats["mixed_words"] += len(row[1].split())
                try:
                    stats["mixed_conf_sum"] += float(row[4])
                except: pass

    # 4. Rejected (with reason breakdown)
    rj_path = output_dir / CSV_REJECTED
    if rj_path.exists() and rj_path.stat().st_size > 0:
        with open(rj_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row: continue
                stats["rejected"] += 1
                if len(row) >= 3:
                    reason = row[2].lower()
                    if "too_short" in reason: stats["rej_too_short"] += 1
                    elif "low_confidence" in reason: stats["rej_low_confidence"] += 1
                    elif "low_quality" in reason: stats["rej_low_quality"] += 1
                    elif "unsupported_lang" in reason: stats["rej_unsupported_lang"] += 1

    # 5. Dialect breakdown
    for fname, is_msa in [(CSV_ARABIC_MSA, True), (CSV_ARABIC_DIA, False)]:
        fpath = output_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if is_msa:
                        stats["ar_msa"] += 1
                    else:
                        stats["ar_dia"] += 1
                        region = row.get("dialect_region", "").upper()
                        region_key = f"ar_{region.lower()}"
                        if region_key in stats:
                            stats[region_key] += 1

    stats["total"] = stats["ar"] + stats["en"] + stats["fr"] + stats["mixed"] + stats["rejected"]
    return stats


# ------------------------------------------------------------------
# CSV writer helper
# ------------------------------------------------------------------
def open_csv_writer(path: Path, header: list):
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(header)
    return fh, writer


# ------------------------------------------------------------------
# Merge worker results into counts + CSV writers
# ------------------------------------------------------------------
def flush_results(results: list, writers: dict, counts: dict) -> None:
    for item in results:
        dest = item["dest"]
        counts["total"] += 1

        if dest == "ar":
            writers["ar"].writerow(item["row"])
            counts["ar"] += 1
        elif dest == "en":
            writers["en"].writerow(item["row"])
            counts["en"] += 1
        elif dest == "fr":
            writers["fr"].writerow(item["row"])
            counts["fr"] += 1
        elif dest == "mixed":
            writers["mixed"].writerow(item["row"])
            counts["mixed"] += 1
        elif dest == "rejected":
            writers["rejected"].writerow(item["row"])
            counts["rejected"] += 1
            counts[item.get("reason", "rej_unsupported_lang")] += 1


# (Dialect classification moved to check_dialect/ensemble_annotator.py)

# ------------------------------------------------------------------
# Stats helper
# ------------------------------------------------------------------
def save_processing_report(output_dir: Path) -> None:
    counts = load_comprehensive_stats(output_dir)
    total    = counts["total"]
    accepted = counts["ar"] + counts["en"] + counts["fr"] + counts["mixed"]
    rejected = counts["rejected"]
    acc_rate = (accepted / total * 100) if total else 0
    rej_rate = (rejected / total * 100) if total else 0
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def pct(n, d=total):
        return f"{n:>10,}  ({n/d*100:5.1f}%)" if d else f"{n:>10,}  (  N/A )"

    def pct2(n, d):
        return f"{n:>10,}  ({n/d*100:5.1f}%)" if d else f"{n:>10,}  (  N/A )"

    sep  = "=" * 60
    sep2 = "-" * 60
    ar   = max(counts["ar"], 1)

    lines = [
        sep,
        f"  PROCESSING REPORT",
        f"  Run date     : {run_time}",
        f"  Workers      : {NUM_WORKERS}   Sub-batch : {SUB_BATCH_SIZE}",
        f"  Total        : {total:,} articles",
        sep,
        "",
        f"  -- ACCEPTED  {accepted:,} / {total:,}  ({acc_rate:.1f}%) --",
        sep2,
        f"    Arabic  (>= {ARABIC_RATIO_THRESHOLD*100:.0f}% AR chars) : {pct(counts['ar'])}",
        f"      -> Total words                 : {counts.get('ar_words', 0):,}",
        f"      -> Avg words/article           : {counts.get('ar_words', 0) / max(1, counts['ar']):.1f}",
        f"    English                          : {pct(counts['en'])}",
        f"      -> Total words                 : {counts.get('en_words', 0):,}",
        f"      -> Avg words/article           : {counts.get('en_words', 0) / max(1, counts['en']):.1f}",
        f"      -> Avg confidence              : {counts.get('en_conf_sum', 0) / max(1, counts['en']):.3f}",
        f"    French                           : {pct(counts['fr'])}",
        f"      -> Total words                 : {counts.get('fr_words', 0):,}",
        f"      -> Avg words/article           : {counts.get('fr_words', 0) / max(1, counts['fr']):.1f}",
        f"      -> Avg confidence              : {counts.get('fr_conf_sum', 0) / max(1, counts['fr']):.3f}",
        f"    Mixed   (< {ARABIC_RATIO_THRESHOLD*100:.0f}% AR chars) : {pct(counts['mixed'])}",
        f"      -> Total words                 : {counts.get('mixed_words', 0):,}",
        f"      -> Avg confidence              : {counts.get('mixed_conf_sum', 0) / max(1, counts['mixed']):.3f}",
        "",
        f"  -- REJECTED  {rejected:,} / {total:,}  ({rej_rate:.1f}%) --",
        sep2,
        f"    Too short     (<= 2 words)       : {pct(counts['rej_too_short'])}",
        f"    Low confidence (< {MIN_CONFIDENCE_THRESHOLD})          : {pct(counts['rej_low_confidence'])}",
        f"    Low quality   (alpha < 0.65)     : {pct(counts['rej_low_quality'])}",
        f"    Unsupported language             : {pct(counts['rej_unsupported_lang'])}",
        "",
        sep,
        f"  Output dir : {output_dir}",
        sep,
        "",
        "  FILE SUMMARY (Article Counts)",
        sep2,
        f"  {CSV_ARABIC:<25} : {counts.get('ar', 0):>10,}",
        f"  {CSV_ENGLISH:<25} : {counts.get('en', 0):>10,}",
        f"  {CSV_FRENCH:<25} : {counts.get('fr', 0):>10,}",
        f"  {CSV_MIXED:<25} : {counts.get('mixed', 0):>10,}",
        f"  {CSV_REJECTED:<25} : {counts.get('rejected', 0):>10,}",
        sep,
    ]

    report = "\n".join(lines)
    print(f"\n{report}")

    stats_path = output_dir / "processing_stats.txt"
    with open(stats_path, mode="w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n  Stats saved to: {stats_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
import pandas as pd # Import needed for splitting

def main():
    # Setup DB
    try:
        db = DatabaseConnection()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    # Pre-download language model to avoid race conditions in parallel workers
    print("\n[!] Pre-warming language detector...")
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "fasttext-langdetect"
    model_path = temp_dir / "lid.176.bin"
    
    if model_path.exists():
        try:
            # FastText lid.176 is ~126MB. Anything drastically different is corrupted.
            if abs(model_path.stat().st_size - 131272845) > 1024 * 1024: 
                print(f"[!] Corrupted model detected ({model_path.stat().st_size} bytes). Deleting...")
                model_path.unlink()
        except:
            pass

    try:
        from src.language_detection import FastTextLanguageDetector
        dummy_detector = FastTextLanguageDetector(model="auto")
        dummy_detector.detect("test")
        print("[!] Language model ready.")
    except Exception as e:
        print(f"[!] Warning during pre-warm: {e}")
        # If it fails even now, delete and try one last time
        if "vector too long" in str(e) and model_path.exists():
            print("[!] Critical corruption. Forcing delete...")
            model_path.unlink(missing_ok=True)
            # Re-try once
            dummy_detector = FastTextLanguageDetector(model="auto")
            dummy_detector.detect("test")

    output_dir = Path(__file__).resolve().parent

    counts = {
        "ar": 0, "en": 0, "fr": 0, "mixed": 0,
        "rejected": 0,
        "rej_too_short": 0, "rej_low_confidence": 0,
        "rej_low_quality": 0, "rej_unsupported_lang": 0,
        "total": 0,
    }

    # --- Resume Logic: Find current offset ---
    current_total_processed = get_total_processed_count(output_dir)
    if current_total_processed > 0:
        print(f"\n[!] Detected {current_total_processed:,} articles already separated. Resuming from offset...")
    else:
        # If starting fresh, delete existing small/empty files
        for fname in [CSV_ARABIC, CSV_ENGLISH, CSV_FRENCH, CSV_MIXED, CSV_REJECTED]:
            fpath = output_dir / fname
            if fpath.exists():
                fpath.unlink()
                print(f"  Deleted existing file: {fname}")
        print(
            f"Starting fresh from offset 0  |  "
            f"workers={NUM_WORKERS}  sub_batch={SUB_BATCH_SIZE}"
        )

    chunk_index = 0
    articles_processed_this_run = 0

    fh_ar, wr_ar = open_csv_writer(output_dir / CSV_ARABIC,   HEADER_ARABIC)
    fh_en, wr_en = open_csv_writer(output_dir / CSV_ENGLISH,  HEADER_LATIN)
    fh_fr, wr_fr = open_csv_writer(output_dir / CSV_FRENCH,   HEADER_LATIN)
    fh_mx, wr_mx = open_csv_writer(output_dir / CSV_MIXED,    HEADER_MIXED)
    fh_rj, wr_rj = open_csv_writer(output_dir / CSV_REJECTED, HEADER_REJECTED)

    writers = {
        "ar":       wr_ar,
        "en":       wr_en,
        "fr":       wr_fr,
        "mixed":    wr_mx,
        "rejected": wr_rj,
    }

    offset = current_total_processed
    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                if MAX_ARTICLES_PER_RUN and articles_processed_this_run >= MAX_ARTICLES_PER_RUN:
                    print(f"\n[!] Reached run limit of {MAX_ARTICLES_PER_RUN}. Stopping separation...")
                    break

                sql = (
                    f"SELECT id, body FROM article "
                    f"WHERE body IS NOT NULL AND body != '' "
                    f"ORDER BY id ASC LIMIT {CHUNK_SIZE} OFFSET {offset}"
                )
                result = db.execute_query(sql)
                rows = result.fetchall()

                if not rows:
                    print("No more articles to process.")
                    break

                chunk_index += 1
                chunk_start_total    = counts["total"]
                chunk_start_rejected = counts["rejected"]
                
                print(f"\n[Chunk {chunk_index}] {len(rows)} articles (Offset {offset})...")

                sub_batches = [rows[i : i + SUB_BATCH_SIZE] for i in range(0, len(rows), SUB_BATCH_SIZE)]
                futures = {executor.submit(process_sub_batch, list(sb)): idx for idx, sb in enumerate(sub_batches)}

                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        flush_results(batch_results, writers, counts)
                    except Exception as e:
                        print(f"  [WORKER ERROR] {e}")

                articles_processed_this_run += len(rows)
                offset += CHUNK_SIZE

    finally:
        fh_ar.close()
        fh_en.close()
        fh_fr.close()
        fh_mx.close()
        fh_rj.close()

    # Final stats
    # Save final statistics report
    save_processing_report(output_dir)

    print("\n[✔] Processing complete.")

if __name__ == "__main__":
    main()