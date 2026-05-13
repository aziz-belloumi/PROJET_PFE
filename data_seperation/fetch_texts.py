import sys
import csv
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
MAX_ARTICLES_PER_RUN     = 100000  # Set to None for no limit

# CSV filenames
CSV_ARABIC     = "arabic_texts.csv"
CSV_ARABIC_MSA = "arabic_msa.csv"
CSV_ARABIC_DIA = "arabic_dialectal.csv"
CSV_ENGLISH    = "english_texts.csv"
CSV_FRENCH     = "french_texts.csv"
CSV_MIXED      = "mixed_texts.csv"
CSV_REJECTED   = "rejected_texts.csv"

# Dialect model config
# IbrahimAmin/marbertv2-arabic-written-dialect-classifier
# Labels: MSA | EGY | LEV | GLF | MGR
# Libyan dialect falls under MGR (Maghrebi)
DIALECT_MODEL = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
DIALECT_BATCH = 32

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


# ------------------------------------------------------------------
# Dialect Classification using MARBERTv2
# ------------------------------------------------------------------
def classify_arabic_dialects(output_dir: Path, counts: dict):
    """
    Reads arabic_texts.csv and splits into MSA vs dialectal using
    IbrahimAmin/marbertv2-arabic-written-dialect-classifier.

    Labels: MSA | EGY (Egyptian) | LEV (Levantine) | GLF (Gulf) | MGR (Maghrebi)
    Libyan dialect falls under MGR (Maghrebi).

    MSA articles  -> arabic_msa.csv
    Dialectal     -> arabic_dialectal.csv  (with dialect region label)
    """
    input_path = output_dir / CSV_ARABIC
    msa_path   = output_dir / CSV_ARABIC_MSA
    dia_path   = output_dir / CSV_ARABIC_DIA

    if not input_path.exists():
        print(f"\n[Dialect ID] Skip: {CSV_ARABIC} not found.")
        return

    # --- Device setup ---
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        f"GPU ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda" else "CPU"
    )
    print(f"\n[Dialect ID] Loading {DIALECT_MODEL} on {device_name}...")

    # --- Load model ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(DIALECT_MODEL)
        model     = AutoModelForSequenceClassification.from_pretrained(
            DIALECT_MODEL
        ).half().to(device)
        model.eval()
        print(f"[Dialect ID] Model loaded successfully on {device_name}.")
        print(f"[Dialect ID] Labels: {list(model.config.id2label.values())}")
    except Exception as e:
        print(f"\n[Dialect ID] ERROR: Could not load model {DIALECT_MODEL}.")
        print(f"Details: {e}")
        return

    # --- Resume Logic: Load already processed IDs ---
    processed_ids = set()
    for path in [msa_path, dia_path]:
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_ids.add(row["id"])
    
    if processed_ids:
        print(f"[Dialect ID] Found {len(processed_ids):,} already processed articles. Resuming...")

    # --- Initialize counters ---
    counts["ar_msa"] = 0
    counts["ar_dia"] = 0
    counts["ar_egy"] = 0
    counts["ar_lev"] = 0
    counts["ar_glf"] = 0
    counts["ar_mgr"] = 0

    # --- Open writers in append mode ---
    msa_exists = msa_path.exists() and msa_path.stat().st_size > 0
    dia_exists = dia_path.exists() and dia_path.stat().st_size > 0

    with open(msa_path, "a", newline="", encoding="utf-8") as f_msa, \
         open(dia_path, "a", newline="", encoding="utf-8") as f_dia, \
         open(input_path, "r", encoding="utf-8") as f_in:

        reader     = csv.DictReader(f_in)
        writer_msa = csv.writer(f_msa)
        writer_dia = csv.writer(f_dia)

        if not msa_exists:
            writer_msa.writerow(HEADER_ARABIC_DIAL)
        if not dia_exists:
            writer_dia.writerow(HEADER_ARABIC_DIAL)

        buffer_rows = []
        row_count   = 0
        batch_count = 0

        def process_buffer(rows: list):
            nonlocal batch_count
            texts  = [r["text"] for r in rows]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type):
                    try:
                        logits = model(**inputs).logits
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        half = len(rows) // 2
                        process_buffer(rows[:half])
                        process_buffer(rows[half:])
                        return

            preds = torch.argmax(logits, dim=-1).tolist()

            for row, pred_id in zip(rows, preds):
                region = model.config.id2label[pred_id]  # MSA/EGY/LEV/GLF/MGR
                # Note: Model labels might vary, mapping to consistent keys
                # Labels: ['MAGHREB', 'LEV', 'MSA', 'GLF', 'EGY']
                
                label_map = {
                    "MSA": "msa",
                    "MAGHREB": "dialect",
                    "LEV": "dialect",
                    "GLF": "dialect",
                    "EGY": "dialect"
                }
                
                region_map = {
                    "MAGHREB": "MGR",
                    "LEV": "LEV",
                    "GLF": "GLF",
                    "EGY": "EGY",
                    "MSA": "MSA"
                }

                region_code = region_map.get(region, region)
                label       = label_map.get(region, "dialect")

                if label == "msa":
                    writer_msa.writerow([
                        row["id"], row["text"], row["arabic_ratio"],
                        label, region_code
                    ])
                    counts["ar_msa"] += 1
                else:
                    writer_dia.writerow([
                        row["id"], row["text"], row["arabic_ratio"],
                        label, region_code
                    ])
                    counts["ar_dia"] += 1
                    # Track dialect region breakdown
                    region_key = f"ar_{region_code.lower()}"
                    if region_key in counts:
                        counts[region_key] += 1
            
            batch_count += 1
            if batch_count % 500 == 0:
                torch.cuda.empty_cache()

        print(f"[Dialect ID] Classifying articles (batch={DIALECT_BATCH})...")

        newly_processed_in_run = 0
        for row in reader:
            if row["id"] in processed_ids:
                continue

            buffer_rows.append(row)
            row_count += 1
            newly_processed_in_run += 1

            if len(buffer_rows) >= DIALECT_BATCH:
                process_buffer(buffer_rows)
                buffer_rows = []

            if MAX_ARTICLES_PER_RUN and newly_processed_in_run >= MAX_ARTICLES_PER_RUN:
                print(f"[Dialect ID] Reached run limit of {MAX_ARTICLES_PER_RUN}. Stopping...")
                break

            if row_count % 5000 == 0:
                print(
                    f"  Processed {row_count:,} new Arabic articles "
                    f"(MSA={counts['ar_msa']:,} | "
                    f"EGY={counts['ar_egy']:,} | "
                    f"LEV={counts['ar_lev']:,} | "
                    f"GLF={counts['ar_glf']:,} | "
                    f"MGR={counts['ar_mgr']:,})..."
                )

        # Final flush
        if buffer_rows:
            process_buffer(buffer_rows)

    print(f"\n[Dialect ID] Done!")
    print(f"  -> MSA             : {counts['ar_msa']:,}")
    print(f"  -> Dialectal total : {counts['ar_dia']:,}")
    print(f"       EGY (Egyptian)  : {counts['ar_egy']:,}")
    print(f"       LEV (Levantine) : {counts['ar_lev']:,}")
    print(f"       GLF (Gulf)      : {counts['ar_glf']:,}")
    print(f"       MGR (Maghrebi)  : {counts['ar_mgr']:,}  <- Libyan dialect")


# ------------------------------------------------------------------
# Stats helper
# ------------------------------------------------------------------
def save_stats(counts: dict, chunk_index, output_dir: Path) -> None:
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
        f"  Chunks       : {chunk_index}   Workers : {NUM_WORKERS}   Sub-batch : {SUB_BATCH_SIZE}",
        f"  Total        : {total:,} articles",
        sep,
        "",
        f"  -- ACCEPTED  {accepted:,} / {total:,}  ({acc_rate:.1f}%) --",
        sep2,
        f"    Arabic  (>= {ARABIC_RATIO_THRESHOLD*100:.0f}% AR chars) : {pct(counts['ar'])}",
        f"      -> Total words                 : {counts.get('ar_words', 0):,}",
        f"      -> Avg words/article           : {counts.get('ar_words', 0) / max(1, counts['ar']):.1f}",
        f"      -> MSA (Standard)              : {pct2(counts.get('ar_msa', 0), ar)}",
        f"      -> Dialectal total             : {pct2(counts.get('ar_dia', 0), ar)}",
        f"           EGY (Egyptian)            : {pct2(counts.get('ar_egy', 0), ar)}",
        f"           LEV (Levantine)           : {pct2(counts.get('ar_lev', 0), ar)}",
        f"           GLF (Gulf)               : {pct2(counts.get('ar_glf', 0), ar)}",
        f"           MGR (Maghrebi/Libyan)     : {pct2(counts.get('ar_mgr', 0), ar)}",
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
        f"  {CSV_ARABIC_MSA:<25} : {counts.get('ar_msa', 0):>10,}",
        f"  {CSV_ARABIC_DIA:<25} : {counts.get('ar_dia', 0):>10,}",
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
def main():
    # Setup DB
    try:
        db = DatabaseConnection()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    output_dir = Path(__file__).resolve().parent

    counts = {
        "ar": 0, "ar_msa": 0, "ar_dia": 0,
        "ar_egy": 0, "ar_lev": 0, "ar_glf": 0, "ar_mgr": 0,
        "en": 0, "fr": 0, "mixed": 0,
        "rejected": 0,
        "rej_too_short": 0, "rej_low_confidence": 0,
        "rej_low_quality": 0, "rej_unsupported_lang": 0,
        "total": 0,
    }

    ar_csv_path = output_dir / CSV_ARABIC

    # --- Resume Logic: Find current offset ---
    current_total_processed = get_total_processed_count(output_dir)
    if current_total_processed > 0:
        print(f"\n[!] Detected {current_total_processed:,} articles already separated. Resuming from offset...")
    else:
        # If starting fresh, delete existing small/empty files
        for fname in [CSV_ARABIC, CSV_ARABIC_MSA, CSV_ARABIC_DIA,
                      CSV_ENGLISH, CSV_FRENCH, CSV_MIXED, CSV_REJECTED]:
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

    if True:  # Always enter separation phase to check for more data
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
                    # Check run limit
                    if MAX_ARTICLES_PER_RUN and articles_processed_this_run >= MAX_ARTICLES_PER_RUN:
                        print(f"\n[!] Reached run limit of {MAX_ARTICLES_PER_RUN} articles. Stopping separation...")
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
                    
                    print(
                        f"\n[Chunk {chunk_index}] {len(rows)} articles (Offset {offset}) "
                        f"-> splitting into sub-batches of {SUB_BATCH_SIZE} "
                        f"across {NUM_WORKERS} workers..."
                    )

                    sub_batches = [
                        rows[i : i + SUB_BATCH_SIZE]
                        for i in range(0, len(rows), SUB_BATCH_SIZE)
                    ]
                    futures = {
                        executor.submit(process_sub_batch, list(sb)): idx
                        for idx, sb in enumerate(sub_batches)
                    }

                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            flush_results(batch_results, writers, counts)
                        except Exception as e:
                            print(f"  [WORKER ERROR] Sub-batch {futures[future]}: {e}")

                    chunk_processed = counts["total"]    - chunk_start_total
                    chunk_rejected  = counts["rejected"] - chunk_start_rejected
                    chunk_accepted  = chunk_processed - chunk_rejected
                    acc_r = chunk_accepted / chunk_processed * 100 if chunk_processed else 0
                    rej_r = chunk_rejected / chunk_processed * 100 if chunk_processed else 0

                    print(
                        f"  └─ Chunk {chunk_index} done | "
                        f"processed={chunk_processed:,} | "
                        f"accepted={chunk_accepted:,} ({acc_r:.1f}%) | "
                        f"rejected={chunk_rejected:,} ({rej_r:.1f}%)"
                    )
                    print(
                        f"       Running totals -> "
                        f"total={counts['total']:,} | "
                        f"ar={counts['ar']:,} | en={counts['en']:,} | "
                        f"fr={counts['fr']:,} | mixed={counts['mixed']:,} | "
                        f"rejected={counts['rejected']:,}"
                    )
                    offset += CHUNK_SIZE
                    articles_processed_this_run += len(rows)

        finally:
            fh_ar.close()
            fh_en.close()
            fh_fr.close()
            fh_mx.close()
            fh_rj.close()

    # --- Dialect Classification Step ---
    # We ensure we run this if there are Arabic articles to process.
    # classify_arabic_dialects has its own resume logic.
    ar_path = output_dir / CSV_ARABIC
    if ar_path.exists() and ar_path.stat().st_size > 0:
        classify_arabic_dialects(output_dir, counts)

    # --- Final stats: Recalculate from all files for global view ---
    print("\n[!] Recalculating comprehensive statistics from all CSV files...")
    final_counts = load_comprehensive_stats(output_dir)

    save_stats(
        final_counts,
        chunk_index if articles_processed_this_run > 0 else "N/A (Resumed)",
        output_dir,
    )


if __name__ == "__main__":
    main()