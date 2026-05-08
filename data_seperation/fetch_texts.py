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
SUB_BATCH_SIZE           = 2000

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
DIALECT_BATCH = 64

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
        ).to(device)
        model.eval()
        print(f"[Dialect ID] Model loaded successfully on {device_name}.")
        print(f"[Dialect ID] Labels: {list(model.config.id2label.values())}")
    except Exception as e:
        print(f"\n[Dialect ID] ERROR: Could not load model {DIALECT_MODEL}.")
        print(f"Details: {e}")
        return

    # --- Initialize counters ---
    counts["ar_msa"] = 0
    counts["ar_dia"] = 0
    counts["ar_egy"] = 0
    counts["ar_lev"] = 0
    counts["ar_glf"] = 0
    counts["ar_mgr"] = 0

    # --- Open writers ---
    with open(msa_path, "w", newline="", encoding="utf-8") as f_msa, \
         open(dia_path, "w", newline="", encoding="utf-8") as f_dia, \
         open(input_path, "r", encoding="utf-8") as f_in:

        reader     = csv.DictReader(f_in)
        writer_msa = csv.writer(f_msa)
        writer_dia = csv.writer(f_dia)

        writer_msa.writerow(HEADER_ARABIC_DIAL)
        writer_dia.writerow(HEADER_ARABIC_DIAL)

        buffer_rows = []
        row_count   = 0

        def process_buffer(rows: list):
            texts  = [r["text"] for r in rows]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits

            preds = torch.argmax(logits, dim=-1).tolist()

            for row, pred_id in zip(rows, preds):
                region = model.config.id2label[pred_id]  # MSA/EGY/LEV/GLF/MGR

                if region == "MSA":
                    label = "msa"
                    writer_msa.writerow([
                        row["id"], row["text"], row["arabic_ratio"],
                        label, region
                    ])
                    counts["ar_msa"] += 1
                else:
                    label = "dialect"
                    writer_dia.writerow([
                        row["id"], row["text"], row["arabic_ratio"],
                        label, region
                    ])
                    counts["ar_dia"] += 1
                    # Track dialect region breakdown
                    region_key = f"ar_{region.lower()}"
                    if region_key in counts:
                        counts[region_key] += 1

        print(f"[Dialect ID] Classifying articles (batch={DIALECT_BATCH})...")

        for row in reader:
            buffer_rows.append(row)
            row_count += 1

            if len(buffer_rows) >= DIALECT_BATCH:
                process_buffer(buffer_rows)
                buffer_rows = []

            if row_count % 5000 == 0:
                print(
                    f"  Processed {row_count:,} Arabic articles "
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
        f"      -> MSA (Standard)              : {pct2(counts.get('ar_msa', 0), ar)}",
        f"      -> Dialectal total             : {pct2(counts.get('ar_dia', 0), ar)}",
        f"           EGY (Egyptian)            : {pct2(counts.get('ar_egy', 0), ar)}",
        f"           LEV (Levantine)           : {pct2(counts.get('ar_lev', 0), ar)}",
        f"           GLF (Gulf)               : {pct2(counts.get('ar_glf', 0), ar)}",
        f"           MGR (Maghrebi/Libyan)     : {pct2(counts.get('ar_mgr', 0), ar)}",
        f"    English                          : {pct(counts['en'])}",
        f"    French                           : {pct(counts['fr'])}",
        f"    Mixed   (< {ARABIC_RATIO_THRESHOLD*100:.0f}% AR chars) : {pct(counts['mixed'])}",
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

    # --- Check for existing separation results ---
    skip_separation = False
    if ar_csv_path.exists() and ar_csv_path.stat().st_size > 0:
        print(f"\n[!] Detected existing {CSV_ARABIC}. Skipping separation phase...")
        skip_separation = True
        with open(ar_csv_path, "r", encoding="utf-8") as f:
            counts["ar"] = sum(1 for _ in f) - 1
        counts["total"] = counts["ar"]
    else:
        for fname in [CSV_ARABIC, CSV_ARABIC_MSA, CSV_ARABIC_DIA,
                      CSV_ENGLISH, CSV_FRENCH, CSV_MIXED, CSV_REJECTED]:
            fpath = output_dir / fname
            if fpath.exists():
                fpath.unlink()
                print(f"  Deleted existing file: {fname}")
        print(
            f"Starting fresh from article ID 0  |  "
            f"workers={NUM_WORKERS}  sub_batch={SUB_BATCH_SIZE}"
        )

    chunk_index = 0

    if not skip_separation:
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

        current_last_id = 0

        try:
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                while True:
                    sql = (
                        f"SELECT id, body FROM article "
                        f"WHERE id > {current_last_id} "
                        f"AND body IS NOT NULL AND body != '' "
                        f"ORDER BY id ASC LIMIT {CHUNK_SIZE}"
                    )
                    result = db.execute_query(sql)
                    rows = result.fetchall()

                    if not rows:
                        print("No more articles to process.")
                        break

                    chunk_index += 1
                    chunk_start_total    = counts["total"]
                    chunk_start_rejected = counts["rejected"]
                    current_last_id      = rows[-1][0]

                    print(
                        f"\n[Chunk {chunk_index}] {len(rows)} articles "
                        f"(up to ID {current_last_id}) "
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

        finally:
            fh_ar.close()
            fh_en.close()
            fh_fr.close()
            fh_mx.close()
            fh_rj.close()

    # --- Dialect Classification Step ---
    if counts["ar"] > 0:
        classify_arabic_dialects(output_dir, counts)

    # --- Final stats ---
    save_stats(
        counts,
        chunk_index if not skip_separation else "N/A (Skipped)",
        output_dir,
    )


if __name__ == "__main__":
    main()