import os
import sys
import pandas as pd
import torch
import gc
import queue
import threading
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import logging
from pathlib import Path
from tqdm import tqdm

# FORCE GUEST ACCESS
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

# ==========================================
# CONFIGURATION
# ==========================================
PC_ID = 1  # Set to 1 or 2 to run on a specific partitioned subset. Set to None to process all.

DATA_DIR      = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation")
OUTPUT_DIR    = DATA_DIR / "check_dialect"

if PC_ID in [1, 2]:
    INPUT_FILE    = DATA_DIR / f"arabic_texts_pc{PC_ID}.csv"
    ACCEPTED_FILE = OUTPUT_DIR / f"accepted_articles_pc{PC_ID}.csv"
    REJECTED_FILE = OUTPUT_DIR / f"rejected_articles_pc{PC_ID}.csv"
    STATS_FILE    = OUTPUT_DIR / f"ensemble_stats_pc{PC_ID}.json"
    REPORT_FILE   = OUTPUT_DIR / f"ensemble_report_pc{PC_ID}.txt"
else:
    INPUT_FILE    = DATA_DIR / "arabic_texts.csv"
    ACCEPTED_FILE = OUTPUT_DIR / "accepted_articles.csv"
    REJECTED_FILE = OUTPUT_DIR / "rejected_articles.csv"
    STATS_FILE    = OUTPUT_DIR / "ensemble_stats.json"
    REPORT_FILE   = OUTPUT_DIR / "ensemble_report.txt"

INFERENCE_BATCH_SIZE = 128     # Sweet spot for 6GB VRAM to avoid Windows virtual VRAM paging
TEXT_BATCH_SIZE      = 1000    # Optimal batch granularity for frequent saves and fast visual updates
MAX_CHUNK_LENGTH     = 512
MAX_CHUNKS_PER_TEXT  = 1       # 512 tokens max (perfect dialect identification, yields immediate 2x speedup)
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# MODELS CONFIG
# ==========================================
MODELS_CONFIG = [
    {
        "id":   "m1",
        "name": "Ibrahim-MARBERTv2",
        "path": "IbrahimAmin/marbertv2-arabic-written-dialect-classifier",
        "mapping": {
            "MSA": "MSA", "EGY": "EGY", "LEV": "LEV", "GLF": "GLF", "MAG": "MGR", "MGR": "MGR",
        }
    },
    {
        "id":   "m2",
        "name": "CAMeL-MADAR6",
        "path": "CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus6",
        "mapping": {
            "MSA": "MSA",
            "BEI": "LEV", "JER": "LEV", "DAM": "LEV", "ALE": "LEV", "AMM": "LEV",
            "CAI": "EGY", "KHA": "EGY", "SAN": "EGY",
            "DOH": "GLF", "BAG": "GLF", "BAS": "GLF", "MUS": "GLF", "RIY": "GLF", "JED": "GLF", "KUW": "GLF", "ABU": "GLF",
            "TUN": "MGR", "SFX": "MGR", "ALG": "MGR", "FEZ": "MGR", "CAS": "MGR", "TRI": "MGR", "BEN": "MGR", "RAB": "MGR",
        }
    },
    {
        "id":   "m3",
        "name": "Keleg-NADI-2023",
        "path": "AMR-KELEG/ADI-NADI-2023",
        "mapping": {
            "MSA":          "MSA",
            "Egypt":        "EGY", "Sudan":        "EGY",
            "Iraq":         "GLF", "Saudi_Arabia": "GLF", "UAE":          "GLF", "Kuwait":       "GLF", "Qatar":        "GLF", "Oman":         "GLF", "Bahrain":      "GLF", "Yemen":        "GLF",
            "Jordan":       "LEV", "Lebanon":      "LEV", "Palestine":    "LEV", "Syria":        "LEV",
            "Libya":        "MGR", "Morocco":      "MGR", "Tunisia":      "MGR", "Algeria":      "MGR", "Mauritania":   "MGR",
        }
    },
    {
        "id":   "m4",
        "name": "Lafifi-ARBERT",
        "path": "lafifi-24/arbert_arabic_dialect_identification",
        "mapping": {
            "MSA": "MSA", "EG": "EGY", "SD": "EGY",
            "SY": "LEV", "PL": "LEV", "LB": "LEV", "JO": "LEV",
            "KW": "GLF", "QA": "GLF", "AE": "GLF", "BH": "GLF", "SA": "GLF", "OM": "GLF", "IQ": "GLF", "YE": "GLF",
            "LY": "MGR", "DZ": "MGR", "MA": "MGR", "TN": "MGR"
        }
    },
    {
        "id":   "m5",
        "name": "Oddadmix-Router",
        "path": "oddadmix/dialect-router-v0.1",
        "mapping": {
            "ar": "MSA", "eg": "EGY", "sd": "EGY", "iq": "GLF", "sa": "GLF", "lb": "LEV", "ps": "LEV", "sy": "LEV", "ly": "MGR", "ma": "MGR", "tn": "MGR", "en": "OTHER"
        }
    }
]

# ==========================================
# VALID OUTPUT LABELS (unified label space)
# ==========================================
VALID_LABELS = {"MSA", "EGY", "LEV", "GLF", "MGR"}

# ==========================================
# M1 MODEL MAPPING
# ==========================================
M1_MAPPING = {
    "MSA":     "MSA",
    "EGY":     "EGY",
    "LEV":     "LEV",
    "GLF":     "GLF",
    "MAG":     "MGR",
    "MGR":     "MGR",
    "MAGHREB": "MGR",
}

# ==========================================
# EVALUATION & STATS
# ==========================================
def evaluate_ensemble(item: dict) -> dict:
    votes = {}
    confs = {}
    
    for m in ["m1", "m2", "m3", "m4", "m5"]:
        lbl = item.get(f"{m}_label")
        conf = item.get(f"{m}_conf")
        
        # Only count labels that belong to the unified valid label space
        if lbl and not pd.isna(lbl) and lbl in VALID_LABELS:
            if lbl not in votes:
                votes[lbl] = 0
                confs[lbl] = []
            votes[lbl] += 1
            confs[lbl].append(float(conf) if conf is not None and not pd.isna(conf) else 0.0)

    if not votes:
        # All 5 models returned invalid / unmapped labels
        return {
            "status": "rejected",
            "reason": "complete_disagreement_low_confidence",
            "agreement_count": 0,
            "average_confidence": 0.0,
            "final_label": None
        }

    max_votes = max(votes.values())
    winners = [l for l, c in votes.items() if c == max_votes]

    # Pick the winner; break ties by highest average confidence
    best_label = winners[0]
    best_avg_conf = sum(confs[best_label]) / len(confs[best_label]) if confs[best_label] else 0.0

    if len(winners) > 1:
        for w in winners[1:]:
            avg_c = sum(confs[w]) / len(confs[w]) if confs[w] else 0.0
            if avg_c > best_avg_conf:
                best_label = w
                best_avg_conf = avg_c

    # ─── REJECTION RULE ───────────────────────────────────────────────────────
    # Reject ONLY when every model voted for a different label (max_votes == 1)
    # AND even the best confidence is below 0.5 — truly ambiguous text.
    # Every other case (any agreement ≥ 2, or disagreement with conf ≥ 0.5) is accepted.
    # ─────────────────────────────────────────────────────────────────────────
    if max_votes == 1 and best_avg_conf < 0.5:
        return {
            "status": "rejected",
            "reason": "complete_disagreement_low_confidence",
            "agreement_count": max_votes,
            "average_confidence": round(best_avg_conf, 4),
            "final_label": best_label
        }

    return {
        "status": "accepted",
        "reason": None,
        "agreement_count": max_votes,
        "average_confidence": round(best_avg_conf, 4),
        "final_label": best_label
    }

def generate_stats():
    print("\n" + "=" * 50)
    print("GENERATING DETAILED REPORT...")
    print("=" * 50)

    total_accepted = 0
    total_rejected = 0
    accepted_words = 0
    dialect_dist = {"MSA": 0, "EGY": 0, "LEV": 0, "GLF": 0, "MGR": 0}
    rejection_reasons = {"complete_disagreement_low_confidence": 0}
    
    agreement_ratios = {"5/5": 0, "4/5": 0, "3/5": 0, "2/5": 0, "1/5": 0}
    model_conf_sums = {f"m{i}": {"sum": 0.0, "count": 0} for i in range(1, 6)}

    if ACCEPTED_FILE.exists():
        for chunk in pd.read_csv(ACCEPTED_FILE, chunksize=50000):
            total_accepted += len(chunk)
            accepted_words += chunk["text"].astype(str).str.split().str.len().sum()
            lbl_col = "final_label" if "final_label" in chunk.columns else "ensemble_label"
            for k, v in chunk[lbl_col].value_counts().to_dict().items():
                dialect_dist[k] = dialect_dist.get(k, 0) + v
            for k, v in chunk["agreement_count"].value_counts().to_dict().items():
                if k in [1, 2, 3, 4, 5]:
                    agreement_ratios[f"{k}/5"] += v
            for i in range(1, 6):
                col = f"m{i}_conf"
                if col in chunk.columns:
                    valid = chunk[col].dropna()
                    model_conf_sums[f"m{i}"]["sum"] += float(valid.sum())
                    model_conf_sums[f"m{i}"]["count"] += len(valid)

    if REJECTED_FILE.exists():
        for chunk in pd.read_csv(REJECTED_FILE, chunksize=50000):
            total_rejected += len(chunk)
            for k, v in chunk["rejection_reason"].value_counts().to_dict().items():
                rejection_reasons[k] = rejection_reasons.get(k, 0) + v
            for i in range(1, 6):
                col = f"m{i}_conf"
                if col in chunk.columns:
                    valid = chunk[col].dropna()
                    model_conf_sums[f"m{i}"]["sum"] += float(valid.sum())
                    model_conf_sums[f"m{i}"]["count"] += len(valid)

    total_articles = total_accepted + total_rejected
    if total_articles == 0:
        print("No articles processed yet.")
        return

    avg_words = accepted_words / total_accepted if total_accepted > 0 else 0

    lines = [
        "=" * 55,
        "     ARABIC ENSEMBLE DIALECT ANNOTATION REPORT",
        "=" * 55,
        f"Date:             {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Processed:  {total_articles:,}",
        f"Accepted:         {total_accepted:,} ({(total_accepted/total_articles)*100:.2f}%)",
        f"Rejected:         {total_rejected:,} ({(total_rejected/total_articles)*100:.2f}%)",
        "-" * 55,
        "",
        "ACCEPTED DIALECT DISTRIBUTION:",
    ]
    
    for label in ["MSA", "EGY", "LEV", "GLF", "MGR"]:
        count = dialect_dist.get(label, 0)
        pct = (count / total_accepted * 100) if total_accepted > 0 else 0
        lines.append(f"  {label:6}: {count:10,}  ({pct:6.2f}%)")

    lines += [
        "",
        "AGREEMENT RATIOS (ACCEPTED):",
    ]
    for k in [5, 4, 3, 2, 1]:
        key = f"{k}/5"
        cnt = agreement_ratios[key]
        pct = (cnt / total_accepted * 100) if total_accepted > 0 else 0
        lines.append(f"  {key} Agreement: {cnt:10,}  ({pct:6.2f}%)")

    lines += ["", "AVERAGE MODEL CONFIDENCE (ALL):"]
    for i, config in enumerate(MODELS_CONFIG):
        m = f"m{i+1}"
        mean_conf = model_conf_sums[m]["sum"] / model_conf_sums[m]["count"] if model_conf_sums[m]["count"] > 0 else 0.0
        lines.append(f"  {config['name']:25}: {mean_conf:.4f}")

    lines += ["", "REJECTION REASONS:"]
    for reason, count in rejection_reasons.items():
        pct = (count / total_rejected * 100) if total_rejected > 0 else 0
        lines.append(f"  {reason:25}: {count:10,}  ({pct:6.2f}%)")

    lines += ["", "=" * 55]
    report = "\n".join(lines)
    print(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    serializable = {
        "total_processed": int(total_articles),
        "total_accepted": int(total_accepted),
        "total_rejected": int(total_rejected),
        "avg_words_accepted": float(avg_words),
        "dialect_distribution": {k: int(v) for k, v in dialect_dist.items()},
        "agreement_ratios": {k: int(v) for k, v in agreement_ratios.items()},
        "average_confidence": {
            f"m{i+1}": float(model_conf_sums[f"m{i+1}"]["sum"] / model_conf_sums[f"m{i+1}"]["count"]) if model_conf_sums[f"m{i+1}"]["count"] > 0 else 0.0
            for i in range(5)
        },
        "rejection_reasons": {k: int(v) for k, v in rejection_reasons.items()}
    }
    
    with open(STATS_FILE, "w") as f:
        json.dump(serializable, f, indent=4)

    print(f"\nFiles saved:\n  Report: {REPORT_FILE}\n  Stats:  {STATS_FILE}")

# ==========================================
# SINGLE MODEL INFERENCE BATCH RUNNER (C++ RUST OPTIMIZED)
# ==========================================
def run_single_model(config, model, tokenizer, batch_records):
    texts = [str(row["text"]) if str(row["text"]).strip() else "empty" for row in batch_records]
    
    # Let Hugging Face C++ Rust-backed tokenizer handle chunking and padding instantly!
    # It automatically reserves space for CLS and SEP tokens and splits texts.
    encodings = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt"
    )
    
    chunk_input_ids_tensor = encodings["input_ids"]
    chunk_attention_mask_tensor = encodings["attention_mask"]
    chunk_to_text_idx = encodings["overflow_to_sample_mapping"].numpy().tolist()
    
    # Filter to keep at most MAX_CHUNKS_PER_TEXT chunks per text index
    if MAX_CHUNKS_PER_TEXT is not None and len(chunk_to_text_idx) > 0:
        keep_indices = []
        current_idx = -1
        chunk_count = 0
        for i, text_idx in enumerate(chunk_to_text_idx):
            if text_idx != current_idx:
                current_idx = text_idx
                chunk_count = 0
            if chunk_count < MAX_CHUNKS_PER_TEXT:
                keep_indices.append(i)
                chunk_count += 1
                
        if len(keep_indices) < len(chunk_to_text_idx):
            chunk_input_ids_tensor = chunk_input_ids_tensor[keep_indices]
            chunk_attention_mask_tensor = chunk_attention_mask_tensor[keep_indices]
            chunk_to_text_idx = [chunk_to_text_idx[i] for i in keep_indices]
            
    if len(chunk_to_text_idx) == 0:
        return [{"label": "OTHER", "conf": 0.0} for _ in batch_records]
        
    # Direct batch GPU inference (extremely fast)
    all_logits = []
    for k in range(0, len(chunk_input_ids_tensor), INFERENCE_BATCH_SIZE):
        input_tensor = chunk_input_ids_tensor[k:k+INFERENCE_BATCH_SIZE].to(DEVICE)
        mask_tensor = chunk_attention_mask_tensor[k:k+INFERENCE_BATCH_SIZE].to(DEVICE)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", enabled=(DEVICE == "cuda")):
                outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_logits.extend(probs.cpu().numpy())
                
    # Aggregate chunk probabilities back to unified labels
    id2label = model.config.id2label
    class_id_to_unified = {
        cid: config["mapping"].get(lbl, "OTHER") for cid, lbl in id2label.items()
    }
    
    text_unified_probs = [ {} for _ in batch_records ]
    text_chunk_counts = [ 0 for _ in batch_records ]
    
    for idx, prob_dist in enumerate(all_logits):
        txt_idx = chunk_to_text_idx[idx]
        text_chunk_counts[txt_idx] += 1
        
        for cid, prob in enumerate(prob_dist):
            unified_lbl = class_id_to_unified[cid]
            if unified_lbl not in text_unified_probs[txt_idx]:
                text_unified_probs[txt_idx][unified_lbl] = 0.0
            text_unified_probs[txt_idx][unified_lbl] += float(prob)
            
    out_predictions = []
    for idx in range(len(batch_records)):
        unified_probs = text_unified_probs[idx]
        counts = text_chunk_counts[idx]
        
        if not unified_probs:
            best_label = "OTHER"
            best_score = 0.0
        else:
            avg_preds = {k: v / counts for k, v in unified_probs.items()}
            best_label = max(avg_preds, key=avg_preds.get)
            best_score = avg_preds[best_label]
            
        out_predictions.append({
            "label": best_label,
            "conf": round(best_score, 4)
        })
        
    return out_predictions


# ==========================================
# FIX OTHER LABELS (Post-Processing)
# ==========================================
def fix_other_labels(models: dict, tokenizers: dict):
    """Fix rows where m1_label == 'OTHER' by re-running Ibrahim model"""
    print("\n" + "="*60)
    print("  POST-PROCESSING: FIXING M1_LABEL == 'OTHER' ROWS")
    print("="*60)
    
    if not ACCEPTED_FILE.exists():
        print(f"  Accepted file not found: {ACCEPTED_FILE}")
        return
    
    print(f"\n[1] Loading {ACCEPTED_FILE.name}...")
    df = pd.read_csv(ACCEPTED_FILE)
    print(f"    Total rows: {len(df):,}")
    
    other_mask = df["m1_label"] == "OTHER"
    other_count = other_mask.sum()
    print(f"\n[2] Rows with m1_label == 'OTHER': {other_count:,}")
    
    if other_count == 0:
        print("    Nothing to fix.")
        return
    
    print(f"\n[3] Re-running Ibrahim on {other_count:,} articles (batch={INFERENCE_BATCH_SIZE})...")
    other_indices = df[other_mask].index.tolist()
    other_texts = df.loc[other_mask, "text"].tolist()
    
    m1_config = MODELS_CONFIG[0]
    m1_model = models["m1"]
    m1_tokenizer = tokenizers["m1"]
    id2label = m1_model.config.id2label
    
    new_labels = []
    new_confs = []
    
    for i in tqdm(range(0, len(other_texts), INFERENCE_BATCH_SIZE), desc="Ibrahim re-run"):
        batch_texts = other_texts[i : i + INFERENCE_BATCH_SIZE]
        inputs = m1_tokenizer(batch_texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            logits = m1_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_confs = probs.max(dim=-1).values.tolist()
        for pred_id, pred_conf in zip(pred_ids, pred_confs):
            raw_label = id2label[pred_id]
            mapped_label = M1_MAPPING.get(raw_label, "OTHER")
            new_labels.append(mapped_label)
            new_confs.append(round(float(pred_conf), 4))
    
    print(f"\n[4] Updating m1_label and m1_conf...")
    for i, idx in enumerate(other_indices):
        df.at[idx, "m1_label"] = new_labels[i]
        df.at[idx, "m1_conf"] = new_confs[i]
    
    new_label_dist = pd.Series(new_labels).value_counts().to_dict()
    print(f"    New m1 distribution for previously-OTHER rows:")
    for lbl, cnt in sorted(new_label_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"      {lbl:<10}: {cnt:>10,}")
    
    print(f"\n[5] Recalculating ensemble verdicts...")
    for idx in tqdm(other_indices, desc="Verdicts"):
        row = df.loc[idx].to_dict()
        verdict = evaluate_ensemble(row)
        df.at[idx, "agreement_count"] = verdict["agreement_count"]
        df.at[idx, "average_confidence"] = verdict["average_confidence"]
        df.at[idx, "final_label"] = verdict["final_label"]
        df.at[idx, "ensemble_label"] = verdict["final_label"]
        df.at[idx, "ensemble_status"] = verdict["status"]
        df.at[idx, "rejection_reason"] = verdict.get("reason", "")
    
    print(f"\n[6] Saving corrected {ACCEPTED_FILE.name}...")
    df.to_csv(ACCEPTED_FILE, index=False)
    print(f"    Saved {len(df):,} rows.")
    
    print(f"\n{'='*60}")
    print(f"  DONE — FIXED SUMMARY")
    print(f"{'='*60}")
    print(f"  Articles fixed : {other_count:,}")
    print(f"  Remaining OTHER: {(df['m1_label'] == 'OTHER').sum():,}")
    print(f"{'='*60}\n")

# ==========================================
# MAIN ENTRYPOINT
# ==========================================
def main():
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    logger.info(f"Loading {INPUT_FILE.name}...")
    input_df = pd.read_csv(INPUT_FILE)
    logger.info(f"  {len(input_df):,} total articles in input.")

    # 1. Load all 5 models into GPU Memory at once
    tokenizers = {}
    models = {}
    
    print("\n=============================================")
    print(" LOADING ALL 5 MODELS INTO GPU MEMORY AT ONCE")
    print("=============================================")
    for config in MODELS_CONFIG:
        m_id = config["id"]
        print(f"Loading tokenizer & model: {config['name']}...")
        tokenizers[m_id] = AutoTokenizer.from_pretrained(config["path"], use_fast=True)
        
        # Load with Native PyTorch Scaled Dot Product Attention (FlashAttention/SDPA) for 2x-3x speedup!
        try:
            models[m_id] = AutoModelForSequenceClassification.from_pretrained(
                config["path"],
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                attn_implementation="sdpa"
            ).to(DEVICE)
        except Exception:
            models[m_id] = AutoModelForSequenceClassification.from_pretrained(
                config["path"],
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE)
            
        models[m_id].eval()

    # 2. Setup Resume mechanism based on already fully-annotated files
    processed_ids = set()
    if ACCEPTED_FILE.exists():
        processed_ids.update(pd.read_csv(ACCEPTED_FILE, usecols=["id"])["id"].tolist())
    if REJECTED_FILE.exists():
        processed_ids.update(pd.read_csv(REJECTED_FILE, usecols=["id"])["id"].tolist())
        
    df_missing = input_df[~input_df["id"].isin(processed_ids)]
    logger.info(f"  {len(df_missing):,} remaining articles to process.")

    if len(df_missing) == 0:
        print("Pipeline already fully completed on all data!")
        generate_stats()
        return

    records = df_missing.to_dict("records")
    
    print(f"\n=============================================")
    print(f" RUNNING ENSEMBLE PIPELINE BATCH-BY-BATCH")
    print(f"=============================================")
    
    pbar = tqdm(total=len(records), desc="Processing Ensemble")
    
    for i in range(0, len(records), TEXT_BATCH_SIZE):
        batch_records = records[i:i+TEXT_BATCH_SIZE]
        
        # Prepare batch structure
        batch_results = [
            {
                "id": row["id"],
                "text": row["text"]
            }
            for row in batch_records
        ]
        
        # Run inference across all 5 loaded models sequentially on this batch
        for config in MODELS_CONFIG:
            m_id = config["id"]
            preds = run_single_model(config, models[m_id], tokenizers[m_id], batch_records)
            
            for idx, p in enumerate(preds):
                batch_results[idx][f"{m_id}_label"] = p["label"]
                batch_results[idx][f"{m_id}_conf"] = p["conf"]
                
        # Evaluate ensemble immediately in memory
        accepted_batch = []
        rejected_batch = []
        
        for row in batch_results:
            eval_res = evaluate_ensemble(row)
            
            # Compile final database output row
            out_row = {
                "id": row["id"],
                "text": row["text"]
            }
            for config in MODELS_CONFIG:
                m_id = config["id"]
                out_row[f"{m_id}_label"] = row[f"{m_id}_label"]
                out_row[f"{m_id}_conf"]  = row[f"{m_id}_conf"]
                
            out_row["ensemble_label"] = eval_res["final_label"]
            out_row["final_label"] = eval_res["final_label"]
            out_row["ensemble_status"] = eval_res["status"]
            out_row["rejection_reason"] = eval_res.get("reason", "")
            out_row["agreement_count"] = eval_res["agreement_count"]
            out_row["average_confidence"] = eval_res["average_confidence"]
            
            if eval_res["status"] == "accepted":
                accepted_batch.append(out_row)
            else:
                rejected_batch.append(out_row)
                
        # Incremental save to final files
        if accepted_batch:
            df_acc = pd.DataFrame(accepted_batch)
            df_acc.to_csv(ACCEPTED_FILE, mode='a', header=not ACCEPTED_FILE.exists(), index=False)
        if rejected_batch:
            df_rej = pd.DataFrame(rejected_batch)
            df_rej.to_csv(REJECTED_FILE, mode='a', header=not REJECTED_FILE.exists(), index=False)
            
        pbar.update(len(batch_records))
        
    pbar.close()

    # Free memory (but keep models for post-processing)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Post-process: Fix rows where m1_label == 'OTHER'
    fix_other_labels(models, tokenizers)
    
    # Free memory
    del models
    del tokenizers
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n=============================================")
    print(" PIPELINE FINISHED - GENERATING FINAL REPORT")
    print("=============================================")
    generate_stats()


if __name__ == "__main__":
    main()