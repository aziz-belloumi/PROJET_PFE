import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm

# FORCE GUEST ACCESS
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

# ==========================================
# CONFIGURATION — edit PC_ID to 1 or 2
# ==========================================
PC_ID = 1  # Set to 1 on PC1, set to 2 on PC2

OUTPUT_DIR    = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation\check_dialect")
ACCEPTED_FILE = OUTPUT_DIR / f"accepted_articles_pc{PC_ID}.csv"
BACKUP_FILE   = OUTPUT_DIR / f"accepted_articles_pc{PC_ID}_backup.csv"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512

M1_MAPPING = {
    "MSA":     "MSA",
    "EGY":     "EGY",
    "LEV":     "LEV",
    "GLF":     "GLF",
    "MAG":     "MGR",
    "MGR":     "MGR",
    "MAGHREB": "MGR",
}

VALID_LABELS = {"MSA", "EGY", "LEV", "GLF", "MGR"}


def get_verdict(row: dict) -> dict:
    votes = {}
    confs = {}

    for m in ["m1", "m2", "m3", "m4", "m5"]:
        lbl  = row.get(f"{m}_label")
        conf = row.get(f"{m}_conf")
        if lbl and isinstance(lbl, str) and lbl in VALID_LABELS:
            votes[lbl] = votes.get(lbl, 0) + 1
            confs.setdefault(lbl, []).append(
                float(conf) if conf is not None and not pd.isna(conf) else 0.0
            )

    if not votes:
        return {"status": "rejected", "reason": "complete_disagreement_low_confidence",
                "agreement_count": 0, "average_confidence": 0.0, "final_label": None}

    max_votes = max(votes.values())
    winners   = [l for l, c in votes.items() if c == max_votes]
    best_label    = winners[0]
    best_avg_conf = sum(confs[best_label]) / len(confs[best_label])

    if len(winners) > 1:
        for w in winners[1:]:
            avg_c = sum(confs[w]) / len(confs[w])
            if avg_c > best_avg_conf:
                best_label    = w
                best_avg_conf = avg_c

    if max_votes == 1 and best_avg_conf < 0.5:
        return {"status": "rejected", "reason": "complete_disagreement_low_confidence",
                "agreement_count": max_votes, "average_confidence": round(best_avg_conf, 4),
                "final_label": best_label}

    return {"status": "accepted", "reason": None,
            "agreement_count": max_votes, "average_confidence": round(best_avg_conf, 4),
            "final_label": best_label}


def main():
    print("=" * 60)
    print(f"  FIX IBRAHIM M1 — PC{PC_ID} — REMAP MAGHREB -> MGR")
    print("=" * 60)

    print(f"\n[1] Loading {ACCEPTED_FILE.name}...")
    df = pd.read_csv(ACCEPTED_FILE)
    print(f"    Total rows: {len(df):,}")

    # Always overwrite backup with current state before fixing
    print(f"[2] Creating backup {BACKUP_FILE.name}...")
    df.to_csv(BACKUP_FILE, index=False)
    print(f"    Backup saved.")

    other_mask  = df["m1_label"] == "OTHER"
    other_count = other_mask.sum()
    print(f"\n[3] Articles with m1_label == OTHER: {other_count:,}")

    if other_count == 0:
        print("    Nothing to fix. Exiting.")
        return

    print(f"\n[4] Loading Ibrahim model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(
        "IbrahimAmin/marbertv2-arabic-written-dialect-classifier", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "IbrahimAmin/marbertv2-arabic-written-dialect-classifier",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()
    print(f"    Labels: {list(model.config.id2label.values())}")

    print(f"\n[5] Running Ibrahim on {other_count:,} articles (batch={BATCH_SIZE})...")
    other_indices = df[other_mask].index.tolist()
    other_texts   = df.loc[other_mask, "text"].tolist()
    id2label      = model.config.id2label

    new_labels = []
    new_confs  = []

    for i in tqdm(range(0, len(other_texts), BATCH_SIZE), desc="Ibrahim re-run"):
        batch_texts = other_texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs      = torch.softmax(logits, dim=-1)
        pred_ids   = torch.argmax(probs, dim=-1).tolist()
        pred_confs = probs.max(dim=-1).values.tolist()
        for pred_id, pred_conf in zip(pred_ids, pred_confs):
            raw_label    = id2label[pred_id]
            mapped_label = M1_MAPPING.get(raw_label, "OTHER")
            new_labels.append(mapped_label)
            new_confs.append(round(pred_conf, 4))

    print(f"\n[6] Updating m1_label and m1_conf...")
    for i, idx in enumerate(other_indices):
        df.at[idx, "m1_label"] = new_labels[i]
        df.at[idx, "m1_conf"]  = new_confs[i]

    new_label_dist = pd.Series(new_labels).value_counts().to_dict()
    print(f"    New m1 distribution for previously-OTHER rows:")
    for lbl, cnt in sorted(new_label_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"      {lbl:<10}: {cnt:>10,}")

    print(f"\n[7] Recalculating ensemble verdicts...")
    for idx in tqdm(other_indices, desc="Verdicts"):
        row     = df.loc[idx].to_dict()
        verdict = get_verdict(row)
        df.at[idx, "agreement_count"]    = verdict["agreement_count"]
        df.at[idx, "average_confidence"] = verdict["average_confidence"]
        df.at[idx, "final_label"]        = verdict["final_label"]
        df.at[idx, "ensemble_label"]     = verdict["final_label"]
        df.at[idx, "ensemble_status"]    = verdict["status"]
        df.at[idx, "rejection_reason"]   = verdict.get("reason", "")

    print(f"\n[8] Saving corrected {ACCEPTED_FILE.name}...")
    df.to_csv(ACCEPTED_FILE, index=False)
    print(f"    Saved {len(df):,} rows.")

    print(f"\n{'='*60}")
    print(f"  DONE — PC{PC_ID} SUMMARY")
    print(f"{'='*60}")
    print(f"  Articles fixed : {other_count:,}")
    print(f"  Final label distribution:")
    for lbl in ["MSA", "EGY", "LEV", "GLF", "MGR"]:
        cnt = (df["final_label"] == lbl).sum()
        pct = cnt / len(df) * 100
        print(f"    {lbl:<6}: {cnt:>10,}  ({pct:.2f}%)")
    print(f"\n  Remaining OTHER : {(df['m1_label'] == 'OTHER').sum():,}")
    print(f"{'='*60}")
    print(f"\n  -> Now run merge_and_report.py to get the final merged stats.")


if __name__ == "__main__":
    main()