import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import json
import logging
import os

# FORCE GUEST ACCESS: Remove any broken or expired Hugging Face tokens from this process
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

# ==========================================
# CONFIGURATION
# ==========================================
PC_ID = "pc1"  # CHANGE THIS to 'pc1', 'pc2', or 'pc3' on each machine

DATA_DIR = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation")
INPUT_FILE = DATA_DIR / f"arabic_texts_{PC_ID}.csv"
OUTPUT_DIR = DATA_DIR / "check_dialect"
RESULTS_FILE = OUTPUT_DIR / f"ensemble_results_{PC_ID}.csv"
PROGRESS_FILE = OUTPUT_DIR / f"progress_{PC_ID}.json"

BATCH_SIZE = 32 # Reduced for 3 models on 6GB VRAM
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_CONFIG = [
    {
        "id": "m1",
        "name": "Ibrahim-V2",
        "path": "IbrahimAmin/marbertv2-arabic-written-dialect-classifier",
        "mapping": {"MSA": "MSA", "EGY": "EGY", "LEV": "LEV", "GLF": "GLF", "MAG": "MGR", "MGR": "MGR"}
    },
    {
        "id": "m2",
        "name": "Camel-Mix-DID",
        "path": "CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus6",
        "mapping": {
            "MSA": "MSA", "BEI": "LEV", "JER": "LEV", "DAM": "LEV", "ALE": "LEV",
            "CAI": "EGY", "BAG": "GLF", "BAS": "GLF", "KHA": "EGY", 
            "CAS": "MGR", "TUN": "MGR", "ALG": "MGR", "BEN": "MGR", "TRI": "MGR"
        }
    },
    {
        "id": "m3",
        "name": "Camel-NADI-DID",
        "path": "CAMeL-Lab/bert-base-arabic-camelbert-msa-did-nadi",
        "mapping": {
            "MSA": "MSA", "Egypt": "EGY", "Iraq": "GLF", "Jordan": "LEV", "Lebanon": "LEV",
            "Palestine": "LEV", "Syria": "LEV", "Saudi_Arabia": "GLF", "UAE": "GLF",
            "Kuwait": "GLF", "Qatar": "GLF", "Oman": "GLF", "Bahrain": "GLF", "Yemen": "GLF",
            "Libya": "MGR", "Morocco": "MGR", "Tunisia": "MGR", "Algeria": "MGR", "Mauritania": "MGR",
            "Sudan": "EGY"
        }
    }
]

# ==========================================
# CORE CLASSES
# ==========================================
class ArabicDataset(Dataset):
    def __init__(self, texts, ids):
        self.texts = texts
        self.ids = ids
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return str(self.texts[idx]), self.ids[idx]

def load_models():
    pipes = []
    for config in MODELS_CONFIG:
        print(f"Loading {config['name']}...")
        pipe = pipeline(
            "text-classification",
            model=config["path"],
            tokenizer=config["path"],
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            token=None
        )
        pipes.append(pipe)
    return pipes

def get_verdict(preds):
    """Majority vote with confidence tie-breaker."""
    labels = [preds["m1_label"], preds["m2_label"], preds["m3_label"]]
    confs = [preds["m1_conf"], preds["m2_conf"], preds["m3_conf"]]
    
    # 1. Check for absolute majority (2 or 3 models agree)
    counts = {}
    for l in labels:
        if l == "DIA": continue # Skip the generic "Dialect" from Camel for voting if others are specific
        counts[l] = counts.get(l, 0) + 1
    
    if counts:
        max_count = max(counts.values())
        potential_winners = [l for l, c in counts.items() if c == max_count]
        
        if max_count >= 2: # Majority found
            return potential_winners[0]

    # 2. If no majority or total disagreement, use the highest confidence model
    # Note: We prefer specific regions over Camel's generic "DIA"
    best_idx = 0
    max_conf = -1.0
    for i in range(3):
        score = confs[i]
        # Slight bonus to specific models (m1, m2) over binary model (m3)
        if i < 2: score += 0.05 
        if score > max_conf:
            max_conf = score
            best_idx = i
            
    return labels[best_idx]

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    if not OUTPUT_DIR.exists(): OUTPUT_DIR.mkdir(parents=True)
    if not INPUT_FILE.exists():
        logger.error(f"Input file {INPUT_FILE} not found. Run fetch_texts.py first.")
        return

    # Resume Progress
    start_idx = 0
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            start_idx = json.load(f).get("processed", 0)

    # Load Data
    logger.info(f"Loading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    if start_idx > 0:
        logger.info(f"Resuming from index {start_idx}...")
        df = df.iloc[start_idx:]

    dataset = ArabicDataset(df['text'].tolist(), df['id'].tolist())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Pipelines
    pipes = load_models()
    
    logger.info(f"Starting Ensemble Annotation for {PC_ID}...")
    
    all_results = []
    
    for i, (batch_texts, batch_ids) in enumerate(tqdm(dataloader, desc="Annotating")):
        batch_data = [{"id": bid.item(), "text": txt} for bid, txt in zip(batch_ids, batch_texts)]
        
        for m_idx, config in enumerate(MODELS_CONFIG):
            pipe = pipes[m_idx]
            
            # Pipeline can handle batches directly
            pipe_results = pipe(list(batch_texts), batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)
            
            for b_idx, res in enumerate(pipe_results):
                raw_label = res["label"]
                mapped_label = config["mapping"].get(raw_label, "MSA")
                
                batch_data[b_idx][f"{config['id']}_label"] = mapped_label
                batch_data[b_idx][f"{config['id']}_conf"] = round(res["score"], 4)

        # Apply Verdict
        for item in batch_data:
            item["final_verdict"] = get_verdict(item)
            all_results.append(item)

        # Periodic Save (every 10 batches)
        if (i + 1) % 10 == 0:
            save_df = pd.DataFrame(all_results)
            save_df.to_csv(RESULTS_FILE, mode='a', header=not RESULTS_FILE.exists(), index=False)
            all_results = []
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"processed": start_idx + (i + 1) * BATCH_SIZE}, f)

    # Final Save
    if all_results:
        save_df = pd.DataFrame(all_results)
        save_df.to_csv(RESULTS_FILE, mode='a', header=not RESULTS_FILE.exists(), index=False)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({"processed": start_idx + len(df)}, f)

    logger.info(f"Done! Results saved to {RESULTS_FILE.name}")

if __name__ == "__main__":
    main()
