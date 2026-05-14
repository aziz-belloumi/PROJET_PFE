import pandas as pd
from pathlib import Path
import json

# Paths
DATA_DIR = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation\check_dialect")
PARALLEL_DIR = DATA_DIR / "parallel_data"
PROGRESS_FILE_PC1 = DATA_DIR / "annotation_progress_pc1.json"

# 1. Load PC1 Progress
with open(PROGRESS_FILE_PC1, 'r') as f:
    progress_pc1 = json.load(f)

done_msa = progress_pc1.get("MSA", 0)
done_dialectal = progress_pc1.get("Dialectal", 0)

print(f"PC1 Progress - MSA: {done_msa}, Dialectal: {done_dialectal}")

def split_pc1_workload():
    # Load current PC1 files
    msa_pc1_path = PARALLEL_DIR / "arabic_msa_pc1.csv"
    dialectal_pc1_path = PARALLEL_DIR / "arabic_dialectal_pc1.csv"

    print("Loading PC1 MSA data...")
    df_msa = pd.read_csv(msa_pc1_path)
    print("Loading PC1 Dialectal data...")
    df_dialectal = pd.read_csv(dialectal_pc1_path)

    # --- Split MSA ---
    df_msa_done = df_msa.iloc[:done_msa]
    df_msa_remaining = df_msa.iloc[done_msa:]
    
    mid_msa = len(df_msa_remaining) // 2
    
    df_msa_pc1_new = pd.concat([df_msa_done, df_msa_remaining.iloc[:mid_msa]])
    df_msa_pc3 = df_msa_remaining.iloc[mid_msa:]

    # --- Split Dialectal ---
    df_dialectal_done = df_dialectal.iloc[:done_dialectal]
    df_dialectal_remaining = df_dialectal.iloc[done_dialectal:]
    
    mid_dialectal = len(df_dialectal_remaining) // 2
    
    df_dialectal_pc1_new = pd.concat([df_dialectal_done, df_dialectal_remaining.iloc[:mid_dialectal]])
    df_dialectal_pc3 = df_dialectal_remaining.iloc[mid_dialectal:]

    # Save files
    print(f"Saving updated PC1 MSA ({len(df_msa_pc1_new)} rows, {done_msa} already done)...")
    df_msa_pc1_new.to_csv(PARALLEL_DIR / "arabic_msa_pc1.csv", index=False)
    
    print(f"Saving PC3 MSA ({len(df_msa_pc3)} rows)...")
    df_msa_pc3.to_csv(PARALLEL_DIR / "arabic_msa_pc3.csv", index=False)

    print(f"Saving updated PC1 Dialectal ({len(df_dialectal_pc1_new)} rows, {done_dialectal} already done)...")
    df_dialectal_pc1_new.to_csv(PARALLEL_DIR / "arabic_dialectal_pc1.csv", index=False)
    
    print(f"Saving PC3 Dialectal ({len(df_dialectal_pc3)} rows)...")
    df_dialectal_pc3.to_csv(PARALLEL_DIR / "arabic_dialectal_pc3.csv", index=False)

    # Initialize PC3 progress file
    progress_pc3_file = DATA_DIR / "annotation_progress_pc3.json"
    with open(progress_pc3_file, 'w') as f:
        json.dump({"MSA": 0, "Dialectal": 0}, f)

    print("\nSplit Complete!")
    print(f"PC1 now has {len(df_msa_pc1_new) - done_msa} MSA rows and {len(df_dialectal_pc1_new) - done_dialectal} Dialectal rows left.")
    print(f"PC3 has {len(df_msa_pc3)} MSA rows and {len(df_dialectal_pc3)} Dialectal rows to start from scratch.")

if __name__ == "__main__":
    split_pc1_workload()
