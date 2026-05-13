import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation\check_dialect")
PARALLEL_DIR = DATA_DIR / "parallel_data"
PARALLEL_DIR.mkdir(exist_ok=True)

# Original files
MSA_FILE = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation\arabic_msa.csv")
DIALECTAL_FILE = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation\arabic_dialectal.csv")

# Progress on PC1 (Last line was 35081, so 35080 data rows finished)
processed_msa = 35080
processed_dialectal = 0 # Assuming we haven't started Dialectal yet

def split_data():
    print("Loading MSA...")
    df_msa = pd.read_csv(MSA_FILE)
    print("Loading Dialectal...")
    df_dialectal = pd.read_csv(DIALECTAL_FILE)

    # 1. Split MSA
    # Remaining MSA after skipping what's already done
    df_msa_remaining = df_msa.iloc[processed_msa:]
    mid_msa = len(df_msa_remaining) // 2
    
    df_msa_pc1 = df_msa_remaining.iloc[:mid_msa]
    df_msa_pc2 = df_msa_remaining.iloc[mid_msa:]

    # 2. Split Dialectal
    df_dialectal_remaining = df_dialectal.iloc[processed_dialectal:]
    mid_dialectal = len(df_dialectal_remaining) // 2

    df_dialectal_pc1 = df_dialectal_remaining.iloc[:mid_dialectal]
    df_dialectal_pc2 = df_dialectal_remaining.iloc[mid_dialectal:]

    # Save files
    print(f"Saving PC1 MSA ({len(df_msa_pc1)} rows)...")
    df_msa_pc1.to_csv(PARALLEL_DIR / "arabic_msa_pc1.csv", index=False)
    print(f"Saving PC2 MSA ({len(df_msa_pc2)} rows)...")
    df_msa_pc2.to_csv(PARALLEL_DIR / "arabic_msa_pc2.csv", index=False)

    print(f"Saving PC1 Dialectal ({len(df_dialectal_pc1)} rows)...")
    df_dialectal_pc1.to_csv(PARALLEL_DIR / "arabic_dialectal_pc1.csv", index=False)
    print(f"Saving PC2 Dialectal ({len(df_dialectal_pc2)} rows)...")
    df_dialectal_pc2.to_csv(PARALLEL_DIR / "arabic_dialectal_pc2.csv", index=False)

    print("Done! Files created in parallel_data/")

if __name__ == "__main__":
    split_data()
