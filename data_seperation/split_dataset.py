import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation")
    input_file = data_dir / "arabic_texts.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
        
    print("Loading arabic_texts.csv...")
    df = pd.read_csv(input_file)
    total_len = len(df)
    print(f"Total articles: {total_len:,}")
    
    # Split into 2 equal parts
    chunk_size = total_len // 2 + 1
    
    print("\nSplitting and saving into 2 balanced parts...")
    for pc_id in range(1, 3):
        start_idx = (pc_id - 1) * chunk_size
        end_idx = min(pc_id * chunk_size, total_len)
        
        df_part = df.iloc[start_idx:end_idx]
        output_file = data_dir / f"arabic_texts_pc{pc_id}.csv"
        df_part.to_csv(output_file, index=False)
        print(f"  PC {pc_id}: Saved {len(df_part):,} rows to {output_file.name}")
        
    print("\nDone! You can now set `PC_ID = 1` or `2` inside ensemble_annotator.py on each PC and run it concurrently!")

if __name__ == "__main__":
    main()
