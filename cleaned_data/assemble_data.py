"""
Script to assemble and distribute data across multiple PCs.

This script:
1. Combines english_texts.csv, french_texts.csv, and accepted_articles.csv
2. Saves the combined data as global_data.csv
3. Splits the combined data into 5 CSV files for 5 PCs
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent  # PROJET_PFE directory
DATA_SEPERATION = PROJECT_ROOT / "data_seperation"
GLOBAL_DATA_DIR = DATA_SEPERATION / "global_data"
CHECK_DIALECT_DIR = DATA_SEPERATION / "check_dialect"
OUTPUT_DIR = Path(__file__).parent  # gather_assemble_data folder

# Input files
ENGLISH_FILE = GLOBAL_DATA_DIR / "english_texts.csv"
FRENCH_FILE = GLOBAL_DATA_DIR / "french_texts.csv"
ACCEPTED_FILE = CHECK_DIALECT_DIR / "accepted_articles.csv"

# Output files
COMBINED_FILE = OUTPUT_DIR / "global_data.csv"
PC1_FILE = OUTPUT_DIR / "global_data_pc1.csv"
PC2_FILE = OUTPUT_DIR / "global_data_pc2.csv"
PC3_FILE = OUTPUT_DIR / "global_data_pc3.csv"
PC4_FILE = OUTPUT_DIR / "global_data_pc4.csv"
PC5_FILE = OUTPUT_DIR / "global_data_pc5.csv"


def validate_input_files():
    """Verify that all input files exist."""
    logger.info("Validating input files...")
    
    files_to_check = {
        "English texts": ENGLISH_FILE,
        "French texts": FRENCH_FILE,
        "Accepted articles": ACCEPTED_FILE
    }
    
    for name, file_path in files_to_check.items():
        if not file_path.exists():
            logger.error(f"❌ {name} not found: {file_path}")
            return False
        logger.info(f"✓ {name} found: {file_path}")
    
    return True


def load_and_combine_data():
    """Load and combine the three CSV files with standardized structure.
    
    Structure: id, text, language
    - English: language = 'en'
    - French: language = 'fr'
    - Arabic: language = final_label (from accepted_articles.csv after ensemble validation)
    """
    logger.info("\nLoading CSV files...")
    
    try:
        # Load English texts
        english_df = pd.read_csv(ENGLISH_FILE)
        english_df = english_df[['id', 'text']].copy()
        english_df['language'] = 'en'
        logger.info(f"English texts: {len(english_df):,} rows")
        
        # Load French texts
        french_df = pd.read_csv(FRENCH_FILE)
        french_df = french_df[['id', 'text']].copy()
        french_df['language'] = 'fr'
        logger.info(f"French texts: {len(french_df):,} rows")
        
        # Load Arabic texts (from accepted_articles - uses final_label for language)
        # These are the validated Arabic articles from the ensemble annotation process
        logger.info("Loading accepted articles (validated Arabic)...")
        arabic_df = pd.read_csv(ACCEPTED_FILE)
        arabic_df = arabic_df[['id', 'text', 'final_label']].copy()
        arabic_df.rename(columns={'final_label': 'language'}, inplace=True)
        logger.info(f"Accepted articles (Arabic): {len(arabic_df):,} rows")
        
        # Combine all dataframes
        logger.info("\nCombining dataframes...")
        combined_df = pd.concat([english_df, french_df, arabic_df], ignore_index=True)
        
        # Ensure correct column order: id, text, language
        combined_df = combined_df[['id', 'text', 'language']]
        
        logger.info(f"✓ Combined data: {len(combined_df):,} rows")
        logger.info(f"  - English: {len(english_df):,}")
        logger.info(f"  - French: {len(french_df):,}")
        logger.info(f"  - Arabic (accepted): {len(arabic_df):,}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"❌ Error loading files: {e}")
        raise


def save_combined_data(df):
    """Save the combined data to a single CSV file."""
    logger.info(f"\nSaving combined data to {COMBINED_FILE}...")
    
    try:
        df.to_csv(COMBINED_FILE, index=False)
        logger.info(f"✓ Combined data saved: {COMBINED_FILE}")
    except Exception as e:
        logger.error(f"❌ Error saving combined data: {e}")
        raise


def split_into_pcs(df):
    """Split the combined data into 5 PC files."""
    logger.info("\nSplitting data into 5 PC files...")

    try:
        # Calculate split points
        total_rows = len(df)
        rows_per_pc = total_rows // 5

        logger.info(f"Total rows: {total_rows}")
        logger.info(f"Rows per PC: ~{rows_per_pc}")

        # Split data
        pc1_df = df.iloc[:rows_per_pc]
        pc2_df = df.iloc[rows_per_pc:2*rows_per_pc]
        pc3_df = df.iloc[2*rows_per_pc:3*rows_per_pc]
        pc4_df = df.iloc[3*rows_per_pc:4*rows_per_pc]
        pc5_df = df.iloc[4*rows_per_pc:]

        # Save split files
        files_and_data = [
            (PC1_FILE, pc1_df),
            (PC2_FILE, pc2_df),
            (PC3_FILE, pc3_df),
            (PC4_FILE, pc4_df),
            (PC5_FILE, pc5_df)
        ]

        for file_path, data_df in files_and_data:
            data_df.to_csv(file_path, index=False)
            logger.info(f"✓ {file_path.name}: {len(data_df)} rows")

        logger.info("\n✓ All PC files created successfully!")

        return {
            "pc1": len(pc1_df),
            "pc2": len(pc2_df),
            "pc3": len(pc3_df),
            "pc4": len(pc4_df),
            "pc5": len(pc5_df)
        }
        
    except Exception as e:
        logger.error(f"❌ Error splitting data: {e}")
        raise


def main():
    """Main function to orchestrate the data assembly and distribution."""
    logger.info("=" * 70)
    logger.info("DATA ASSEMBLY AND DISTRIBUTION SCRIPT")
    logger.info("=" * 70)
    
    # Validate input files
    if not validate_input_files():
        logger.error("❌ Validation failed. Exiting.")
        return False
    
    # Load and combine data
    combined_df = load_and_combine_data()
    
    # Save combined data
    save_combined_data(combined_df)
    
    # Split into PC files
    split_results = split_into_pcs(combined_df)
    
    # Language distribution
    logger.info("\n" + "=" * 70)
    logger.info("LANGUAGE DISTRIBUTION")
    logger.info("=" * 70)
    lang_dist = combined_df['language'].value_counts().to_dict()
    for lang in sorted(lang_dist.keys()):
        count = lang_dist[lang]
        percentage = (count / len(combined_df)) * 100
        logger.info(f"  {lang:10s} : {count:>10,} ({percentage:>5.1f}%)")
    logger.info("=" * 70)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Combined file: {COMBINED_FILE}")
    logger.info(f"Total rows: {len(combined_df):,}")
    logger.info(f"PC1 file: {split_results['pc1']:,} rows")
    logger.info(f"PC2 file: {split_results['pc2']:,} rows")
    logger.info(f"PC3 file: {split_results['pc3']:,} rows")
    logger.info(f"PC4 file: {split_results['pc4']:,} rows")
    logger.info(f"PC5 file: {split_results['pc5']:,} rows")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("\n✓ Script completed successfully!")
        else:
            logger.error("\n❌ Script failed.")
    except Exception as e:
        logger.error(f"\n❌ Script execution failed: {e}")
        raise
