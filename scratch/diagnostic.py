import pandas as pd
import unicodedata
import sys
from transformers import AutoTokenizer

# Configure output file
out_path = "scratch/diagnostic_results.txt"
with open(out_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        f.write(str(msg) + "\n")
        try:
            print(msg)
        except UnicodeEncodeError:
            print(str(msg).encode("ascii", errors="replace").decode("ascii"))

    log("Loading data...")
    df = pd.read_csv("fine_tune_data/global_data_merged.csv")
    log(f"Total rows in CSV: {len(df)}")
    log(f"Columns: {list(df.columns)}")

    VALID_LANG = {"msa", "egy", "lev", "glf", "mgr"}
    df["text"] = df["text"].astype(str)
    df["language"] = df["language"].astype(str).str.lower().str.strip()
    df = df[df["language"].isin(VALID_LANG)].copy()
    log(f"Rows after language filter: {len(df)}")

    CATEGORY_DISPLAY = {
        0:  {"ar": "السياسة",    "en": "Politics"},
        1:  {"ar": "الاقتصاد",   "en": "Economy"},
        2:  {"ar": "الأمن",      "en": "Security"},
        3:  {"ar": "الطاقة",     "en": "Energy"},
        4:  {"ar": "النزاع",     "en": "Conflict"},
        5:  {"ar": "الانتخابات", "en": "Elections"},
        6:  {"ar": "العدالة",    "en": "Justice"},
        7:  {"ar": "الصحة",      "en": "Health"},
        8:  {"ar": "الطقس",      "en": "Weather"},
        9:  {"ar": "الرياضة",    "en": "Sports"},
        10: {"ar": "الثقافة",    "en": "Culture"},
        11: {"ar": "التعليم",    "en": "Education"},
        12: {"ar": "التكنولوجيا","en": "Technology"},
        13: {"ar": "البيئة",     "en": "Environment"},
        14: {"ar": "الدبلوماسية","en": "Diplomacy"},
        15: {"ar": "الدين",      "en": "Religion"},
        16: {"ar": "الهجرة",     "en": "Migration"},
        17: {"ar": "عام",        "en": "General"},
    }

    def norm(x: str) -> str:
        if not isinstance(x, str):
            return ""
        x = unicodedata.normalize("NFKD", x)
        x = "".join(c for c in x if not unicodedata.combining(c))
        return x.strip().lower()

    TOPIC_MAP = {}
    for cid, langs in CATEGORY_DISPLAY.items():
        TOPIC_MAP[norm(langs["ar"])] = cid
        TOPIC_MAP[norm(langs["en"])] = cid
    TOPIC_MAP[norm("العدل")] = 6

    def parse_topic(x) -> int:
        if pd.isna(x):
            return -1
        s = str(x).strip()
        if s.isdigit():
            i = int(s)
            return i if 0 <= i <= 17 else -1
        return TOPIC_MAP.get(norm(s), -1)

    df["label"] = df["topic"].apply(parse_topic).astype(int)

    log("\nValue counts for topic column:")
    log(df["topic"].value_counts(dropna=False).head(20).to_string())

    log("\nValue counts for parsed label:")
    log(df["label"].value_counts(dropna=False).to_string())

    log(f"\nRows with label == -1: {len(df[df['label'] == -1])}")

    # Check tokenization
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    log(f"\nLoading tokenizer {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    log("Tokenizing a sample text:")
    sample_text = df["text"].iloc[0]
    log(f"Raw text: {sample_text[:120]}")
    log(f"Tokens: {tok.tokenize(sample_text)[:20]}")
    log(f"Token IDs: {tok.encode(sample_text)[:20]}")

    log("\nChecking label quality samples:")
    for label_id in sorted(CATEGORY_DISPLAY.keys()):
        subset = df[df["label"] == label_id]
        if len(subset) > 0:
            log(f"Label {label_id} ({CATEGORY_DISPLAY[label_id]['en']}): count={len(subset)}")
            log(f"  Sample Text: {subset['text'].iloc[0][:120]}...")
        else:
            log(f"Label {label_id} ({CATEGORY_DISPLAY[label_id]['en']}) has NO SAMPLES!")
