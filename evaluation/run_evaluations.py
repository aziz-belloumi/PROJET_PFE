import pandas as pd
import requests
import re
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import time

# -------------------------------
# Configuration
# -------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SENTIMENT_CSV = PROJECT_ROOT / "scripts" / "manual_eval_sample_sentiment.csv"
TOPIC_CSV = PROJECT_ROOT / "scripts" / "manual_eval_sample_topic.csv"

# Chunking configuration (important for long articles)
CHUNK_SIZE = 4000       # characters per chunk
CHUNK_OVERLAP = 400     # overlap to avoid boundary loss
MAX_RETRIES = 3

# Model version identifiers (mv0..mv3) — no alias names
MODEL_VERSIONS = [0, 1, 2, 3]

# Multilingual topic label mapping — matches the database CATEGORY_DISPLAY exactly
CATEGORY_DISPLAY = {
    0: {"ar": "السياسة",    "fr": "Politique",  "en": "Politics"},
    1: {"ar": "الاقتصاد",  "fr": "Économie",   "en": "Economy"},
    2: {"ar": "الأمن",      "fr": "Sécurité",   "en": "Security"},
    3: {"ar": "الطاقة",    "fr": "Énergie",    "en": "Energy"},
    4: {"ar": "النزاع",    "fr": "Conflit",    "en": "Conflict"},
    5: {"ar": "الانتخابات","fr": "Élections",  "en": "Elections"},
    6: {"ar": "العدالة",   "fr": "Justice",    "en": "Justice"},
    7: {"ar": "الصحة",     "fr": "Santé",      "en": "Health"},
    8: {"ar": "الطقس",     "fr": "Météo",      "en": "Weather"},
    9: {"ar": "الرياضة",    "fr": "Sport",    "en": "Sports"},
    10: {"ar": "الثقافة",    "fr": "Culture",  "en": "Culture"},
}

# Flat set of every valid label across all languages (for validation)
TOPIC_ALLOWED_LABELS = {v for row in CATEGORY_DISPLAY.values() for v in row.values()}

# Sentiments allowed
SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
SENTIMENT_LIST_TEXT = "\n".join(SENTIMENTS)

# Sentiment normalization map
SENTIMENT_NORM = {
    "POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL",
    "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL",
}

# -------------------------------
# Article Chunking
# -------------------------------

def chunk_article(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long article into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


# -------------------------------
# Prompt Builders
# -------------------------------

def build_topic_prompt(language, article):
    # Build the multilingual label table for the prompt
    table_lines = ["Arabic | French | English"]
    for row in CATEGORY_DISPLAY.values():
        table_lines.append(f"{row['ar']} | {row['fr']} | {row['en']}")
    label_table = "\n".join(table_lines)

    return f"""You are evaluating topic classification for news articles.

The article may be written in Arabic, English, or French.
Article language: {language}

Article:
{article}

You MUST choose the topic from the following table and return it EXACTLY as written.
Do not translate, paraphrase, or invent new labels.

{label_table}

Rules:
- If the article is in Arabic, return the Arabic label.
- If the article is in English, return the English label.
- If the article is in French, return the French label.
- Respond ONLY with the two lines below. No explanations. No additional text.

true_prediction: <exact label from the table above>
true_language: <Arabic | English | French>"""

def build_sentiment_prompt(language, article, predicted_sentiment):
    return f"""
You are evaluating sentiment classification for news articles.

The article may be written in Arabic, English, or French.

Article language: {language}

Article:
{article}

Possible sentiments:
{SENTIMENT_LIST_TEXT}

Tasks:
1. Determine the TRUE sentiment of the article.

Output STRICTLY in this format:

true_prediction: one of the sentiment names
true_language: one of [Arabic, English, French]
"""


# -------------------------------
# LLM Call
# -------------------------------

def call_llm(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "temperature": 0,
            "num_predict": 100
        }
    }

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=300
            )

            if response.status_code == 200:
                return response.json()["response"]

        except Exception:
            pass

        time.sleep(2)

    raise RuntimeError("LLM request failed after retries")


# -------------------------------
# Parse LLM Output
# -------------------------------

def parse_llm_response(response_text):
    """
    Parses the LLM output to extract 'true_prediction' and 'true_language'.
    """
    prediction = ""
    language = ""
    
    # Extract prediction
    match_p = re.search(r"true_prediction:\s*(.*)", response_text, re.IGNORECASE)
    if match_p:
        prediction = match_p.group(1).strip()
    
    # Extract language
    match_l = re.search(r"true_language:\s*(.*)", response_text, re.IGNORECASE)
    if match_l:
        language = match_l.group(1).strip().lower()
        # Map back to codes
        if "arabic" in language: language = "ar"
        elif "english" in language: language = "en"
        elif "french" in language: language = "fr"

    return prediction, language


# -------------------------------
# Topic Evaluation
# -------------------------------

def evaluate_topics():

    if not TOPIC_CSV.exists():
        print(f"Topic CSV not found: {TOPIC_CSV}")
        return

    df = pd.read_csv(TOPIC_CSV).fillna("")

    if "true_prediction" not in df.columns:
        df["true_prediction"] = ""
    if "true_language" not in df.columns:
        df["true_language"] = ""

    # Ensure verdict columns exist (may be absent in old CSVs)
    for mv in MODEL_VERSIONS:
        if f"topic_verdict_mv{mv}" not in df.columns:
            df[f"topic_verdict_mv{mv}"] = ""

    topic_label_cols = sorted([c for c in df.columns if c.startswith("topic_label_")])

    print(f"Total topic articles: {len(df)}")

    # ---------------------------------------------------------------
    # LLM pass: the model ONLY determines true_prediction + true_language
    # ---------------------------------------------------------------
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if str(row["true_prediction"]).strip():
            continue  # already evaluated — skip

        prompt = build_topic_prompt(row["language"], row["body"])
        resp = call_llm(prompt)
        pred, lang_true = parse_llm_response(resp)
        df.at[idx, "true_prediction"] = pred
        df.at[idx, "true_language"] = lang_true

    # ---------------------------------------------------------------
    # Python verdict computation — no LLM involvement
    # ---------------------------------------------------------------
    print("Computing topic verdicts (Python)...")
    for col in topic_label_cols:
        mv_id = col.replace("topic_label_", "")          # e.g. "mv0"
        verdict_col = f"topic_verdict_{mv_id}"            # e.g. "topic_verdict_mv0"

        df[verdict_col] = df.apply(
            lambda row, c=col: (
                str(row[c]).strip() == str(row["true_prediction"]).strip()
            ) if str(row["true_prediction"]).strip() else "",
            axis=1
        )

    df.to_csv(TOPIC_CSV, index=False, encoding="utf-8-sig")
    print("Topic evaluation finished.")


# -------------------------------
# Sentiment Evaluation
# -------------------------------

def evaluate_sentiment():

    if not SENTIMENT_CSV.exists():
        print(f"Sentiment CSV not found: {SENTIMENT_CSV}")
        return

    df = pd.read_csv(SENTIMENT_CSV).fillna("")

    if "true_prediction" not in df.columns:
        df["true_prediction"] = ""
    if "true_language" not in df.columns:
        df["true_language"] = ""

    # Ensure verdict columns exist (may be absent in old CSVs)
    for mv in MODEL_VERSIONS:
        if f"sentiment_verdict_mv{mv}" not in df.columns:
            df[f"sentiment_verdict_mv{mv}"] = ""

    sentiment_label_cols = sorted([c for c in df.columns if c.startswith("sentiment_label_")])
    primary_sentiment_col = sentiment_label_cols[0] if sentiment_label_cols else None

    print(f"Total sentiment articles: {len(df)}")

    # ---------------------------------------------------------------
    # LLM pass: the model ONLY determines true_prediction + true_language
    # ---------------------------------------------------------------
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if str(row["true_prediction"]).strip():
            continue  # already evaluated — skip

        prompt = build_sentiment_prompt(row["language"], row["body"], row.get(primary_sentiment_col, ""))
        resp = call_llm(prompt)
        pred, lang_true = parse_llm_response(resp)
        df.at[idx, "true_prediction"] = pred
        df.at[idx, "true_language"] = lang_true

    # ---------------------------------------------------------------
    # Python verdict computation — no LLM involvement
    # ---------------------------------------------------------------
    print("Computing sentiment verdicts (Python)...")
    for col in sentiment_label_cols:
        mv_id = col.replace("sentiment_label_", "")          # e.g. "mv0"
        verdict_col = f"sentiment_verdict_{mv_id}"            # e.g. "sentiment_verdict_mv0"

        df[verdict_col] = df.apply(
            lambda row, c=col: (
                SENTIMENT_NORM.get(str(row[c]).strip().upper(), str(row[c]).strip().upper()) ==
                SENTIMENT_NORM.get(str(row["true_prediction"]).strip().upper(), str(row["true_prediction"]).strip().upper())
            ) if str(row["true_prediction"]).strip() else "",
            axis=1
        )

    df.to_csv(SENTIMENT_CSV, index=False, encoding="utf-8-sig")
    print("Sentiment evaluation finished.")


# -------------------------------
# Statistics Generator
# -------------------------------

def _accuracy_block(df: pd.DataFrame, true_col: str, pred_col: str, task: str) -> list:
    """Compute overall + per-language + per-label accuracy rows."""
    rows = []
    evaluated = df[df[true_col].astype(str).str.strip() != ""].copy()
    total = len(evaluated)
    if total == 0:
        rows.append({"Task": task, "Category": "Overall", "Metric": "Total Evaluated", "Value": 0, "Note": "No evaluated rows found"})
        return rows, 0

    y_pred = evaluated[pred_col].astype(str).str.upper()
    y_true = evaluated[true_col].astype(str).str.upper()
    is_correct = y_pred == y_true

    correct = is_correct.sum()
    accuracy = correct / total * 100

    rows.append({"Task": task, "Category": "Overall", "Metric": "Total Evaluated", "Value": total, "Note": ""})
    rows.append({"Task": task, "Category": "Overall", "Metric": "Correct", "Value": correct, "Note": ""})
    rows.append({"Task": task, "Category": "Overall", "Metric": "Accuracy (%)", "Value": f"{accuracy:.1f}", "Note": ""})

    # Per language
    if "language" in evaluated.columns:
        for lang in sorted(evaluated["language"].dropna().unique()):
            sub_mask = evaluated["language"] == lang
            sub = evaluated[sub_mask]
            if len(sub) == 0: continue
            lang_acc = is_correct.loc[sub.index].sum() / len(sub) * 100
            rows.append({"Task": task, "Category": f"Language: {lang}", "Metric": "Accuracy (%)", "Value": f"{lang_acc:.1f}", "Note": f"{len(sub)} articles"})

    # Per predicted label overall
    if pred_col and pred_col in evaluated.columns:
        for label in sorted(evaluated[pred_col].dropna().unique()):
            sub_mask = evaluated[pred_col] == label
            sub = evaluated[sub_mask]
            if len(sub) == 0: continue
            lbl_acc = is_correct.loc[sub.index].sum() / len(sub) * 100
            rows.append({"Task": task, "Category": f"Label: {label} (Overall)", "Metric": "Accuracy (%)", "Value": f"{lbl_acc:.1f}", "Note": f"{len(sub)} articles"})

            # Per label AND language
            if "language" in sub.columns:
                for lang in sorted(sub["language"].dropna().unique()):
                    sub_lang_mask = (evaluated[pred_col] == label) & (evaluated["language"] == lang)
                    sub_lang = evaluated[sub_lang_mask]
                    if len(sub_lang) == 0: continue
                    lbl_lang_acc = is_correct.loc[sub_lang.index].sum() / len(sub_lang) * 100
                    rows.append({"Task": task, "Category": f"Label: {label} | Lang: {lang}", "Metric": "Accuracy (%)", "Value": f"{lbl_lang_acc:.1f}", "Note": f"{len(sub_lang)} articles"})

    return rows, accuracy


def generate_statistics():
    print("\n--- Generating Statistics ---")
    rows = []
    topic_acc = None
    sentiment_acc = None

    # ---- Topic ----
    if TOPIC_CSV.exists():
        df_t = pd.read_csv(TOPIC_CSV).fillna("")

        topic_verdict_cols = sorted([c for c in df_t.columns if c.startswith("topic_verdict_mv")])
        topic_label_cols   = sorted([c for c in df_t.columns if c.startswith("topic_label_")])
        primary_topic_col  = topic_label_cols[0] if topic_label_cols else None

        evaluated_t = df_t[df_t["true_prediction"].astype(str).str.strip() != ""].copy()

        if not evaluated_t.empty and topic_verdict_cols:
            rows.append({"Task": "Topic", "Category": "Per Model Accuracy", "Metric": "---", "Value": "---", "Note": ""})
            rows.append({"Task": "Topic", "Category": "Overall", "Metric": "Total Evaluated", "Value": len(evaluated_t), "Note": ""})

            for vcol in topic_verdict_cols:
                mv_id = vcol.replace("topic_verdict_", "")   # e.g. "mv0"
                # Verdict column contains True/False (or empty for unevaluated)
                mask = evaluated_t[vcol].astype(str).str.strip().isin(["True", "False"])
                sub = evaluated_t[mask]
                if sub.empty:
                    continue
                correct = (sub[vcol].astype(str) == "True").sum()
                acc = correct / len(sub) * 100
                rows.append({
                    "Task": "Topic",
                    "Category": f"Model: {mv_id}",
                    "Metric": "Accuracy (%)",
                    "Value": f"{acc:.1f}",
                    "Note": f"Based on {len(sub)} articles"
                })
                if vcol == topic_verdict_cols[0]:
                    topic_acc = acc

            # Language detection accuracy
            if "true_language" in evaluated_t.columns and evaluated_t["true_language"].astype(str).str.strip().any():
                lang_correct = (
                    evaluated_t["language"].astype(str).str.lower() ==
                    evaluated_t["true_language"].astype(str).str.lower()
                ).sum()
                lang_acc = lang_correct / len(evaluated_t) * 100
                rows.append({
                    "Task": "Language",
                    "Category": "DB Language Detection",
                    "Metric": "Accuracy (%)",
                    "Value": f"{lang_acc:.1f}",
                    "Note": "Evaluating the language routing model"
                })

            # Per-language and per-label breakdown using primary model verdict
            if primary_topic_col and topic_verdict_cols:
                primary_vcol = topic_verdict_cols[0]
                sub_ev = evaluated_t[
                    evaluated_t[primary_vcol].astype(str).str.strip().isin(["True", "False"])
                ].copy()
                sub_ev["_correct"] = sub_ev[primary_vcol].astype(str) == "True"

                if "language" in sub_ev.columns:
                    for lang in sorted(sub_ev["language"].dropna().unique()):
                        lang_sub = sub_ev[sub_ev["language"] == lang]
                        if lang_sub.empty: continue
                        la = lang_sub["_correct"].sum() / len(lang_sub) * 100
                        rows.append({"Task": "Topic", "Category": f"Language: {lang}",
                                     "Metric": "Accuracy (%)", "Value": f"{la:.1f}",
                                     "Note": f"{len(lang_sub)} articles"})

                for label in sorted(sub_ev["true_prediction"].dropna().unique()):
                    lbl_sub = sub_ev[sub_ev["true_prediction"] == label]
                    if lbl_sub.empty: continue
                    la = lbl_sub["_correct"].sum() / len(lbl_sub) * 100
                    rows.append({"Task": "Topic", "Category": f"Label: {label} (Overall)",
                                 "Metric": "Accuracy (%)", "Value": f"{la:.1f}",
                                 "Note": f"{len(lbl_sub)} articles"})
        else:
            rows.append({"Task": "Topic", "Category": "Overall", "Metric": "Status",
                         "Value": "Not evaluated yet", "Note": ""})
    else:
        rows.append({"Task": "Topic", "Category": "Overall", "Metric": "Status",
                     "Value": "CSV not found", "Note": ""})

    rows.append({"Task": "", "Category": "", "Metric": "", "Value": "", "Note": ""})  # separator

    # ---- Sentiment ----
    if SENTIMENT_CSV.exists():
        df_s = pd.read_csv(SENTIMENT_CSV).fillna("")

        sent_verdict_cols = sorted([c for c in df_s.columns if c.startswith("sentiment_verdict_mv")])

        evaluated_s = df_s[df_s["true_prediction"].astype(str).str.strip() != ""].copy()

        if not evaluated_s.empty and sent_verdict_cols:
            rows.append({"Task": "Sentiment", "Category": "Per Model Accuracy", "Metric": "---", "Value": "---", "Note": ""})
            rows.append({"Task": "Sentiment", "Category": "Overall", "Metric": "Total Evaluated", "Value": len(evaluated_s), "Note": ""})

            for vcol in sent_verdict_cols:
                mv_id = vcol.replace("sentiment_verdict_", "")   # e.g. "mv0"
                mask = evaluated_s[vcol].astype(str).str.strip().isin(["True", "False"])
                sub = evaluated_s[mask]
                if sub.empty:
                    continue
                correct = (sub[vcol].astype(str) == "True").sum()
                acc = correct / len(sub) * 100
                rows.append({
                    "Task": "Sentiment",
                    "Category": f"Model: {mv_id}",
                    "Metric": "Accuracy (%)",
                    "Value": f"{acc:.1f}",
                    "Note": f"Based on {len(sub)} articles"
                })
                if vcol == sent_verdict_cols[0]:
                    sentiment_acc = acc
        else:
            rows.append({"Task": "Sentiment", "Category": "Overall", "Metric": "Status",
                         "Value": "Not evaluated yet", "Note": ""})
    else:
        rows.append({"Task": "Sentiment", "Category": "Overall", "Metric": "Status",
                     "Value": "CSV not found", "Note": ""})

    rows.append({"Task": "", "Category": "", "Metric": "", "Value": "", "Note": ""})  # separator

    # ---- Recommendation ----
    if topic_acc is not None and sentiment_acc is not None:
        if topic_acc > sentiment_acc:
            better = "Topic Classification"
            note = f"Topic accuracy ({topic_acc:.1f}%) is higher than Sentiment accuracy ({sentiment_acc:.1f}%)"
        elif sentiment_acc > topic_acc:
            better = "Sentiment Classification"
            note = f"Sentiment accuracy ({sentiment_acc:.1f}%) is higher than Topic accuracy ({topic_acc:.1f}%)"
        else:
            better = "Both perform equally"
            note = f"Both models achieved {topic_acc:.1f}% accuracy"
        rows.append({"Task": "Recommendation", "Category": "Best Performing", "Metric": "Model Task", "Value": better, "Note": note})

        # Flag weak labels (accuracy < 60%)
        weak_topic = []
        weak_sent = []
        for r in rows:
            if r["Task"] == "Topic" and r["Category"].startswith("Label:") and r["Value"] not in ("",):
                try:
                    if float(r["Value"]) < 60:
                        weak_topic.append(r["Category"].replace("Label: ", ""))
                except ValueError:
                    pass
            if r["Task"] == "Sentiment" and r["Category"].startswith("Model:") and r["Value"] not in ("",):
                try:
                    if float(r["Value"]) < 60:
                        weak_sent.append(r["Category"].replace("Model: ", ""))
                except ValueError:
                    pass

        if weak_topic:
            rows.append({"Task": "Recommendation", "Category": "Weak Topic Labels (<60%)", "Metric": "Labels",
                         "Value": ", ".join(weak_topic), "Note": "Consider improving training data for these topics"})
        if weak_sent:
            rows.append({"Task": "Recommendation", "Category": "Weak Sentiment Models (<60%)", "Metric": "Models",
                         "Value": ", ".join(weak_sent), "Note": "Consider improving these models"})
    elif topic_acc is not None:
        rows.append({"Task": "Recommendation", "Category": "Info", "Metric": "Note", "Value": "Only topic data available", "Note": ""})
    elif sentiment_acc is not None:
        rows.append({"Task": "Recommendation", "Category": "Info", "Metric": "Note", "Value": "Only sentiment data available", "Note": ""})
    else:
        rows.append({"Task": "Recommendation", "Category": "Info", "Metric": "Note",
                     "Value": "No evaluated data found — run evaluations first", "Note": ""})

    out_path = SCRIPT_DIR / "evaluation.csv"
    pd.DataFrame(rows, columns=["Task", "Category", "Metric", "Value", "Note"]).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    print(f"Statistics saved to: {out_path}")


# -------------------------------
# Main
# -------------------------------

def run_evaluation():

    print("--- Starting Topic Evaluation ---")
    # evaluate_topics()  # DISABLED: Unsupervised dynamic topic labels are no longer compatible with zero-shot evaluations
    print("Zero-shot topic evaluation is currently disabled after moving to dynamic batch topic models (BERTopic / Top2Vec).")

    # --- Sentiment evaluation DISABLED ---
    # print("\n--- Starting Sentiment Evaluation ---")
    # evaluate_sentiment()

    print("\n--- Generating Evaluation Statistics ---")
    generate_statistics()

    print("\nAll evaluations complete!")


if __name__ == "__main__":
    run_evaluation()