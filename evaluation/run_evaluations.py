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

# Topics allowed
TOPICS = [
    "Politics", "Economy", "Security", "Energy", "Conflict",
    "Elections", "Justice", "Health", "Weather", "Sports"
]
TOPIC_LIST_TEXT = "\n".join(TOPICS)

# Sentiments allowed
SENTIMENTS = ["POSITIVE","NEGATIVE","NEUTRAL"]
SENTIMENT_LIST_TEXT = "\n".join(SENTIMENTS)


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

def build_topic_prompt(language, article, predicted_topic):
    return f"""
You are evaluating topic classification for news articles.

The article may be written in Arabic, English, or French.

Article language: {language}

Article:
{article}

Predicted topic:
{predicted_topic}

Possible topics:
{TOPIC_LIST_TEXT}

Tasks:
1. Determine the TRUE topic of the article from the list above.
2. Check if the predicted topic matches the true topic.

Output STRICTLY in this format:

topic_verdict: TRUE or FALSE
true_prediction: one of the topic names
"""

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

def parse_topic_output(text):

    verdict_match = re.search(r"topic_verdict:\s*(TRUE|FALSE)", text, re.IGNORECASE)

    topics_regex = "|".join(TOPICS)

    topic_match = re.search(
        rf"true_prediction:\s*({topics_regex})",
        text,
        re.IGNORECASE,
    )

    verdict = verdict_match.group(1).upper() if verdict_match else None
    topic = topic_match.group(1).title() if topic_match else None

    return verdict, topic


def parse_sentiment_output(text):

    sentiment_match = re.search(
        r"true_prediction:\s*(POSITIVE|NEGATIVE|NEUTRAL)",
        text,
        re.IGNORECASE,
    )

    sentiment = sentiment_match.group(1).upper() if sentiment_match else None

    return sentiment


# -------------------------------
# Topic Evaluation
# -------------------------------

def evaluate_topics():

    if not TOPIC_CSV.exists():
        print(f"Topic CSV not found: {TOPIC_CSV}")
        return

    df = pd.read_csv(TOPIC_CSV).fillna("")

    if "topic_verdict" not in df.columns:
        df["topic_verdict"] = ""

    if "true_prediction" not in df.columns:
        df["true_prediction"] = ""

    print(f"Total topic articles: {len(df)}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        if str(row["topic_verdict"]).strip() and str(row["true_prediction"]).strip():
            continue

        article = str(row["body"])
        language = str(row["language"])
        predicted_topic = str(row["topic_label"])

        chunks = chunk_article(article)

        chunk_topics = []
        chunk_verdicts = []

        for chunk in chunks:

            prompt = build_topic_prompt(language, chunk, predicted_topic)

            try:

                llm_response = call_llm(prompt)
                verdict, topic = parse_topic_output(llm_response)

                if topic:
                    chunk_topics.append(topic)

                if verdict:
                    chunk_verdicts.append(verdict)

            except Exception as e:
                print(f"Chunk error row {idx}: {e}")

        final_topic = Counter(chunk_topics).most_common(1)[0][0] if chunk_topics else None
        final_verdict = Counter(chunk_verdicts).most_common(1)[0][0] if chunk_verdicts else None

        df.at[idx,"topic_verdict"] = final_verdict
        df.at[idx,"true_prediction"] = final_topic

    df.to_csv(TOPIC_CSV,index=False)

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

    sentiment_cols = [c for c in df.columns if c.startswith("sentiment_label")]
    primary_sentiment_col = sentiment_cols[0] if sentiment_cols else None

    print(f"Total sentiment articles: {len(df)}")

    for idx,row in tqdm(df.iterrows(),total=len(df)):

        if str(row["true_prediction"]).strip():
            continue

        article = str(row["body"])
        language = str(row["language"])

        predicted_sentiment = ""
        if primary_sentiment_col:
            predicted_sentiment = str(row[primary_sentiment_col])

        chunks = chunk_article(article)

        sentiments = []

        for chunk in chunks:

            prompt = build_sentiment_prompt(language,chunk,predicted_sentiment)

            try:

                llm_response = call_llm(prompt)
                sentiment = parse_sentiment_output(llm_response)

                if sentiment:
                    sentiments.append(sentiment)

            except Exception as e:
                print(f"Chunk error row {idx}: {e}")

        final_sentiment = Counter(sentiments).most_common(1)[0][0] if sentiments else None

        df.at[idx,"true_prediction"] = final_sentiment

    df.to_csv(SENTIMENT_CSV,index=False)

    print("Sentiment evaluation finished.")


# -------------------------------
# Statistics Generator
# -------------------------------

def _accuracy_block(df: pd.DataFrame, verdict_col: str, label_col: str, task: str) -> list:
    """Compute overall + per-language + per-label accuracy rows."""
    rows = []
    evaluated = df[df[verdict_col].astype(str).str.upper().isin(["TRUE", "FALSE"])].copy()
    total = len(evaluated)
    if total == 0:
        rows.append({"Task": task, "Category": "Overall", "Metric": "Total Evaluated", "Value": 0, "Note": "No evaluated rows found"})
        return rows

    correct = (evaluated[verdict_col].astype(str).str.upper() == "TRUE").sum()
    accuracy = correct / total * 100

    rows.append({"Task": task, "Category": "Overall", "Metric": "Total Evaluated", "Value": total, "Note": ""})
    rows.append({"Task": task, "Category": "Overall", "Metric": "Correct", "Value": correct, "Note": ""})
    rows.append({"Task": task, "Category": "Overall", "Metric": "Accuracy (%)", "Value": f"{accuracy:.1f}", "Note": ""})

    # Per language
    if "language" in evaluated.columns:
        for lang in sorted(evaluated["language"].dropna().unique()):
            sub = evaluated[evaluated["language"] == lang]
            if len(sub) == 0: continue
            lang_acc = (sub[verdict_col].astype(str).str.upper() == "TRUE").sum() / len(sub) * 100
            rows.append({"Task": task, "Category": f"Language: {lang}", "Metric": "Accuracy (%)", "Value": f"{lang_acc:.1f}", "Note": f"{len(sub)} articles"})

    # Per predicted label overall
    if label_col and label_col in evaluated.columns:
        for label in sorted(evaluated[label_col].dropna().unique()):
            sub = evaluated[evaluated[label_col] == label]
            if len(sub) == 0: continue
            lbl_acc = (sub[verdict_col].astype(str).str.upper() == "TRUE").sum() / len(sub) * 100
            rows.append({"Task": task, "Category": f"Label: {label} (Overall)", "Metric": "Accuracy (%)", "Value": f"{lbl_acc:.1f}", "Note": f"{len(sub)} articles"})

            # Per label AND language
            if "language" in sub.columns:
                for lang in sorted(sub["language"].dropna().unique()):
                    sub_lang = sub[sub["language"] == lang]
                    if len(sub_lang) == 0: continue
                    lbl_lang_acc = (sub_lang[verdict_col].astype(str).str.upper() == "TRUE").sum() / len(sub_lang) * 100
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
        if "topic_verdict" in df_t.columns:
            result = _accuracy_block(df_t, "topic_verdict", "topic_label", "Topic")
            if isinstance(result, tuple):
                block_rows, topic_acc = result
            else:
                block_rows = result
            rows.extend(block_rows)
        else:
            rows.append({"Task": "Topic", "Category": "Overall", "Metric": "Status", "Value": "Not evaluated yet", "Note": ""})
    else:
        rows.append({"Task": "Topic", "Category": "Overall", "Metric": "Status", "Value": "CSV not found", "Note": ""})

    rows.append({"Task": "", "Category": "", "Metric": "", "Value": "", "Note": ""})  # blank separator

    # ---- Sentiment ----
    if SENTIMENT_CSV.exists():
        df_s = pd.read_csv(SENTIMENT_CSV).fillna("")
        # Find all sentiment label columns
        sent_label_cols = sorted([c for c in df_s.columns if c.startswith("sentiment_label")])
        primary_label_col = sent_label_cols[0] if sent_label_cols else None
        
        if "true_prediction" in df_s.columns:
            
            # Per-Model Accuracy
            rows.append({"Task": "Sentiment", "Category": "Per Model Accuracy", "Metric": "---", "Value": "---", "Note": ""})
            
            evaluated = df_s[df_s["true_prediction"].astype(str).str.strip() != ""].copy()
            
            if not evaluated.empty:
                # Mapping for label normalization
                label_map = {
                    "POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL",
                    "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL"
                }
                
                rows.append({"Task": "Sentiment", "Category": "Overall", "Metric": "Total Evaluated", "Value": len(evaluated), "Note": ""})
                
                for col in sent_label_cols:
                    model_id = col.replace("sentiment_label_", "")
                    
                    # Normalize both sides for comparison
                    y_pred = evaluated[col].astype(str).str.upper().map(lambda x: label_map.get(x, x))
                    y_true = evaluated["true_prediction"].astype(str).str.upper().map(lambda x: label_map.get(x, x))
                    
                    correct = (y_pred == y_true).sum()
                    acc = correct / len(evaluated) * 100
                    rows.append({
                        "Task": "Sentiment",
                        "Category": f"Model: {model_id}",
                        "Metric": "Accuracy (%)",
                        "Value": f"{acc:.1f}",
                        "Note": f"Based on {len(evaluated)} articles"
                    })
                    
                    # Store primary model accuracy for overall recommendation comparison
                    if col == primary_label_col:
                        sentiment_acc = acc
            else:
                rows.append({"Task": "Sentiment", "Category": "Overall", "Metric": "Status", "Value": "Not evaluated yet", "Note": ""})
        else:
            rows.append({"Task": "Sentiment", "Category": "Overall", "Metric": "Status", "Value": "Not evaluated yet", "Note": ""})
    else:
        rows.append({"Task": "Sentiment", "Category": "Overall", "Metric": "Status", "Value": "CSV not found", "Note": ""})

    rows.append({"Task": "", "Category": "", "Metric": "", "Value": "", "Note": ""})  # blank separator

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
            rows.append({"Task": "Recommendation", "Category": "Weak Topic Labels (<60%)", "Metric": "Labels", "Value": ", ".join(weak_topic), "Note": "Consider improving training data for these topics"})
        if weak_sent:
            rows.append({"Task": "Recommendation", "Category": "Weak Sentiment Models (<60%)", "Metric": "Models", "Value": ", ".join(weak_sent), "Note": "Consider improving these models"})
    elif topic_acc is not None:
        rows.append({"Task": "Recommendation", "Category": "Info", "Metric": "Note", "Value": "Only topic data available", "Note": ""})
    elif sentiment_acc is not None:
        rows.append({"Task": "Recommendation", "Category": "Info", "Metric": "Note", "Value": "Only sentiment data available", "Note": ""})
    else:
        rows.append({"Task": "Recommendation", "Category": "Info", "Metric": "Note", "Value": "No evaluated data found — run evaluations first", "Note": ""})

    out_path = SCRIPT_DIR / "evaluation.csv"
    pd.DataFrame(rows, columns=["Task", "Category", "Metric", "Value", "Note"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Statistics saved to: {out_path}")


# -------------------------------
# Main
# -------------------------------

def run_evaluation():

    print("--- Starting Topic Evaluation ---")
    evaluate_topics()

    # print("\n--- Starting Sentiment Evaluation ---")
    # evaluate_sentiment()

    print("\n--- Generating Evaluation Statistics ---")
    generate_statistics()

    print("\nAll evaluations complete!")


if __name__ == "__main__":
    run_evaluation()