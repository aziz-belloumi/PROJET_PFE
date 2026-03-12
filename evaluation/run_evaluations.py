import pandas as pd
import requests
import re
from tqdm import tqdm
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SENTIMENT_CSV = PROJECT_ROOT / "scripts" / "manual_eval_sample_sentiment.csv"
TOPIC_CSV = PROJECT_ROOT / "scripts" / "manual_eval_sample_topic.csv"

# Topics allowed
TOPICS = [
    "Politics",
    "Economy",
    "Security",
    "Energy",
    "Conflict",
    "Elections",
    "Justice",
    "Health",
    "Weather",
]
TOPIC_LIST_TEXT = "\n".join(TOPICS)

# Sentiments allowed
SENTIMENTS = [
    "POSITIVE",
    "NEGATIVE",
    "NEUTRAL",
]
SENTIMENT_LIST_TEXT = "\n".join(SENTIMENTS)


# -------------------------------
# Prompt Builders
# -------------------------------

def build_topic_prompt(language: str, article: str, predicted_topic: str) -> str:
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

def build_sentiment_prompt(language: str, article: str, predicted_sentiment: str) -> str:
    return f"""
You are evaluating sentiment classification for news articles.

The article may be written in Arabic, English, or French.

Article language: {language}

Article:
{article}

Predicted sentiment:
{predicted_sentiment}

Possible sentiments:
{SENTIMENT_LIST_TEXT}

Tasks:

1. Determine the TRUE sentiment of the article from the list above.
2. Check if the predicted sentiment matches the true sentiment.

Output STRICTLY in this format:

sentiment_verdict: TRUE or FALSE
true_prediction: one of the sentiment names
"""


# -------------------------------
# LLM Call
# -------------------------------

def call_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 32000,
            "temperature": 0 
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error: {response.text}")

    return response.json()["response"]


# -------------------------------
# Parse LLM Output
# -------------------------------

def parse_topic_output(text: str):
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

def parse_sentiment_output(text: str):
    verdict_match = re.search(r"sentiment_verdict:\s*(TRUE|FALSE)", text, re.IGNORECASE)
    sentiment_match = re.search(
        r"true_prediction:\s*(POSITIVE|NEGATIVE|NEUTRAL)",
        text,
        re.IGNORECASE,
    )

    verdict = verdict_match.group(1).upper() if verdict_match else None
    sentiment = sentiment_match.group(1).upper() if sentiment_match else None

    return verdict, sentiment


# -------------------------------
# Evaluation Pipelines
# -------------------------------

def evaluate_topics():
    if not TOPIC_CSV.exists():
        print(f"Topic CSV not found: {TOPIC_CSV}")
        return

    print(f"Loading topic dataset from {TOPIC_CSV}...")
    df = pd.read_csv(TOPIC_CSV)

    if "topic_verdict" not in df.columns:
        df["topic_verdict"] = pd.Series("", index=df.index, dtype=object)
    else:
        df["topic_verdict"] = df["topic_verdict"].astype(object)

    if "true_prediction" not in df.columns:
        df["true_prediction"] = pd.Series("", index=df.index, dtype=object)
    else:
        df["true_prediction"] = df["true_prediction"].astype(object)

    print(f"Total topic articles: {len(df)}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Skip already evaluated rows if resuming
        verdict_val = row.get("topic_verdict", "")
        pred_val = row.get("true_prediction", "")
        if isinstance(verdict_val, str) and verdict_val.strip() and isinstance(pred_val, str) and pred_val.strip():
            continue
            
        article = str(row["body"])
        language = str(row["language"])
        predicted_topic = str(row["topic_label"])

        prompt = build_topic_prompt(language, article, predicted_topic)

        try:
            llm_response = call_llm(prompt)
            verdict, true_topic = parse_topic_output(llm_response)

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            verdict, true_topic = None, None

        df.at[idx, "topic_verdict"] = verdict
        df.at[idx, "true_prediction"] = true_topic

    print(f"Saving results to {TOPIC_CSV}...")
    df.to_csv(TOPIC_CSV, index=False)
    print("Topic evaluation finished.")


def evaluate_sentiment():
    if not SENTIMENT_CSV.exists():
        print(f"Sentiment CSV not found: {SENTIMENT_CSV}")
        return

    print(f"Loading sentiment dataset from {SENTIMENT_CSV}...")
    df = pd.read_csv(SENTIMENT_CSV)

    if "sentiment_verdict" not in df.columns:
        df["sentiment_verdict"] = pd.Series("", index=df.index, dtype=object)
    else:
        df["sentiment_verdict"] = df["sentiment_verdict"].astype(object)

    if "true_prediction" not in df.columns:
        df["true_prediction"] = pd.Series("", index=df.index, dtype=object)
    else:
        df["true_prediction"] = df["true_prediction"].astype(object)

    print(f"Total sentiment articles: {len(df)}")

    sentiment_cols = [c for c in df.columns if c.startswith("sentiment_label")]
    primary_sentiment_col = sentiment_cols[0] if sentiment_cols else None

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Skip already evaluated rows if resuming
        verdict_val = row.get("sentiment_verdict", "")
        pred_val = row.get("true_prediction", "")
        if isinstance(verdict_val, str) and verdict_val.strip() and isinstance(pred_val, str) and pred_val.strip():
            continue
            
        article = str(row["body"])
        language = str(row["language"])
        
        predicted_sentiment = ""
        if primary_sentiment_col:
            predicted_sentiment = str(row[primary_sentiment_col])

        prompt = build_sentiment_prompt(language, article, predicted_sentiment)

        try:
            llm_response = call_llm(prompt)
            verdict, true_sentiment = parse_sentiment_output(llm_response)

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            verdict, true_sentiment = None, None

        df.at[idx, "sentiment_verdict"] = verdict
        df.at[idx, "true_prediction"] = true_sentiment

    print(f"Saving results to {SENTIMENT_CSV}...")
    df.to_csv(SENTIMENT_CSV, index=False)
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
            lang_acc = (sub[verdict_col].astype(str).str.upper() == "TRUE").sum() / len(sub) * 100
            rows.append({"Task": task, "Category": f"Language: {lang}", "Metric": "Accuracy (%)", "Value": f"{lang_acc:.1f}", "Note": f"{len(sub)} articles"})

    # Per predicted label
    if label_col and label_col in evaluated.columns:
        for label in sorted(evaluated[label_col].dropna().unique()):
            sub = evaluated[evaluated[label_col] == label]
            lbl_acc = (sub[verdict_col].astype(str).str.upper() == "TRUE").sum() / len(sub) * 100
            rows.append({"Task": task, "Category": f"Label: {label}", "Metric": "Accuracy (%)", "Value": f"{lbl_acc:.1f}", "Note": f"{len(sub)} articles"})

    return rows, accuracy


def generate_statistics():
    print("\n--- Generating Statistics ---")
    rows = []
    topic_acc = None
    sentiment_acc = None

    # ---- Topic ----
    if TOPIC_CSV.exists():
        df_t = pd.read_csv(TOPIC_CSV)
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
        df_s = pd.read_csv(SENTIMENT_CSV)
        # Find primary sentiment label column
        sent_label_cols = [c for c in df_s.columns if c.startswith("sentiment_label")]
        primary_label_col = sent_label_cols[0] if sent_label_cols else None
        if "sentiment_verdict" in df_s.columns:
            result = _accuracy_block(df_s, "sentiment_verdict", primary_label_col, "Sentiment")
            if isinstance(result, tuple):
                block_rows, sentiment_acc = result
            else:
                block_rows = result
            rows.extend(block_rows)
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
            if r["Task"] == "Sentiment" and r["Category"].startswith("Label:") and r["Value"] not in ("",):
                try:
                    if float(r["Value"]) < 60:
                        weak_sent.append(r["Category"].replace("Label: ", ""))
                except ValueError:
                    pass

        if weak_topic:
            rows.append({"Task": "Recommendation", "Category": "Weak Topic Labels (<60%)", "Metric": "Labels", "Value": ", ".join(weak_topic), "Note": "Consider improving training data for these topics"})
        if weak_sent:
            rows.append({"Task": "Recommendation", "Category": "Weak Sentiment Labels (<60%)", "Metric": "Labels", "Value": ", ".join(weak_sent), "Note": "Consider improving training data for these sentiments"})
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
    print("\n--- Starting Sentiment Evaluation ---")
    evaluate_sentiment()
    print("\n--- Generating Evaluation Statistics ---")
    generate_statistics()
    print("\nAll evaluations complete!")


if __name__ == "__main__":
    run_evaluation()
