import pandas as pd
import numpy as np
from collections import Counter
import re

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("fine_tune_data/global_data_merged.csv")

SEP = "=" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── 1. Shape & Schema ─────────────────────────────────────────────────────────
section("1. SHAPE & SCHEMA")
print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"\nColumn names & dtypes:")
print(df.dtypes.to_string())
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# ── 2. Missing Values ─────────────────────────────────────────────────────────
section("2. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
print(missing_df.to_string())
print(f"\nRows with at least one missing value: {df.isnull().any(axis=1).sum():,}")
print(f"Completely empty rows               : {df.isnull().all(axis=1).sum():,}")

# ── 3. Duplicate Analysis ─────────────────────────────────────────────────────
section("3. DUPLICATE ANALYSIS")
print(f"Duplicate rows (all columns)     : {df.duplicated().sum():,}")
print(f"Duplicate IDs                    : {df['id'].duplicated().sum():,}")
print(f"Duplicate texts (exact)          : {df['text'].duplicated().sum():,}")
print(f"Duplicate (text+sentiment+topic) : {df.duplicated(subset=['text','sentiment','topic']).sum():,}")

# ── 4. ID Column ──────────────────────────────────────────────────────────────
section("4. ID COLUMN")
print(f"Min ID      : {df['id'].min():,}")
print(f"Max ID      : {df['id'].max():,}")
print(f"ID range    : {df['id'].max() - df['id'].min():,}")
print(f"Unique IDs  : {df['id'].nunique():,}")
gaps = df['id'].sort_values().diff().dropna()
print(f"Max gap between consecutive IDs: {gaps.max():.0f}")
print(f"Mean gap between consecutive IDs: {gaps.mean():.2f}")

# ── 5. Language Distribution ──────────────────────────────────────────────────
section("5. LANGUAGE DISTRIBUTION")
lang_counts = df['language'].value_counts()
lang_pct    = df['language'].value_counts(normalize=True).mul(100).round(2)
print(pd.DataFrame({"count": lang_counts, "%": lang_pct}).to_string())

# ── 6. Sentiment Distribution ─────────────────────────────────────────────────
section("6. SENTIMENT DISTRIBUTION")
sent_counts = df['sentiment'].value_counts()
sent_pct    = df['sentiment'].value_counts(normalize=True).mul(100).round(2)
print(pd.DataFrame({"count": sent_counts, "%": sent_pct}).to_string())
print(f"\nClass imbalance ratio (max/min): {sent_counts.max() / sent_counts.min():.2f}x")

# ── 7. Topic Distribution ─────────────────────────────────────────────────────
section("7. TOPIC DISTRIBUTION")
topic_counts = df['topic'].value_counts()
topic_pct    = df['topic'].value_counts(normalize=True).mul(100).round(2)
print(pd.DataFrame({"count": topic_counts, "%": topic_pct}).to_string())
print(f"\nTotal unique topics : {df['topic'].nunique()}")

# ── 8. Cross-tab: Sentiment × Language ───────────────────────────────────────
section("8. CROSS-TAB: SENTIMENT × LANGUAGE")
ct = pd.crosstab(df['sentiment'], df['language'])
print(ct.to_string())

# ── 9. Cross-tab: Topic × Sentiment ──────────────────────────────────────────
section("9. CROSS-TAB: TOPIC × SENTIMENT")
ct2 = pd.crosstab(df['topic'], df['sentiment'])
ct2['total'] = ct2.sum(axis=1)
print(ct2.sort_values('total', ascending=False).to_string())

# ── 10. Cross-tab: Topic × Language ──────────────────────────────────────────
section("10. CROSS-TAB: TOPIC × LANGUAGE")
ct3 = pd.crosstab(df['topic'], df['language'])
print(ct3.to_string())

# ── 11. Text Length Stats ─────────────────────────────────────────────────────
section("11. TEXT LENGTH STATS (characters)")
df['text_len_chars'] = df['text'].str.len()
print(df['text_len_chars'].describe().round(2).to_string())
print(f"\nMedian  : {df['text_len_chars'].median():.1f}")
print(f"Skewness: {df['text_len_chars'].skew():.4f}")
print(f"Kurtosis: {df['text_len_chars'].kurt():.4f}")

# ── 12. Word Count Stats ──────────────────────────────────────────────────────
section("12. WORD COUNT STATS")
df['word_count'] = df['text'].str.split().str.len()
print(df['word_count'].describe().round(2).to_string())
print(f"\nMedian  : {df['word_count'].median():.1f}")
print(f"Skewness: {df['word_count'].skew():.4f}")
print(f"Kurtosis: {df['word_count'].kurt():.4f}")

# ── 13. Word Count by Sentiment ───────────────────────────────────────────────
section("13. WORD COUNT BY SENTIMENT")
print(df.groupby('sentiment')['word_count'].agg(['mean','median','std','min','max']).round(2).to_string())

# ── 14. Word Count by Language ────────────────────────────────────────────────
section("14. WORD COUNT BY LANGUAGE")
print(df.groupby('language')['word_count'].agg(['mean','median','std','min','max']).round(2).to_string())

# ── 15. Word Count by Topic ───────────────────────────────────────────────────
section("15. WORD COUNT BY TOPIC")
print(df.groupby('topic')['word_count'].agg(['mean','median','std','min','max']).round(2).sort_values('mean', ascending=False).to_string())

# ── 16. Char Count by Sentiment ───────────────────────────────────────────────
section("16. CHAR COUNT BY SENTIMENT")
print(df.groupby('sentiment')['text_len_chars'].agg(['mean','median','std','min','max']).round(2).to_string())

# ── 17. Percentile Breakdown (word count) ─────────────────────────────────────
section("17. WORD COUNT PERCENTILES")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
wc_p = df['word_count'].quantile([p/100 for p in percentiles])
wc_p.index = [f"p{p}" for p in percentiles]
print(wc_p.round(1).to_string())

# ── 18. Empty / Very Short Texts ─────────────────────────────────────────────
section("18. SHORT / EMPTY TEXT FLAGS")
print(f"Empty texts (NaN)     : {df['text'].isnull().sum():,}")
print(f"Blank strings         : {(df['text'].str.strip() == '').sum():,}")
print(f"Texts ≤ 10 chars      : {(df['text_len_chars'] <= 10).sum():,}")
print(f"Texts ≤ 5 words       : {(df['word_count'] <= 5).sum():,}")
print(f"Texts > 200 chars     : {(df['text_len_chars'] > 200).sum():,}")
print(f"Texts > 50 words      : {(df['word_count'] > 50).sum():,}")

# ── 19. Special Character Analysis ───────────────────────────────────────────
section("19. SPECIAL CHARACTER ANALYSIS")
df['has_arabic']  = df['text'].str.contains(r'[؀-ۿ]', regex=True, na=False)
df['has_url']     = df['text'].str.contains(r'https?://', regex=True, na=False)
df['has_mention'] = df['text'].str.contains(r'@\w+', regex=True, na=False)
df['has_hashtag'] = df['text'].str.contains(r'#\w+', regex=True, na=False)
df['has_number']  = df['text'].str.contains(r'\d', regex=True, na=False)
df['has_punct_arabic'] = df['text'].str.contains('،', na=False)  # Arabic comma

flags = ['has_arabic','has_url','has_mention','has_hashtag','has_number','has_punct_arabic']
for f in flags:
    n = df[f].sum()
    print(f"{f:<25}: {n:,}  ({n/len(df)*100:.2f}%)")

# ── 20. Unique Values per Column ──────────────────────────────────────────────
section("20. UNIQUE VALUE COUNTS")
for col in df.columns:
    if col not in ['text_len_chars','word_count','has_arabic','has_url',
                   'has_mention','has_hashtag','has_number','has_punct_arabic']:
        print(f"{col:<15}: {df[col].nunique():,} unique values")

# ── 21. Most Common Words (English) ──────────────────────────────────────────
section("21. TOP 30 WORDS (English texts only, stopwords excluded)")
STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','has','have','had','it','its','this',
    'that','they','he','she','we','i','you','by','as','from','not','no','s',
    'after','before','into','over','than','then','their','there','about',
    'up','out','will','would','could','should','also','which','when','who'
}
en_texts = df[df['language'] == 'en']['text'].dropna()
all_words = []
for t in en_texts:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
    all_words.extend([w for w in words if w not in STOPWORDS])
top_words = Counter(all_words).most_common(30)
print(f"{'Word':<20} {'Count':>8}")
print("-" * 30)
for word, count in top_words:
    print(f"{word:<20} {count:>8,}")

# ── 22. Label Consistency Checks ─────────────────────────────────────────────
section("22. LABEL CONSISTENCY CHECKS")
print("Known sentiment labels :", sorted(df['sentiment'].unique()))
print("Known language labels  :", sorted(df['language'].unique()))
print("Known topic labels     :", sorted(df['topic'].unique()))
unexpected_sent = df[~df['sentiment'].isin(['POSITIVE','NEGATIVE','NEUTRAL'])]
print(f"\nRows with unexpected sentiment: {len(unexpected_sent):,}")
if len(unexpected_sent):
    print(unexpected_sent['sentiment'].value_counts().to_string())

# ── 23. Sample Rows ───────────────────────────────────────────────────────────
section("23. SAMPLE ROWS (5 per sentiment)")
for sent in df['sentiment'].unique():
    print(f"\n--- {sent} ---")
    sample = df[df['sentiment'] == sent][['id','language','topic','word_count','text']].head(5)
    for _, row in sample.iterrows():
        print(f"  [{row['id']}] ({row['language']}, {row['topic']}, {row['word_count']}w) {str(row['text'])[:100]}")

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"Total rows           : {len(df):,}")
print(f"Languages            : {df['language'].nunique()} — {list(df['language'].unique())}")
print(f"Sentiments           : {df['sentiment'].nunique()} — {list(df['sentiment'].unique())}")
print(f"Topics               : {df['topic'].nunique()}")
print(f"Avg words/text       : {df['word_count'].mean():.1f}")
print(f"Avg chars/text       : {df['text_len_chars'].mean():.1f}")
print(f"Missing values       : {df.isnull().sum().sum():,}")
print(f"Exact duplicates     : {df.duplicated().sum():,}")
print(f"Texts with Arabic    : {df['has_arabic'].sum():,}")
