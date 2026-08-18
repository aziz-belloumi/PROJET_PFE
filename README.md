# 📊 Multilingual Data Exploration & Arabic Human Libellisation

A clean, modular two-phase pipeline for building a large-scale, human-annotated multilingual news corpus (Arabic, English, French):

1. **Phase 1 — `data_exploration/`**: Parallel MySQL extraction, fastText language detection, preprocessing, and exploratory data analysis.
2. **Phase 2 — `data_libelisation/`**: Interactive CLI annotator for dual-task human labeling (topic + sentiment) with automatic plot generation on completion.
3. **Core Package — `src/`**: Shared utilities covering database connection, language detection, Arabic/Latin text preprocessing, and console encoding.

> **Status: ✅ COMPLETE** — All 1,484,845 articles have been fully extracted, preprocessed, and human-annotated (100% libelised).

---

## 📄 Dataset Schemas

| File | Path | Columns | Records | Description |
| :--- | :--- | :--- | ---: | :--- |
| **`global_data.csv`** | `data_exploration/global_data.csv` | `id`, `text`, `language` | 1,484,845 | Clean preprocessed articles (language: `da`, `ar`, `en`, `fr`). |
| **`global_data_libelised.csv`** | `data_libelisation/global_data_libelised.csv` | `id`, `text`, `language`, `sentiment`, `topic` | 1,484,845 | Fully human-annotated — **3 sentiment classes** × **18 topic categories** with Dialectal (`da`) and MSA (`ar`) distinction. |

### Taxonomy

**Language Tags:**

| Tag | Language Variety | Description |
| :---: | :--- | :--- |
| `da` | Dialectal Arabic | Egyptian, Gulf, Levantine, Maghrebi dialects |
| `ar` | Modern Standard Arabic | Standard / Classical Arabic news (MSA) |
| `en` | English | English articles |
| `fr` | French | French articles |

**Sentiment Classes (3):**

| Code | Label | Meaning |
| :---: | :--- | :--- |
| `1` | `POSITIVE` | Positive news tone |
| `2` | `NEGATIVE` | Negative news tone |
| `3` | `NEUTRAL` | Neutral / factual report |

**Topic Categories (18):**

| # | Code | # | Code | # | Code |
| :---: | :--- | :---: | :--- | :---: | :--- |
| 0 | Politics | 6 | Justice | 12 | Technology |
| 1 | Economy | 7 | Health | 13 | Environment |
| 2 | Security | 8 | Weather | 14 | Diplomacy |
| 3 | Energy | 9 | Sports | 15 | Religion |
| 4 | Conflict | 10 | Culture | 16 | Migration |
| 5 | Elections | 11 | Education | 17 | General |

---

## 🏗️ Architecture & Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       MySQL Database (nlp_mena)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼ fetch_and_preprocess_sql.py
                     [Parallel 100k batches / N CPU cores]
                     [fastText language detection per row]
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│              data_exploration/global_data.csv                       │
│                   1,484,845 rows  (id, text, language)              │
└────────────────┬────────────────────────────────┬───────────────────┘
                 │                                │
                 ▼                                ▼
  data_exploration.ipynb                  cli_annotator.py
  (Phase 1 — Visual EDA)                 (Phase 2 — Human Labeling)
                 │                                │
                 ▼                                ▼
  data_exploration/plots/        data_libelisation/global_data_libelised.csv
  ├── Language Pie                              (1,484,845 fully annotated rows)
  ├── Language Bar                                        │
  ├── Word Counts                                         ▼
  ├── Rejection Reasons                  data_libelisation/plots/
  └── Accepted vs Rejected               ├── Sentiment Bar (3 classes)
                                         ├── Sentiment Pie
                                         ├── 18 Topic Categories
                                         └── Topic × Sentiment Heatmap
```

---

## 📁 Repository Structure

```
Data_Exploring/
│
├── data_exploration/
│   ├── data_exploration.ipynb          # EDA notebook — language stats, word counts, quality metrics
│   ├── fetch_and_preprocess_sql.py     # Parallel MySQL extractor (100k batches, all CPU cores)
│   ├── global_data.csv                 # Clean dataset  (id, text, language: ar/en/fr)
│   ├── processing_stats.txt            # Extraction report — acceptance/rejection breakdown
│   └── plots/                          # Auto-generated EDA diagrams
│       ├── 01_language_distribution_pie.png
│       ├── 02_language_distribution_bar.png
│       ├── 03_word_counts_per_language.png
│       ├── 04_word_count_distribution.png
│       ├── 05_rejection_reasons_pie.png
│       └── 06_accepted_vs_rejected_donut.png
│
├── data_libelisation/
│   ├── cli_annotator.py                # Interactive console annotator (topic + sentiment)
│   ├── global_data_libelised.csv       # Fully annotated dataset (id, text, lang, sentiment, topic)
│   └── plots/                          # Auto-generated libellisation diagrams (on 100% completion)
│       ├── 01_sentiment_distribution_bar.png
│       ├── 02_sentiment_distribution_pie.png
│       ├── 03_topic_distribution_18_classes.png
│       └── 04_sentiment_by_topic_heatmap.png
│
├── src/                                # Shared core Python package
│   ├── config/                         # Environment & DB configuration
│   │   ├── base.py                     # .env loading, path helpers
│   │   ├── db.py                       # MySQL connection config
│   │   ├── preprocessing.py            # Preprocessing presets (lang-detect params)
│   │   └── __init__.py                 # Unified Config class
│   ├── preprocessing/                  # Text cleaning pipelines
│   │   ├── arabic.py                   # Arabic normalisation, diacritics, noise removal
│   │   ├── latin.py                    # Latin/French/English cleaning
│   │   ├── router.py                   # Single preprocess(text, lang, task) entry point
│   │   └── __init__.py
│   ├── text_utils.py                   # Windows UTF-8 console encoding helper
│   ├── db_config.py                    # DatabaseConnection (SQLAlchemy pool)
│   ├── language_detection.py           # fastText-based article classifier (ar/en/fr)
│   └── __init__.py                     # Package public exports
│
├── .env.local                          # MySQL credentials (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📈 Phase 1: Data Exploration (`data_exploration/`)

### 1a. Parallel Extraction — `fetch_and_preprocess_sql.py`

Extracts all raw articles from MySQL in **100,000-row parallel batches** across all available CPU cores, applies language detection via `fastText`, and streams valid articles into `global_data.csv`.

```bash
# Extract all articles (default: 100k chunks, all CPU cores)
python data_exploration/fetch_and_preprocess_sql.py

# Resume from a specific row offset
python data_exploration/fetch_and_preprocess_sql.py --offset 500000
```

**Output files:**
- `data_exploration/global_data.csv` — accepted articles (`ar`, `en`, `fr`)
- `data_exploration/processing_stats.txt` — acceptance rate, rejection reasons, confidence stats

### 1b. Exploratory Analysis — `data_exploration.ipynb`

Open in Jupyter or VS Code. Analyzes `global_data.csv` and auto-exports diagrams to `data_exploration/plots/`:

- Language distribution (Pie & Bar)
- Word count distributions per language (p50, p95, p99 percentiles)
- Rejection reasons breakdown
- Accepted vs Rejected article counts

---

## 🏷️ Phase 2: Human Annotation (`data_libelisation/`)

Terminal-based annotation tool with Arabic RTL rendering support.

**Features:**
- **Arabic BiDi Rendering**: `arabic_reshaper` + `python-bidi` for correct Right-to-Left cursive display in Windows terminal.
- **Arabic Variety Selection**: Prompts `[1] da  [2] ar` for Arabic articles at the first step to distinguish Dialectal Arabic (`da`) from Modern Standard Arabic (`ar`).
- **Multi-Task Labeling**: Language Variety (`[1] da / [2] ar`), Topic `[0-17]`, and Sentiment `[1/2/3]` — sequential prompts, auto-skips completed tasks.
- **Smart Resume**: Scans `global_data_libelised.csv` on startup, isolates only unlabeled articles, and starts directly there.
- **Zero-Lag Saves**: Labels and language tags are stored in memory instantly, flushed to disk only when the session ends or the queue is fully processed.
- **Automatic Plots**: On 100% completion, generates 4 summary charts to `data_libelisation/plots/`.

```bash
# Run annotator (auto-resumes from first unlabeled article)
python data_libelisation/cli_annotator.py
```

**In-Session Commands:**

| Input | Action |
| :--- | :--- |
| `1` / `2` | Select Arabic Variety (`1`: `da` Dialectal, `2`: `ar` MSA) when prompted |
| `0`–`17` | Assign topic category |
| `1` / `2` / `3` | Assign POSITIVE / NEGATIVE / NEUTRAL sentiment |
| `s` or `n` | Skip current article |
| `u` or `b` | Undo (go back one article) |
| `q` | Quit and save all progress |

---

## ⚙️ Setup & Configuration

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure MySQL Connection

Create `.env.local` in the project root:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=nlp_mena
RAW_TABLE=article
```

### 3. (Optional) Install Arabic Rendering Libraries

Required only for correct Arabic display in `cli_annotator.py`:

```bash
pip install arabic-reshaper python-bidi
```

---

## 📦 Dependencies

| Library | Purpose |
| :--- | :--- |
| `SQLAlchemy`, `PyMySQL` | MySQL connection pool & streaming |
| `fast-langdetect` | fastText-based language detection (lid.176.bin) |
| `arabic-reshaper`, `python-bidi` | Arabic BiDi terminal rendering |
| `pandas`, `numpy` | Data processing and statistics |
| `matplotlib`, `seaborn` | Chart generation for EDA and libellisation plots |
| `python-dotenv` | `.env.local` configuration loading |

---

## ✅ Project Completion Status

| Phase | Task | Status |
| :--- | :--- | :---: |
| Phase 1 | MySQL parallel extraction (1.5M articles) | ✅ Done |
| Phase 1 | fastText language detection & preprocessing | ✅ Done |
| Phase 1 | EDA notebook & plot generation | ✅ Done |
| Phase 2 | CLI annotator (topic labeling — 18 categories) | ✅ Done |
| Phase 2 | CLI annotator (sentiment labeling — 3 classes) | ✅ Done |
| Phase 2 | Final libellisation plots generation | ✅ Done |
| **Overall** | **1,484,845 / 1,484,845 articles fully annotated** | **✅ 100%** |
