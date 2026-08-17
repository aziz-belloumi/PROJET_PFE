# Testing_Models — NLP Benchmark Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-zone)

A production-grade multilingual benchmark and evaluation suite for **Sentiment Analysis**, **Named Entity Recognition (NER)**, and **Zero-Shot Topic Classification**. 

The suite systematically evaluates **31 transformer models** and **Qwen2.5:7b (Ollama)** on a curated, manually verified dataset of **600 news articles** spanning **Modern Standard Arabic (MSA)**, **Dialectal Arabic**, **English**, and **French**.

---

## Table of Contents

- [Project Architecture](#project-architecture)
- [Dataset Composition](#dataset-composition)
- [Model Catalog (31 Benchmarked Models)](#model-catalog-31-benchmarked-models)
- [Quickstart & Execution Guide](#quickstart--execution-guide)
  - [1. Master Benchmark Runner (`run_all.py`)](#1-master-benchmark-runner-run_allpy)
  - [2. Individual Task Runners](#2-individual-task-runners)
- [Evaluation Metrics & Methodology](#evaluation-metrics--methodology)
- [Hardware & Resource Benchmarking](#hardware--resource-benchmarking)
- [Annotation & Quality Assurance Tools](#annotation--quality-assurance-tools)
- [Translation Pipeline](#translation-pipeline)
- [Installation & Setup](#installation--setup)

---

## Project Architecture

```
Testing_Models/
├── run_all.py                          # Master entrypoint: runs inference, benchmarks & generates reports
├── requirements.txt                    # Pinned Python package dependencies
├── .gitignore                          # Git exclusion rules
│
├── data/
│   └── manual_eval_global.csv          # 600 annotated articles with all ground-truth & model predictions
│
├── models/
│   ├── ner/
│   │   └── ner_extraction.py           # Unified TransformersNER & GLiNER extraction engines
│   │
│   ├── transformers/                   # Modular HuggingFace Transformers pipeline
│   │   ├── bert_runner.py              # Runner for BERT Sentiment & NER models
│   │   ├── nli_runner.py               # Runner for Zero-Shot NLI Topic classification models
│   │   ├── evaluation.py               # Accuracy, F1 Macro, Precision, Recall & Report generator
│   │   └── shared.py                   # Shared constants, label maps, chunking & resource tracker
│   │
│   └── qwen/                           # Qwen2.5 / Ollama LLM integration
│       ├── sentiment_analysis.py       # LLMSentiment extraction class
│       ├── topic_generation.py         # LLMTopic classification class
│       ├── run_qwen.py                 # Offline batch annotator for Qwen2.5:7b
│       └── run_qwen_pass.py            # Database pass updater (requires external DB)
│
├── benchmark/
│   ├── master_benchmark.py             # Standalone CPU/GPU resource benchmark runner
│   ├── resource_usage_report.csv       # Latency, throughput, and memory records per model
│   └── score_remarques.md              # Metric interpretation and behavioral notes
│
├── annotation/
│   ├── cli_annotator.py                # Interactive terminal manual annotator with RTL Arabic support
│   ├── check_inconsistencies.py        # Consistency validator for annotations and ground-truth
│   └── sample_extraction.py            # Stratified sampler from production SQL database (*)
│
├── translation/
│   ├── translation.py                  # MarianMT (Helsinki-NLP/opus-mt-en-fr) translation engine
│   └── translate_manual_eval.py        # Batch English-to-French dataset translator
│
├── utils/
│   └── chunking.py                     # Token-aware overlapping text chunking utility
│
└── reports/
    └── report.txt                      # Comprehensive generated evaluation report
```

> `(*)` Requires external database environment credentials (`src.db_config`).

---

## Dataset Composition

The ground-truth dataset ([data/manual_eval_global.csv](data/manual_eval_global.csv)) contains **600 balanced news articles**:

| Language Variant | Rows | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Arabic (MSA)** | 150 | 25.0% | Modern Standard Arabic news articles (indices 0–149) |
| **Arabic (Dialectal)** | 150 | 25.0% | Dialectal Arabic news articles (indices 150–299) |
| **English** | 150 | 25.0% | English news articles (indices 300–449) |
| **French** | 150 | 25.0% | French news articles (indices 450–599) |
| **Total** | **600** | **100.0%** | Ground-truth annotated across 18 topics & 3 sentiment classes |

### Topic Classes (18 Categories)
`Politics`, `Economy`, `Security`, `Energy`, `Conflict`, `Elections`, `Justice`, `Health`, `Weather`, `Sports`, `Culture`, `Education`, `Technology`, `Environment`, `Diplomacy`, `Religion`, `Migration`, `General`.

---

## Model Catalog (31 Benchmarked Models)

### 1. Sentiment Analysis (11 Models)
- **Arabic (4)**:
  - `CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment`
  - `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`
  - `CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment`
  - `PRAli22/AraBert-Arabic-Sentiment-Analysis`
- **English (3)**:
  - `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - `j-hartmann/sentiment-roberta-large-english-3-classes`
  - `finiteautomata/bertweet-base-sentiment-analysis`
- **French (2)**:
  - `cmarkea/distilcamembert-base-sentiment`
  - `nlptown/bert-base-multilingual-uncased-sentiment`
- **Multilingual (2)**:
  - `cardiffnlp/twitter-xlm-roberta-base-sentiment`
  - `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

### 2. Named Entity Recognition (12 Models)
- **Arabic (4)**:
  - `hatmimoha/arabic-ner`
  - `CAMeL-Lab/bert-base-arabic-camelbert-msa-ner`
  - `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner`
  - `MostafaAhmed98/AraBert-Arabic-NER-CoNLLpp`
- **English (3)**:
  - `dslim/bert-base-NER`
  - `dslim/bert-large-NER`
  - `Jean-Baptiste/roberta-large-ner-english`
- **French (2)**:
  - `Jean-Baptiste/camembert-ner`
  - `cmarkea/distilcamembert-base-ner`
- **Multilingual (3)**:
  - `urchade/gliner_multi-v2.1`
  - `Davlan/bert-base-multilingual-cased-ner-hrl`
  - `Babelscape/wikineural-multilingual-ner`

### 3. Zero-Shot NLI Topic Classification (8 Models)
- **Arabic (2)**:
  - `AhmedZaky1/arabic-bert-nli-matryoshka`
  - `HassanB4/s04-arbert-nli`
- **English (3)**:
  - `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
  - `roberta-large-mnli`
  - `facebook/bart-large-mnli`
- **French (1)**:
  - `cmarkea/distilcamembert-base-nli`
- **Multilingual (2)**:
  - `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
  - `joeddav/xlm-roberta-large-xnli`

---

## Quickstart & Execution Guide

### 1. Master Benchmark Runner (`run_all.py`)

The master runner orchestrates the entire benchmark, executes all model pipelines, tracks hardware metrics, verifies missing values, and produces [reports/report.txt](reports/report.txt).

```bash
# Run complete benchmark (BERT sentiment + NER + NLI topics) and generate report
python run_all.py

# Optional CLI flags:
python run_all.py --bert-only       # Run only BERT sentiment & NER models
python run_all.py --nli-only        # Run only Zero-Shot NLI topic models
python run_all.py --report-only     # Skip inference and re-generate report from existing CSV
python run_all.py --force-rerun     # Force re-execution of all models
python run_all.py --csv <path>      # Custom input CSV path
python run_all.py --report-out <p>  # Custom report output path
```

### 2. Individual Task Runners

You can also run individual tasks independently:

```bash
# Run only BERT Sentiment & NER models
python models/transformers/bert_runner.py

# Run only Zero-Shot NLI Topic models
python models/transformers/nli_runner.py

# Run standalone CPU/GPU resource benchmark
python benchmark/master_benchmark.py

# Run Qwen2.5:7b offline predictions (requires local Ollama server)
python models/qwen/run_qwen.py
```

---

## Evaluation Metrics & Methodology

The benchmark reports comprehensive multi-class metrics:

1. **Accuracy**: Total percentage of correct predictions across labelled rows.
2. **F1 Macro**: Unweighted mean of F1 scores across all classes. Crucial for handling class imbalance (e.g. rare topics or skewed sentiment classes).
3. **Precision (Macro)**: Measures false alarm rates per class.
4. **Recall (Macro)**: Measures model coverage of true positive instances per class.
5. **MSA vs. Dialectal Breakdown**: All Arabic models are evaluated independently on the MSA subset (150 rows) and Dialectal subset (150 rows) to expose dialect generalization gaps.
6. **NER Metrics**: Percentage of documents with identified entities, average entities extracted per document, and empty prediction rates.

---

## Hardware & Resource Benchmarking

During every run, the `BenchmarkTracker` monitors:
- **Load Time (`sec`)**: Time required to instantiate the model and move weights to VRAM/RAM.
- **Inference Latency (`ms/doc`)**: Average processing time per document including chunk aggregation.
- **Throughput (`docs/sec`)**: Number of documents processed per second.
- **Peak CPU Delta (`MB`)**: Maximum resident set size memory overhead.
- **Peak GPU Delta (`MB`)**: Peak CUDA allocated memory.

All benchmark results are automatically saved to [benchmark/resource_usage_report.csv](benchmark/resource_usage_report.csv).

---

## Annotation & Quality Assurance Tools

### Validate Dataset Consistency
Checks that predicted flags, expected themes, and sentiment labels in the CSV are logically aligned:
```bash
python annotation/check_inconsistencies.py
```

### Interactive CLI Annotator
Provides a terminal UI with bi-directional Arabic reshaper support for manual validation:
```bash
python annotation/cli_annotator.py
```

---

## Translation Pipeline

To generate the French evaluation dataset from English articles, a MarianMT translation pipeline is provided:
```bash
# Run translation test
python translation/translation.py

# Batch translate English dataset to French
python translation/translate_manual_eval.py
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone <repository_url>
cd Testing_Models
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Setup Ollama for Qwen2.5:7b
If running Qwen evaluations:
1. Install [Ollama](https://ollama.com/)
2. Pull the model:
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
