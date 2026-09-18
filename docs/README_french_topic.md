# French Topic Classification — Fine-Tuning Documentation

This document provides a comprehensive technical reference for [`fine_tune_french_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/french/fine_tune_french_topic.py). It covers the CamemBERT-large custom architecture, automated MarianMT offline translation pipeline, class-specific focal loss gammas, minority class oversampling, and the full multi-class ROC/AUC evaluation suite.

---

## 1. Overview & Objective

* **Target Task**: 18-Category French Topic Classification.
* **Target Script**: [`training/french/fine_tune_french_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/french/fine_tune_french_topic.py)
* **Base Model**: [`camembert/camembert-large`](https://huggingface.co/camembert/camembert-large) (435M Parameters).
* **Dataset**: French original articles augmented via cross-lingual MarianMT translations from English.
* **Hardware Profile**: Optimized for NVIDIA RTX 3050 with automatic `bf16`/`fp16` mixed precision and expandable memory management.

---

## 2. Target Categories & Label Mapping

The model classifies French texts into 18 standardized topics:

| ID | French Topic | English Translation | ID | French Topic | English Translation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | **Politique** | Politics | `9` | **Sports** | Sports |
| `1` | **Économie** | Economy | `10` | **Culture** | Culture |
| `2` | **Sécurité** | Security | `11` | **Éducation** | Education |
| `3` | **Énergie** | Energy | `12` | **Technologie** | Technology |
| `4` | **Conflit** | Conflict | `13` | **Environnement** | Environment |
| `5` | **Élections** | Elections | `14` | **Diplomatie** | Diplomacy |
| `6` | **Justice** | Justice | `15` | **Religion** | Religion |
| `7` | **Santé** | Health | `16` | **Migration** | Migration |
| `8` | **Météo** | Weather | `17` | **Général** | General |

Text normalization strips diacritics and unifies spelling variations for robust topic assignment.

---

## 3. Theoretical & Architectural Rationale

### 3.1 Custom Architecture: `LargeClassifierFocal`
Rather than using generic sequence classification heads, a tailored module (`LargeClassifierFocal`) wraps `camembert-large`:
* **Feature Extraction**: Extracts the `<s>` (CLS) token representation from `last_hidden_state[:, 0, :]` (1024-dimensional vector).
* **Dropout Regularization**: Passes the pooled vector through `nn.Dropout(0.3)`.
* **Linear Projection**: Projects the 1024-d representation directly to 18 class logits.
* **Internal Loss Computation**: Evaluates class-specific Focal Loss internally within the forward pass.

### 3.2 Offline Cross-Lingual Translation Pipeline
* Uses `Helsinki-NLP/opus-mt-en-fr` to translate English news articles into French in batches of 128.
* Translations are cached to `translated_data/french_topic_translated_data.csv`.
* Increases dataset volume and lexical diversity while CamemBERT-large provides high tolerance to translation noise.

### 3.3 Environment Class Oversampling
Class 13 (`Environnement`) is chronically scarce in news corpora. To prevent zero-gradient representation during early training epochs, articles belonging to class 13 are synthetically oversampled via integer replication until reaching a floor of at least 500 training samples.

---

## 4. Mathematical Formulations & Loss Functions

### 4.1 Class-Specific Focal Gammas ($\gamma_c \in [2.0, 4.0]$)
Instead of applying a static gamma across all classes, the script dynamically computes a custom gamma for each category based on sample frequency:

$$\gamma_c = \text{clip}\left(2.0 + \frac{2.0}{\sqrt{N_c}}, 2.0, 4.0\right)$$

* **Frequent Classes (e.g. Politique, Général)**: $\gamma_c \to 2.0$ (standard focal moderation).
* **Rare Classes (e.g. Environnement, Religion)**: $\gamma_c \to 4.0$ (extreme focusing exponent penalizing misclassifications).

### 4.2 Soft Inverse Square-Root Class Weights
To avoid extreme weight penalties that destabilize 435M-parameter optimization:

$$w_c = \frac{1}{\sqrt{N_c}}, \quad \tilde{w}_c = \frac{w_c}{\sum_{k=0}^{K-1} w_k} \cdot K$$
$$w_c^{\text{final}} = \text{clip}(\tilde{w}_c, 0.8, 2.0)$$

### 4.3 Integrated Focal Loss
$$\mathcal{L}_{\text{Focal}} = \frac{1}{B} \sum_{i=1}^B w_{y_i} \cdot (1 - p_{t,i})^{\gamma_{y_i}} \cdot \text{CE}(\mathbf{z}_i, y_i)$$

---

## 5. Hyperparameter Choices & Empirical Justifications

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| `MAX_LEN` | `512` | Full Transformer window capturing complete article context. |
| `BATCH_SIZE` | `4` | Maximum batch fitting 435M parameters in VRAM at 512 tokens. |
| `GRAD_ACCUM` | `4` | Delivers an effective batch size of $4 \times 4 = 16$. |
| `LR` | `5e-6` | Halved learning rate for final cooling-down fine-tuning phase. |
| `EPOCHS` | `6` | Sufficient for convergence with pre-translated augmented data. |
| `WARMUP_STEPS` | `10%` | 10% of total optimization steps. |
| `WEIGHT_DECAY` | `0.01` | Weight decay regularizing classifier weights. |
| `MAX_GRAD_NORM` | `0.3` | Tight gradient clipping preventing gradient spikes. |
| `EARLY_STOPPING_PATIENCE` | `5` | Evaluation patience steps. |
| `EARLY_STOPPING_THRESHOLD`| `0.001`| Minimum threshold in `eval_f1_macro`. |

---

## 6. Comprehensive Evaluation & Visualization Suite

The script generates a comprehensive set of diagnostic evaluation artifacts in `plots/`:

1. **Multi-Class ROC Curves (`roc_curves_validation.png` / `roc_curves_test.png`)**:
   * One-vs-Rest ROC curve for all 18 classes.
   * Computes and logs Macro-average and Micro-average AUC scores.
2. **AUC Bar Charts (`auc_barchart_validation.png` / `auc_barchart_test.png`)**:
   * Per-class AUC performance bar chart.
   * Color coded: Green ($\text{AUC} \ge 0.8$), Orange ($0.6 \le \text{AUC} < 0.8$), Red ($\text{AUC} < 0.6$).
3. **Dual Confusion Matrices (`confusion_matrix_validation.png` / `confusion_matrix_test.png`)**:
   * Side-by-side heatmaps displaying raw sample counts and normalized percentage accuracies.
4. **Training Trajectories (`loss_curves.png` & `metrics_curves.png`)**:
   * Step-by-step training and validation loss curves.
   * Dual-axis trajectory of Validation Accuracy and Macro F1.

---

## 7. Code Walkthrough & Component Reference

### 7.1 Key Functions
* `normalize_text(x)`: Unicode NFKD normalization and diacritic removal.
* `parse_topic(x)`: Maps numeric or text strings to category IDs $[0, 17]$.
* `balance_training_frame(train_df)`: Shuffles training subsets to eliminate ordering bias.
* `build_class_weights(train_df, num_labels, device)`: Generates clipped soft inverse-sqrt class weights.
* `translate_texts(texts, batch_size)`: Batched MarianMT translation engine.
* `load_or_create_translated_data()`: Manages Parquet/CSV translation caching.
* `save_run_artifacts(trainer, run_dir, eval_ds, test_ds)`: Generates ROC curves, AUC bar charts, confusion matrices, and exports `eval_results.json`.

### 7.2 Custom Classes
* `SimpleTextDataset(TorchDataset)`: Tokenizes texts to length 512 with `tqdm` progress tracking.
* `FocalLoss(nn.Module)`: PyTorch module supporting sample-level dynamic class gammas.
* `LargeClassifierFocal(nn.Module)`: Custom sequence classification model wrapping `camembert-large`.

---

## 8. Execution Guide

### Run Script
```powershell
python training/french/fine_tune_french_topic.py
```

### Outputs
Stored in `experiments/topic_french/<timestamp>/`:
* `pytorch_model.bin` & `model_config.json`: Exported custom model weights and configuration.
* `hyperparameters.json`: Configuration snapshot.
* `eval_results.json`: Comprehensive metric scores across validation and test splits.
* `plots/`:
  * `roc_curves_validation.png` & `roc_curves_test.png`
  * `auc_barchart_validation.png` & `auc_barchart_test.png`
  * `confusion_matrix_validation.png` & `confusion_matrix_test.png`
  * `loss_curves.png` & `metrics_curves.png`
