from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_SEP_DIR = SCRIPT_DIR.parent
STATS_TXT    = DATA_SEP_DIR / "processing_stats.txt"
ENSEMBLE_JSON = DATA_SEP_DIR / "check_dialect" / "ensemble_stats.json"
OUTPUT_DIR   = SCRIPT_DIR          # diagrams land next to this script

# ──────────────────────────────────────────────────────────────
# Colour palettes
# ──────────────────────────────────────────────────────────────
LANG_COLORS = {
    "Arabic":   "#2E86AB",
    "English":  "#A23B72",
    "French":   "#F18F01",
    "Mixed":    "#C73E1D",
    "Rejected": "#8B8B8B",
}

DIALECT_COLORS = {
    "MSA": "#1B4965",
    "MGR": "#5FA8D3",
    "GLF": "#62B6CB",
    "LEV": "#BEE9E8",
    "EGY": "#CAE9FF",
}

REJECT_COLORS = {
    "Too short":     "#E63946",
    "Low confidence": "#457B9D",
    "Low quality":   "#A8DADC",
    "Unsupported":   "#F4A261",
}

MODEL_COLORS = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
]

# ──────────────────────────────────────────────────────────────
# Parse processing_stats.txt
# ──────────────────────────────────────────────────────────────

def _parse_int(s: str) -> int:
    return int(s.replace(",", "").strip())


def parse_processing_stats(path: Path) -> dict:
    """Extract key numbers from the free-text processing report."""
    text = path.read_text(encoding="utf-8")

    def _grab(pattern: str) -> int:
        m = re.search(pattern, text)
        return _parse_int(m.group(1)) if m else 0

    total    = _grab(r"Total\s*:\s*([\d,]+)\s*articles")
    arabic   = _grab(r"Arabic\s*\(.*?\)\s*:\s*([\d,]+)")
    english  = _grab(r"English\s*:\s*([\d,]+)")
    french   = _grab(r"French\s*:\s*([\d,]+)")
    mixed    = _grab(r"Mixed\s*\(.*?\)\s*:\s*([\d,]+)")
    rejected = _grab(r"REJECTED\s+([\d,]+)")

    # Words
    ar_words  = _grab(r"Arabic.*?Total words\s*:\s*([\d,]+)")
    en_words  = _grab(r"English.*?Total words\s*:\s*([\d,]+)")
    fr_words  = _grab(r"French.*?Total words\s*:\s*([\d,]+)")
    mx_words  = _grab(r"Mixed.*?Total words\s*:\s*([\d,]+)")

    # Rejection reasons
    too_short = _grab(r"Too short.*?:\s*([\d,]+)")
    low_conf  = _grab(r"Low confidence.*?:\s*([\d,]+)")
    low_qual  = _grab(r"Low quality.*?:\s*([\d,]+)")
    unsup     = _grab(r"Unsupported language\s*:\s*([\d,]+)")

    return {
        "total": total,
        "languages": {
            "Arabic":   arabic,
            "English":  english,
            "French":   french,
            "Mixed":    mixed,
            "Rejected": rejected,
        },
        "words": {
            "Arabic":  ar_words,
            "English": en_words,
            "French":  fr_words,
            "Mixed":   mx_words,
        },
        "rejection_reasons": {
            "Too short":      too_short,
            "Low confidence": low_conf,
            "Low quality":    low_qual,
            "Unsupported":    unsup,
        },
    }


# ──────────────────────────────────────────────────────────────
# Diagram helpers
# ──────────────────────────────────────────────────────────────

def _save(fig, name: str) -> Path:
    p = OUTPUT_DIR / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {p.name}")
    return p


def _fmt_k(n: int) -> str:
    """Human-friendly count."""
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}K"
    return str(n)


# ──────────────────────────────────────────────────────────────
# DIAGRAM 1 — Language distribution (pie)
# ──────────────────────────────────────────────────────────────

def fig_language_pie(stats: dict) -> Path:
    langs = stats["languages"]
    labels  = list(langs.keys())
    sizes   = list(langs.values())
    colors  = [LANG_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 12},
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    ax.set_title(
        f"Language Distribution of Fetched Articles\n(Total: {stats['total']:,})",
        fontsize=14, fontweight="bold", pad=20,
    )
    return _save(fig, "01_language_distribution_pie.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 2 — Language distribution (bar)
# ──────────────────────────────────────────────────────────────

def fig_language_bar(stats: dict) -> Path:
    langs = stats["languages"]
    labels = list(langs.keys())
    sizes  = list(langs.values())
    colors = [LANG_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, sizes, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + stats["total"] * 0.008,
            f"{_fmt_k(v)}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_title("Article Counts per Language Category", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "02_language_distribution_bar.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 3 — Word counts per language (bar)
# ──────────────────────────────────────────────────────────────

def fig_word_counts(stats: dict) -> Path:
    words = stats["words"]
    labels = list(words.keys())
    sizes  = list(words.values())
    colors = [LANG_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, sizes, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sizes):
        ax.text(
            bar.get_width() + max(sizes) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            _fmt_k(v), va="center", fontsize=11, fontweight="bold",
        )
    ax.set_xlabel("Total Word Count", fontsize=12)
    ax.set_title("Corpus Size per Language (Total Words)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, "03_word_counts_per_language.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 4 — Rejection reasons (pie)
# ──────────────────────────────────────────────────────────────

def fig_rejection_reasons(stats: dict) -> Path:
    reasons = stats["rejection_reasons"]
    labels = list(reasons.keys())
    sizes  = list(reasons.values())
    colors = [REJECT_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 11},
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    total_rej = sum(sizes)
    ax.set_title(
        f"Rejection Reasons Breakdown\n(Total Rejected: {total_rej:,})",
        fontsize=14, fontweight="bold", pad=20,
    )
    return _save(fig, "04_rejection_reasons_pie.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 5 — Accepted vs Rejected (donut)
# ──────────────────────────────────────────────────────────────

def fig_accepted_rejected(stats: dict) -> Path:
    accepted = stats["total"] - stats["languages"]["Rejected"]
    rejected = stats["languages"]["Rejected"]

    fig, ax = plt.subplots(figsize=(7, 7))
    sizes  = [accepted, rejected]
    labels = ["Accepted", "Rejected"]
    colors = ["#2E86AB", "#8B8B8B"]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 13},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(13)
    centre = plt.Circle((0, 0), 0.35, fc="white")
    ax.add_artist(centre)
    ax.text(0, 0.05, f"{stats['total']:,}", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, -0.12, "Total Articles", ha="center", va="center", fontsize=10, color="#555")
    ax.set_title("Accepted vs Rejected Articles", fontsize=14, fontweight="bold", pad=20)
    return _save(fig, "05_accepted_vs_rejected_donut.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 6 — Dialect distribution (pie)
# ──────────────────────────────────────────────────────────────

def fig_dialect_pie(ens: dict) -> Path:
    dist = ens["dialect_distribution"]
    dialect_labels_map = {
        "MSA": "MSA (Modern Standard Arabic)",
        "MGR": "Maghrebi",
        "GLF": "Gulf",
        "LEV": "Levantine",
        "EGY": "Egyptian",
    }
    labels = [dialect_labels_map.get(k, k) for k in dist]
    sizes  = list(dist.values())
    colors = [DIALECT_COLORS[k] for k in dist]

    fig, ax = plt.subplots(figsize=(9, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 11},
        pctdistance=0.80,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    total = sum(sizes)
    ax.set_title(
        f"Arabic Dialect Distribution (Ensemble)\n(Total: {total:,} articles)",
        fontsize=14, fontweight="bold", pad=20,
    )
    return _save(fig, "06_dialect_distribution_pie.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 7 — Dialect distribution (bar)
# ──────────────────────────────────────────────────────────────

def fig_dialect_bar(ens: dict) -> Path:
    dist = ens["dialect_distribution"]
    labels = list(dist.keys())
    sizes  = list(dist.values())
    colors = [DIALECT_COLORS[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, sizes, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.015,
            f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_title("Article Counts per Arabic Dialect", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    fig.tight_layout()
    return _save(fig, "07_dialect_distribution_bar.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 8 — Text length distribution (bar)
# ──────────────────────────────────────────────────────────────

def fig_text_length_dist(ens: dict) -> Path:
    dist = ens["text_length_distribution"]
    labels = list(dist.keys())
    sizes  = list(dist.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, sizes, color="#2E86AB", edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.012,
            _fmt_k(v), ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("Word Count Range", fontsize=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_title("Text Length Distribution (Arabic Corpus)", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, "08_text_length_distribution.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 9 — Per-dialect word statistics (grouped bar)
# ──────────────────────────────────────────────────────────────

def fig_dialect_word_stats(ens: dict) -> Path:
    dw = ens["dialect_word_stats"]
    dialects = list(dw.keys())
    means   = [dw[d]["mean"] for d in dialects]
    medians = [dw[d]["median"] for d in dialects]

    x = np.arange(len(dialects))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - w/2, means,   w, label="Mean", color="#264653", edgecolor="white")
    bars2 = ax.bar(x + w/2, medians, w, label="Median", color="#2A9D8F", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dialects, fontsize=11)
    ax.set_ylabel("Words per Article", fontsize=12)
    ax.set_title("Average vs Median Article Length by Dialect", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "09_dialect_word_stats.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 10 — Ensemble agreement distribution (bar)
# ──────────────────────────────────────────────────────────────

def fig_agreement_distribution(ens: dict) -> Path:
    raw = ens["raw_agreement_counts"]
    # Sorted by agreement level descending
    labels = ["5/5", "4/5", "3/5", "2/5", "1/5"]
    keys   = ["5", "4", "3", "2", "1"]
    sizes  = [raw.get(k, 0) for k in keys]
    colors = ["#1B4965", "#5FA8D3", "#62B6CB", "#BEE9E8", "#CAE9FF"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, sizes, color=colors, edgecolor="white", linewidth=0.8)
    total = sum(sizes)
    for bar, v in zip(bars, sizes):
        pct = v / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.008,
            f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("Number of Models that Agreed", fontsize=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_title("Ensemble Model Agreement Distribution", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "10_ensemble_agreement_distribution.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 11 — Model confidence comparison (box-style bar)
# ──────────────────────────────────────────────────────────────

def fig_model_confidence(ens: dict) -> Path:
    mc = ens["model_confidence_stats"]
    model_names = {
        "m1": "Ibrahim\nMARBERTv2",
        "m2": "CAMeL\nMADAR6",
        "m3": "Keleg\nNADI-2023",
        "m4": "Lafifi\nARBERT",
        "m5": "Oddadmix\nRouter",
    }
    keys = list(mc.keys())
    labels  = [model_names.get(k, k) for k in keys]
    means   = [mc[k]["mean"] for k in keys]
    medians = [mc[k]["median"] for k in keys]
    p25s    = [mc[k]["p25"] for k in keys]
    p75s    = [mc[k]["p75"] for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars (mean), with error-bar-like range from p25 to p75
    yerr_low  = [abs(m - p) for m, p in zip(means, p25s)]
    yerr_high = [abs(p - m) for m, p in zip(means, p75s)]
    bars = ax.bar(x, means, color=MODEL_COLORS, edgecolor="white", linewidth=0.8,
                  yerr=[yerr_low, yerr_high], capsize=5, error_kw={"lw": 1.5})

    # Overlay median markers
    ax.scatter(x, medians, color="white", zorder=5, s=60, edgecolors="black", linewidths=1.2, label="Median")

    for i, (m, md) in enumerate(zip(means, medians)):
        ax.text(x[i], m + max(yerr_high) * 0.15 + 0.01, f"μ={m:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Confidence Score", fontsize=12)
    ax.set_title("Individual Model Confidence Statistics\n(Bar = Mean ± IQR, ◆ = Median)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "11_model_confidence_comparison.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 12 — Per-model label distributions (stacked bar)
# ──────────────────────────────────────────────────────────────

def fig_per_model_labels(ens: dict) -> Path:
    pm = ens["per_model_label_distributions"]
    model_names = {
        "m1": "Ibrahim-MARBERTv2",
        "m2": "CAMeL-MADAR6",
        "m3": "Keleg-NADI-2023",
        "m4": "Lafifi-ARBERT",
        "m5": "Oddadmix-Router",
    }
    all_dialects = ["MSA", "EGY", "LEV", "GLF", "MGR"]
    models = list(pm.keys())
    labels = [model_names.get(m, m) for m in models]

    # Compute percentages
    data_pct = {}
    for d in all_dialects:
        data_pct[d] = []
        for m in models:
            total_m = sum(pm[m].values())
            data_pct[d].append(pm[m].get(d, 0) / total_m * 100 if total_m else 0)

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 7))

    bottom = np.zeros(len(models))
    for d in all_dialects:
        vals = data_pct[d]
        ax.bar(x, vals, bottom=bottom, label=d, color=DIALECT_COLORS[d],
               edgecolor="white", linewidth=0.5)
        # Add percentage text on segments > 8%
        for i, v in enumerate(vals):
            if v > 8:
                ax.text(x[i], bottom[i] + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Dialect Label Distribution per Model (Stacked %)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return _save(fig, "12_per_model_label_distribution.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 13 — Inter-model pairwise agreement (heatmap)
# ──────────────────────────────────────────────────────────────

def fig_pairwise_agreement(ens: dict) -> Path:
    pairs = ens["inter_model_pairwise_agreement"]
    model_short = {
        "Ibrahim-MARBERTv2": "Ibrahim",
        "CAMeL-MADAR6":      "CAMeL",
        "Keleg-NADI-2023":   "Keleg",
        "Lafifi-ARBERT":     "Lafifi",
        "Oddadmix-Router":   "Oddadmix",
    }
    names = list(model_short.values())
    n = len(names)
    matrix = np.full((n, n), np.nan)

    # Fill diagonal with 100%
    for i in range(n):
        matrix[i][i] = 100.0

    # Map full names to indices
    full_to_idx = {full: i for i, full in enumerate(model_short.keys())}

    for pair_key, info in pairs.items():
        parts = pair_key.split(" vs ")
        i = full_to_idx.get(parts[0].strip())
        j = full_to_idx.get(parts[1].strip())
        if i is not None and j is not None:
            matrix[i][j] = info["pct"]
            matrix[j][i] = info["pct"]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=10, rotation=35, ha="right")
    ax.set_yticklabels(names, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            if not np.isnan(v):
                color = "white" if v > 60 else "black"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_title("Inter-Model Pairwise Agreement (%)", fontsize=14, fontweight="bold", pad=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Agreement %")
    fig.tight_layout()
    return _save(fig, "13_pairwise_agreement_heatmap.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 14 — Ensemble confidence distribution (bar)
# ──────────────────────────────────────────────────────────────

def fig_ensemble_confidence_dist(ens: dict) -> Path:
    dist = ens["ensemble_confidence_distribution"]
    labels = list(dist.keys())
    sizes  = list(dist.values())
    total  = sum(sizes)

    colors_gradient = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, sizes, color=colors_gradient, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sizes):
        pct = v / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.008,
            f"{_fmt_k(v)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("Ensemble Confidence Range", fontsize=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_title("Ensemble Confidence Score Distribution", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return _save(fig, "14_ensemble_confidence_distribution.png")


# ──────────────────────────────────────────────────────────────
# DIAGRAM 15 — Per-dialect agreement breakdown (grouped bar)
# ──────────────────────────────────────────────────────────────

def fig_per_dialect_agreement(ens: dict) -> Path:
    pda = ens["per_dialect_agreement"]
    dialects = list(pda.keys())
    levels   = ["5/5", "4/5", "3/5", "2/5", "1/5"]
    level_colors = ["#1B4965", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

    x = np.arange(len(dialects))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (level, color) in enumerate(zip(levels, level_colors)):
        vals = []
        for d in dialects:
            total_d = sum(pda[d].values())
            vals.append(pda[d].get(level, 0) / total_d * 100 if total_d else 0)
        ax.bar(x + i * width, vals, width, label=level, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(dialects, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Per-Dialect Model Agreement Breakdown", fontsize=14, fontweight="bold")
    ax.legend(title="Agreement Level", fontsize=10, title_fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "15_per_dialect_agreement_breakdown.png")


# ──────────────────────────────────────────────────────────────
# descriptions.txt
# ──────────────────────────────────────────────────────────────

DESCRIPTIONS = {
    "01_language_distribution_pie.png": (
        "Language Distribution of Fetched Articles (Pie Chart)",
        "Shows the proportional breakdown of all articles fetched from the database "
        "by fetch_texts.py across the five categories: Arabic, English, French, Mixed, "
        "and Rejected. Gives a quick visual overview of the linguistic composition of "
        "the entire corpus."
    ),
    "02_language_distribution_bar.png": (
        "Article Counts per Language Category (Bar Chart)",
        "Displays the absolute number of articles in each language category. Unlike "
        "the pie chart, this allows precise comparison of magnitudes, clearly showing "
        "that Arabic dominates the corpus while French is extremely rare."
    ),
    "03_word_counts_per_language.png": (
        "Corpus Size per Language — Total Words (Horizontal Bar)",
        "Measures corpus size by total word count per language. This is important "
        "because article count alone can be misleading — a language with fewer but "
        "longer articles may contribute more text. Arabic has by far the largest word "
        "volume."
    ),
    "04_rejection_reasons_pie.png": (
        "Rejection Reasons Breakdown (Pie Chart)",
        "Breaks down the reasons why articles were rejected during fetch_texts.py "
        "processing. The main categories are: too short (<= 2 words), low language "
        "detection confidence (< 0.5), low text quality (alpha ratio < 0.65), and "
        "unsupported language. Helps identify the dominant cause of data loss."
    ),
    "05_accepted_vs_rejected_donut.png": (
        "Accepted vs Rejected Articles (Donut Chart)",
        "A high-level overview showing what percentage of the total fetched articles "
        "passed the quality and language filters (Accepted) versus those that were "
        "discarded (Rejected). The center displays the total article count."
    ),
    "06_dialect_distribution_pie.png": (
        "Arabic Dialect Distribution — Ensemble Result (Pie Chart)",
        "Shows the final dialect labels assigned by the 5-model ensemble annotator "
        "to the accepted Arabic articles. The five dialect categories are: MSA (Modern "
        "Standard Arabic), Maghrebi (MGR), Gulf (GLF), Levantine (LEV), and Egyptian "
        "(EGY). MSA and Maghrebi together dominate the corpus."
    ),
    "07_dialect_distribution_bar.png": (
        "Article Counts per Arabic Dialect (Bar Chart)",
        "Provides the exact article counts for each dialect category. This makes it "
        "easy to see the class imbalance — MSA and MGR have ~540K articles each, while "
        "EGY has only ~31K, which is important for downstream model training decisions."
    ),
    "08_text_length_distribution.png": (
        "Text Length Distribution — Arabic Corpus (Bar Chart)",
        "Plots the distribution of article lengths (in word count ranges) for the "
        "Arabic corpus. The majority of texts are short (11–50 words), followed by "
        "very short texts (1–10 words). Only a tiny fraction exceeds 1000 words. "
        "This distribution impacts model performance and chunking strategies."
    ),
    "09_dialect_word_stats.png": (
        "Average vs Median Article Length by Dialect (Grouped Bar)",
        "Compares the mean and median word counts per dialect. A large gap between "
        "mean and median (e.g., MSA: mean=61 vs median=17) indicates a heavily "
        "right-skewed distribution with a few very long articles pulling the average "
        "up. Levantine has the shortest texts overall."
    ),
    "10_ensemble_agreement_distribution.png": (
        "Ensemble Model Agreement Distribution (Bar Chart)",
        "Shows how many of the 5 ensemble models agreed on the winning dialect label. "
        "3/5 agreement is the most common (majority vote), while full 5/5 unanimous "
        "agreement is rare (~5%). A high proportion of 2/5 or 1/5 disagreement "
        "indicates inherent difficulty in dialect classification."
    ),
    "11_model_confidence_comparison.png": (
        "Individual Model Confidence Statistics (Bar + IQR)",
        "Compares the confidence levels of all 5 dialect classification models. Each "
        "bar represents the mean confidence, with error bars showing the interquartile "
        "range (P25–P75). The white dot marks the median. Keleg-NADI-2023 has notably "
        "lower confidence than the others."
    ),
    "12_per_model_label_distribution.png": (
        "Dialect Label Distribution per Model (Stacked %)",
        "Reveals how each model distributes its predictions across dialect labels. "
        "Significant differences exist: e.g., Keleg-NADI-2023 never predicts MSA, "
        "while CAMeL-MADAR6 assigns MSA to ~57% of articles. These biases are why "
        "an ensemble approach is necessary."
    ),
    "13_pairwise_agreement_heatmap.png": (
        "Inter-Model Pairwise Agreement Heatmap",
        "A matrix showing the percentage of articles where each pair of models "
        "assigned the same dialect label. Darker cells indicate higher agreement. "
        "The highest agreement is between Keleg and Lafifi (62.5%), while Keleg "
        "and CAMeL agree only 17.2% of the time, reflecting fundamentally different "
        "classification strategies."
    ),
    "14_ensemble_confidence_distribution.png": (
        "Ensemble Confidence Score Distribution (Bar Chart)",
        "Plots how the average ensemble confidence scores are distributed across the "
        "corpus. 65% of articles have ensemble confidence above 0.90, indicating "
        "strong model consensus. Only ~1% fall below 0.50, meaning very few articles "
        "are truly ambiguous across all models."
    ),
    "15_per_dialect_agreement_breakdown.png": (
        "Per-Dialect Model Agreement Breakdown (Grouped Bar)",
        "For each dialect, shows what fraction of articles had 5/5, 4/5, 3/5, 2/5, "
        "or 1/5 model agreement. MGR has the highest proportion of 5/5 unanimous "
        "agreement (~12%), while EGY has the highest 2/5 disagreement (~54%), "
        "suggesting Egyptian dialect is the hardest to classify reliably."
    ),
}


def write_descriptions(output_dir: Path):
    """Write a descriptions.txt file explaining each diagram."""
    lines = [
        "=" * 70,
        "  DIAGRAM DESCRIPTIONS",
        "  Auto-generated by generate_diagrams.py",
        "=" * 70,
        "",
    ]

    for i, (filename, (title, desc)) in enumerate(DESCRIPTIONS.items(), 1):
        lines.append(f"  {i}. {filename}")
        lines.append(f"     Title: {title}")
        lines.append(f"     Description:")
        for wrapped in textwrap.wrap(desc, width=65):
            lines.append(f"       {wrapped}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF DESCRIPTIONS")
    lines.append("=" * 70)
    lines.append("")

    path = output_dir / "descriptions.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ saved {path.name}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Diagram Generator — Language Separation & Dialect Data")
    print("=" * 60)

    # ---- Validate inputs ----
    if not STATS_TXT.exists():
        print(f"\n  ✗ ERROR: {STATS_TXT} not found.")
        print("    Run fetch_texts.py first to generate processing stats.")
        sys.exit(1)

    if not ENSEMBLE_JSON.exists():
        print(f"\n  ✗ ERROR: {ENSEMBLE_JSON} not found.")
        print("    Run ensemble_annotator.py first to generate dialect stats.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("\n  Loading data sources...")
    stats = parse_processing_stats(STATS_TXT)
    print(f"    processing_stats.txt  → {stats['total']:,} total articles")

    with open(ENSEMBLE_JSON, "r", encoding="utf-8") as f:
        ens = json.load(f)
    print(f"    ensemble_stats.json   → {ens['total_processed']:,} Arabic articles")

    # ---- Generate diagrams ----
    print("\n  Generating diagrams...\n")

    # fetch_texts.py diagrams
    fig_language_pie(stats)
    fig_language_bar(stats)
    fig_word_counts(stats)
    fig_rejection_reasons(stats)
    fig_accepted_rejected(stats)

    # Dialect ensemble diagrams
    fig_dialect_pie(ens)
    fig_dialect_bar(ens)
    fig_text_length_dist(ens)
    fig_dialect_word_stats(ens)
    fig_agreement_distribution(ens)
    fig_model_confidence(ens)
    fig_per_model_labels(ens)
    fig_pairwise_agreement(ens)
    fig_ensemble_confidence_dist(ens)
    fig_per_dialect_agreement(ens)

    # ---- Write descriptions ----
    print()
    write_descriptions(OUTPUT_DIR)

    print(f"\n  ✓ All done! {len(DESCRIPTIONS)} diagrams + descriptions.txt")
    print(f"    Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
