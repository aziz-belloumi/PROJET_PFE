#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive CLI Annotator for Multilingual News Articles (Arabic, English, French)
================================================================================
Simple, fast console annotator that searches directly for unlibeled articles:
  - Scans global_data_libelised.csv and isolates only the unlibeled articles
  - Prompts directly for missing label by number [1: Pos, 2: Neg, 3: Neu]
  - Saves progress immediately to global_data_libelised.csv
  - Generates final plots to data_libelisation/plots/ when 100% finished

Input  : global_data.csv / global_data_libelised.csv
Output : global_data_libelised.csv
Plots  : data_libelisation/plots/
"""

from __future__ import annotations

import os
import sys
import csv
import re
import shutil
import textwrap
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Configure console UTF-8 output on Windows immediately
if os.name == "nt":
    try:
        os.system("chcp 65001 > nul 2>&1")
    except Exception:
        pass

try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if sys.stdin and hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Try importing Arabic reshaper & BiDi
ARABIC_AVAILABLE = False
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_AVAILABLE = True
except ImportError:
    pass

# Try importing Matplotlib & Seaborn for plot export
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


# ==============================================================================
# TAXONOMY (STRICTLY 3 SENTIMENT CLASSES + 18 TOPICS)
# ==============================================================================

ARABIC_CHAR_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

SENTIMENTS = {
    "1": "POSITIVE",
    "2": "NEGATIVE",
    "3": "NEUTRAL",
    "pos": "POSITIVE",
    "neg": "NEGATIVE",
    "neu": "NEUTRAL",
    "positive": "POSITIVE",
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
}

SENTIMENT_COLORS = {
    "POSITIVE": "#2ecc71",
    "NEGATIVE": "#e74c3c",
    "NEUTRAL":  "#3498db",
}

TOPIC_CATEGORIES = {
    0:  {"code": "Politics",       "ar": "السياسة",       "en": "Politics",      "fr": "Politique"},
    1:  {"code": "Economy",        "ar": "الاقتصاد",      "en": "Economy",       "fr": "Économie"},
    2:  {"code": "Security",       "ar": "الأمن",         "en": "Security",      "fr": "Sécurité"},
    3:  {"code": "Energy",         "ar": "الطاقة",        "en": "Energy",        "fr": "Énergie"},
    4:  {"code": "Conflict",       "ar": "النزاع",        "en": "Conflict",      "fr": "Conflit"},
    5:  {"code": "Elections",      "ar": "الانتخابات",    "en": "Elections",     "fr": "Élections"},
    6:  {"code": "Justice",        "ar": "العدالة",       "en": "Justice",       "fr": "Justice"},
    7:  {"code": "Health",         "ar": "الصحة",         "en": "Health",        "fr": "Santé"},
    8:  {"code": "Weather",        "ar": "الطقس",         "en": "Weather",       "fr": "Météo"},
    9:  {"code": "Sports",         "ar": "الرياضة",       "en": "Sports",        "fr": "Sport"},
    10: {"code": "Culture",        "ar": "الثقافة",      "en": "Culture",       "fr": "Culture"},
    11: {"code": "Education",      "ar": "التعليم",      "en": "Education",     "fr": "Éducation"},
    12: {"code": "Technology",     "ar": "التكنولوجيا",  "en": "Technology",    "fr": "Technologie"},
    13: {"code": "Environment",    "ar": "البيئة",       "en": "Environment",   "fr": "Environnement"},
    14: {"code": "Diplomacy",      "ar": "الدبلوماسية",  "en": "Diplomacy",     "fr": "Diplomatie"},
    15: {"code": "Religion",       "ar": "الدين",        "en": "Religion",      "fr": "Religion"},
    16: {"code": "Migration",      "ar": "الهجرة",       "en": "Migration",     "fr": "Migration"},
    17: {"code": "General",        "ar": "عام",          "en": "General",       "fr": "Général"},
}

TOPIC_CODE_MAP = {data["code"].lower(): data["code"] for data in TOPIC_CATEGORIES.values()}
TOPIC_INDEX_MAP = {idx: data["code"] for idx, data in TOPIC_CATEGORIES.items()}


def normalize_topic_code(val: str) -> str:
    """Robustly normalizes topic string (Arabic, French, English, index, accented) to standard Code."""
    if not val:
        return ""
    v = str(val).strip()
    if not v or v.lower() == "nan":
        return ""

    for cat in TOPIC_CATEGORIES.values():
        if v.lower() in (cat["code"].lower(), cat["en"].lower(), cat["fr"].lower(), cat["ar"]):
            return cat["code"]

    v_norm = "".join(c for c in unicodedata.normalize("NFD", v) if unicodedata.category(c) != "Mn").lower()
    for cat in TOPIC_CATEGORIES.values():
        fr_norm = "".join(c for c in unicodedata.normalize("NFD", cat["fr"]) if unicodedata.category(c) != "Mn").lower()
        if v_norm == fr_norm:
            return cat["code"]

    if v.isdigit() and int(v) in TOPIC_INDEX_MAP:
        return TOPIC_INDEX_MAP[int(v)]

    return v


def format_text_lines(text: str, width: int = 80) -> List[str]:
    """Wraps text cleanly and formats Arabic BiDi reshaping if Arabic glyphs exist."""
    if not text:
        return ["  (empty text)"]
    lines_out = []
    for para in str(text).split("\n"):
        p_clean = " ".join(para.split())
        if not p_clean:
            continue
        for chunk in textwrap.wrap(p_clean, width=width):
            if ARABIC_CHAR_PATTERN.search(chunk) and ARABIC_AVAILABLE:
                try:
                    display_chunk = get_display(arabic_reshaper.reshape(chunk))
                except Exception:
                    display_chunk = chunk
            else:
                display_chunk = chunk
            lines_out.append("  " + display_chunk)
    return lines_out or ["  (empty text)"]


# ==============================================================================
# CLI ANNOTATOR CLASS (DIRECT UNLIBELED SEARCH & FAST ANNOTATION)
# ==============================================================================

class CLIAnnotator:
    def __init__(
        self,
        input_csv: str = "global_data.csv",
        output_csv: str = "global_data_libelised.csv",
    ):
        self.input_path = self._resolve_path(input_csv)
        if Path(output_csv).is_absolute():
            self.output_path = Path(output_csv)
        else:
            self.output_path = (CURRENT_DIR / output_csv).resolve()

        self.plots_dir = CURRENT_DIR / "plots"
        
        self.all_rows: List[Dict[str, str]] = []
        self.all_data_map: Dict[str, Dict[str, str]] = {}
        
        self.unlibeled_queue: List[Dict[str, str]] = []
        self.current_queue_idx: int = 0
        self.history: List[str] = []
        self.dirty: bool = False

    def _resolve_path(self, filename: str) -> Path:
        p = Path(filename)
        if p.exists():
            return p.resolve()
        candidates = [
            CURRENT_DIR / filename,
            PROJECT_ROOT / filename,
            PROJECT_ROOT / "data_exploration" / filename,
            PROJECT_ROOT / "data_libelisation" / filename,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return (CURRENT_DIR / filename).resolve()

    def load_data(self):
        """Scans dataset, loads existing labels, and builds the unlibeled queue."""
        if self.output_path.exists():
            print(f"Scanning {self.output_path.name} to find unlibeled articles...")
            with open(self.output_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    aid = str(r.get("id", "")).strip()
                    if not aid:
                        continue
                    
                    raw_sent = str(r.get("sentiment", "")).strip().upper()
                    norm_sent = raw_sent if raw_sent in ("POSITIVE", "NEGATIVE", "NEUTRAL") else SENTIMENTS.get(raw_sent.lower(), "")
                    norm_top = normalize_topic_code(r.get("topic"))
                    text = r.get("text", "")
                    lang = r.get("language", "")

                    row_entry = {
                        "id": aid,
                        "text": text,
                        "language": lang,
                        "sentiment": norm_sent,
                        "topic": norm_top,
                    }
                    self.all_rows.append(row_entry)
                    self.all_data_map[aid] = row_entry

                    # Check if unlibeled (missing sentiment or missing topic)
                    if not norm_sent or not norm_top:
                        self.unlibeled_queue.append(row_entry)

        else:
            print(f"No {self.output_path.name} found. Loading from {self.input_path.name}...")
            if not self.input_path.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
            with open(self.input_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    aid = str(r.get("id", "")).strip()
                    if not aid:
                        continue
                    row_entry = {
                        "id": aid,
                        "text": r.get("text", ""),
                        "language": r.get("language", ""),
                        "sentiment": "",
                        "topic": "",
                    }
                    self.all_rows.append(row_entry)
                    self.all_data_map[aid] = row_entry
                    self.unlibeled_queue.append(row_entry)

            # Initialize file with headers
            with open(self.output_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "text", "language", "sentiment", "topic"])

        total_articles = len(self.all_rows)
        unlibeled_count = len(self.unlibeled_queue)
        libelised_count = total_articles - unlibeled_count
        pct = (libelised_count / total_articles * 100) if total_articles else 0.0

        print(f"✓ Total Dataset Size: {total_articles:,} articles")
        print(f"✓ Already Libelised : {libelised_count:,} ({pct:.3f}%)")
        print(f"✓ Unlibeled Queue   : {unlibeled_count:,} articles remaining to libelise.")

    def _save_all_to_csv(self):
        """Rewrites the master CSV with all updated labels."""
        tmp_file = self.output_path.with_suffix(".tmp")
        with open(tmp_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text", "language", "sentiment", "topic"])
            for r in self.all_rows:
                aid = r["id"]
                data = self.all_data_map.get(aid, r)
                writer.writerow([
                    aid,
                    data.get("text", ""),
                    data.get("language", ""),
                    data.get("sentiment", ""),
                    data.get("topic", ""),
                ])
        if tmp_file.exists():
            shutil.move(str(tmp_file), str(self.output_path))

    def _save_single_label(self, aid: str, sentiment: str, topic: str):
        """Updates in-memory record instantly (0ms latency)."""
        if aid in self.all_data_map:
            self.all_data_map[aid]["sentiment"] = sentiment
            self.all_data_map[aid]["topic"] = topic
        self.dirty = True

    # ==========================================================================
    # DISPLAY METHODS
    # ==========================================================================

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
        total = len(self.all_rows)
        remaining = sum(1 for d in self.all_data_map.values() if not d.get("sentiment") or not d.get("topic"))
        completed = total - remaining
        pct = (completed / total * 100) if total else 0.0
        print("=" * 75)
        print(f"  MULTILINGUAL NEWS CLI ANNOTATOR (AR / EN / FR)")
        print(f"  Overall Progress: {completed:,} / {total:,} ({pct:.3f}%) | Remaining: {remaining:,}")
        print("=" * 75)

    def print_topics_menu(self):
        print("\n--- TOPIC CATEGORIES (18) ---")
        items = []
        for idx in range(18):
            data = TOPIC_CATEGORIES[idx]
            items.append(f"[{idx:2d}] {data['en']} ({data['ar']})")

        for i in range(0, len(items), 3):
            line_items = items[i:i+3]
            print("  " + "   ".join(f"{it:<23}" for it in line_items))

        print("\nCommands: [s] Skip | [u] Undo | [q] Quit")

    def print_sentiment_menu(self):
        print("\n--- SENTIMENT CHOICES (3 Classes) ---")
        print("  [1] POSITIVE (إيجابي)    [2] NEGATIVE (سلبي)    [3] NEUTRAL (محايد)")
        print("\nCommands: [s] Skip | [u] Undo | [q] Quit")

    def display_article(self, row: Dict[str, str], current_num: int, total_num: int):
        aid = str(row.get("id", "")).strip()
        lang = str(row.get("language", "unknown")).upper()
        text = str(row.get("text", "")).strip()
        top = row.get("topic", "—") or "—"
        sent = row.get("sentiment", "—") or "—"

        print(f"\nUnlibeled Article #{current_num} of {total_num} | ID: {aid} | Language: {lang}")
        print(f"Current Status -> Topic: [{top}] | Sentiment: [{sent}]")
        print("-" * 75)
        print("ARTICLE TEXT:")
        print()
        for line in format_text_lines(text, width=80):
            print(line)
        print()
        print("-" * 75)

    # ==========================================================================
    # FINAL PLOTS (100% COMPLETE)
    # ==========================================================================

    def generate_libelisation_plots(self):
        """Generates all 8 diagrams into data_libelisation/plots/."""
        if not MATPLOTLIB_AVAILABLE:
            print("\nMatplotlib is not installed. Skipping plot generation.")
            return

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating comprehensive libelisation plots in: {self.plots_dir}...")

        records = [
            {
                "id": aid,
                "text": d.get("text", ""),
                "language": d.get("language", ""),
                "sentiment": d.get("sentiment", ""),
                "topic": d.get("topic", ""),
            }
            for aid, d in self.all_data_map.items()
            if d.get("sentiment") and d.get("topic")
        ]

        df_lab = pd.DataFrame(records)
        df_lab["word_count"] = df_lab["text"].astype(str).str.split().apply(len)
        total_labeled = len(df_lab)

        # 1. Sentiment Bar
        sent_counts = df_lab["sentiment"].value_counts()
        sent_order = [s for s in ["POSITIVE", "NEGATIVE", "NEUTRAL"] if s in sent_counts.index]
        colors_sent = [SENTIMENT_COLORS.get(s, "#3498db") for s in sent_order]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(sent_order, [sent_counts[s] for s in sent_order], color=colors_sent, edgecolor="white", linewidth=0.8)
        for bar, s in zip(bars, sent_order):
            cnt = sent_counts[s]
            pct = (cnt / total_labeled) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + total_labeled * 0.01,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax.set_ylabel("Number of Articles", fontsize=12)
        ax.set_title(f"Sentiment Distribution across Libelised Corpus (Total: {total_labeled:,})", fontsize=14, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p1 = self.plots_dir / "01_sentiment_distribution_bar.png"
        fig.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ saved {p1.name}")

        # 2. Sentiment Pie
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            [sent_counts[s] for s in sent_order], labels=sent_order, autopct="%1.1f%%",
            colors=colors_sent, startangle=140, textprops={"fontsize": 12}, pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontweight("bold")
            t.set_color("white")
        ax.set_title(f"Sentiment Proportions (3 Classes)\n(Total Libelised: {total_labeled:,})", fontsize=14, fontweight="bold", pad=20)
        p2 = self.plots_dir / "02_sentiment_distribution_pie.png"
        fig.savefig(p2, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ saved {p2.name}")

        # 3. Topic Bar
        topic_counts = df_lab["topic"].value_counts()
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(topic_counts.index, topic_counts.values, color="#2b5c8f", edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, topic_counts.values):
            pct = (v / total_labeled) * 100
            ax.text(
                bar.get_width() + max(topic_counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({pct:.1f}%)", va="center", fontsize=10, fontweight="bold",
            )
        ax.set_xlabel("Number of Articles", fontsize=12)
        ax.set_title(f"18 Topic Categories Distribution (Total: {total_labeled:,})", fontsize=15, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        fig.tight_layout()
        p3 = self.plots_dir / "03_topic_distribution_18_classes.png"
        fig.savefig(p3, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ saved {p3.name}")

        # 4. Sentiment by Topic Heatmap
        clean_subset = df_lab[df_lab["sentiment"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"]) & df_lab["topic"].notna()]
        if not clean_subset.empty:
            cross_tab = pd.crosstab(clean_subset["topic"], clean_subset["sentiment"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Percentage (%)"}, ax=ax)
            ax.set_title("Sentiment Breakdown by Topic Category (% per Topic)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Topic Category")
            fig.tight_layout()
            p4 = self.plots_dir / "04_sentiment_by_topic_heatmap.png"
            fig.savefig(p4, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  ✓ saved {p4.name}")

        # Descriptions
        desc = [
            "=" * 70,
            "  DATA LIBELLISATION DIAGRAM DESCRIPTIONS",
            "  Auto-generated by cli_annotator.py upon 100% dataset labeling",
            "=" * 70,
            "",
            "1. 01_sentiment_distribution_bar.png: Absolute counts & percentages of POSITIVE, NEGATIVE, NEUTRAL.",
            "2. 02_sentiment_distribution_pie.png: Proportional breakdown across 3 sentiment classes.",
            "3. 03_topic_distribution_18_classes.png: Article counts across the 18 topic categories.",
            "4. 04_sentiment_by_topic_heatmap.png: Cross-tabulation (% sentiment within each topic).",
            "",
            "=" * 70,
        ]
        p_desc = self.plots_dir / "descriptions.txt"
        p_desc.write_text("\n".join(desc), encoding="utf-8")
        print(f"  ✓ saved {p_desc.name}")
        print(f"\n🎉 All libellisation plots successfully generated in: {self.plots_dir.resolve()}\n")

    # ==========================================================================
    # MAIN RUN LOOP
    # ==========================================================================

    def run(self):
        """Processes unlibeled articles directly."""
        self.load_data()

        if not self.unlibeled_queue:
            self.clear_screen()
            self.print_header()
            print("\n🎉 ALL 1,484,845 ARTICLES ARE ALREADY FULLY LIBELISED (100.0%)!")
            self.generate_libelisation_plots()
            return

        while self.current_queue_idx < len(self.unlibeled_queue):
            row = self.unlibeled_queue[self.current_queue_idx]
            aid = row["id"]
            
            # Fetch latest in-memory state
            data = self.all_data_map.get(aid, row)
            top = data.get("topic", "")
            sent = data.get("sentiment", "")

            # If already completed in this session, skip
            if top and sent:
                self.current_queue_idx += 1
                continue

            self.clear_screen()
            self.print_header()
            self.display_article(data, self.current_queue_idx + 1, len(self.unlibeled_queue))

            # Case 1: Topic exists -> ONLY Sentiment needed (The 60 articles case!)
            if top and not sent:
                self.print_sentiment_menu()
                user_inp = input(f"\nSelect Sentiment [1: Pos, 2: Neg, 3: Neu] for Article #{aid}: ").strip()
                if not user_inp:
                    continue

                cmd = user_inp.lower()
                if cmd in ("q", "quit", "exit"):
                    print(f"\nSaving progress...")
                    self._save_all_to_csv()
                    print("Done! Progress saved. Goodbye.")
                    return
                if cmd in ("s", "skip", "next", "n"):
                    self.current_queue_idx += 1
                    continue
                if cmd in ("u", "undo", "prev", "p", "b", "back"):
                    self.current_queue_idx = max(0, self.current_queue_idx - 1)
                    continue

                chosen_sent = SENTIMENTS.get(cmd) or (cmd.upper() if cmd.upper() in ("POSITIVE", "NEGATIVE", "NEUTRAL") else None)
                if chosen_sent:
                    self._save_single_label(aid, chosen_sent, top)
                    self.history.append(aid)
                    self.current_queue_idx += 1
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    input("Press Enter to continue...")

            # Case 2: Sentiment exists -> ONLY Topic needed
            elif sent and not top:
                self.print_topics_menu()
                user_inp = input(f"\nSelect Topic [0-17] for Article #{aid}: ").strip()
                if not user_inp:
                    continue

                cmd = user_inp.lower()
                if cmd in ("q", "quit", "exit"):
                    print(f"\nSaving progress...")
                    self._save_all_to_csv()
                    print("Done! Progress saved. Goodbye.")
                    return
                if cmd in ("s", "skip", "next", "n"):
                    self.current_queue_idx += 1
                    continue
                if cmd in ("u", "undo", "prev", "p", "b", "back"):
                    self.current_queue_idx = max(0, self.current_queue_idx - 1)
                    continue

                chosen_top = normalize_topic_code(user_inp)
                if chosen_top:
                    self._save_single_label(aid, sent, chosen_top)
                    self.history.append(aid)
                    self.current_queue_idx += 1
                else:
                    print("Invalid choice. Please enter Topic number [0-17].")
                    input("Press Enter to continue...")

            # Case 3: Both needed
            else:
                self.print_topics_menu()
                user_inp = input(f"\n[1/2] Select Topic [0-17] for Article #{aid} (or combo '0 1'): ").strip()
                if not user_inp:
                    continue

                cmd = user_inp.lower()
                if cmd in ("q", "quit", "exit"):
                    self._save_all_to_csv()
                    return
                if cmd in ("s", "skip", "next", "n"):
                    self.current_queue_idx += 1
                    continue
                if cmd in ("u", "undo", "prev", "p", "b", "back"):
                    self.current_queue_idx = max(0, self.current_queue_idx - 1)
                    continue

                tokens = user_inp.split()
                if len(tokens) >= 2:
                    t1 = normalize_topic_code(tokens[0])
                    s1 = SENTIMENTS.get(tokens[1].lower())
                    if t1 and s1:
                        self._save_single_label(aid, s1, t1)
                        self.history.append(aid)
                        self.current_queue_idx += 1
                        continue

                t = normalize_topic_code(tokens[0])
                if t:
                    self.print_sentiment_menu()
                    s_inp = input(f"\n[2/2] Topic [{t}] selected. Select Sentiment [1: Pos, 2: Neg, 3: Neu]: ").strip()
                    s = SENTIMENTS.get(s_inp.lower())
                    if s:
                        self._save_single_label(aid, s, t)
                        self.history.append(aid)
                        self.current_queue_idx += 1
                    else:
                        print("Invalid sentiment. Skipping.")
                        input("Press Enter to continue...")

        # When loop finishes: all unlibeled articles completed
        if self.dirty:
            print("\nSaving updated annotations to global_data_libelised.csv...")
            self._save_all_to_csv()
            print("✓ Progress saved.")

        self.clear_screen()
        self.print_header()
        print("\n🎉 ALL ARTICLES HAVE BEEN FULLY LIBELISED (100.0%)!")
        self.generate_libelisation_plots()


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLI Annotator for Arabic & Multilingual News Articles")
    parser.add_argument("--input", "-i", default="global_data.csv", help="Input CSV path (default: global_data.csv)")
    parser.add_argument("--output", "-o", default="global_data_libelised.csv", help="Output labeled CSV path (default: global_data_libelised.csv)")
    args = parser.parse_args()

    annotator = CLIAnnotator(input_csv=args.input, output_csv=args.output)

    try:
        annotator.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving session progress...")
        annotator._save_all_to_csv()
        print("Progress saved. Goodbye!")


if __name__ == "__main__":
    main()
