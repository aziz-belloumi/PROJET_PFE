"""
test_topic.py
-------------
Runs inference on a set of Arabic test sentences using the BEST fine-tuned topic model,
as determined by report_search.py (ranked by average F1 macro across MSA + Dialect).

Results are saved inside the best model's run folder:
    experiments/topic/<best_run_timestamp>/test_runs/<test_timestamp>/
        results.csv
        summary.json
"""
import sys
import csv
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to sys.path to allow importing report_search
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from transformers import pipeline

from report.report_search import find_best_run

# UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Arabic shaping (fix disconnected letters in some terminals)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    def ar(s: str) -> str:
        return get_display(arabic_reshaper.reshape(s))
except Exception:
    def ar(s: str) -> str:
        return s

# ── Find best model via report_search ─────────────────────────────────────────
TASK   = "topic"
DEVICE = 0 if torch.cuda.is_available() else -1

try:
    best_run_dir = find_best_run(task=TASK, metric="avg_f1")
    print(f"Best model: {best_run_dir}")
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

p = pipeline("text-classification", model=str(best_run_dir), device=DEVICE)

# ── Save test results inside the best run's folder ────────────────────────────
test_ts  = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
test_dir = best_run_dir / "test_runs" / test_ts
test_dir.mkdir(parents=True, exist_ok=True)

# ── Test sentences representing different topics ──────────────────────────────
tests = [
    "عقد رئيس الوزراء اجتماعاً مع السفير الفرنسي لمناقشة العلاقات الثنائية.", # Politics / Diplomacy
    "سجلت أسعار النفط تراجعاً طفيفاً في التداولات العالمية صباح اليوم.",       # Economy / Energy
    "تأهل المنتخب الوطني إلى نهائي بطولة كأس الأمم الأفريقية بعد فوزه المثير.",   # Sports
    "أعلنت شركة أبل عن إطلاق هاتفها الجديد المزود بتقنيات الذكاء الاصطناعي.",   # Technology
    "أكدت وزارة الصحة على أهمية تلقي اللقاحات للوقاية من الأنفلونزا الموسمية.",   # Health
    "تتأثر المنطقة بموجة حر شديدة تؤدي إلى ارتفاع درجات الحرارة بشكل غير مسبوق.", # Environment / Weather
    "انطلقت اليوم مراسم الحج في مكة المكرمة وسط إجراءات تنظيمية مكثفة.",         # Religion
    "أعلنت القوات الأمنية عن إحباط مخطط تخريبي واعتقال شبكة إرهابية.",         # Security / Conflict
    "تبدأ الامتحانات النهائية لطلبة الجامعات الأسبوع المقبل بمختلف التخصصات.",      # Education
    "افتتح مهرجان الكتاب الدولي بمشاركة مئات دور النشر العربية والأجنبية.",     # Culture
]

# ── Run inference ──────────────────────────────────────────────────────────────
rows = []
for t in tests:
    out = p(t, truncation=True)[0]
    label, score = out["label"], out["score"]
    rows.append({"text": t, "label": label, "score": round(score, 6)})
    print(f"{ar(t)}\n -> {label} ({score:.4f})\n")

# ── Save results.csv ──────────────────────────────────────────────────────────
results_csv = test_dir / "results.csv"
with results_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label", "score"])
    writer.writeheader()
    writer.writerows(rows)

# ── Save summary.json ─────────────────────────────────────────────────────────
label_counts: dict = {}
for r in rows:
    label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

summary = {
    "test_timestamp":  test_ts,
    "best_model_dir":  str(best_run_dir),
    "num_samples":     len(rows),
    "label_counts":    label_counts,
    "results":         rows,
}
with (test_dir / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {test_dir}")
