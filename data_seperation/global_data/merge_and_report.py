import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

def main():
    data_dir = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation\global_data")
    output_dir = Path(r"c:\Users\bello\Desktop\PROJET_PFE\data_seperation\check_dialect")
    
    # Define paths
    acc_pc1 = output_dir / "accepted_articles_pc1.csv"
    acc_pc2 = output_dir / "accepted_articles_pc2.csv"
    rej_pc1 = output_dir / "rejected_articles_pc1.csv"
    rej_pc2 = output_dir / "rejected_articles_pc2.csv"
    
    master_acc = output_dir / "accepted_articles.csv"
    master_rej = output_dir / "rejected_articles.csv"
    
    print("\n" + "="*60)
    print("          MERGING PC1 AND PC2 ENSEMBLE RESULTS")
    print("="*60)
    
    # Verify files
    if not acc_pc1.exists():
        logger.error(f"PC 1 accepted file not found at {output_dir}! Make sure accepted_articles_pc1.csv is in the folder.")
        return
    if not acc_pc2.exists():
        logger.error(f"PC 2 accepted file not found at {output_dir}! Please copy accepted_articles_pc2.csv from PC 2 into this folder.")
        return

    # 1. Merge Accepted Articles
    logger.info("Merging accepted articles...")
    df_acc1 = pd.read_csv(acc_pc1)
    df_acc2 = pd.read_csv(acc_pc2)
    df_master_acc = pd.concat([df_acc1, df_acc2], ignore_index=True)
    
    # Fill empty final_label values with ensemble_label if columns exist
    if "final_label" in df_master_acc.columns and "ensemble_label" in df_master_acc.columns:
        df_master_acc["final_label"] = df_master_acc["final_label"].fillna(df_master_acc["ensemble_label"])
    elif "ensemble_label" in df_master_acc.columns:
        df_master_acc["final_label"] = df_master_acc["ensemble_label"]
        
    df_master_acc.to_csv(master_acc, index=False)
    logger.info(f"  Master Accepted Database saved: {len(df_master_acc):,} rows to {master_acc.name}")
    
    # 2. Merge Rejected Articles (Optional, since 0 rejected articles means files don't exist)
    logger.info("Merging rejected articles...")
    rejected_dfs = []
    if rej_pc1.exists():
        rejected_dfs.append(pd.read_csv(rej_pc1))
    if rej_pc2.exists():
        rejected_dfs.append(pd.read_csv(rej_pc2))
        
    if rejected_dfs:
        df_master_rej = pd.concat(rejected_dfs, ignore_index=True)
    else:
        df_master_rej = pd.DataFrame(columns=df_master_acc.columns)
        
    df_master_rej.to_csv(master_rej, index=False)
    logger.info(f"  Master Rejected Database saved: {len(df_master_rej):,} rows to {master_rej.name}")
    

    # 3. Generate Master Report
    logger.info("Generating global master report...")
    total_accepted = len(df_master_acc)
    total_rejected = len(df_master_rej)
    total_articles = total_accepted + total_rejected

    import numpy as np

    models_names = {
        "m1": "Ibrahim-MARBERTv2",
        "m2": "CAMeL-MADAR6",
        "m3": "Keleg-NADI-2023",
        "m4": "Lafifi-ARBERT",
        "m5": "Oddadmix-Router"
    }
    lbl_col = "final_label" if "final_label" in df_master_acc.columns else "ensemble_label"
    dialects_order = ["MSA", "EGY", "LEV", "GLF", "MGR"]

    # ------- SECTION 1: GLOBAL OVERVIEW -------
    word_counts = df_master_acc["text"].astype(str).str.split().str.len()
    total_words = int(word_counts.sum())
    avg_words = word_counts.mean()
    median_words = word_counts.median()
    min_words = int(word_counts.min())
    max_words = int(word_counts.max())
    std_words = word_counts.std()

    # ------- SECTION 2: DIALECT DISTRIBUTION -------
    dialect_dist = df_master_acc[lbl_col].value_counts().to_dict()
    dialect_dist = {d: dialect_dist.get(d, 0) for d in dialects_order}

    # ------- SECTION 3: AGREEMENT RATIOS (TRUE = based on valid voters) -------
    # Compute the number of valid voters per article (models that did NOT output OTHER)
    valid_labels_set = {"MSA", "EGY", "LEV", "GLF", "MGR"}
    valid_voter_counts = pd.Series(0, index=df_master_acc.index)
    for i in range(1, 6):
        col = f"m{i}_label"
        if col in df_master_acc.columns:
            valid_voter_counts += df_master_acc[col].isin(valid_labels_set).astype(int)

    # True agreement ratio = agreement_count / valid_voter_count
    true_agreement = df_master_acc["agreement_count"] / valid_voter_counts.replace(0, 1)
    
    # Raw agreement count distribution (as stored in CSV)
    raw_agreement_ratios = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
    for k, v in df_master_acc["agreement_count"].value_counts().to_dict().items():
        if k in [1, 2, 3, 4, 5]:
            raw_agreement_ratios[str(k)] += v

    # Agreement value coverage diagnostics: account for rows not in 1..5
    agreement_value_counts = df_master_acc["agreement_count"].value_counts(dropna=False)
    accounted_for = sum(v for k, v in agreement_value_counts.items() if k in [1, 2, 3, 4, 5])
    nan_count = int(df_master_acc["agreement_count"].isna().sum())
    other_values = {str(k): int(v) for k, v in agreement_value_counts.items() if (k not in [1, 2, 3, 4, 5]) and (not pd.isna(k))}
    missing_or_other = int(total_accepted - accounted_for)

    # Valid voter count distribution
    voter_count_dist = valid_voter_counts.value_counts().sort_index().to_dict()

    # True agreement ratio buckets
    true_agree_bins = [0, 0.201, 0.401, 0.601, 0.801, 1.01]
    true_agree_labels = ["<=20%", "21-40%", "41-60%", "61-80%", "81-100%"]
    true_agree_dist = pd.cut(true_agreement, bins=true_agree_bins, labels=true_agree_labels, right=False).value_counts().sort_index().to_dict()

    # ------- SECTION 4: PER-DIALECT WORD COUNT STATS -------
    dialect_word_stats = {}
    for d in dialects_order:
        mask = df_master_acc[lbl_col] == d
        wc = word_counts[mask]
        if len(wc) > 0:
            dialect_word_stats[d] = {
                "count": int(len(wc)),
                "total_words": int(wc.sum()),
                "mean": round(float(wc.mean()), 1),
                "median": round(float(wc.median()), 1),
                "min": int(wc.min()),
                "max": int(wc.max()),
                "std": round(float(wc.std()), 1),
            }

    # ------- SECTION 5: TEXT LENGTH DISTRIBUTION (BUCKETS) -------
    bins = [0, 10, 50, 100, 200, 500, 1000, 5000, float("inf")]
    bin_labels = ["1-10", "11-50", "51-100", "101-200", "201-500", "501-1000", "1001-5000", "5000+"]
    text_len_dist = pd.cut(word_counts, bins=bins, labels=bin_labels).value_counts().sort_index().to_dict()

    # ------- SECTION 6: MODEL CONFIDENCE STATS -------
    model_conf_stats = {}
    for i in range(1, 6):
        col = f"m{i}_conf"
        if col in df_master_acc.columns:
            vals = df_master_acc[col].dropna()
            model_conf_stats[f"m{i}"] = {
                "mean": round(float(vals.mean()), 4),
                "median": round(float(vals.median()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "p25": round(float(vals.quantile(0.25)), 4),
                "p75": round(float(vals.quantile(0.75)), 4),
                "p95": round(float(vals.quantile(0.95)), 4),
            }

    # ------- SECTION 7: PER-MODEL LABEL DISTRIBUTIONS -------
    model_label_dists = {}
    for i in range(1, 6):
        col = f"m{i}_label"
        if col in df_master_acc.columns:
            model_label_dists[f"m{i}"] = df_master_acc[col].value_counts().to_dict()

    # ------- SECTION 8: ENSEMBLE CONFIDENCE STATS -------
    ens_conf = df_master_acc["average_confidence"]
    ens_conf_stats = {
        "mean": round(float(ens_conf.mean()), 4),
        "median": round(float(ens_conf.median()), 4),
        "std": round(float(ens_conf.std()), 4),
        "min": round(float(ens_conf.min()), 4),
        "max": round(float(ens_conf.max()), 4),
        "p25": round(float(ens_conf.quantile(0.25)), 4),
        "p75": round(float(ens_conf.quantile(0.75)), 4),
        "p95": round(float(ens_conf.quantile(0.95)), 4),
    }
    # Ensemble confidence buckets
    conf_bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    conf_labels = ["<0.30", "0.30-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    ens_conf_dist = pd.cut(ens_conf, bins=conf_bins, labels=conf_labels, right=False).value_counts().sort_index().to_dict()

    # ------- SECTION 9: PER-DIALECT AGREEMENT BREAKDOWN -------
    dialect_agreement = {}
    for d in dialects_order:
        mask = df_master_acc[lbl_col] == d
        ag = df_master_acc.loc[mask, "agreement_count"].value_counts().to_dict()
        dialect_agreement[d] = {f"{k}/5": ag.get(k, 0) for k in [5, 4, 3, 2, 1]}

    # ------- SECTION 10: INTER-MODEL PAIRWISE AGREEMENT -------
    pairwise_agreement = {}
    for i in range(1, 6):
        for j in range(i + 1, 6):
            col_i = f"m{i}_label"
            col_j = f"m{j}_label"
            if col_i in df_master_acc.columns and col_j in df_master_acc.columns:
                agree_count = int((df_master_acc[col_i] == df_master_acc[col_j]).sum())
                pct = round(agree_count / total_accepted * 100, 2)
                key = f"{models_names[f'm{i}']} vs {models_names[f'm{j}']}"
                pairwise_agreement[key] = {"agree": agree_count, "pct": pct}

    # ------- SECTION 11: REJECTION REASONS -------
    rejection_reasons = {}
    if total_rejected > 0:
        rejection_reasons = df_master_rej["rejection_reason"].value_counts().to_dict()

    # =====================================================================
    # BUILD REPORT TEXT
    # =====================================================================
    W = 65
    lines = []
    lines.append("=" * W)
    lines.append("  CONSOLIDATED ARABIC ENSEMBLE DIALECT ANNOTATION REPORT")
    lines.append("=" * W)
    lines.append(f"Date:               {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Processed:    {total_articles:,}")
    lines.append(f"Accepted:           {total_accepted:,} ({(total_accepted/total_articles)*100:.2f}%)")
    lines.append(f"Rejected:           {total_rejected:,} ({(total_rejected/total_articles)*100:.2f}%)")
    lines.append("-" * W)

    # --- 1. CORPUS TEXT STATISTICS ---
    lines.append("")
    lines.append("1. CORPUS TEXT STATISTICS")
    lines.append("-" * W)
    lines.append(f"  Total Words:        {total_words:,}")
    lines.append(f"  Average Words/Text: {avg_words:,.1f}")
    lines.append(f"  Median Words/Text:  {median_words:,.1f}")
    lines.append(f"  Std Dev:            {std_words:,.1f}")
    lines.append(f"  Min Words:          {min_words:,}")
    lines.append(f"  Max Words:          {max_words:,}")

    # --- 2. TEXT LENGTH DISTRIBUTION ---
    lines.append("")
    lines.append("2. TEXT LENGTH DISTRIBUTION (by word count)")
    lines.append("-" * W)
    for bucket, count in text_len_dist.items():
        pct = (count / total_accepted) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"  {bucket:>10} words: {count:>10,}  ({pct:>5.1f}%)  {bar}")

    # --- 3. DIALECT DISTRIBUTION ---
    lines.append("")
    lines.append("3. DIALECT DISTRIBUTION")
    lines.append("-" * W)
    for d in dialects_order:
        v = dialect_dist.get(d, 0)
        pct = (v / total_accepted) * 100 if total_accepted > 0 else 0
        bar = "#" * int(pct / 2)
        lines.append(f"  {d:<6}: {v:>10,}  ({pct:>6.2f}%)  {bar}")

    # --- 4. PER-DIALECT WORD COUNT STATISTICS ---
    lines.append("")
    lines.append("4. PER-DIALECT WORD COUNT STATISTICS")
    lines.append("-" * W)
    lines.append(f"  {'Dialect':<6}  {'Count':>10}  {'TotalWords':>12}  {'Mean':>7}  {'Median':>7}  {'Std':>7}")
    for d in dialects_order:
        s = dialect_word_stats.get(d, {})
        if s:
            lines.append(f"  {d:<6}  {s['count']:>10,}  {s['total_words']:>12,}  {s['mean']:>7.1f}  {s['median']:>7.1f}  {s['std']:>7.1f}")

    # --- 5. VALID VOTER DISTRIBUTION ---
    lines.append("")
    lines.append("5. VALID VOTER COUNT DISTRIBUTION (models that gave MSA/EGY/LEV/GLF/MGR)")
    lines.append("-" * W)
    lines.append("  (Models that output OTHER are excluded from voting)")
    for nv, cnt in sorted(voter_count_dist.items()):
        pct = (cnt / total_accepted) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"  {nv} valid voters: {cnt:>10,}  ({pct:>5.1f}%)  {bar}")

    # --- 6. RAW AGREEMENT COUNT ---
    lines.append("")
    lines.append("6. RAW AGREEMENT COUNT (max votes for winning label)")
    lines.append("-" * W)
    for k in ["5", "4", "3", "2", "1"]:
        v = raw_agreement_ratios.get(k, 0)
        pct = (v / total_accepted) * 100 if total_accepted > 0 else 0
        bar = "#" * int(pct / 2)
        lines.append(f"  {k} models agreed: {v:>10,}  ({pct:>6.2f}%)  {bar}")

    # --- 6b. AGREEMENT COUNT COVERAGE ---
    lines.append("")
    lines.append("6b. AGREEMENT COUNT COVERAGE")
    lines.append("-" * W)
    lines.append(f"  Rows with agreement_count in 1..5 (accounted): {accounted_for:,}")
    lines.append(f"  Remaining accepted rows not accounted: {missing_or_other:,}  (NaN: {nan_count:,}, other values: {sum(other_values.values()):,})")
    if other_values:
        ev = ", ".join([f"{k}:{v:,}" for k, v in list(other_values.items())[:10]])
        lines.append(f"  Unexpected agreement_count values: {ev}")

    # --- 7. TRUE AGREEMENT RATIO ---
    lines.append("")
    lines.append("7. TRUE AGREEMENT RATIO (agreement_count / valid_voters)")
    lines.append("-" * W)
    lines.append("  (Corrected for models that abstained by outputting OTHER)")
    for bucket, count in true_agree_dist.items():
        pct = (count / total_accepted) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"  {bucket:>10}: {count:>10,}  ({pct:>5.1f}%)  {bar}")
    lines.append(f"  Mean true agreement ratio: {true_agreement.mean():.4f}")
    lines.append(f"  Median true agreement ratio: {true_agreement.median():.4f}")

    # --- 8. PER-DIALECT AGREEMENT BREAKDOWN ---
    lines.append("")
    lines.append("8. PER-DIALECT AGREEMENT BREAKDOWN (raw agreement count)")
    lines.append("-" * W)
    for d in dialects_order:
        d_total = dialect_dist.get(d, 0)
        lines.append(f"  {d}:")
        ag = dialect_agreement.get(d, {})
        for level in ["5/5", "4/5", "3/5", "2/5", "1/5"]:
            cnt = ag.get(level, 0)
            pct = (cnt / d_total) * 100 if d_total > 0 else 0
            lines.append(f"    {level}: {cnt:>10,}  ({pct:>6.2f}%)")

    # --- 9. MODEL CONFIDENCE STATISTICS ---
    lines.append("")
    lines.append("9. INDIVIDUAL MODEL CONFIDENCE STATISTICS")
    lines.append("-" * W)
    lines.append(f"  {'Model':<25}  {'Mean':>6}  {'Med':>6}  {'Std':>6}  {'P25':>6}  {'P75':>6}  {'P95':>6}  {'Min':>6}  {'Max':>6}")
    for m_id, name in models_names.items():
        s = model_conf_stats.get(m_id, {})
        if s:
            lines.append(f"  {name:<25}  {s['mean']:>6.4f}  {s['median']:>6.4f}  {s['std']:>6.4f}  {s['p25']:>6.4f}  {s['p75']:>6.4f}  {s['p95']:>6.4f}  {s['min']:>6.4f}  {s['max']:>6.4f}")

    # --- 10. ENSEMBLE CONFIDENCE STATISTICS ---
    lines.append("")
    lines.append("10. ENSEMBLE CONFIDENCE STATISTICS (average_confidence)")
    lines.append("-" * W)
    lines.append(f"  Mean:   {ens_conf_stats['mean']:.4f}")
    lines.append(f"  Median: {ens_conf_stats['median']:.4f}")
    lines.append(f"  Std:    {ens_conf_stats['std']:.4f}")
    lines.append(f"  P25:    {ens_conf_stats['p25']:.4f}    P75: {ens_conf_stats['p75']:.4f}    P95: {ens_conf_stats['p95']:.4f}")
    lines.append(f"  Min:    {ens_conf_stats['min']:.4f}    Max: {ens_conf_stats['max']:.4f}")
    lines.append("")
    lines.append("  Confidence Distribution:")
    for bucket, count in ens_conf_dist.items():
        pct = (count / total_accepted) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"    {bucket:>10}: {count:>10,}  ({pct:>5.1f}%)  {bar}")

    # --- 11. PER-MODEL LABEL DISTRIBUTIONS ---
    lines.append("")
    lines.append("11. PER-MODEL LABEL DISTRIBUTIONS")
    lines.append("-" * W)
    for m_id, name in models_names.items():
        dist = model_label_dists.get(m_id, {})
        lines.append(f"  {name}:")
        for d in dialects_order:
            cnt = dist.get(d, 0)
            pct = (cnt / total_accepted) * 100 if total_accepted > 0 else 0
            lines.append(f"    {d:<6}: {cnt:>10,}  ({pct:>6.2f}%)")
        # Show any OTHER labels
        for lbl, cnt in dist.items():
            if lbl not in dialects_order:
                pct = (cnt / total_accepted) * 100
                lines.append(f"    {lbl:<6}: {cnt:>10,}  ({pct:>6.2f}%)")
        lines.append("")

    # --- 12. INTER-MODEL PAIRWISE AGREEMENT ---
    lines.append("12. INTER-MODEL PAIRWISE AGREEMENT")
    lines.append("-" * W)
    for pair, data in pairwise_agreement.items():
        bar = "#" * int(data["pct"] / 2)
        lines.append(f"  {pair:<45}: {data['agree']:>10,}  ({data['pct']:>5.2f}%)  {bar}")

    # --- 13. REJECTION REASONS ---
    lines.append("")
    lines.append("13. REJECTION REASONS")
    lines.append("-" * W)
    if rejection_reasons:
        for k, v in rejection_reasons.items():
            pct = (v / total_rejected) * 100 if total_rejected > 0 else 0
            lines.append(f"  {k:<40}: {v:>10,}  ({pct:>6.2f}%)")
    else:
        lines.append("  None (0 articles rejected)")

    lines.append("")
    lines.append("=" * W)
    lines.append("  END OF REPORT")
    lines.append("=" * W)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # Save files
    master_report = output_dir / "ensemble_report.txt"
    master_stats = output_dir / "ensemble_stats.json"

    with open(master_report, "w", encoding="utf-8") as f:
        f.write(report_text)

    stats_json = {
        "total_processed": total_articles,
        "accepted_count": total_accepted,
        "rejected_count": total_rejected,
        "corpus_text_stats": {
            "total_words": total_words,
            "avg_words": round(avg_words, 1),
            "median_words": round(median_words, 1),
            "std_words": round(std_words, 1),
            "min_words": min_words,
            "max_words": max_words,
        },
        "text_length_distribution": {k: int(v) for k, v in text_len_dist.items()},
        "dialect_distribution": dialect_dist,
        "dialect_word_stats": dialect_word_stats,
        "valid_voter_count_distribution": {str(k): int(v) for k, v in voter_count_dist.items()},
        "raw_agreement_counts": raw_agreement_ratios,
        "agreement_count_coverage": {
            "accounted": int(accounted_for),
            "missing_or_other": int(missing_or_other),
            "nan_count": int(nan_count),
            "other_values": {k: int(v) for k, v in other_values.items()},
        },
        "true_agreement_ratio": {
            "mean": round(float(true_agreement.mean()), 4),
            "median": round(float(true_agreement.median()), 4),
            "distribution": {k: int(v) for k, v in true_agree_dist.items()},
        },
        "per_dialect_agreement": dialect_agreement,
        "model_confidence_stats": model_conf_stats,
        "ensemble_confidence_stats": ens_conf_stats,
        "ensemble_confidence_distribution": {k: int(v) for k, v in ens_conf_dist.items()},
        "per_model_label_distributions": model_label_dists,
        "inter_model_pairwise_agreement": pairwise_agreement,
        "rejection_reasons": rejection_reasons
    }
    with open(master_stats, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print("          ALL PC RESULTS SUCCESSFULLY MERGED!")
    print(f"Master files saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
