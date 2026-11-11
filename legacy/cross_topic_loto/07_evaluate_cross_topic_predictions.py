# -*- coding: utf-8 -*-
"""
07_evaluate_cross_topic_predictions.py
计算每个 held-out topic × method 的指标、复杂子集指标、跨话题迁移增益，并生成论文总表。

默认 --eval_on_intersection true：同一 held-out topic 下，只在所有 method 均有有效预测的样本上比较，
保证不同方法的比较样本完全一致（未预测样本数会在报告中列出）。
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ct_common import (ALL_METHODS, LABELS, RAG_METHODS, add_common_args, complex_subset_flags, compute_metrics,
                       load_gold, load_predictions, load_retrieval, log_args, report_path, setup_logger,
                       stage_guard, str2bool, write_json)

SUBSET_NAMES = {
    "short_text": "短文本", "is_ambiguous_current": "模糊评论", "depth_ge_3": "深层回复(depth>=3)",
    "prop_stance_reversal": "语义传导=stance_reversal", "prop_sarcastic_reversal": "语义传导=sarcastic_reversal",
    "trel_reply_author_only": "目标关系=reply_author_only", "trel_topic_shift": "目标关系=topic_shift",
    "reply_mock_parent": "回复关系=mock_parent", "reply_disagree_with_parent": "回复关系=disagree_with_parent",
}


def parse_args():
    ap = argparse.ArgumentParser(description="评估跨话题预测")
    ap.add_argument("--predictions_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--retrieval_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--baseline", default="dynamic_semantic", help="transfer_gain 的基线方法")
    ap.add_argument("--eval_on_intersection", type=str2bool, default=True)
    ap.add_argument("--short_text_len", type=int, default=10)
    ap.add_argument("--min_subset_size", type=int, default=5, help="子集样本数少于此值时仍输出但标记 small_sample")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    add_common_args(ap)
    return ap.parse_args()


def paired_bootstrap(golds, pa, pb, n, seed):
    """macro-F1(a) - macro-F1(b) 的配对 bootstrap；返回 delta、95%CI、p(Δ<=0)。"""
    lid = {l: i for i, l in enumerate(LABELS)}
    golds = np.array([lid[x] for x in golds])
    pa = np.array([lid[x] for x in pa])
    pb = np.array([lid[x] for x in pb])
    k = len(golds)
    if k == 0:
        return None

    def f(g, p):  # 向量化 macro-F1（与 sklearn zero_division=0 一致）
        cm = np.bincount(g * 3 + p, minlength=9).reshape(3, 3)
        tp = np.diag(cm).astype(float)
        denom = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * tp, denom, out=np.zeros(3), where=denom > 0)
        return f1.mean()
    delta = f(golds, pa) - f(golds, pb)
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        ds.append(f(golds[idx], pa[idx]) - f(golds[idx], pb[idx]))
    ds = np.array(ds)
    return {"delta_macro_f1": float(delta), "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
            "p_delta_le_0": float((ds <= 0).mean()), "n": k}


def mcnemar(golds, pa, pb):
    from scipy.stats import binomtest
    a_right = [x == g for x, g in zip(pa, golds)]
    b_right = [x == g for x, g in zip(pb, golds)]
    n01 = sum(1 for a, b in zip(a_right, b_right) if a and not b)
    n10 = sum(1 for a, b in zip(a_right, b_right) if b and not a)
    if n01 + n10 == 0:
        return {"a_only_correct": 0, "b_only_correct": 0, "p_value": 1.0}
    return {"a_only_correct": n01, "b_only_correct": n10,
            "p_value": float(binomtest(n01, n01 + n10, 0.5).pvalue)}


def main():
    args = parse_args()
    out = Path(args.output_dir)
    logger = setup_logger("07_evaluate_cross_topic_predictions", out / "logs", args.debug)
    log_args(logger, args)
    files = [out / n for n in ("summary_by_topic_method.csv", "summary_average.csv", "complex_subset_metrics.json",
                               "confusion_matrices.json", "transfer_gain_report.json", "final_summary_table.md")]
    if not stage_guard(files, args.overwrite, args.skip_existing, logger, "07"):
        return

    gold = load_gold(args.splits_dir, logger)
    preds = load_predictions(args.predictions_dir, logger)
    rets = load_retrieval(args.retrieval_dir, logger)
    topics = sorted({t for t, _ in gold})
    if args.only_heldout_topic:
        topics = [t for t in topics if t in args.only_heldout_topic]
    methods = [m for m in args.methods if any((t, m) in preds for t in topics)]
    logger.info("held-out topics=%s | 有预测的 methods=%s", topics, methods)
    if not methods:
        raise SystemExit("没有任何预测结果可评估")

    rows, cms, subset_out, coverage_info = [], {}, {}, {}
    eval_sets = {}
    pooled = defaultdict(lambda: {"g": [], "p": [], "flags": []})
    for t in topics:
        uids_all = sorted(u for (tt, u) in gold if tt == t)
        avail = {m: set(preds.get((t, m), {})) for m in methods}
        if args.eval_on_intersection:
            common = set(uids_all)
            for m in methods:
                if avail[m]:
                    common &= avail[m]
            eval_uids = sorted(common)
        else:
            eval_uids = None
        coverage_info[t] = {"n_gold": len(uids_all), "n_eval_intersection": len(eval_uids) if eval_uids is not None else None,
                            "n_pred_by_method": {m: len(avail[m]) for m in methods}}
        eval_sets[t] = eval_uids
        for m in methods:
            pm = preds.get((t, m), {})
            if not pm:
                continue
            uids = eval_uids if eval_uids is not None else sorted(u for u in uids_all if u in pm)
            g = [gold[(t, u)]["_gold"] for u in uids]
            p = [pm[u]["predicted_label"] for u in uids]
            met = compute_metrics(g, p)
            rm = rets.get((t, m), {})
            rm = rm or rets.get((t, {"context_memory": "source_rag"}.get(m, m)), {})
            if (m in RAG_METHODS or m == "context_memory") and rm:
                feats = [rm[u]["rag_features"] for u in uids if u in rm]
                cov = float(np.mean([f["rag_memory_count"] > 0 for f in feats])) if feats else 0.0
                htr = float(np.mean([f.get("rag_htr", 0.0) for f in feats])) if feats else 0.0
                avg_tr = float(np.mean([f["rag_avg_transferability"] for f in feats])) if feats else 0.0
            else:
                cov = htr = avg_tr = float("nan")
            used_rag = [bool(pm[u].get("used_rag")) for u in uids]
            rows.append({"heldout_topic": t, "method": m, "n": met.get("n", 0),
                         "n_missing_pred": len(uids_all) - len(set(pm) & set(uids_all)),
                         "accuracy": met.get("accuracy"), "macro_f1": met.get("macro_f1"), "f_avg": met.get("f_avg"),
                         "weighted_f1": met.get("weighted_f1"),
                         **{f"{l}_{k}": met.get(f"{l}_{k}") for l in LABELS for k in ("precision", "recall", "f1")},
                         "coverage_test": cov, "avg_htr": htr, "avg_transrel": avg_tr,
                         "used_rag_rate": float(np.mean(used_rag)) if (used_rag and m in RAG_METHODS) else float("nan")})
            cms[f"{t}::{m}"] = met.get("confusion_matrix")
            # 复杂子集
            flags = [complex_subset_flags(gold[(t, u)], args.short_text_len) for u in uids]
            pooled[m]["g"] += g
            pooled[m]["p"] += p
            pooled[m]["flags"] += flags
            sub = {}
            for s in SUBSET_NAMES:
                idx = [i for i, f in enumerate(flags) if f[s]]
                if not idx:
                    continue
                mm = compute_metrics([g[i] for i in idx], [p[i] for i in idx])
                mm.pop("confusion_matrix", None)
                mm["small_sample"] = len(idx) < args.min_subset_size
                sub[s] = mm
            subset_out.setdefault(t, {})[m] = sub

    df = pd.DataFrame(rows)
    # transfer gain
    base = {r["heldout_topic"]: r["macro_f1"] for r in rows if r["method"] == args.baseline}
    df["transfer_gain"] = [r["macro_f1"] - base[r["heldout_topic"]] if r["heldout_topic"] in base else float("nan")
                           for r in rows]
    order = {m: i for i, m in enumerate(ALL_METHODS)}
    df = df.sort_values(["heldout_topic", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
    df.to_csv(files[0], index=False, encoding="utf-8-sig")

    avg = df.groupby("method").agg(
        n_topics=("heldout_topic", "nunique"), avg_accuracy=("accuracy", "mean"), avg_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"), avg_weighted_f1=("weighted_f1", "mean"), avg_f_avg=("f_avg", "mean"),
        avg_favor_f1=("favor_f1", "mean"), avg_against_f1=("against_f1", "mean"), avg_none_f1=("none_f1", "mean"),
        avg_transfer_gain=("transfer_gain", "mean"), avg_coverage=("coverage_test", "mean"),
        avg_htr=("avg_htr", "mean")).reset_index()
    avg = avg.sort_values("method", key=lambda s: s.map(order))
    # pooled（所有 held-out 样本合并）作为补充
    pooled_rows = {}
    for m, d in pooled.items():
        mm = compute_metrics(d["g"], d["p"])
        pooled_rows[m] = {"pooled_accuracy": mm["accuracy"], "pooled_macro_f1": mm["macro_f1"]}
    avg["pooled_accuracy"] = avg["method"].map(lambda m: pooled_rows.get(m, {}).get("pooled_accuracy"))
    avg["pooled_macro_f1"] = avg["method"].map(lambda m: pooled_rows.get(m, {}).get("pooled_macro_f1"))
    avg.to_csv(files[1], index=False, encoding="utf-8-sig")

    # pooled 复杂子集
    pooled_subsets = {}
    for m, d in pooled.items():
        pooled_subsets[m] = {}
        for s in SUBSET_NAMES:
            idx = [i for i, f in enumerate(d["flags"]) if f[s]]
            if idx:
                mm = compute_metrics([d["g"][i] for i in idx], [d["p"][i] for i in idx])
                mm.pop("confusion_matrix", None)
                mm["small_sample"] = len(idx) < args.min_subset_size
                pooled_subsets[m][s] = mm
    # 子集上的 RAG 增益
    subset_gain = {}
    if args.baseline in pooled_subsets:
        for m in pooled_subsets:
            if m == args.baseline:
                continue
            subset_gain[m] = {s: {"macro_f1_gain": v["macro_f1"] - pooled_subsets[args.baseline][s]["macro_f1"],
                                  "accuracy_gain": v["accuracy"] - pooled_subsets[args.baseline][s]["accuracy"],
                                  "n": v["n"]}
                              for s, v in pooled_subsets[m].items() if s in pooled_subsets[args.baseline]}
    write_json(files[2], {"subset_definitions": SUBSET_NAMES, "short_text_len": args.short_text_len,
                          "by_topic_method": subset_out, "pooled_by_method": pooled_subsets,
                          f"pooled_gain_vs_{args.baseline}": subset_gain})
    write_json(files[3], cms)

    # 配对显著性
    comparisons = [("source_rag", args.baseline), ("pattern_only_rag", args.baseline),
                   ("transferable_rag", args.baseline), ("transferable_rag", "source_rag"),
                   ("source_rag", "random_memory_rag"), ("pattern_only_rag", "random_memory_rag"),
                   ("transferable_rag", "random_memory_rag"), ("random_memory_rag", args.baseline),
                   (args.baseline, "current_only"),
                   # rag_empty 对照：指令效应 = rag_empty - baseline；记忆内容效应 = X_rag - rag_empty
                   ("rag_empty", args.baseline), ("source_rag", "rag_empty"), ("pattern_only_rag", "rag_empty"),
                   ("transferable_rag", "rag_empty"), ("random_memory_rag", "rag_empty"),
                   # 路线 B：去语义字段消融 & 示例式检索
                   ("context_only", "current_only"), ("context_memory", "context_only"),
                   (args.baseline, "context_only"), ("context_memory", args.baseline),
                   ("example_rag", args.baseline), ("example_rag", "rag_empty"), ("example_rag", "source_rag"),
                   ("example_rag", "random_memory_rag"),
                   # 同话题对照：intopic_rag − baseline = 同话题记忆增益；intopic_rag − source_rag = 跨话题迁移损失
                   ("intopic_rag", args.baseline), ("intopic_rag", "source_rag"), ("intopic_rag", "rag_empty"),
                   ("intopic_rag", "random_memory_rag")]
    comp_out = {}
    for a, b in comparisons:
        if a not in pooled or b not in pooled:
            continue
        ga, gb = [], []
        pa, pb = [], []
        for t in topics:
            A, B = preds.get((t, a), {}), preds.get((t, b), {})
            uids = eval_sets[t] if eval_sets[t] is not None else sorted(set(A) & set(B))
            uids = [u for u in uids if u in A and u in B]
            ga += [gold[(t, u)]["_gold"] for u in uids]
            pa += [A[u]["predicted_label"] for u in uids]
            pb += [B[u]["predicted_label"] for u in uids]
        per_topic = {}
        for t in topics:
            ra = df[(df.heldout_topic == t) & (df.method == a)]
            rb = df[(df.heldout_topic == t) & (df.method == b)]
            if len(ra) and len(rb):
                per_topic[t] = float(ra.macro_f1.iloc[0] - rb.macro_f1.iloc[0])
        wins = sum(1 for v in per_topic.values() if v > 0)
        comp_out[f"{a}_vs_{b}"] = {
            "avg_macro_f1_diff_over_topics": float(np.mean(list(per_topic.values()))) if per_topic else None,
            "per_topic_diff": per_topic, "wins": wins, "n_topics": len(per_topic),
            "holds_on_average": bool(np.mean(list(per_topic.values())) > 0) if per_topic else None,
            "pooled_paired_bootstrap": paired_bootstrap(ga, pa, pb, args.n_bootstrap, args.seed),
            "pooled_mcnemar": mcnemar(ga, pa, pb),
        }
    tg = {"baseline": args.baseline,
          "definition": "transfer_gain = macro_f1(method) - macro_f1(baseline)，按 held-out topic 计算",
          "by_topic": {t: {r["method"]: r["transfer_gain"] for _, r in df[df.heldout_topic == t].iterrows()}
                       for t in topics},
          "average": {r["method"]: r["avg_transfer_gain"] for _, r in avg.iterrows()},
          "pairwise_comparisons": comp_out, "eval_coverage": coverage_info,
          "eval_on_intersection": args.eval_on_intersection}
    write_json(files[4], tg)

    # Markdown 总表
    def fmt(x, nd=4):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
    lines = ["# 跨目标泛化实验总表（Leave-One-Topic-Out）", "",
             f"基线：`{args.baseline}`；评估口径：{'各方法共同预测样本交集' if args.eval_on_intersection else '各方法自身预测样本'}", "",
             "| heldout_topic | method | accuracy | macro_f1 | f_avg | weighted_f1 | favor_f1 | against_f1 | none_f1 | transfer_gain | coverage_test | avg_htr |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.heldout_topic} | {r.method} | {fmt(r.accuracy)} | {fmt(r.macro_f1)} | {fmt(r.f_avg)} | {fmt(r.weighted_f1)} | "
                     f"{fmt(r.favor_f1)} | {fmt(r.against_f1)} | {fmt(r.none_f1)} | {fmt(r.transfer_gain)} | "
                     f"{fmt(r.coverage_test)} | {fmt(r.avg_htr)} |")
    lines += ["", "## Average（对 held-out topic 取宏平均）", "",
              "| method | avg_accuracy | avg_macro_f1 | avg_f_avg | avg_weighted_f1 | avg_transfer_gain | avg_coverage | avg_htr |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in avg.iterrows():
        lines.append(f"| {r.method} | {fmt(r.avg_accuracy)} | {fmt(r.avg_macro_f1)} | {fmt(r.avg_f_avg)} | {fmt(r.avg_weighted_f1)} | "
                     f"{fmt(r.avg_transfer_gain)} | {fmt(r.avg_coverage)} | {fmt(r.avg_htr)} |")
    lines += ["", "## 配对比较（pooled bootstrap / McNemar）", "",
              "| 比较 | 平均ΔmacroF1 | 胜出topic数 | pooled Δ | 95%CI | p(Δ≤0) | McNemar p |", "|---|---|---|---|---|---|---|"]
    for k, v in comp_out.items():
        bs = v["pooled_paired_bootstrap"] or {}
        lines.append(f"| {k} | {fmt(v['avg_macro_f1_diff_over_topics'])} | {v['wins']}/{v['n_topics']} | "
                     f"{fmt(bs.get('delta_macro_f1'))} | [{fmt((bs.get('ci95') or [None, None])[0])}, "
                     f"{fmt((bs.get('ci95') or [None, None])[1])}] | {fmt(bs.get('p_delta_le_0'), 3)} | "
                     f"{fmt(v['pooled_mcnemar']['p_value'], 3)} |")
    files[5].write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_path(out / "evaluation_report.json", args.overwrite),
               {"topics": topics, "methods": methods, "baseline": args.baseline, "eval_coverage": coverage_info,
                "outputs": [str(f) for f in files]})
    logger.info("\n%s", avg[["method", "avg_accuracy", "avg_macro_f1", "avg_transfer_gain", "avg_coverage", "avg_htr"]]
                .to_string(index=False))
    logger.info("完成，总表见 %s", files[5])


if __name__ == "__main__":
    main()