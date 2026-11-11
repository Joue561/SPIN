# -*- coding: utf-8 -*-
"""
03_compute_memory_transferability.py
为每个 held-out fold 的 source memory 计算迁移可靠性：

  Cov(κ)  = |{q ∈ Q_src : N_{κ,q} ≥ m}| / |Q_src|
  Stab(κ) = 1 - mean_{q ∈ Q_src^cov} JSD(P_q(y|κ) || P_src(y|κ))     （JSD 以 2 为底，∈[0,1]）
  TransRel(κ) = Rel(κ) · Cov(κ) · Stab(κ)

说明：公式中对所有 Q_src 求平均，但无样本的话题 P_q 无定义；本实现默认只在覆盖话题
（N_{κ,q} ≥ m）上平均（--stab_over covered），未覆盖话题的惩罚已由 Cov 体现。
也可用 --stab_over all：未覆盖话题按 JSD=1 计（最保守）。

Rel(κ) = w_level·level_score + w_prob·norm(dom_prob) + w_conf·avg_conf + w_n·size_score
  level_score: high=1, medium=0.7, low=0.4；norm(p)=(p-1/3)/(2/3)；size_score=min(1, log(1+n)/log(1+n_ref))
注意：只使用 source topics 的统计量，绝不使用 held-out topic 的任何标签。
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ct_common import (LABELS, add_common_args, dist_from_counts, jsd, list_topic_dirs, log_args, read_json,
                       read_jsonl, report_path, select_topics, setup_logger, stage_guard, write_json, write_jsonl)

LEVEL_SCORE = {"high": 1.0, "medium": 0.7, "low": 0.4}


def parse_args():
    ap = argparse.ArgumentParser(description="计算 memory 迁移可靠性 TransRel")
    ap.add_argument("--memory_dir", required=True)
    ap.add_argument("--output_dir", default=None, help="默认与 memory_dir 相同")
    ap.add_argument("--min_topic_count", type=int, default=2, help="公式中的 m：话题内至少多少样本才算覆盖")
    ap.add_argument("--high_transfer_threshold", type=float, default=0.45)
    ap.add_argument("--low_transfer_threshold", type=float, default=0.20)
    ap.add_argument("--stab_over", choices=["covered", "all"], default="covered")
    ap.add_argument("--jsd_form", choices=["divergence", "distance"], default="divergence",
                    help="divergence=JSD 本身（与公式一致）；distance=sqrt(JSD)（区分度更高，可作敏感性分析）")
    ap.add_argument("--smoothing_alpha", type=float, default=0.5, help="话题内分布的加性平滑")
    ap.add_argument("--rel_weights", type=float, nargs=4, default=[0.4, 0.3, 0.2, 0.1],
                    help="Rel 组合权重：level, dom_prob, avg_conf, size")
    ap.add_argument("--size_ref", type=int, default=50)
    add_common_args(ap)
    return ap.parse_args()


def rel_score(u, args):
    wl, wp, wc, wn = args.rel_weights
    level = LEVEL_SCORE.get(u.get("reliability_level"), 0.4)
    p = float(u.get("dominant_label_probability", 1 / 3))
    pn = max(0.0, min(1.0, (p - 1 / 3) / (2 / 3)))
    conf = float(u.get("average_confidence") or 0.0)
    n = int(u.get("source_sample_count_raw", 0))
    size = min(1.0, math.log1p(n) / math.log1p(args.size_ref))
    return float(wl * level + wp * pn + wc * conf + wn * size), {"level_score": level, "dom_prob_norm": pn,
                                                                 "avg_conf": conf, "size_score": size}


def transfer_scores(u, fold_topics, args):
    td = u.get("topic_distribution", {})
    m = args.min_topic_count
    Q = list(fold_topics) if fold_topics else sorted(td)
    covered = [q for q in Q if td.get(q, {}).get("count_raw", 0) >= m]
    cov = len(covered) / len(Q) if Q else 0.0
    pooled = {l: sum(td[q]["label_counts_weighted"][l] for q in td) for l in LABELS}
    p_src = dist_from_counts(pooled, args.smoothing_alpha)
    per_topic = {}
    jsds = []
    for q in Q:
        if q in covered:
            pq = dist_from_counts(td[q]["label_counts_weighted"], args.smoothing_alpha)
            d = jsd(pq, p_src)
            if args.jsd_form == "distance":
                d = math.sqrt(d)
            jsds.append(d)
            per_topic[q] = {"n": td[q]["count_raw"], "dist": dict(zip(LABELS, pq)), "jsd": d,
                            "dominant": LABELS[int(np.argmax(pq))]}
        else:
            if args.stab_over == "all":
                jsds.append(1.0)
            per_topic[q] = {"n": td.get(q, {}).get("count_raw", 0), "covered": False}
    stab = 1.0 - float(np.mean(jsds)) if jsds else 0.0
    doms = [v["dominant"] for v in per_topic.values() if "dominant" in v]
    label_consistency = (Counter(doms).most_common(1)[0][1] / len(doms)) if doms else 0.0
    return cov, stab, per_topic, label_consistency, covered


def main():
    args = parse_args()
    mem_root = Path(args.memory_dir)
    out_root = Path(args.output_dir or args.memory_dir)
    logger = setup_logger("03_compute_memory_transferability", out_root / "logs", args.debug)
    log_args(logger, args)
    dirs = select_topics(list_topic_dirs(mem_root, "memory_units.jsonl"), args.only_heldout_topic)
    if not dirs:
        raise SystemExit(f"{mem_root} 下没有 memory_units.jsonl")
    summary = {}
    for d in dirs:
        build_rep = read_json(d / "memory_build_report.json", {})
        heldout = build_rep.get("heldout_topic", d.name)
        fold_topics = build_rep.get("source_topics", [])
        if heldout in fold_topics and not build_rep.get("intopic"):
            raise RuntimeError(f"[{heldout}] source_topics 含 held-out，终止")
        odir = out_root / d.name
        files = [odir / "memory_transferability.jsonl", odir / "memory_transferability_report.json",
                 odir / "high_transfer_memory.jsonl", odir / "low_transfer_memory.jsonl"]
        if not stage_guard(files, args.overwrite, args.skip_existing, logger, f"03:{heldout}"):
            summary[heldout] = {"skipped": True}
            continue
        units = read_jsonl(d / "memory_units.jsonl", logger)
        rows = []
        for u in units:
            try:
                rel, rel_parts = rel_score(u, args)
                cov, stab, per_topic, cons, covered = transfer_scores(u, fold_topics, args)
            except Exception as e:
                logger.warning("memory %s 计算失败: %s", u.get("memory_id"), e)
                continue
            tr = rel * cov * stab
            rows.append({
                "memory_id": u["memory_id"], "memory_type": u["memory_type"], "pattern_key": u["pattern_key"],
                "pattern_name": u["pattern_name"], "heldout_topic": heldout, "source_topics": fold_topics,
                "dominant_label": u["dominant_label"], "dominant_label_probability": u["dominant_label_probability"],
                "reliability_level": u["reliability_level"], "suggested_usage": u["suggested_usage"],
                "source_sample_count_raw": u["source_sample_count_raw"],
                "Rel": rel, "Rel_components": rel_parts, "Cov": cov, "Stab": stab, "TransRel": tr,
                "covered_topics": covered, "n_covered_topics": len(covered),
                "cross_topic_label_consistency": cons, "per_topic": per_topic,
                "transfer_level": ("high" if tr >= args.high_transfer_threshold else
                                   "low" if tr < args.low_transfer_threshold else "medium"),
            })
        rows.sort(key=lambda r: -r["TransRel"])
        high = [r for r in rows if r["TransRel"] >= args.high_transfer_threshold]
        low = [r for r in rows if r["TransRel"] < args.low_transfer_threshold]
        write_jsonl(files[0], rows)
        write_jsonl(files[2], high)
        write_jsonl(files[3], low)
        tr_arr = np.array([r["TransRel"] for r in rows]) if rows else np.zeros(1)
        rep = {
            "heldout_topic": heldout, "source_topics": fold_topics, "n_memory": len(rows),
            "n_high_transfer": len(high), "n_low_transfer": len(low),
            "high_transfer_ratio": len(high) / len(rows) if rows else 0.0,
            "TransRel_stats": {"mean": float(tr_arr.mean()), "median": float(np.median(tr_arr)),
                               "p25": float(np.percentile(tr_arr, 25)), "p75": float(np.percentile(tr_arr, 75)),
                               "max": float(tr_arr.max())},
            "mean_Cov": float(np.mean([r["Cov"] for r in rows])) if rows else 0.0,
            "mean_Stab": float(np.mean([r["Stab"] for r in rows])) if rows else 0.0,
            "mean_Rel": float(np.mean([r["Rel"] for r in rows])) if rows else 0.0,
            "by_type_mean_TransRel": {t: float(np.mean([r["TransRel"] for r in rows if r["memory_type"] == t]))
                                      for t in sorted({r["memory_type"] for r in rows})},
            "top10": [{k: r[k] for k in ("memory_id", "pattern_name", "dominant_label", "TransRel", "Cov", "Stab", "Rel")}
                      for r in rows[:10]],
            "params": {"min_topic_count": args.min_topic_count, "high_transfer_threshold": args.high_transfer_threshold,
                       "low_transfer_threshold": args.low_transfer_threshold, "stab_over": args.stab_over, "jsd_form": args.jsd_form,
                       "smoothing_alpha": args.smoothing_alpha, "rel_weights": args.rel_weights},
            "note": "仅基于 source topics 统计；held-out topic 标签未参与任何计算。",
        }
        write_json(files[1], rep)
        summary[heldout] = {k: rep[k] for k in ("source_topics", "n_memory", "n_high_transfer", "high_transfer_ratio",
                                                "mean_Cov", "mean_Stab")}
        logger.info("[%s] memory=%d | high(>=%.2f)=%d | low=%d | mean TransRel=%.3f Cov=%.3f Stab=%.3f",
                    heldout, len(rows), args.high_transfer_threshold, len(high), len(low),
                    rep["TransRel_stats"]["mean"], rep["mean_Cov"], rep["mean_Stab"])
    write_json(report_path(out_root / "all_transferability_summary.json", args.overwrite), summary)
    logger.info("完成")


if __name__ == "__main__":
    main()