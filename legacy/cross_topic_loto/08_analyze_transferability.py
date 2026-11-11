# -*- coding: utf-8 -*-
"""
08_analyze_transferability.py
分析 memory 迁移可靠性与预测正确性的关系，并自动生成论文结论所需数据：

 1. TransRel 分桶准确率（样本级：按检索结果平均 TransRel；memory 级：按被检索 memory 的 TransRel）
 2. HTR（高迁移 memory 占比）与预测正确率的相关性
 3. 高 TransRel memory 的 top patterns（跨 fold 汇总）
 4. 低 TransRel 检索下的失败案例
 5. 不同 memory_type 的贡献
 6. RAG 帮助案例（baseline 错 → RAG 对）
 7. RAG 误导案例（baseline 对 → RAG 错）
 + RAG 证据有效性（memory 主导立场与 gold 一致率、模型跟随率），用于证明“RAG 不是装饰”
 + analysis_report.json / analysis_report.md：九个论文问题的自动结论
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ct_common import (ALL_METHODS, F_PAT, F_PROP, F_REPLY, F_TREL, LABELS, RAG_METHODS, add_common_args,
                       complex_subset_flags, compute_metrics, list_topic_dirs, load_gold, load_predictions,
                       load_retrieval, log_args, read_json, read_jsonl, report_path, safe_mean, setup_logger,
                       short_text, stage_guard, str2bool, write_json, write_jsonl)

SUBSET_ZH = {
    "short_text": "短文本", "is_ambiguous_current": "模糊评论", "depth_ge_3": "深层回复",
    "prop_stance_reversal": "立场反转", "prop_sarcastic_reversal": "反讽反转",
    "trel_reply_author_only": "仅针对回复者", "trel_topic_shift": "话题转移",
    "reply_mock_parent": "嘲讽父评论", "reply_disagree_with_parent": "反驳父评论",
}


def parse_args():
    ap = argparse.ArgumentParser(description="迁移可靠性深入分析")
    ap.add_argument("--predictions_dir", required=True)
    ap.add_argument("--retrieval_dir", required=True)
    ap.add_argument("--memory_dir", required=True)
    ap.add_argument("--intopic_memory_dir", default=None,
                    help="intopic 记忆目录，默认为 memory_dir 同级的 memories_intopic（不存在则忽略）")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--splits_dir", default=None, help="默认 predictions_dir 的同级 splits 目录")
    ap.add_argument("--baseline", default="dynamic_semantic")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="参与分析的方法（默认：有预测的全部方法）。与 07 一致，只在这些方法共同预测的样本上分析")
    ap.add_argument("--eval_on_intersection", type=str2bool, default=True)
    ap.add_argument("--rag_methods", nargs="+", default=RAG_METHODS)
    ap.add_argument("--structured_methods", nargs="+", default=["source_rag", "pattern_only_rag", "transferable_rag"])
    ap.add_argument("--bins", type=float, nargs="+", default=[0.0, 0.2, 0.35, 0.5, 0.65, 1.01],
                    help="TransRel 分桶边界")
    ap.add_argument("--low_transfer_threshold", type=float, default=0.35)
    ap.add_argument("--short_text_len", type=int, default=10)
    ap.add_argument("--max_cases", type=int, default=300, help="每类案例最多输出条数")
    ap.add_argument("--min_bucket_n", type=int, default=10)
    add_common_args(ap)
    return ap.parse_args()


def bucket_of(x, bins):
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= x < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f">={bins[-1]:.2f}"


def corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None, "n": int(len(x))}
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {"pearson_r(point_biserial)": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]), "n": int(len(x))}


def case_record(g, base_p, rag_p, ret, method, mems):
    return {
        "heldout_topic": g.get("topic"), "sample_uid": g["sample_uid"], "method": method,
        "gold": g["_gold"], "baseline_pred": base_p.get("predicted_label") if base_p else None,
        "rag_pred": rag_p.get("predicted_label") if rag_p else None,
        "main_target": g.get("main_target"),
        "parent_text": short_text(g.get("parent_text", ""), 120), "current_text": short_text(g.get("current_text", ""), 160),
        "depth": g.get("depth"), F_REPLY: g.get(F_REPLY), F_PROP: g.get(F_PROP), F_TREL: g.get(F_TREL),
        F_PAT: g.get(F_PAT),
        "baseline_reason": (base_p or {}).get("reason_short"), "rag_reason": (rag_p or {}).get("reason_short"),
        "used_memory_ids": (rag_p or {}).get("used_memory_ids"),
        "retrieved_memory_ids": (ret or {}).get("retrieved_memory_ids"),
        "retrieved_memory_names": [mems.get(i, {}).get("pattern_name") for i in (ret or {}).get("retrieved_memory_ids", [])],
        "retrieved_memory_dominant_labels": (ret or {}).get("retrieved_memory_dominant_labels"),
        "retrieved_transrel": (ret or {}).get("retrieved_transrel"),
        "rag_features": {k: v for k, v in ((ret or {}).get("rag_features") or {}).items()
                         if k in ("rag_dominant_label", "rag_dominant_label_probability", "rag_avg_transferability",
                                  "rag_htr", "rag_evidence_consistency", "rag_memory_count")},
        "complex_flags": [k for k, v in complex_subset_flags(g).items() if v],
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    logger = setup_logger("08_analyze_transferability", out / "logs", args.debug)
    log_args(logger, args)
    names = ["transferability_bucket_metrics.json", "top_transferable_patterns.csv", "low_transfer_failure_cases.jsonl",
             "rag_help_cases.jsonl", "rag_harm_cases.jsonl", "memory_type_contribution.json",
             "rag_evidence_validity.json", "htr_correlation.json", "analysis_report.json", "analysis_report.md"]
    files = {n: out / n for n in names}
    if not stage_guard(list(files.values()), args.overwrite, args.skip_existing, logger, "08"):
        return
    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(args.predictions_dir).parent / "splits"
    gold = load_gold(splits_dir, logger)
    preds = load_predictions(args.predictions_dir, logger)
    rets = load_retrieval(args.retrieval_dir, logger)
    topics = sorted({t for t, _ in gold})
    if args.only_heldout_topic:
        topics = [t for t in topics if t in args.only_heldout_topic]

    # ---- memory 与 TransRel（按 fold）----
    mem_by_topic, tr_by_topic, tr_rows_all = {}, {}, []
    for d in list_topic_dirs(args.memory_dir, "memory_units.jsonl"):
        rep = read_json(d / "memory_build_report.json", {})
        t = rep.get("heldout_topic", d.name)
        mem_by_topic[t] = {m["memory_id"]: m for m in read_jsonl(d / "memory_units.jsonl", logger)}
        rows = read_jsonl(d / "memory_transferability.jsonl", logger)
        tr_by_topic[t] = {r["memory_id"]: r for r in rows}
        tr_rows_all += rows
    # intopic 记忆（memory_id 带 IN_ 前缀，不与 source 记忆冲突；不参与跨话题稳定模式统计）
    idir = Path(args.intopic_memory_dir or (Path(args.memory_dir).parent / "memories_intopic"))
    for d in list_topic_dirs(idir, "memory_units.jsonl"):
        rep = read_json(d / "memory_build_report.json", {})
        t = rep.get("heldout_topic", d.name)
        mem_by_topic.setdefault(t, {}).update({m["memory_id"]: m for m in read_jsonl(d / "memory_units.jsonl", logger)})
        tr_by_topic.setdefault(t, {}).update({r["memory_id"]: r for r in read_jsonl(d / "memory_transferability.jsonl", logger)})

    wanted = args.methods or ALL_METHODS
    preds = {k: v for k, v in preds.items() if k[1] in wanted}
    if args.eval_on_intersection:
        # 与 07 相同：每个 held-out topic 只保留所有方法都有预测的样本，保证各方法比较的样本一致
        for t in topics:
            ms = [m for m in wanted if (t, m) in preds]
            if not ms:
                continue
            common = set.intersection(*[set(preds[(t, m)]) for m in ms])
            for m in ms:
                preds[(t, m)] = {u: p for u, p in preds[(t, m)].items() if u in common}
            logger.info("[%s] 交集样本数=%d（方法：%s）", t, len(common), ms)
    rag_methods = [m for m in args.rag_methods if any((t, m) in preds for t in topics)]
    methods = [m for m in ALL_METHODS if any((t, m) in preds for t in topics)]
    logger.info("topics=%s | methods=%s", topics, methods)

    # ---- 样本级表 ----
    srows = []
    for t in topics:
        for m in rag_methods:
            pm, rm = preds.get((t, m), {}), rets.get((t, m), {})
            for uid, p in pm.items():
                g = gold.get((t, uid))
                r = rm.get(uid)
                if g is None or r is None:
                    continue
                f = r["rag_features"]
                srows.append({"topic": t, "method": m, "uid": uid, "correct": int(p["predicted_label"] == g["_gold"]),
                              "avg_transrel": f["rag_avg_transferability"], "htr": f.get("rag_htr", 0.0),
                              "k": f["rag_memory_count"], "consistency": f["rag_evidence_consistency"],
                              "rag_dom": f.get("rag_dominant_label"), "gold": g["_gold"], "pred": p["predicted_label"],
                              "used_rag": bool(p.get("used_rag")), "entropy": f["rag_evidence_entropy"]})
    sdf = pd.DataFrame(srows)

    # 1. 分桶
    bucket_out = {"bins": args.bins, "sample_level": {}, "memory_level": {}}
    if len(sdf):
        sdf["bucket"] = sdf["avg_transrel"].map(lambda x: bucket_of(x, args.bins))
        for m, sub in sdf.groupby("method"):
            bucket_out["sample_level"][m] = {
                b: {"n": int(len(s)), "accuracy": float(s.correct.mean()), "small_sample": len(s) < args.min_bucket_n}
                for b, s in sorted(sub.groupby("bucket"), key=lambda x: x[0])}
        # memory 级：每次检索到某 memory，记录该次预测是否正确、memory 主导立场是否等于 gold
        mrows = []
        for t in topics:
            for m in rag_methods:
                for uid, r in rets.get((t, m), {}).items():
                    p = preds.get((t, m), {}).get(uid)
                    g = gold.get((t, uid))
                    if p is None or g is None:
                        continue
                    for rank, mid in enumerate(r["retrieved_memory_ids"]):
                        if mid not in mem_by_topic.get(t, {}):
                            continue  # example_rag 的示例不是 memory 单元，不参与 memory 级分析
                        tr = tr_by_topic.get(t, {}).get(mid, {})
                        mem = mem_by_topic.get(t, {}).get(mid, {})
                        mrows.append({"topic": t, "method": m, "memory_id": mid, "rank": rank,
                                      "memory_type": mem.get("memory_type"), "pattern_name": mem.get("pattern_name"),
                                      "transrel": tr.get("TransRel", 0.0),
                                      "pred_correct": int(p["predicted_label"] == g["_gold"]),
                                      "evidence_correct": int(mem.get("dominant_label") == g["_gold"]),
                                      "cited": int(mid in (p.get("used_memory_ids") or []))})
        mdf = pd.DataFrame(mrows)
        if len(mdf):
            mdf["bucket"] = mdf["transrel"].map(lambda x: bucket_of(x, args.bins))
            for m, sub in mdf.groupby("method"):
                bucket_out["memory_level"][m] = {
                    b: {"n_retrievals": int(len(s)), "n_unique_memory": int(s.memory_id.nunique()),
                        "prediction_accuracy": float(s.pred_correct.mean()),
                        "evidence_accuracy(memory主导立场==gold)": float(s.evidence_correct.mean()),
                        "cite_rate": float(s.cited.mean())}
                    for b, s in sorted(sub.groupby("bucket"), key=lambda x: x[0])}
            st = mdf[mdf.method.isin(args.structured_methods)]
            bucket_out["memory_level_structured_pooled_corr"] = {
                "transrel_vs_evidence_correct": corr(st.transrel, st.evidence_correct),
                "transrel_vs_pred_correct": corr(st.transrel, st.pred_correct)}
        else:
            mdf = pd.DataFrame(columns=["method", "memory_type", "transrel", "evidence_correct", "pred_correct", "rank"])
    else:
        mdf = pd.DataFrame(columns=["method", "memory_type", "transrel", "evidence_correct", "pred_correct", "rank"])
    write_json(files["transferability_bucket_metrics.json"], bucket_out)

    # 2. HTR 相关性
    htr_out = {}
    if len(sdf):
        for m, sub in sdf.groupby("method"):
            htr_out[m] = {"htr_vs_correct": corr(sub.htr, sub.correct),
                          "avg_transrel_vs_correct": corr(sub.avg_transrel, sub.correct),
                          "accuracy_htr_ge_0.5": float(sub[sub.htr >= 0.5].correct.mean()) if (sub.htr >= 0.5).any() else None,
                          "accuracy_htr_lt_0.5": float(sub[sub.htr < 0.5].correct.mean()) if (sub.htr < 0.5).any() else None,
                          "n_htr_ge_0.5": int((sub.htr >= 0.5).sum()), "n_htr_lt_0.5": int((sub.htr < 0.5).sum())}
        st = sdf[sdf.method.isin(args.structured_methods)]
        htr_out["structured_pooled"] = {"htr_vs_correct": corr(st.htr, st.correct),
                                        "avg_transrel_vs_correct": corr(st.avg_transrel, st.correct)}
    write_json(files["htr_correlation.json"], htr_out)

    # 3. top transferable patterns
    pat_rows = []
    if tr_rows_all:
        tdf = pd.DataFrame([{k: r.get(k) for k in ("memory_type", "pattern_key", "pattern_name", "heldout_topic",
                                                     "dominant_label", "dominant_label_probability", "TransRel", "Cov",
                                                     "Stab", "Rel", "cross_topic_label_consistency",
                                                     "source_sample_count_raw")} for r in tr_rows_all])
        for (mt, pk), s in tdf.groupby(["memory_type", "pattern_key"]):
            labs = Counter(s.dominant_label)
            pat_rows.append({"memory_type": mt, "pattern_key": pk, "pattern_name": s.pattern_name.iloc[0],
                             "n_folds": int(s.heldout_topic.nunique()),
                             "mean_TransRel": float(s.TransRel.mean()), "min_TransRel": float(s.TransRel.min()),
                             "mean_Cov": float(s.Cov.mean()), "mean_Stab": float(s.Stab.mean()),
                             "mean_Rel": float(s.Rel.mean()),
                             "dominant_label": labs.most_common(1)[0][0],
                             "dominant_label_consistent_across_folds": len(labs) == 1,
                             "mean_dominant_prob": float(s.dominant_label_probability.mean()),
                             "mean_within_fold_topic_label_consistency": float(s.cross_topic_label_consistency.mean()),
                             "mean_sample_count": float(s.source_sample_count_raw.mean())})
    pdf = pd.DataFrame(pat_rows)
    if len(pdf):
        # 结合检索效果：该 pattern 被检索时 memory 证据正确率
        if len(mdf):
            mm = mdf.merge(pd.DataFrame([{"memory_id": r["memory_id"], "pattern_key": r["pattern_key"],
                                          "memory_type2": r["memory_type"]} for r in tr_rows_all]).drop_duplicates(),
                           on="memory_id", how="left")
            ev = mm.groupby(["memory_type2", "pattern_key"]).agg(
                n_retrieved=("evidence_correct", "size"), retrieved_evidence_accuracy=("evidence_correct", "mean"),
                retrieved_pred_accuracy=("pred_correct", "mean")).reset_index().rename(columns={"memory_type2": "memory_type"})
            pdf = pdf.merge(ev, on=["memory_type", "pattern_key"], how="left")
        pdf = pdf.sort_values(["mean_TransRel", "n_folds"], ascending=False)
        pdf.insert(0, "rank", range(1, len(pdf) + 1))
    pdf.to_csv(files["top_transferable_patterns.csv"], index=False, encoding="utf-8-sig")

    # 4/6/7. 案例
    low_cases, help_cases, harm_cases = [], [], []
    help_cnt, harm_cnt, subset_size = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for t in topics:
        base = preds.get((t, args.baseline), {})
        mems = mem_by_topic.get(t, {})
        for m in rag_methods:
            pm, rm = preds.get((t, m), {}), rets.get((t, m), {})
            for uid, p in pm.items():
                g = gold.get((t, uid))
                if g is None:
                    continue
                g = {**g, "topic": t}
                r = rm.get(uid)
                ok = p["predicted_label"] == g["_gold"]
                if r and not ok and r["rag_features"]["rag_avg_transferability"] < args.low_transfer_threshold:
                    low_cases.append(case_record(g, base.get(uid), p, r, m, mems))
                b = base.get(uid)
                if b is None:
                    continue
                flags = [k for k, v in complex_subset_flags(g, args.short_text_len).items() if v] + ["ALL"]
                for fl in flags:
                    subset_size[m][fl] += 1
                bok = b["predicted_label"] == g["_gold"]
                if ok and not bok:
                    help_cases.append(case_record(g, b, p, r, m, mems))
                    for fl in flags:
                        help_cnt[m][fl] += 1
                elif bok and not ok:
                    harm_cases.append(case_record(g, b, p, r, m, mems))
                    for fl in flags:
                        harm_cnt[m][fl] += 1
    write_jsonl(files["low_transfer_failure_cases.jsonl"], low_cases[: args.max_cases])
    write_jsonl(files["rag_help_cases.jsonl"], help_cases[: args.max_cases])
    write_jsonl(files["rag_harm_cases.jsonl"], harm_cases[: args.max_cases])
    help_harm = {}
    for m in rag_methods:
        help_harm[m] = {}
        for fl in subset_size[m]:
            n = subset_size[m][fl]
            h, x = help_cnt[m][fl], harm_cnt[m][fl]
            help_harm[m][fl] = {"subset_zh": SUBSET_ZH.get(fl, "全部"), "n": n, "help": h, "harm": x, "net": h - x,
                                "help_rate": h / n if n else 0.0, "harm_rate": x / n if n else 0.0,
                                "net_rate": (h - x) / n if n else 0.0}

    # 5. memory_type 贡献
    type_out = {}
    if len(mdf):
        for m, sub in mdf.groupby("method"):
            type_out[m] = {}
            for mt, s in sub.groupby("memory_type"):
                top1 = s[s["rank"] == 0]
                type_out[m][mt] = {"n_retrievals": int(len(s)), "evidence_accuracy": float(s.evidence_correct.mean()),
                                   "pred_accuracy_when_retrieved": float(s.pred_correct.mean()),
                                   "n_top1": int(len(top1)),
                                   "pred_accuracy_when_top1": float(top1.pred_correct.mean()) if len(top1) else None,
                                   "mean_TransRel": float(s.transrel.mean()), "cite_rate": float(s.cited.mean())}
    write_json(files["memory_type_contribution.json"], type_out)

    # 证据有效性：RAG 不是装饰
    ev_out = {}
    if len(sdf):
        for m, sub in sdf.groupby("method"):
            has = sub[sub.k > 0]
            follow = has[has.pred == has.rag_dom]
            nofollow = has[has.pred != has.rag_dom]
            ev_out[m] = {
                "n": int(len(sub)), "n_with_memory": int(len(has)),
                "evidence_accuracy(rag主导立场==gold)": float((has.rag_dom == has.gold).mean()) if len(has) else None,
                "follow_rate(pred==rag主导立场)": float((has.pred == has.rag_dom).mean()) if len(has) else None,
                "accuracy_when_follow": float(follow.correct.mean()) if len(follow) else None,
                "accuracy_when_not_follow": float(nofollow.correct.mean()) if len(nofollow) else None,
                "accuracy_high_consistency(>=0.8)": float(has[has.consistency >= 0.8].correct.mean()) if (has.consistency >= 0.8).any() else None,
                "accuracy_low_consistency(<0.6)": float(has[has.consistency < 0.6].correct.mean()) if (has.consistency < 0.6).any() else None,
                "model_reported_used_rag_rate": float(has.used_rag.mean()) if len(has) else None,
            }
    write_json(files["rag_evidence_validity.json"], ev_out)

    # ---- 论文结论 ----
    mf = {}
    for t in topics:
        for m in methods:
            pm = preds.get((t, m), {})
            uids = [u for u in pm if (t, u) in gold]
            if uids:
                mf[(t, m)] = compute_metrics([gold[(t, u)]["_gold"] for u in uids], [pm[u]["predicted_label"] for u in uids])["macro_f1"]

    def avg_f1(m):
        return safe_mean(mf.get((t, m)) for t in topics)

    def compare(a, b):
        diffs = {t: mf[(t, a)] - mf[(t, b)] for t in topics if (t, a) in mf and (t, b) in mf}
        if not diffs:
            return {"available": False}
        mean = float(np.mean(list(diffs.values())))
        return {"available": True, "avg_a": avg_f1(a), "avg_b": avg_f1(b), "avg_diff": mean, "holds": mean > 0,
                "wins": sum(v > 0 for v in diffs.values()), "n_topics": len(diffs), "per_topic_diff": diffs}

    structured_avg = safe_mean(avg_f1(m) for m in args.structured_methods if avg_f1(m) is not None)
    rand = avg_f1("random_memory_rag")
    q4 = {"random_avg_macro_f1": rand, "structured_avg_macro_f1": structured_avg,
          "gap": (structured_avg - rand) if (structured_avg is not None and rand is not None) else None,
          "per_method_vs_random": {m: compare(m, "random_memory_rag") for m in args.structured_methods}}
    q4["random_clearly_lower(gap>=0.02 且各结构化方法多数话题胜出)"] = bool(
        q4["gap"] is not None and q4["gap"] >= 0.02 and all(
            v.get("available") and v["wins"] > v["n_topics"] / 2 for v in q4["per_method_vs_random"].values()))

    q5 = {}
    for m, bk in bucket_out.get("memory_level", {}).items():
        items = [(b, v) for b, v in bk.items() if v["n_retrievals"] >= args.min_bucket_n]
        if len(items) >= 2:
            lo, hi = items[0], items[-1]
            q5[m] = {"lowest_bucket": lo[0], "lowest_evidence_acc": lo[1]["evidence_accuracy(memory主导立场==gold)"],
                     "highest_bucket": hi[0], "highest_evidence_acc": hi[1]["evidence_accuracy(memory主导立场==gold)"],
                     "monotonic_increase_holds": hi[1]["evidence_accuracy(memory主导立场==gold)"] >
                     lo[1]["evidence_accuracy(memory主导立场==gold)"]}
    q5["corr_structured"] = bucket_out.get("memory_level_structured_pooled_corr")
    q5["htr_corr_structured"] = htr_out.get("structured_pooled")

    stable = []
    if len(pdf):
        stable = pdf[(pdf.n_folds >= max(2, len(topics) - 1)) & (pdf.dominant_label_consistent_across_folds)].head(15)
        stable = stable[["memory_type", "pattern_name", "dominant_label", "mean_TransRel", "mean_Stab", "mean_Cov",
                         "n_folds"]].to_dict("records")

    hard = []
    for t in topics:
        rag_f1 = safe_mean(mf.get((t, m)) for m in args.structured_methods)
        gains = [mf[(t, m)] - mf[(t, args.baseline)] for m in args.structured_methods
                 if (t, m) in mf and (t, args.baseline) in mf]
        hard.append({"heldout_topic": t, "avg_structured_rag_macro_f1": rag_f1,
                     "baseline_macro_f1": mf.get((t, args.baseline)),
                     "avg_transfer_gain": float(np.mean(gains)) if gains else None})
    hard.sort(key=lambda x: (x["avg_structured_rag_macro_f1"] if x["avg_structured_rag_macro_f1"] is not None else 9))

    def rank_subsets(key):
        agg = defaultdict(lambda: {"n": 0, "help": 0, "harm": 0})
        for m in args.structured_methods:
            for fl, v in help_harm.get(m, {}).items():
                if fl == "ALL":
                    continue
                for k in ("n", "help", "harm"):
                    agg[fl][k] += v[k]
        rows = [{"subset": fl, "subset_zh": SUBSET_ZH.get(fl), **v,
                 "help_rate": v["help"] / v["n"] if v["n"] else 0, "harm_rate": v["harm"] / v["n"] if v["n"] else 0,
                 "net_rate": (v["help"] - v["harm"]) / v["n"] if v["n"] else 0} for fl, v in agg.items()]
        return sorted(rows, key=lambda r: -r[key])

    report = {
        "baseline": args.baseline, "topics": topics, "methods": methods,
        "avg_macro_f1": {m: avg_f1(m) for m in methods},
        "Q0a_instruction_effect(rag_empty - baseline)": compare("rag_empty", args.baseline),
        "Q0b_memory_content_effect(source_rag - rag_empty)": compare("source_rag", "rag_empty"),
        "Q0c_memory_content_effect(pattern_only_rag - rag_empty)": compare("pattern_only_rag", "rag_empty"),
        "Q0d_memory_content_effect(transferable_rag - rag_empty)": compare("transferable_rag", "rag_empty"),
        "Q0e_random_memory_effect(random_memory_rag - rag_empty)": compare("random_memory_rag", "rag_empty"),
        "QI1_intopic_memory_gain(intopic_rag - baseline)": compare("intopic_rag", args.baseline),
        "QI2_crosstopic_memory_gain(source_rag - baseline)": compare("source_rag", args.baseline),
        "QI3_transfer_loss(intopic_rag - source_rag)": compare("intopic_rag", "source_rag"),
        "QB1_memory_without_semantic(context_memory - context_only)": compare("context_memory", "context_only"),
        "QB2_semantic_fields_effect(dynamic_semantic - context_only)": compare(args.baseline, "context_only"),
        "QB3_example_rag_vs_baseline": compare("example_rag", args.baseline),
        "QB4_example_rag_vs_rag_empty": compare("example_rag", "rag_empty"),
        "QB5_example_rag_vs_random_memory": compare("example_rag", "random_memory_rag"),
        "Q1_source_rag_gt_dynamic_semantic": compare("source_rag", args.baseline),
        "Q2_pattern_only_rag_gt_dynamic_semantic": compare("pattern_only_rag", args.baseline),
        "Q3_transferable_rag_gt_source_rag": compare("transferable_rag", "source_rag"),
        "Q4_random_memory_rag_lower_than_structured": q4,
        "Q5_high_transrel_higher_accuracy": q5,
        "Q6_most_stable_patterns": stable,
        "Q7_hardest_heldout_topics(按结构化RAG平均macroF1升序)": hard,
        "Q8_rag_helps_most(按 net_rate 降序)": rank_subsets("net_rate"),
        "Q9_rag_misleads_most(按 harm_rate 降序)": rank_subsets("harm_rate"),
        "help_harm_by_method_subset": help_harm,
        "rag_evidence_validity": ev_out,
        "n_cases": {"help": len(help_cases), "harm": len(harm_cases), "low_transfer_failure": len(low_cases)},
    }
    write_json(files["analysis_report.json"], report)

    # Markdown
    def esc(x):
        return str(x).replace("|", "\\|")

    def f4(x):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

    def yn(c):
        if not c.get("available"):
            return "数据不足"
        return f"{'成立' if c['holds'] else '不成立'}（平均 Δ={c['avg_diff']:+.4f}，{c['wins']}/{c['n_topics']} 个 held-out topic 胜出）"

    md = ["# 跨目标泛化 · 迁移可靠性分析报告", "",
          f"基线：`{args.baseline}`；结构化 RAG：{', '.join(args.structured_methods)}", "",
          "## 指令效应 vs 记忆内容效应（rag_empty 对照）", "",
          f"- 指令效应 rag_empty > {args.baseline}：{yn(report['Q0a_instruction_effect(rag_empty - baseline)'])}",
          f"- 记忆内容效应 source_rag > rag_empty：{yn(report['Q0b_memory_content_effect(source_rag - rag_empty)'])}",
          f"- 记忆内容效应 pattern_only_rag > rag_empty：{yn(report['Q0c_memory_content_effect(pattern_only_rag - rag_empty)'])}",
          f"- 记忆内容效应 transferable_rag > rag_empty：{yn(report['Q0d_memory_content_effect(transferable_rag - rag_empty)'])}",
          f"- 随机记忆 random_memory_rag > rag_empty：{yn(report['Q0e_random_memory_effect(random_memory_rag - rag_empty)'])}",
          "", "## 同话题 vs 跨话题记忆（intopic_rag 对照，同一批评估样本）", "",
          f"- 同话题记忆增益 intopic_rag > {args.baseline}：{yn(report['QI1_intopic_memory_gain(intopic_rag - baseline)'])}",
          f"- 跨话题记忆增益 source_rag > {args.baseline}：{yn(report['QI2_crosstopic_memory_gain(source_rag - baseline)'])}",
          f"- 迁移损失 intopic_rag > source_rag：{yn(report['QI3_transfer_loss(intopic_rag - source_rag)'])}",
          "", "## 路线 B：去语义字段消融 & 示例式检索", "",
          f"- 无语义字段时记忆有效 context_memory > context_only：{yn(report['QB1_memory_without_semantic(context_memory - context_only)'])}",
          f"- 语义字段本身的作用 {args.baseline} > context_only：{yn(report['QB2_semantic_fields_effect(dynamic_semantic - context_only)'])}",
          f"- 示例式检索 example_rag > {args.baseline}：{yn(report['QB3_example_rag_vs_baseline'])}",
          f"- 示例内容效应 example_rag > rag_empty：{yn(report['QB4_example_rag_vs_rag_empty'])}",
          f"- example_rag > random_memory_rag：{yn(report['QB5_example_rag_vs_random_memory'])}",
          "", "## 核心假设检验", "",
          f"1. source_rag > dynamic_semantic：{yn(report['Q1_source_rag_gt_dynamic_semantic'])}",
          f"2. pattern_only_rag > dynamic_semantic：{yn(report['Q2_pattern_only_rag_gt_dynamic_semantic'])}",
          f"3. transferable_rag > source_rag：{yn(report['Q3_transferable_rag_gt_source_rag'])}",
          f"4. random_memory_rag 明显低于结构化检索：{'是' if q4['random_clearly_lower(gap>=0.02 且各结构化方法多数话题胜出)'] else '否'}"
          f"（random={f4(rand)}，结构化平均={f4(structured_avg)}，gap={f4(q4['gap'])}）", "",
          "## 5. 高 TransRel memory 是否对应更高准确率", ""]
    for m, v in q5.items():
        if isinstance(v, dict) and "lowest_bucket" in v:
            md.append(f"- {m}：最低桶 {v['lowest_bucket']} 证据正确率 {f4(v['lowest_evidence_acc'])} → 最高桶 "
                      f"{v['highest_bucket']} {f4(v['highest_evidence_acc'])}（{'成立' if v['monotonic_increase_holds'] else '不成立'}）")
    c = (q5.get("corr_structured") or {}).get("transrel_vs_evidence_correct") or {}
    md.append(f"- 结构化方法 pooled：TransRel 与证据正确的 Spearman ρ={f4(c.get('spearman_rho'))}（p={f4(c.get('spearman_p'))}）")
    md += ["", "## 6. 跨 topic 最稳定的模式（Top）", "", "| memory_type | pattern | 主导立场 | mean TransRel | mean Stab | folds |",
           "|---|---|---|---|---|---|"]
    for r in stable[:10]:
        md.append(f"| {r['memory_type']} | {esc(r['pattern_name'])} | {r['dominant_label']} | {f4(r['mean_TransRel'])} | "
                  f"{f4(r['mean_Stab'])} | {r['n_folds']} |")
    md += ["", "## 7. 最难迁移的 held-out topic", "", "| held-out | 结构化RAG平均macroF1 | 基线macroF1 | 平均迁移增益 |", "|---|---|---|---|"]
    for r in hard:
        md.append(f"| {r['heldout_topic']} | {f4(r['avg_structured_rag_macro_f1'])} | {f4(r['baseline_macro_f1'])} | {f4(r['avg_transfer_gain'])} |")
    md += ["", "## 8/9. RAG 帮助 / 误导的复杂样本（结构化 RAG 合并）", "",
           "| 子集 | n | 帮助 | 误导 | 净帮助率 | 误导率 |", "|---|---|---|---|---|---|"]
    for r in report["Q8_rag_helps_most(按 net_rate 降序)"]:
        md.append(f"| {r['subset_zh']} | {r['n']} | {r['help']} | {r['harm']} | {f4(r['net_rate'])} | {f4(r['harm_rate'])} |")
    md += ["", "## RAG 证据有效性（是否为装饰）", "",
           "| method | 证据正确率 | 跟随率 | 跟随时准确率 | 不跟随时准确率 | 高一致性准确率 |", "|---|---|---|---|---|---|"]
    for m, v in ev_out.items():
        md.append(f"| {m} | {f4(v['evidence_accuracy(rag主导立场==gold)'])} | {f4(v['follow_rate(pred==rag主导立场)'])} | "
                  f"{f4(v['accuracy_when_follow'])} | {f4(v['accuracy_when_not_follow'])} | "
                  f"{f4(v['accuracy_high_consistency(>=0.8)'])} |")
    md += ["", "> 解读提示：若结构化 RAG 的证据正确率显著高于 random_memory_rag，且跟随证据时准确率更高，"
           "说明检索到的 memory 为预测提供了有效证据而非装饰性解释。"]
    files["analysis_report.md"].write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("帮助案例 %d，误导案例 %d，低迁移失败案例 %d", len(help_cases), len(harm_cases), len(low_cases))
    logger.info("完成，报告见 %s", files["analysis_report.md"])


if __name__ == "__main__":
    main()