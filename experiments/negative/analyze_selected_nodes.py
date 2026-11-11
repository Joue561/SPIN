# -*- coding: utf-8 -*-
"""
analyze_selected_nodes.py
读取 analyze_conflict_detection.py 输出的 conflict_per_node.jsonl（不调用 API），回答：
在被挑中的节点上，不调用 LLM、只是“更相信知识库”（加大这些节点的 λ，或直接取先验最大类）能否改对？
如果能，说明问题出在推断本身；如果不能，复核才有必要。

对 post_margin / conflict 两种打分、各预算，输出挑中节点上：
  当前最终预测（KB）的准确率、基础预测准确率、先验最大类准确率、不同 λ' 下 观测×先验^λ' 的准确率，
以及“只在挑中节点上换成该 λ'”后全体测试集的 MacroF1 / F_avg（λ' 在这里按 test 选只作诊断，正式实验须在 valid 上选）。

用法：
python spin/experiments/negative/analyze_selected_nodes.py --per_node conflict_detection_knn/conflict_per_node.jsonl
"""
import argparse
import json

import numpy as np

LABELS = ["favor", "against", "none"]


def scores(g, p):
    f = []
    for l in LABELS:
        tp = sum(a == l and b == l for a, b in zip(g, p))
        fp = sum(a != l and b == l for a, b in zip(g, p))
        fn = sum(a == l and b != l for a, b in zip(g, p))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f)), (f[0] + f[1]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_node", required=True)
    ap.add_argument("--budgets", default="5,10,20,30")
    ap.add_argument("--lams", default="0.5,1,1.5,2,3,5")
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.per_node, encoding="utf-8") if l.strip()]
    for r in R:
        p = np.array(r["post"]); s = np.sort(p)
        r["post_margin"] = 1 - (s[-1] - s[-2])
    lams = [float(x) for x in a.lams.split(",")]
    g_all = [r["stance_label"] for r in R]
    base_mf, base_fa = scores(g_all, [r["kb_prediction"] for r in R])
    print(f"全体（n={len(R)}）当前 KB 预测：MacroF1 {base_mf:.4f}  F_avg {base_fa:.4f}")
    for score in ("post_margin", "conflict"):
        print(f"\n【按 {score} 挑选】只统计有传导先验的节点；准确率均在挑中节点上计算")
        print(f"  {'预算':>5s} {'n':>5s} {'基础':>6s} {'KB当前':>7s} {'先验最大':>8s} " +
              " ".join(f"{'λ='+format(l, 'g'):>7s}" for l in lams) + " | 最好 λ 换上后全体 MacroF1 / F_avg")
        cand = [r for r in R if r["prior"] is not None]
        for B in [float(x) for x in a.budgets.split(",")]:
            k = int(round(len(R) * B / 100))
            sel = sorted(cand, key=lambda r: -r[score])[:k]
            ids = {(r["topic"], r["node_id"]) for r in sel}
            acc = lambda labs: np.mean([x == r["stance_label"] for x, r in zip(labs, sel)])
            a_base = acc([r["base_prediction"] for r in sel])
            a_kb = acc([r["kb_prediction"] for r in sel])
            a_prior = acc([LABELS[int(np.argmax(r["prior"]))] for r in sel])
            per_l, best = [], (base_mf, base_fa, None)
            for l in lams:
                labs = {(r["topic"], r["node_id"]): LABELS[int(np.argmax(np.array(r["obs"]) * np.array(r["prior"]) ** l))]
                        for r in sel}
                per_l.append(acc([labs[(r["topic"], r["node_id"])] for r in sel]))
                mf, fa = scores(g_all, [labs.get((r["topic"], r["node_id"]), r["kb_prediction"]) for r in R])
                if mf > best[0]:
                    best = (mf, fa, l)
            bl = "无提升" if best[2] is None else f"λ={best[2]:g}: {best[0]:.4f} / {best[1]:.4f}"
            print(f"  {B:4g}% {len(sel):5d} {a_base:6.3f} {a_kb:7.3f} {a_prior:8.3f} " +
                  " ".join(f"{x:7.3f}" for x in per_l) + f" | {bl}")


if __name__ == "__main__":
    main()