# -*- coding: utf-8 -*-
"""
zscsd_breakdown.py
ZS-CSD 结果分解（不调用 API）：读取 analyze_combined_prior.py 输出的 pred_combined.jsonl，
按目标类型（名词短语 n / 观点句 c）、深度、是否同说话人回复分组，比较
  基础预测 / 只用相对 / 只用绝对 / 组合 四种预测的 F1-macro 与 F_avg，
并输出基础预测与组合的混淆矩阵，以及各分组上组合相对基础预测的 bootstrap 置信区间。

用法（make_rag_data 目录下）：
python spin/experiments/analysis/zscsd_breakdown.py \
  --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --pred zscsd_exp/outputs/gpt4omini/combined_prior/pred_combined.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

LABELS = ["favor", "against", "none"]
FIELDS = [("基础", "base_prediction"), ("相对", "relative_only_prediction"),
          ("绝对", "absolute_only_prediction"), ("组合", "predicted_stance")]


def f1s(g, p):
    out = []
    for c in LABELS:
        tp = sum(1 for a, b in zip(g, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(g, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(g, p) if a == c and b != c)
        out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return out


def scores(g, p):
    f = f1s(g, p)
    return float(np.mean(f)), (f[0] + f[1]) / 2, f


def boot(g, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g, a, b = np.array(g), np.array(a), np.array(b)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(g), len(g))
        d.append(scores(g[i], b[i])[0] - scores(g[i], a[i])[0])
    return np.percentile(d, [2.5, 97.5]) * 100


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    ins = {r["node_id"]: r for r in load(a.instances)}
    rows = load(a.pred)
    for r in rows:
        m = ins.get(r["node_id"], {})
        r["target_type"] = {"n": "名词短语目标", "c": "观点句目标"}.get(m.get("target_type"), "未知")
        d = int(r.get("depth") or m.get("depth") or 0)
        r["depth_b"] = f"深度 {min(d, 6)}" + ("+" if d >= 6 else "")
        r["spk"] = "同说话人再次回复" if m.get("same_speaker_ancestor_ids") else "新说话人"
        r["sem"] = "语义键缺失" if m.get("llm_annotation_source") == "missing" else "有语义键"
        r["target"] = m.get("main_target")

    def report(name, key):
        print(f"\n【按{name}】  F1-macro（×100）：基础 / 相对 / 绝对 / 组合   ΔF1(组合−基础) [95% CI]   F_avg 基础→组合")
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for k in sorted(groups):
            g = [r["stance_label"] for r in groups[k]]
            vals = [scores(g, [r[f] for r in groups[k]]) for _, f in FIELDS]
            ci = boot(g, [r["base_prediction"] for r in groups[k]], [r["predicted_stance"] for r in groups[k]])
            print(f"  {k:12s} n={len(g):5d}  " + " / ".join(f"{v[0] * 100:5.2f}" for v in vals) +
                  f"   {(vals[3][0] - vals[0][0]) * 100:+5.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
                  f"   {vals[0][1] * 100:5.2f}→{vals[3][1] * 100:5.2f}")

    g = [r["stance_label"] for r in rows]
    print(f"全部 n={len(rows)}，gold 分布 {dict(Counter(g))}")
    for nm, f in FIELDS:
        m, fa, per = scores(g, [r[f] for r in rows])
        print(f"  {nm}：F1-macro {m * 100:.2f}  F_avg {fa * 100:.2f}  各类 F1 favor {per[0]:.3f} against {per[1]:.3f} none {per[2]:.3f}"
              f"  预测分布 {dict(Counter(r[f] for r in rows))}")
    for nm, f in (FIELDS[0], FIELDS[3]):
        print(f"\n【混淆矩阵：{nm}】行=gold，列=预测 (favor / against / none)")
        cm = Counter((r["stance_label"], r[f]) for r in rows)
        for y in LABELS:
            print(f"  {y:8s} " + " ".join(f"{cm[(y, p)]:6d}" for p in LABELS))
    report("目标类型", "target_type")
    report("深度", "depth_b")
    report("说话人", "spk")
    report("语义键", "sem")

    # 各目标的变化（看提升是否集中在少数目标）
    per_t = defaultdict(list)
    for r in rows:
        per_t[r["target"]].append(r)
    d = []
    for t, rs in per_t.items():
        gg = [r["stance_label"] for r in rs]
        d.append((scores(gg, [r["predicted_stance"] for r in rs])[0] - scores(gg, [r["base_prediction"] for r in rs])[0], t, len(rs)))
    d.sort()
    up = sum(1 for x in d if x[0] > 0.005); dn = sum(1 for x in d if x[0] < -0.005)
    print(f"\n【各目标】{len(d)} 个目标中，组合比基础提升 {up} 个、下降 {dn} 个、基本不变 {len(d) - up - dn} 个")
    print("  下降最多：" + "；".join(f"{t}({n}) {x * 100:+.1f}" for x, t, n in d[:3]))
    print("  提升最多：" + "；".join(f"{t}({n}) {x * 100:+.1f}" for x, t, n in d[-3:]))


if __name__ == "__main__":
    main()