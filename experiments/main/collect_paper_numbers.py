# -*- coding: utf-8 -*-
"""
collect_paper_numbers.py
从各设定的 pred_combined.jsonl 一次性汇总论文表 2 / 表 3 需要的数字，不调用 API。

对每个文件，在 test 节点上比较 base_prediction（基础模型）与 predicted_stance（+SPIN）：
  合并 Macro-F1、合并 F_avg、话题平均 Macro-F1、话题平均 F_avg（基础 / +SPIN / Δ）
  合并 ΔMacro-F1 与 ΔF_avg 的配对 bootstrap 95% 置信区间（与 paired_significance.py 同一实现）
  近似随机化 p 值（Macro-F1）与 McNemar p 值
最后打印一张可直接抄进表 2 的汇总表。显著：置信区间不含 0。

用法（make_rag_data 目录下）：
python spin/experiments/main/collect_paper_numbers.py \
  --pred C/GPT-4o-mini=combined_prior_wide_gpt4omini/pred_combined.jsonl \
         Z/GPT-4o-mini=zscsd_exp/outputs/gpt4omini/combined_prior_wide/pred_combined.jsonl
"""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("ps", Path(__file__).resolve().parent / "paired_significance.py")
ps = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ps)

LABELS = ("favor", "against", "none")
LID = {l: i for i, l in enumerate(LABELS)}


def scores(g, p):
    g = np.array([LID[x] for x in g]); p = np.array([LID[x] for x in p])
    mf, fa, _ = ps.macro_f1(g, p)
    return mf, fa


def load(path, base_field, pred_field):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        y, b, s = (str(r.get(k) or "").lower() for k in ("stance_label", base_field, pred_field))
        if y in LID and b in LID and s in LID:
            rows.append((str(r.get("topic")), y, b, s))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="名字=pred_combined.jsonl，可给多个")
    ap.add_argument("--base_field", default="base_prediction")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    summary = []
    for item in a.pred:
        name, path = item.split("=", 1)
        rows = load(path, a.base_field, a.pred_field)
        if not rows:
            print(f"\n===== {name}：没有可用的行（检查字段名）"); continue
        _, g, b, s = zip(*rows)
        r = ps.test(list(g), list(s), list(b), a.n_boot, a.seed)   # A = +SPIN，B = 基础
        by = defaultdict(list)
        for t, y, bb, ss in rows:
            by[t].append((y, bb, ss))
        tb = [scores([x[0] for x in v], [x[1] for x in v]) for v in by.values()]
        tsp = [scores([x[0] for x in v], [x[2] for x in v]) for v in by.values()]
        t_mf_b, t_fa_b = np.mean([x[0] for x in tb]), np.mean([x[1] for x in tb])
        t_mf_s, t_fa_s = np.mean([x[0] for x in tsp]), np.mean([x[1] for x in tsp])
        sig = lambda ci: "显著" if ci[0] > 0 or ci[1] < 0 else "不显著"
        print(f"\n===== {name}   n={r['n']}  话题数={len(by)}  （{path}）")
        print(f"  合并 MacroF1   {100 * r['B']['macro_f1']:6.2f} → {100 * r['A']['macro_f1']:6.2f}  "
              f"Δ {100 * r['delta_macro_f1']:+.2f}  95%CI [{100 * r['ci95_macro_f1'][0]:+.2f}, {100 * r['ci95_macro_f1'][1]:+.2f}]  "
              f"{sig(r['ci95_macro_f1'])}  p_AR={r['p_approx_randomization(two-sided)']:.4f}  p_McNemar={r['mcnemar']['p']:.4f}")
        print(f"  合并 F_avg     {100 * r['B']['f_avg']:6.2f} → {100 * r['A']['f_avg']:6.2f}  "
              f"Δ {100 * r['delta_f_avg']:+.2f}  95%CI [{100 * r['ci95_f_avg'][0]:+.2f}, {100 * r['ci95_f_avg'][1]:+.2f}]  {sig(r['ci95_f_avg'])}")
        print(f"  话题平均 MacroF1 {100 * t_mf_b:6.2f} → {100 * t_mf_s:6.2f}  Δ {100 * (t_mf_s - t_mf_b):+.2f}")
        print(f"  话题平均 F_avg   {100 * t_fa_b:6.2f} → {100 * t_fa_s:6.2f}  Δ {100 * (t_fa_s - t_fa_b):+.2f}")
        summary.append((name, r, sig))

    print("\n===== 表 2 汇总（合并口径，%）")
    print(f"{'设定':22s} {'基础MF1':>7s} {'+SPIN':>7s} {'ΔMF1':>7s} {'显著':>4s} {'基础Favg':>8s} {'+SPIN':>7s} {'ΔFavg':>7s} {'显著':>4s}")
    for name, r, sig in summary:
        print(f"{name:22s} {100 * r['B']['macro_f1']:7.2f} {100 * r['A']['macro_f1']:7.2f} {100 * r['delta_macro_f1']:+7.2f} "
              f"{sig(r['ci95_macro_f1']):>4s} {100 * r['B']['f_avg']:8.2f} {100 * r['A']['f_avg']:7.2f} "
              f"{100 * r['delta_f_avg']:+7.2f} {sig(r['ci95_f_avg']):>4s}")


if __name__ == "__main__":
    main()