# -*- coding: utf-8 -*-
"""
depth_breakdown.py
按回复深度统计 SPIN 的 ΔMacro-F1，输出给 make_figures.py 画图 6。不调用 API。

深度分档：1（C-MTCSD 的根帖深度 0 并入此档）、2、3、4、5、6+；节点少于 --min_n 的分档不输出。
每个输入为「名字=文件[:SPIN 字段]」，文件需含 depth、stance_label、base_prediction 和 SPIN 预测字段
（pred_combined.jsonl 默认 predicted_stance；pred_ablation.jsonl 用 pred_full_pred）。

用法（make_rag_data 目录下）：
python depth_breakdown.py --out depth_breakdown.csv --pred \
  "C-MTCSD 4o-mini kNN=combined_prior_wide_gpt4omini/pred_combined.jsonl" \
  "ZS-CSD 4o-mini kNN=zscsd_exp/outputs/gpt4omini/combined_prior_wide/pred_combined.jsonl" \
  "ZS-CSD 4o-mini CoT=zscsd_exp/outputs/gpt4omini/plugin_cot/pred_combined.jsonl" \
  "ZS-CSD GPT-4o CoT=zscsd_exp/outputs/gpt4o/ablation_cot/pred_ablation.jsonl:pred_full_pred"
"""
import argparse
import csv
import json
from collections import defaultdict

LABELS = ("favor", "against", "none")


def macro_f1(g, p):
    f1s = []
    for l in LABELS:
        tp = sum(a == l and b == l for a, b in zip(g, p))
        fp = sum(a != l and b == l for a, b in zip(g, p))
        fn = sum(a == l and b != l for a, b in zip(g, p))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / 3


def bucket(d):
    d = max(int(d), 1)
    return "6+" if d >= 6 else str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--min_n", type=int, default=50)
    ap.add_argument("--out", default="depth_breakdown.csv")
    a = ap.parse_args()
    rows = []
    for item in a.pred:
        name, spec = item.split("=", 1)
        path, field = (spec.rsplit(":", 1) + ["predicted_stance"])[:2] if spec.endswith(("_pred", "_stance")) and ":" in spec \
            else (spec, "predicted_stance")
        by = defaultdict(lambda: ([], [], []))
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            y, b, s = (str(r.get(k) or "").lower() for k in ("stance_label", "base_prediction", field))
            if y in LABELS and b in LABELS and s in LABELS and r.get("depth") is not None:
                g = by[bucket(r["depth"])]
                g[0].append(y); g[1].append(b); g[2].append(s)
        print(f"\n===== {name}（{path}，字段 {field}）")
        for k in sorted(by, key=lambda x: int(x.rstrip("+"))):
            g, b, s = by[k]
            mb, ms = macro_f1(g, b), macro_f1(g, s)
            flag = "" if len(g) >= a.min_n else "（节点太少，不输出）"
            print(f"  深度 {k:>3s}  n={len(g):5d}  基础 {100 * mb:6.2f} → SPIN {100 * ms:6.2f}  Δ {100 * (ms - mb):+6.2f}{flag}")
            if len(g) >= a.min_n:
                rows.append({"setting": name, "depth_bucket": k, "n": len(g), "base_macro_f1": round(100 * mb, 2),
                             "spin_macro_f1": round(100 * ms, 2), "delta_macro_f1": round(100 * (ms - mb), 2)})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n已保存 {a.out}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
