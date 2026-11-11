# -*- coding: utf-8 -*-
"""
summarize_methods.py
把多个方法的逐条预测汇总成论文主表：在所有方法共同预测的样本上计算
  Accuracy / Macro-F1 / F_avg（favor 与 against F1 的均值）/ 各类 F1
并给出三种口径：
  1) 合并（pooled）：所有话题样本合在一起算
  2) 话题平均（topic-macro）：每个话题单独算，再对话题取平均（不受大话题主导）
  3) 逐话题
输出终端表格 + CSV。

示例：
python spin/experiments/main/summarize_methods.py \
  --pred dynamic_semantic=rag_prediction_outputs/predictions_dynamic_semantic_test.jsonl \
         topic_memory=rag_prediction_outputs/predictions_topicmem_rag_test.jsonl \
         mixed_memory=rag_prediction_outputs/predictions_full_rag_test.jsonl \
         crosstopic_memory=rag_prediction_outputs/predictions_crosstopicmem_rag_test.jsonl \
  --id_field topic node_id --pred_field predicted_stance --gold_field stance_label \
  --group_field topic --baseline dynamic_semantic --out_csv main_table.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

LABELS = ["favor", "against", "none"]
ALIAS = {"favor": "favor", "favour": "favor", "support": "favor", "支持": "favor",
         "against": "against", "oppose": "against", "反对": "against",
         "none": "none", "neutral": "none", "中立": "none", "无立场": "none"}


def norm(x):
    return None if x is None else ALIAS.get(str(x).strip().lower())


def load(path):
    if path.endswith(".csv"):
        return list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def metrics(g, p):
    lid = {l: i for i, l in enumerate(LABELS)}
    g = np.array([lid[x] for x in g]); p = np.array([lid[x] for x in p])
    cm = np.bincount(g * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(3), where=denom > 0)
    return {"n": int(len(g)), "acc": float(tp.sum() / len(g)), "macro_f1": float(f1.mean()),
            "f_avg": float((f1[0] + f1[1]) / 2), "favor_f1": float(f1[0]), "against_f1": float(f1[1]),
            "none_f1": float(f1[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="方法名=预测文件路径，可多个")
    ap.add_argument("--id_field", nargs="+", default=["topic", "node_id"])
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--gold_field", default="stance_label")
    ap.add_argument("--group_field", default="topic")
    ap.add_argument("--baseline", default=None, help="计算 Δ 的基线方法名")
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()

    data, gold, grp = {}, {}, {}
    for spec in a.pred:
        name, path = spec.split("=", 1)
        d = {}
        for r in load(path):
            k = "::".join(str(r.get(f)) for f in a.id_field)
            p, g = norm(r.get(a.pred_field)), norm(r.get(a.gold_field))
            if p is None or g is None:
                continue
            d[k] = p
            gold.setdefault(k, g)
            grp.setdefault(k, str(r.get(a.group_field)))
        data[name] = d
        print(f"{name}: {len(d)} 条有效预测  ({path})")
    common = sorted(set.intersection(*[set(d) for d in data.values()]))
    print(f"\n所有方法共同样本：{len(common)} 条（只在这些样本上比较，保证公平）")
    print("gold 分布：", dict(Counter(gold[k] for k in common)))
    groups = sorted({grp[k] for k in common})

    rows = []
    for name, d in data.items():
        pooled = metrics([gold[k] for k in common], [d[k] for k in common])
        per = {t: metrics([gold[k] for k in common if grp[k] == t], [d[k] for k in common if grp[k] == t])
               for t in groups}
        tm = {m: float(np.mean([per[t][m] for t in groups])) for m in ("acc", "macro_f1", "f_avg")}
        rows.append((name, pooled, tm, per))

    base = next((r for r in rows if r[0] == a.baseline), None)

    def d(v, bv):
        return f"{v - bv:+.4f}" if base else ""

    print("\n=== 主表（合并口径 / 话题平均口径）===")
    hdr = f"{'方法':22s} {'Acc':>7s} {'MacroF1':>8s} {'F_avg':>7s} {'favor':>6s} {'against':>7s} {'none':>6s} | {'话题平均Acc':>10s} {'话题平均MacroF1':>14s} {'话题平均F_avg':>12s}"
    if base:
        hdr += f" | {'ΔMacroF1(合并)':>13s} {'ΔMacroF1(话题平均)':>17s}"
    print(hdr)
    for name, p, tm, _ in rows:
        line = (f"{name:22s} {p['acc']:7.4f} {p['macro_f1']:8.4f} {p['f_avg']:7.4f} {p['favor_f1']:6.3f} "
                f"{p['against_f1']:7.3f} {p['none_f1']:6.3f} | {tm['acc']:10.4f} {tm['macro_f1']:14.4f} {tm['f_avg']:12.4f}")
        if base:
            line += f" | {d(p['macro_f1'], base[1]['macro_f1']):>13s} {d(tm['macro_f1'], base[2]['macro_f1']):>17s}"
        print(line)

    print("\n=== 逐话题 Macro-F1（括号内为 F_avg）===")
    print(f"{'方法':22s} " + " ".join(f"{t[:18]:>20s}" for t in groups))
    for name, _, _, per in rows:
        print(f"{name:22s} " + " ".join(f"{per[t]['macro_f1']:.4f} ({per[t]['f_avg']:.4f})".rjust(20) for t in groups))
    print(f"{'n':22s} " + " ".join(f"{rows[0][3][t]['n']:>20d}" for t in groups))

    if a.out_csv:
        with open(a.out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "scope", "n", "acc", "macro_f1", "f_avg", "favor_f1", "against_f1", "none_f1"])
            for name, p, tm, per in rows:
                w.writerow([name, "pooled", p["n"], p["acc"], p["macro_f1"], p["f_avg"], p["favor_f1"], p["against_f1"], p["none_f1"]])
                w.writerow([name, "topic_macro", "", tm["acc"], tm["macro_f1"], tm["f_avg"], "", "", ""])
                for t in groups:
                    m = per[t]
                    w.writerow([name, t, m["n"], m["acc"], m["macro_f1"], m["f_avg"], m["favor_f1"], m["against_f1"], m["none_f1"]])
        print(f"\n已保存 {a.out_csv}")


if __name__ == "__main__":
    main()