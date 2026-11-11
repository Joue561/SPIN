# -*- coding: utf-8 -*-
"""
paired_significance.py
对任意两组逐条预测做配对显著性检验（用于检验你之前同话题 / MT-CSD 实验的结论是否稳健）。

支持 .jsonl / .json(list) / .csv。两个文件按样本 ID 对齐；gold 可来自任一预测文件或单独的 gold 文件。

示例：
python tools/paired_significance.py \
  --a old_results/full_rag_predictions.jsonl --b old_results/dynamic_semantic_predictions.jsonl \
  --id_field node_id --pred_field predicted_label --gold_field stance_label \
  --name_a full_rag --name_b dynamic_semantic

如果 gold 不在预测文件里：加 --gold_file data/test.jsonl（同样用 --id_field 对齐）。
如果一个文件里混有多个数据集/话题，可用 --group_field topic 额外输出分组结果。
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LABELS = ["favor", "against", "none"]
ALIAS = {"favor": "favor", "favour": "favor", "support": "favor", "支持": "favor", "pro": "favor",
         "against": "against", "oppose": "against", "反对": "against", "con": "against",
         "none": "none", "neutral": "none", "neither": "none", "中立": "none", "无立场": "none",
         "0": "favor", "1": "against", "2": "none"}


def norm(x, int_map):
    if x is None:
        return None
    s = str(x).strip().lower()
    if int_map and s in int_map:
        return int_map[s]
    return ALIAS.get(s)


def load(path):
    p = Path(path)
    if p.suffix == ".csv":
        with open(p, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if p.suffix == ".json":
        obj = json.load(open(p, encoding="utf-8"))
        return obj if isinstance(obj, list) else list(obj.values())
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def key_of(r, fields):
    return "::".join(str(r.get(f)) for f in fields)


def macro_f1(g, p):
    cm = np.bincount(g * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(3), where=denom > 0)
    return f1.mean(), (f1[0] + f1[1]) / 2, (tp.sum() / len(g))


def test(g, a, b, n_boot, seed):
    lid = {l: i for i, l in enumerate(LABELS)}
    g = np.array([lid[x] for x in g]); a = np.array([lid[x] for x in a]); b = np.array([lid[x] for x in b])
    ma, fa, aa = macro_f1(g, a)
    mb, fb, ab = macro_f1(g, b)
    rng = np.random.default_rng(seed)
    dm, df = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        x, y = macro_f1(g[i], a[i]), macro_f1(g[i], b[i])
        dm.append(x[0] - y[0]); df.append(x[1] - y[1])
    dm, df = np.array(dm), np.array(df)
    # 近似随机化检验（approximate randomization，NLP 常用）
    obs = abs(ma - mb)
    hits = 0
    for _ in range(n_boot):
        swap = rng.random(len(g)) < 0.5
        aa_, bb_ = np.where(swap, b, a), np.where(swap, a, b)
        if abs(macro_f1(g, aa_)[0] - macro_f1(g, bb_)[0]) >= obs:
            hits += 1
    from scipy.stats import binomtest
    n01 = int(((a == g) & (b != g)).sum()); n10 = int(((b == g) & (a != g)).sum())
    mc = float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return {
        "n": int(len(g)), "gold_dist": dict(Counter(LABELS[x] for x in g)),
        "A": {"macro_f1": ma, "f_avg": fa, "acc": aa}, "B": {"macro_f1": mb, "f_avg": fb, "acc": ab},
        "delta_macro_f1": ma - mb, "ci95_macro_f1": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
        "p_boot(Δ≤0)": float((dm <= 0).mean()),
        "delta_f_avg": fa - fb, "ci95_f_avg": [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))],
        "p_approx_randomization(two-sided)": (hits + 1) / (n_boot + 1),
        "mcnemar": {"A_only_correct": n01, "B_only_correct": n10, "p": mc},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--name_a", default="A"); ap.add_argument("--name_b", default="B")
    ap.add_argument("--id_field", nargs="+", default=["sample_uid"], help="对齐用的 ID 字段（可多个联合）")
    ap.add_argument("--pred_field", default="predicted_label")
    ap.add_argument("--pred_field_b", default=None, help="B 文件的预测字段名（默认同 --pred_field）")
    ap.add_argument("--gold_field", default="stance_label")
    ap.add_argument("--gold_file", default=None)
    ap.add_argument("--group_field", default=None, help="可选：按该字段（如 topic/target/dataset）分组输出")
    ap.add_argument("--int_label_map", default=None, help='整数标签映射，如 \'{"0":"against","1":"favor","2":"none"}\'')
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    imap = {k: v for k, v in json.loads(args.int_label_map).items()} if args.int_label_map else None

    A = {key_of(r, args.id_field): r for r in load(args.a)}
    B = {key_of(r, args.id_field): r for r in load(args.b)}
    G = {key_of(r, args.id_field): r for r in load(args.gold_file)} if args.gold_file else {}
    common = sorted(set(A) & set(B))
    print(f"A={len(A)} 条，B={len(B)} 条，ID 对齐后共同样本 {len(common)} 条")
    if not common:
        sys.exit("没有共同样本：请检查 --id_field 是否在两个文件中都存在且取值一致")
    rows, skipped = [], Counter()
    pb_field = args.pred_field_b or args.pred_field
    for k in common:
        src = G.get(k) or A[k]
        g = norm(src.get(args.gold_field) if src.get(args.gold_field) is not None else B[k].get(args.gold_field), imap)
        a = norm(A[k].get(args.pred_field), imap); b = norm(B[k].get(pb_field), imap)
        if g is None:
            skipped["no_gold"] += 1; continue
        if a is None or b is None:
            skipped["bad_pred"] += 1; continue
        grp = (G.get(k) or A[k]).get(args.group_field) if args.group_field else "ALL"
        rows.append((grp, g, a, b))
    if skipped:
        print("跳过：", dict(skipped))
    groups = defaultdict(list)
    for grp, g, a, b in rows:
        groups["ALL"].append((g, a, b))
        if args.group_field:
            groups[str(grp)].append((g, a, b))
    for name, data in groups.items():
        g, a, b = zip(*data)
        r = test(list(g), list(a), list(b), args.n_boot, args.seed)
        print(f"\n===== {name} | n={r['n']} | gold={r['gold_dist']}")
        print(f"{args.name_a}: macroF1={r['A']['macro_f1']:.4f} F_avg={r['A']['f_avg']:.4f} acc={r['A']['acc']:.4f}")
        print(f"{args.name_b}: macroF1={r['B']['macro_f1']:.4f} F_avg={r['B']['f_avg']:.4f} acc={r['B']['acc']:.4f}")
        print(f"Δ macroF1={r['delta_macro_f1']:+.4f}  95%CI=[{r['ci95_macro_f1'][0]:+.4f}, {r['ci95_macro_f1'][1]:+.4f}]  "
              f"p_boot(Δ≤0)={r['p_boot(Δ≤0)']:.3f}  p_AR(双侧)={r['p_approx_randomization(two-sided)']:.3f}")
        print(f"Δ F_avg  ={r['delta_f_avg']:+.4f}  95%CI=[{r['ci95_f_avg'][0]:+.4f}, {r['ci95_f_avg'][1]:+.4f}]")
        print(f"McNemar: 仅{args.name_a}对={r['mcnemar']['A_only_correct']} 仅{args.name_b}对={r['mcnemar']['B_only_correct']} p={r['mcnemar']['p']:.3f}")


if __name__ == "__main__":
    main()