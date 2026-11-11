# -*- coding: utf-8 -*-
"""
analyze_retrieval_significance.py
离线检验（不调用 API）：语义传导键检索到的知识单元，判断"当前回复与父帖立场的关系"（继承 / 反转 / 中立化）
是否显著比仅深度、随机键、多数类更准。这个检验不依赖 λ 和会话树推断，是语义键独立贡献的直接证据。

评估对象：父帖 gold 立场为 favor/against 的节点（与 analyze_propagation_kb.py 的【检索质量】一致）。
每个节点：检索到的知识单元 → 概率最大的关系 → 与 gold 关系比较。
知识库均为话题内：每个话题只用自己的 train 构建。

检验：
  关系准确率之差：配对 bootstrap（按会话树重采样，95% CI）+ 精确 McNemar
  关系 macro-F1 之差：配对 bootstrap（按会话树重采样）
  按话题分别输出

用法（make_rag_data 目录下）：
python -u spin/experiments/ablation/analyze_retrieval_significance.py \
  --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl \
  --eval_split test --output_dir retrieval_significance
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("kbm", (Path(__file__).resolve().parent / "../../spin_core/analyze_propagation_kb.py").resolve())
kbm = importlib.util.module_from_spec(_s)
_s.loader.exec_module(kbm)
dd = kbm.dd
RELS = dd.RELS


def mcnemar_exact(b, c):
    """b = 只有 A 对，c = 只有 B 对；双侧精确二项检验。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def rel_macro_f1(gold, pred):
    f = []
    for r in RELS:
        tp = sum(g == r and p == r for g, p in zip(gold, pred))
        fp = sum(g != r and p == r for g, p in zip(gold, pred))
        fn = sum(g == r and p != r for g, p in zip(gold, pred))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f))


def compare(items, a, b, n_boot, rng):
    """items: 节点列表，每个含 gold 与各方法预测；按会话树(topic, root)聚类重采样（向量化）。"""
    gold = [x["gold"] for x in items]
    pa, pb = [x[a] for x in items], [x[b] for x in items]
    acc_a = np.mean([g == p for g, p in zip(gold, pa)])
    acc_b = np.mean([g == p for g, p in zip(gold, pb)])
    f_a, f_b = rel_macro_f1(gold, pa), rel_macro_f1(gold, pb)
    only_a = sum(g == x and g != y for g, x, y in zip(gold, pa, pb))
    only_b = sum(g == y and g != x for g, x, y in zip(gold, pa, pb))

    # 每个会话树的计数向量：[n, 对A, 对B, 每个关系 r 的 (tpA, fpA, fnA, tpB, fpB, fnB)]
    cid = {}
    for x in items:
        cid.setdefault(x["tree"], len(cid))
    C = np.zeros((len(cid), 3 + 6 * len(RELS)))
    for x, g, u, v in zip(items, gold, pa, pb):
        row = C[cid[x["tree"]]]
        row[0] += 1; row[1] += g == u; row[2] += g == v
        for j, r in enumerate(RELS):
            o = 3 + 6 * j
            row[o] += g == r and u == r; row[o + 1] += g != r and u == r; row[o + 2] += g == r and u != r
            row[o + 3] += g == r and v == r; row[o + 4] += g != r and v == r; row[o + 5] += g == r and v != r
    k = len(cid)
    W = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)])
    S = W @ C
    d_acc = (S[:, 1] - S[:, 2]) / S[:, 0]

    def mf1(off):
        fs = []
        for j in range(len(RELS)):
            o = 3 + 6 * j + off
            tp, fp, fn = S[:, o], S[:, o + 1], S[:, o + 2]
            den = 2 * tp + fp + fn
            fs.append(np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0))
        return np.mean(fs, axis=0)
    d_f1 = mf1(0) - mf1(3)
    return {"n": len(items), "acc_A": round(float(acc_a), 4), "acc_B": round(float(acc_b), 4),
            "Δacc": round(float(acc_a - acc_b), 4),
            "Δacc_CI": [round(float(np.percentile(d_acc, 2.5)), 4), round(float(np.percentile(d_acc, 97.5)), 4)],
            "p_boot_acc(Δ≤0)": round(float(np.mean(d_acc <= 0)), 4),
            "relF1_A": round(f_a, 4), "relF1_B": round(f_b, 4), "ΔrelF1": round(f_a - f_b, 4),
            "ΔrelF1_CI": [round(float(np.percentile(d_f1, 2.5)), 4), round(float(np.percentile(d_f1, 97.5)), 4)],
            "p_boot_relF1(Δ≤0)": round(float(np.mean(d_f1 <= 0)), 4),
            "只有A对": int(only_a), "只有B对": int(only_b), "McNemar_p": round(mcnemar_exact(only_a, only_b), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--eval_split", default="test", help="test / valid / test,valid")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="retrieval_significance")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    splits = set(a.eval_split.split(","))
    ev = [n for n in nodes.values() if n["split"] in splits and n["y"] and n["pa_gold"] in ("favor", "against")]
    print(f"train {len(train)} | 评估节点（{a.eval_split}，父帖 gold 有立场）{len(ev)}")

    by_t = {t: [n for n in train if n["topic"] == t] for t in topics}
    models = {name: {t: kbm.PatternKB(by_t[t], kf, a.min_count) for t in topics} for name, kf in kbm.KEYS.items()}
    models["KB:随机键"] = {t: kbm.PatternKB(by_t[t], kbm.k_sp_depth, a.min_count,
                                          kbm.random_keyfn(by_t[t], kbm.k_sp_depth, "in" + t)) for t in topics}
    models["LR:表层特征"] = {t: dd.Transition(by_t[t]) for t in topics}
    for ms in models.values():
        for t in topics:
            ms[t].precompute([n for n in ev if n["topic"] == t])

    items = []
    for n in ev:
        x = {"topic": n["topic"], "node_id": n["node_id"], "depth": n["depth"], "tree": (n["topic"], n["root"]),
             "gold": dd.relation(n["pa_gold"], n["y"])}
        for name, ms in models.items():
            pr = ms[n["topic"]].rel_proba(n)
            x[name] = max(RELS, key=pr.get)
        x["多数类"] = models["KB:仅深度"][n["topic"]].r_major
        items.append(x)

    print("\n【各方法关系准确率】（话题内知识库）")
    names = list(models) + ["多数类"]
    for nm in names:
        acc = np.mean([x["gold"] == x[nm] for x in items])
        f1 = rel_macro_f1([x["gold"] for x in items], [x[nm] for x in items])
        print(f"  {nm:16s} 关系准确率 {acc:.4f}  关系macroF1 {f1:.4f}")

    pairs = [("KB:完整组合", "KB:仅深度"), ("KB:传导类型+深度", "KB:仅深度"), ("KB:传导类型", "KB:仅深度"),
             ("KB:完整组合", "KB:随机键"), ("KB:传导类型+深度", "KB:随机键"),
             ("KB:完整组合", "多数类"), ("KB:仅深度", "多数类"),
             ("KB:完整组合", "LR:表层特征"), ("KB:完整组合", "KB:传导类型+深度")]
    rows = []
    print("\n【配对检验】Δ = A − B；CI 为按会话树重采样的 95% 区间；McNemar 为精确检验")
    for A, B in pairs:
        for grp in ["ALL"] + topics:
            its = items if grp == "ALL" else [x for x in items if x["topic"] == grp]
            if not its:
                continue
            r = {"A": A, "B": B, "group": grp, **compare(its, A, B, a.n_boot, rng)}
            rows.append(r)
            if grp == "ALL":
                print(f"  {A} vs {B}  n={r['n']}  准确率 {r['acc_A']:.4f} vs {r['acc_B']:.4f}  "
                      f"Δ={r['Δacc']:+.4f} {r['Δacc_CI']}  | 关系macroF1 Δ={r['ΔrelF1']:+.4f} {r['ΔrelF1_CI']}  "
                      f"| 只有A对 {r['只有A对']} 只有B对 {r['只有B对']} McNemar p={r['McNemar_p']:.4g}")
        for r in rows:
            if r["A"] == A and r["B"] == B and r["group"] != "ALL":
                sig = "显著" if r["McNemar_p"] < 0.05 and r["Δacc"] > 0 else ""
                print(f"      {r['group']:22s} n={r['n']:4d} Δacc={r['Δacc']:+.4f} {r['Δacc_CI']}  McNemar p={r['McNemar_p']:.4g} {sig}")

    # 按深度
    print("\n【按深度】KB:完整组合 vs KB:仅深度 的关系准确率")
    for db in ["1", "2", "3", "4+"]:
        its = [x for x in items if dd.depth_bucket(x["depth"]) == db]
        if its:
            r = compare(its, "KB:完整组合", "KB:仅深度", a.n_boot, rng)
            print(f"  深度{db:3s} n={r['n']:4d} {r['acc_A']:.4f} vs {r['acc_B']:.4f} Δ={r['Δacc']:+.4f} {r['Δacc_CI']}  McNemar p={r['McNemar_p']:.4g}")

    with open(out / "retrieval_significance.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(out / "retrieval_per_node.jsonl", "w", encoding="utf-8") as f:
        for x in items:
            y = dict(x); y["tree"] = list(y["tree"])
            f.write(json.dumps(y, ensure_ascii=False) + "\n")
    print(f"\n结果已保存到 {out.resolve()}")


if __name__ == "__main__":
    main()

    