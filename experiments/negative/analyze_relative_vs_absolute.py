# -*- coding: utf-8 -*-
"""
analyze_relative_vs_absolute.py
离线验证（不调用 API）：结构化记忆跨话题失效，是否因为记忆存的是“相对主目标的绝对立场”，
而真正可迁移的是“相对父评论的立场变换”。

两个分析：
【分析一：跨话题稳定性】对每类模式键（回复关系、语义传导、候选模式、复合模式……），
  在父评论有立场（favor/against）的样本上，比较两种分布在各话题之间的差异（JSD，越小越稳定）：
    绝对分布  P(当前立场 ∈ {favor, against, none} | 模式)
    相对分布  P(变换 ∈ {inherit 继承, reverse 反转, neutralize 中立化} | 模式)
  两者都是 3 类，可以直接比较。

【分析二：留一话题预测】用记忆统计量直接预测 held-out 话题 test 样本的立场，比较三种“记忆”：
    A 绝对记忆        argmax P(当前立场 | 模式)                          ← 现在 RAG 记忆的做法
    B 绝对+父立场     argmax P(当前立场 | 模式, 父评论立场)
    C 相对记忆        argmax P(变换 | 模式) 再结合父评论立场得到当前立场   ← 拟议的新记忆
  每种都在两种记忆来源下计算：
    跨话题：只用其他话题的 train；同话题：只用该话题自己的 train
  迁移损失 = 同话题 − 跨话题。若 C 的迁移损失明显小于 A，说明相对变换才是可迁移的部分。
  模式键采用回退策略：复合模式 → 回复关系+语义传导 → 回复关系 → 全局（各方法相同，保证公平）。

注意：父评论立场使用数据中的 parent_stance_label（gold），因此 B/C 是“已知父立场”的上界分析；
实际 RAG 推理时父立场需由模型从上下文推断。

用法：
python spin/experiments/negative/analyze_relative_vs_absolute.py \
  --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \
  --output_dir relative_vs_absolute_analysis
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LABELS = ["favor", "against", "none"]
RELS = ["inherit", "reverse", "neutralize"]
ALIAS = {"favor": "favor", "favour": "favor", "support": "favor", "支持": "favor",
         "against": "against", "oppose": "against", "反对": "against",
         "none": "none", "neutral": "none", "中立": "none", "无立场": "none"}
OPP = {"favor": "against", "against": "favor"}


def norm_label(x):
    return None if x is None else ALIAS.get(str(x).strip().lower())


def nv(x):
    if x is None:
        return "unknown"
    if isinstance(x, (list, tuple)):
        x = "|".join(map(str, x))
    s = re.sub(r"[\s\-]+", "_", str(x).strip().lower())
    return s or "unknown"


def patterns(x):
    if x is None:
        return []
    items = x if isinstance(x, (list, tuple)) else re.split(r"[|｜;；]", str(x))
    out = []
    for it in items:
        if isinstance(it, dict):
            it = it.get("pattern") or it.get("name") or ""
        v = nv(it)
        if v not in ("", "unknown", "none") and v not in out:
            out.append(v)
    return out


def relation(parent, child):
    """父评论有立场时，当前评论相对父评论的变换。"""
    if parent not in ("favor", "against"):
        return None
    if child == "none":
        return "neutralize"
    return "inherit" if child == parent else "reverse"


def apply_relation(parent, rel):
    if rel == "inherit":
        return parent
    if rel == "reverse":
        return OPP[parent]
    return "none"


def jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist(counter, keys, alpha=0.5):
    v = np.array([counter.get(k, 0) for k in keys], float) + alpha
    return v / v.sum()


def macro_f1(g, p):
    lid = {l: i for i, l in enumerate(LABELS)}
    g = np.array([lid[x] for x in g]); p = np.array([lid[x] for x in p])
    cm = np.bincount(g * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(3), where=den > 0)
    return float(f1.mean()), float(tp.sum() / len(g)), float((f1[0] + f1[1]) / 2)


# ---------------------------------------------------------------------------
KEY_TYPES = {
    "reply_relation": lambda r: [(r["rr"],)],
    "semantic_propagation": lambda r: [(r["sp"],)],
    "reply+propagation": lambda r: [(r["rr"], r["sp"])],
    "target_relation": lambda r: [(r["tr"],)],
    "candidate_pattern": lambda r: [(p,) for p in r["pats"]],
    "composite(pattern,rr,sp,tr)": lambda r: [(p, r["rr"], r["sp"], r["tr"]) for p in r["pats"]],
}
BACKOFF = [("composite", lambda r: [("c", p, r["rr"], r["sp"], r["tr"]) for p in r["pats"]]),
           ("rr+sp", lambda r: [("rs", r["rr"], r["sp"])]),
           ("rr", lambda r: [("r", r["rr"])]),
           ("global", lambda r: [("g",)])]


def load(path, args):
    recs, skipped = [], Counter()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            skipped["bad_json"] += 1
            continue
        y, pa = norm_label(r.get("stance_label")), norm_label(r.get(args.parent_field))
        if y is None or pa is None:
            skipped["no_label"] += 1
            continue
        recs.append({"topic": r.get("topic"), "split": str(r.get("split", "")).lower(), "y": y, "pa": pa,
                     "rr": nv(r.get("llm_reply_relation")), "sp": nv(r.get("llm_semantic_propagation")),
                     "tr": nv(r.get("llm_target_relation")), "pats": patterns(r.get("llm_candidate_patterns")),
                     "llm_map": nv(r.get("llm_stance_mapping_to_parent"))})
    return recs, skipped


# ---------------------------- 分析一 ----------------------------
def stability(train, topics, args):
    rows = []
    for kname, kf in KEY_TYPES.items():
        abs_c = defaultdict(lambda: defaultdict(Counter))   # key -> topic -> label
        rel_c = defaultdict(lambda: defaultdict(Counter))   # key -> topic -> relation
        for r in train:
            if r["pa"] not in ("favor", "against"):
                continue
            for k in kf(r):
                abs_c[k][r["topic"]][r["y"]] += 1
                rel_c[k][r["topic"]][relation(r["pa"], r["y"])] += 1
        ja, jr, ws, nkeys = [], [], [], 0
        for k in abs_c:
            covered = [t for t in topics if sum(abs_c[k][t].values()) >= args.min_topic_count]
            if len(covered) < 2:
                continue
            nkeys += 1
            pooled_a = sum((abs_c[k][t] for t in covered), Counter())
            pooled_r = sum((rel_c[k][t] for t in covered), Counter())
            pa_, pr_ = dist(pooled_a, LABELS), dist(pooled_r, RELS)
            for t in covered:
                n = sum(abs_c[k][t].values())
                ja.append(jsd(dist(abs_c[k][t], LABELS), pa_))
                jr.append(jsd(dist(rel_c[k][t], RELS), pr_))
                ws.append(n)
        if not ws:
            continue
        w = np.array(ws, float)
        a, b = float(np.average(ja, weights=w)), float(np.average(jr, weights=w))
        rows.append({"key_type": kname, "n_keys": nkeys, "JSD_absolute": a, "JSD_relative": b,
                     "relative_more_stable": b < a, "ratio_rel/abs": b / a if a > 0 else float("nan")})
    return rows


# ---------------------------- 分析二 ----------------------------
class Mem:
    def __init__(self, recs, min_count):
        self.m = min_count
        self.a = defaultdict(Counter)                    # key -> label
        self.b = defaultdict(Counter)                    # (key, parent) -> label
        self.c = defaultdict(Counter)                    # key -> relation（仅父有立场）
        self.c_none = defaultdict(Counter)               # 父无立场时：key -> label（相对记忆无法定义，退回绝对）
        for r in recs:
            for _, f in BACKOFF:
                for k in f(r):
                    self.a[k][r["y"]] += 1
                    self.b[(k, r["pa"])][r["y"]] += 1
                    if r["pa"] in ("favor", "against"):
                        self.c[k][relation(r["pa"], r["y"])] += 1
                    else:
                        self.c_none[k][r["y"]] += 1

    def _lookup(self, table, keys_fn, r, wrap=lambda k: k):
        for _, f in BACKOFF:
            agg = Counter()
            for k in f(r):
                agg.update(table.get(wrap(k), Counter()))
            if sum(agg.values()) >= self.m or _ == "global":
                return agg
        return Counter()

    def predict(self, r):
        a = self._lookup(self.a, None, r)
        b = self._lookup(self.b, None, r, wrap=lambda k: (k, r["pa"]))
        pa = max(LABELS, key=lambda l: a.get(l, 0))
        pb = max(LABELS, key=lambda l: b.get(l, 0)) if sum(b.values()) else pa
        if r["pa"] in ("favor", "against"):
            c = self._lookup(self.c, None, r)
            pc = apply_relation(r["pa"], max(RELS, key=lambda x: c.get(x, 0))) if sum(c.values()) else pa
        else:
            c = self._lookup(self.c_none, None, r)
            pc = max(LABELS, key=lambda l: c.get(l, 0)) if sum(c.values()) else pa
        return pa, pb, pc


def loto(recs, topics, args):
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == args.eval_split]
    out = []
    for t in topics:
        test = [r for r in te if r["topic"] == t]
        if not test:
            continue
        for src_name, src in (("cross_topic", [r for r in tr if r["topic"] != t]),
                              ("in_topic", [r for r in tr if r["topic"] == t])):
            mem = Mem(src, args.min_count)
            P = {"A_absolute": [], "B_absolute+parent": [], "C_relative": []}
            for r in test:
                a, b, c = mem.predict(r)
                P["A_absolute"].append(a); P["B_absolute+parent"].append(b); P["C_relative"].append(c)
            g = [r["y"] for r in test]
            polar = [i for i, r in enumerate(test) if r["pa"] in ("favor", "against")]
            for m, p in P.items():
                f, acc, fa = macro_f1(g, p)
                fp, accp, _ = macro_f1([g[i] for i in polar], [p[i] for i in polar]) if polar else (float("nan"),) * 3
                out.append({"heldout_topic": t, "memory_source": src_name, "method": m, "n": len(g),
                            "macro_f1": f, "acc": acc, "f_avg": fa, "n_parent_polar": len(polar),
                            "macro_f1_parent_polar": fp, "acc_parent_polar": accp})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--output_dir", default="relative_vs_absolute_analysis")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--min_count", type=int, default=10, help="回退查表时键的最小样本数（与记忆构建默认值一致）")
    ap.add_argument("--min_topic_count", type=int, default=10, help="稳定性分析中，话题内某模式至少多少样本才计入")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    recs, skipped = load(a.instances, a)
    topics = sorted({r["topic"] for r in recs})
    tr = [r for r in recs if r["split"] == "train"]
    print(f"样本 {len(recs)}（跳过 {dict(skipped)}）| train {len(tr)} | 话题 {topics}")
    print("父评论立场分布（train）：", dict(Counter(r["pa"] for r in tr)))
    rel_cnt = Counter(relation(r["pa"], r["y"]) for r in tr if r["pa"] in ("favor", "against"))
    print("相对变换分布（train，父有立场）：", dict(rel_cnt))
    agree = Counter((r["llm_map"], relation(r["pa"], r["y"])) for r in tr if r["pa"] in ("favor", "against"))
    write_csv(out / "llm_mapping_vs_gold_relation.csv",
              [{"llm_stance_mapping_to_parent": k[0], "gold_relation": k[1], "count": v} for k, v in agree.most_common()])

    # 分析一
    st = stability(tr, topics, a)
    write_csv(out / "stability_absolute_vs_relative.csv", st)
    print("\n【分析一】跨话题差异（加权平均 JSD，越小越稳定；仅父评论有立场的 train 样本）")
    print(f"{'模式键':30s} {'键数':>5s} {'绝对立场JSD':>12s} {'相对变换JSD':>12s} {'相对/绝对':>9s}")
    for r in st:
        print(f"{r['key_type']:30s} {r['n_keys']:5d} {r['JSD_absolute']:12.4f} {r['JSD_relative']:12.4f} {r['ratio_rel/abs']:9.2f}")

    # 分析二
    lo = loto(recs, topics, a)
    write_csv(out / "loto_prediction.csv", lo)
    print(f"\n【分析二】用记忆统计量直接预测 held-out 话题 {a.eval_split} 样本（Macro-F1）")
    methods = ["A_absolute", "B_absolute+parent", "C_relative"]
    summary = []
    print(f"{'话题':24s} {'方法':20s} {'同话题':>8s} {'跨话题':>8s} {'迁移损失':>8s}")
    for t in topics:
        for m in methods:
            g = {x["memory_source"]: x for x in lo if x["heldout_topic"] == t and x["method"] == m}
            if len(g) < 2:
                continue
            i, c = g["in_topic"]["macro_f1"], g["cross_topic"]["macro_f1"]
            summary.append({"topic": t, "method": m, "in": i, "cross": c, "loss": i - c,
                            "in_polar": g["in_topic"]["macro_f1_parent_polar"],
                            "cross_polar": g["cross_topic"]["macro_f1_parent_polar"]})
            print(f"{t[:24]:24s} {m:20s} {i:8.4f} {c:8.4f} {i - c:+8.4f}")
    print("\n话题平均：")
    print(f"{'方法':20s} {'同话题':>8s} {'跨话题':>8s} {'迁移损失':>8s} | {'父有立场子集: 同话题':>18s} {'跨话题':>8s} {'迁移损失':>8s}")
    avg_rows = []
    for m in methods:
        s = [x for x in summary if x["method"] == m]
        if not s:
            continue
        r = {k: float(np.nanmean([x[k] for x in s])) for k in ("in", "cross", "loss", "in_polar", "cross_polar")}
        r["loss_polar"] = r["in_polar"] - r["cross_polar"]
        r["method"] = m
        avg_rows.append(r)
        print(f"{m:20s} {r['in']:8.4f} {r['cross']:8.4f} {r['loss']:+8.4f} | {r['in_polar']:18.4f} {r['cross_polar']:8.4f} {r['loss_polar']:+8.4f}")
    write_csv(out / "loto_summary_topic_avg.csv", avg_rows)

    # 结论
    A = next((r for r in avg_rows if r["method"] == "A_absolute"), None)
    C = next((r for r in avg_rows if r["method"] == "C_relative"), None)
    n_stable = sum(1 for r in st if r["relative_more_stable"])
    concl = {
        "stability_relative_more_stable_keytypes": f"{n_stable}/{len(st)}",
        "transfer_loss_absolute": A["loss"] if A else None,
        "transfer_loss_relative": C["loss"] if C else None,
        "cross_topic_macro_f1_absolute": A["cross"] if A else None,
        "cross_topic_macro_f1_relative": C["cross"] if C else None,
        "note": "B/C 使用 gold 父评论立场，是已知父立场条件下的上界分析",
    }
    json.dump(concl, open(out / "conclusion.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n结论：")
    print(f"  相对变换比绝对立场更稳定的模式键类型：{concl['stability_relative_more_stable_keytypes']}")
    if A and C:
        print(f"  迁移损失（话题平均）：绝对记忆 {A['loss']:+.4f}  vs  相对记忆 {C['loss']:+.4f}")
        print(f"  跨话题 Macro-F1：绝对记忆 {A['cross']:.4f}  vs  相对记忆 {C['cross']:.4f}")
    print(f"\n结果已保存到 {out.resolve()}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()