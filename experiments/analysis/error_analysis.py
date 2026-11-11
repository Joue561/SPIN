# -*- coding: utf-8 -*-
"""
error_analysis.py
知识库会话树推断的误差分析（不调用 API）。输入为 analyze_combined_prior.py 输出的 pred_combined.jsonl
（含 base_prediction 与 predicted_stance），或任意含两列预测的文件。

输出：
  A. 总览：改动数、改对 / 改错 / 错改错，none 预测占比的变化（与 gold 对比）
  B. 改动流向：基础预测 → 修正后预测 的矩阵（只统计被改动的节点），分改对 / 改错
  C. 按父节点预测是否正确：父节点修正后预测对 / 错时，子节点被改对、改错的比例
  D. 按真实父子关系（继承 / 反转 / 中立化 / 表态 / 保持中立）：改对、改错、净收益
  E. 按 LLM 标注的传导类型：净收益，以及该标注与真实关系是否一致时的净收益
  F. 按话题（或目标）：基础与修正后的 Macro-F1 / F_avg、净收益，
     以及 test 与 train 的父子关系分布差异（JS 散度，越大说明知识库学到的规律越不适用）
  G. 按深度：净收益
  H. 案例：随机抽取改对、改错各 N 条（含文本），写入 cases.jsonl，并在屏幕上各显示 3 条

用法（make_rag_data 目录下）：
python spin/experiments/analysis/error_analysis.py \
  --instances processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
  --pred baseline_outputs_llama70b/combined_prior_wide_llama70b/pred_combined.jsonl \
  --group_field topic --name "C-MTCSD Llama-70B" --output_dir error_analysis/cmtcsd_llama70b
ZS-CSD 按目标分组用 --group_field main_target。
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LABELS = ["favor", "against", "none"]
# LLM 传导类型 → 它所暗示的父子关系（用于检查标注是否与真实关系一致）
SP2REL = {"stance_reversal": "reverse", "sarcastic_reversal": "reverse",
          "stance_inheritance": "inherit", "stance_amplification": "inherit",
          "stance_softening": "neutralize"}


def rel(p, c):
    if p in ("favor", "against"):
        return "继承" if c == p else ("中立化" if c == "none" else "反转")
    if p == "none":
        return "表态" if c != "none" else "保持中立"
    return "无父节点"


REL_EN = {"继承": "inherit", "反转": "reverse", "中立化": "neutralize"}


def f1s(g, p):
    out = []
    for c in LABELS:
        tp = sum(a == c == b for a, b in zip(g, p)); fp = sum(a != c and b == c for a, b in zip(g, p))
        fn = sum(a == c and b != c for a, b in zip(g, p))
        out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return np.mean(out), (out[0] + out[1]) / 2


def js(p, q):
    p = np.asarray(p, float) + 1e-9; q = np.asarray(q, float) + 1e-9
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--base_field", default="base_prediction")
    ap.add_argument("--group_field", default="topic", help="topic（C-MTCSD）或 main_target（ZS-CSD）")
    ap.add_argument("--name", default="")
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="error_analysis")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    ins = {}
    for r in load(a.instances):
        ins[(str(r.get("topic")), str(r.get("node_id")))] = r
    rows = load(a.pred)
    P = {}
    for r in rows:
        k = (str(r["topic"]), str(r["node_id"]))
        if k in ins and r.get(a.pred_field) in LABELS and r.get(a.base_field) in LABELS:
            P[k] = (r[a.base_field], r[a.pred_field])
    keys = list(P)
    gold = {k: ins[k]["stance_label"] for k in keys}

    def status(k):
        b, f = P[k]; g = gold[k]
        if b == f:
            return "未改"
        return "改对" if f == g else ("改错" if b == g else "错改错")

    st = {k: status(k) for k in keys}
    c = Counter(st.values())
    print(f"===== {a.name}  评估节点 {len(keys)}")
    print(f"【A. 总览】改动 {len(keys) - c['未改']}：改对 {c['改对']}，改错 {c['改错']}，错改错 {c['错改错']}；"
          f"净收益 {c['改对'] - c['改错']}（{100 * (c['改对'] - c['改错']) / len(keys):+.2f} 个点准确率）")
    for nm, idx in (("基础预测", 0), ("修正后", 1)):
        print(f"  {nm} 预测分布 " + "  ".join(f"{l} {sum(P[k][idx] == l for k in keys) / len(keys):.1%}" for l in LABELS))
    print("  gold 分布     " + "  ".join(f"{l} {sum(gold[k] == l for k in keys) / len(keys):.1%}" for l in LABELS))

    print("\n【B. 改动流向】行 = 基础预测，列 = 修正后（只统计被改动的节点）；括号内为其中改对 / 改错的条数")
    flow = Counter((P[k][0], P[k][1], st[k]) for k in keys if st[k] != "未改")
    for b in LABELS:
        cells = []
        for f in LABELS:
            if b == f:
                cells.append(f"{'—':>16s}"); continue
            n = sum(v for (x, y, s), v in flow.items() if x == b and y == f)
            cells.append(f"{n:5d}（{flow[(b, f, '改对')]}/{flow[(b, f, '改错')]}）".rjust(16))
        print(f"  {b:8s}" + "".join(cells))

    def table(title, groups, extra=None):
        print(f"\n{title}")
        print(f"  {'分组':16s} {'节点':>6s} {'改对':>5s} {'改错':>5s} {'净收益':>6s} {'改错率':>6s}" + (f"  {extra[0]}" if extra else ""))
        for g, ks in sorted(groups.items(), key=lambda x: -len(x[1])):
            cc = Counter(st[k] for k in ks)
            ch = len(ks) - cc["未改"]
            line = (f"  {str(g)[:16]:16s} {len(ks):6d} {cc['改对']:5d} {cc['改错']:5d} {cc['改对'] - cc['改错']:+6d} "
                    f"{(cc['改错'] / ch if ch else 0):6.1%}")
            if extra:
                line += f"  {extra[1](g, ks)}"
            print(line)

    # C. 父节点（修正后）预测是否正确
    grp = defaultdict(list)
    for k in keys:
        pid = ins[k].get("parent_id")
        pk = (k[0], str(pid)) if pid else None
        if not pk or pk not in P:
            grp["无父节点 / 父节点无预测"].append(k)
        else:
            grp["父节点修正后正确" if P[pk][1] == gold[pk] else "父节点修正后错误"].append(k)
    table("【C. 按父节点预测是否正确】", grp)

    # D. 真实父子关系
    grp = defaultdict(list)
    for k in keys:
        pl = ins[k].get("parent_stance_label")
        grp[rel(pl, gold[k])].append(k)
    table("【D. 按真实父子关系】", grp)

    # E. 传导类型及其与真实关系是否一致
    grp, agree = defaultdict(list), defaultdict(list)
    for k in keys:
        sp = str(ins[k].get("llm_semantic_propagation") or "unknown")
        grp[sp].append(k)
        r = rel(ins[k].get("parent_stance_label"), gold[k])
        if sp in SP2REL and r in REL_EN:
            agree["标注与真实关系一致" if SP2REL[sp] == REL_EN[r] else "标注与真实关系不一致"].append(k)
    table("【E1. 按 LLM 标注的传导类型】", grp)
    table("【E2. 传导类型标注是否与真实关系一致】（只统计父节点有立场、类型可映射的节点）", agree)

    # F. 话题 / 目标
    train_rel = defaultdict(Counter)
    for k, r in ins.items():
        if r.get("split") == "train" and r.get("stance_label") in LABELS:
            train_rel[r.get(a.group_field)][rel(r.get("parent_stance_label"), r["stance_label"])] += 1
    pooled_train = sum(train_rel.values(), Counter())
    RELS = ["继承", "反转", "中立化", "表态", "保持中立"]
    grp = defaultdict(list)
    for k in keys:
        grp[ins[k].get(a.group_field)].append(k)
    rows_f = []
    for g, ks in grp.items():
        gg = [gold[k] for k in ks]
        b_mf, b_fa = f1s(gg, [P[k][0] for k in ks]); f_mf, f_fa = f1s(gg, [P[k][1] for k in ks])
        te = Counter(rel(ins[k].get("parent_stance_label"), gold[k]) for k in ks)
        ref = train_rel.get(g) if sum(train_rel.get(g, Counter()).values()) >= 50 else pooled_train
        d = js([te[r] for r in RELS], [ref[r] for r in RELS])
        rows_f.append((f_mf - b_mf, g, len(ks), b_mf, f_mf, b_fa, f_fa, d))
    rows_f.sort()
    print(f"\n【F. 按{a.group_field}】Macro-F1 / F_avg（×100）基础 → 修正后；JS = test 与 train 父子关系分布的差异"
          f"（零样本目标与全部 train 比较）")
    show = rows_f if len(rows_f) <= 12 else rows_f[:6] + rows_f[-6:]
    for dm, g, n, bm, fm, bf, ff, d in show:
        print(f"  {str(g)[:22]:22s} n={n:5d}  Macro-F1 {100 * bm:5.2f}→{100 * fm:5.2f}（{100 * dm:+6.2f}）  "
              f"F_avg {100 * bf:5.2f}→{100 * ff:5.2f}  JS {d:.3f}")
    if len(rows_f) > 4:
        ds = np.array([x[0] for x in rows_f]); js_ = np.array([x[7] for x in rows_f])
        r = np.corrcoef(ds, js_)[0, 1]
        print(f"  分组净变化与 JS 的相关系数 r = {r:+.3f}（负相关说明分布差异越大，知识库越容易失效）")

    # G. 深度
    grp = defaultdict(list)
    for k in keys:
        d = int(ins[k].get("depth") or 0)
        grp[f"深度 {min(d, 6)}" + ("+" if d >= 6 else "")].append(k)
    table("【G. 按深度】", grp)

    # H. 案例
    rng = random.Random(a.seed)
    with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        for s in ("改对", "改错"):
            pool = [k for k in keys if st[k] == s]
            rng.shuffle(pool)
            for i, k in enumerate(pool[:a.cases]):
                r = ins[k]
                case = {"类型": s, "话题": k[0], "目标": r.get("main_target"), "node_id": k[1], "深度": r.get("depth"),
                        "父评论": r.get("parent_text"), "当前评论": r.get("current_text"),
                        "gold": gold[k], "基础预测": P[k][0], "修正后": P[k][1],
                        "父节点 gold": r.get("parent_stance_label"),
                        "父节点修正后预测": P.get((k[0], str(r.get("parent_id"))), (None, None))[1],
                        "传导类型": r.get("llm_semantic_propagation"), "回复关系": r.get("llm_reply_relation"),
                        "目标关系": r.get("llm_target_relation")}
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
                if i < 3:
                    print(f"\n  [{s}] {case['目标']} | 深度 {case['深度']} | gold {case['gold']} | 基础 {case['基础预测']} → {case['修正后']}"
                          f" | 父 gold {case['父节点 gold']} / 父预测 {case['父节点修正后预测']} | {case['传导类型']}")
                    print(f"    父：{str(case['父评论'])[:80]}")
                    print(f"    当前：{str(case['当前评论'])[:80]}")
    print(f"\n【H. 案例】改对、改错各最多 {a.cases} 条 → {out / 'cases.jsonl'}")


if __name__ == "__main__":
    main()