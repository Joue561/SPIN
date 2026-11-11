# -*- coding: utf-8 -*-
"""
analyze_combined_ablation.py
主方法（相对 + 绝对知识的会话树推断，统一 λ 协议）的消融与上界，不调用 API。

在同一份基础预测上比较：
  检索键   full（完整组合，主方法）/ sp_depth（传导类型+深度）/ sp（传导类型）/ depth（仅深度，无语义信息）
           / random（随机键：每个节点用一个随机 train 样本的完整组合键检索）
  父立场   pred（推断时用父节点的后验，主方法）/ gold（用父节点的人工标签，上界；只换父立场，其余不变）
每个变体的 λr、λa 都在 valid 上独立选择（valid 上的父立场取法与 test 相同），test 只用于报告。

输出：
  屏幕：各变体的合并 Macro-F1 / F_avg、话题平均 Macro-F1 / F_avg、改对 / 改错、选出的 λ
  pred_ablation.jsonl：每个节点各变体的预测（字段 pred_<变体>），配合 paired_significance.py 做两两检验

用法（make_rag_data 目录下）：
python -u spin/experiments/ablation/analyze_combined_ablation.py \
  --instances processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
  --pred baseline_outputs/predictions_knn_fewshot_test.jsonl \
  --valid_pred baseline_outputs/predictions_knn_fewshot_valid.jsonl \
  --output_dir ablation_gpt4omini
"""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("cp", (Path(__file__).resolve().parent / "../../spin_core/analyze_combined_prior.py").resolve())
cp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cp)
cd, kc, kbm, dd = cp.cd, cp.kc, cp.kbm, cp.dd
LABELS, LID = dd.LABELS, dd.LID

VARIANTS = [("full", "pred"), ("sp_depth", "pred"), ("sp", "pred"), ("depth", "pred"), ("random", "pred"), ("full", "gold")]
NAMES = {("full", "pred"): "完整组合（主方法）", ("sp_depth", "pred"): "传导类型+深度", ("sp", "pred"): "传导类型",
         ("depth", "pred"): "仅深度（无语义）", ("random", "pred"): "随机键", ("full", "gold"): "完整组合 + 人工父立场（上界）"}


def forward(ev, base, ms, emis, lr, la, parent):
    q, lab = {}, {}
    for n in sorted(ev, key=lambda x: x["depth"]):
        k = (n["topic"], n["node_id"])
        m = ms[n["topic"]]
        post = emis(n)[:, LID[base[k]]].copy()
        if parent == "gold":
            qp = np.eye(3)[LID[n["pa_gold"]]] if n.get("parent_key") and n.get("pa_gold") in LID else None
        else:
            qp = q.get(n["parent_key"]) if n["parent_key"] else None
        if la > 0:
            post = post * m.direct_proba(n) ** la
        if qp is not None and lr > 0:
            post = post * m.prior(n, qp) ** lr
        post = post / post.sum()
        q[k] = post
        lab[k] = LABELS[int(post.argmax())] if (la > 0 or (qp is not None and lr > 0)) else base[k]
    return lab


def metrics(ev, lab, topics):
    g = [n["y"] for n in ev]
    p = [lab[(n["topic"], n["node_id"])] for n in ev]
    mf, fa, _ = cd.scores_of(g, p)
    per = [cd.scores_of([n["y"] for n in ev if n["topic"] == t], [lab[(n["topic"], n["node_id"])] for n in ev if n["topic"] == t])
           for t in topics]
    return mf, fa, float(np.mean([x[0] for x in per])), float(np.mean([x[1] for x in per]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2,3,4,6")
    ap.add_argument("--variants", default="all", help="逗号分隔，如 full-pred,depth-pred；默认全部")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--output_dir", default="ablation")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]
    variants = VARIANTS if a.variants == "all" else [tuple(v.split("-")) for v in a.variants.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, _ = kc.load_pred_conf(a.pred, a.pred_field, a.conf_field)
    vbase, vconf = kc.load_pred_conf(a.valid_pred, a.pred_field, a.conf_field)
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    one = kc.make_binner(None)
    E = {t: kc.confusion([n for n in vev if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}
    emis = lambda n: E[n["topic"]]
    kbs = {}
    for key in {v[0] for v in variants}:
        if key == "random":
            kbs[key] = {t: kbm.PatternKB([n for n in train if n["topic"] == t], kbm.k_full, a.min_count,
                                         eval_keyfn=kbm.random_keyfn([n for n in train if n["topic"] == t], kbm.k_full, "rand"))
                        for t in topics}
        else:
            kbs[key] = {t: kbm.PatternKB([n for n in train if n["topic"] == t], cd.KB_KEYS[key], a.min_count) for t in topics}
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)} | 话题 {len(topics)}")

    keys = [(n["topic"], n["node_id"]) for n in ev]
    b = {k: base[k] for k in keys}
    mf, fa, tmf, tfa = metrics(ev, b, topics)
    print(f"\n{'变体':26s} {'合并MacroF1':>10s} {'合并F_avg':>9s} {'话题平均MacroF1':>14s} {'话题平均F_avg':>12s}  改对/改错   λr/λa（valid）")
    print(f"  {'基础预测':24s} {mf:10.4f} {fa:9.4f} {tmf:14.4f} {tfa:12.4f}")
    preds = {}
    for key, parent in variants:
        ms = kbs[key]
        grid = {c: cp.topic_avg(vev, forward(vev, vbase, ms, emis, *c, parent), topics) for c in itertools.product(lams, lams)}
        c = max(grid, key=lambda c: (grid[c], -(c[0] + c[1])))
        lab = forward(ev, base, ms, emis, *c, parent)
        mf, fa, tmf, tfa = metrics(ev, lab, topics)
        fx = sum(lab[k] == n["y"] != base[k] for k, n in zip(keys, ev))
        br = sum(base[k] == n["y"] != lab[k] for k, n in zip(keys, ev))
        print(f"  {NAMES[(key, parent)]:24s} {mf:10.4f} {fa:9.4f} {tmf:14.4f} {tfa:12.4f}  {fx:4d}/{br:<4d}   {c[0]:g}/{c[1]:g}（{grid[c]:.4f}）", flush=True)
        preds[f"pred_{key}_{parent}"] = lab

    with open(out / "pred_ablation.jsonl", "w", encoding="utf-8") as f:
        for k, n in zip(keys, ev):
            r = {"topic": k[0], "node_id": k[1], "depth": n["depth"], "stance_label": n["y"], "base_prediction": base[k]}
            r.update({fld: lab[k] for fld, lab in preds.items()})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已保存 {out / 'pred_ablation.jsonl'}；字段 {', '.join(preds)}")


if __name__ == "__main__":
    main()