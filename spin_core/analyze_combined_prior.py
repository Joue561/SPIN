# -*- coding: utf-8 -*-
"""
analyze_combined_prior.py
离线（不调用 API）：在会话树推断中同时使用知识库的两种知识，权重在 valid 上选择。

    q_i(y) ∝ E[y, ŷ_i] · p_rel_i(y)^λr · p_abs_i(y)^λa
      p_rel：相对传导知识 + 父节点后验（继承 / 反转 / 中立化 / 表态）
      p_abs：绝对知识，即该知识单元在本话题 train 中的立场分布（不看父节点）
没有父节点后验的节点（原帖、父节点不在评估集）只使用 p_abs。

λr、λa 在 valid 上按话题平均 MacroF1 选择（各话题共用，与 valid_shared 一致），test 只用于报告。
同时报告：只用相对知识（λa=0）、只用绝对知识（λr=0）两种特例（各自也在 valid 上选 λ），便于对比。

用法（make_rag_data 目录下）：
python -u spin/spin_core/analyze_combined_prior.py \
  --instances processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
  --pred baseline_outputs_llama8b/predictions_knn_fewshot_test.jsonl \
  --valid_pred baseline_outputs_llama8b/predictions_knn_fewshot_valid.jsonl \
  --output_dir combined_prior_llama8b
"""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("cd", Path(__file__).resolve().parent / "analyze_conflict_detection.py")
cd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cd)
kc, kbm, dd = cd.kc, cd.kbm, cd.dd
LABELS, LID = dd.LABELS, dd.LID


def forward(ev, base, ms, emis, lr, la):
    q, lab = {}, {}
    for n in sorted(ev, key=lambda x: x["depth"]):
        k = (n["topic"], n["node_id"])
        e = emis(n)[:, LID[base[k]]]
        m = ms[n["topic"]]
        qp = q.get(n["parent_key"]) if n["parent_key"] else None
        post = e.copy()
        if la > 0:
            post = post * m.direct_proba(n) ** la
        if qp is not None and lr > 0:
            post = post * m.prior(n, qp) ** lr
        post = post / post.sum()
        q[k] = post
        lab[k] = LABELS[int(post.argmax())] if (la > 0 or (qp is not None and lr > 0)) else base[k]
    return lab


def topic_avg(ev, lab, topics):
    return float(np.mean([cd.scores_of([n["y"] for n in ev if n["topic"] == t],
                                       [lab[(n["topic"], n["node_id"])] for n in ev if n["topic"] == t])[0]
                          for t in topics]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--kb", default="full", choices=list(cd.KB_KEYS))
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--output_dir", default="combined_prior")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, _ = kc.load_pred_conf(a.pred, a.pred_field, a.conf_field)
    vbase, vconf = kc.load_pred_conf(a.valid_pred, a.pred_field, a.conf_field)
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    ms = {t: kbm.PatternKB([n for n in train if n["topic"] == t], cd.KB_KEYS[a.kb], a.min_count) for t in topics}
    one = kc.make_binner(None)
    E = {t: kc.confusion([n for n in vev if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}
    emis = lambda n: E[n["topic"]]
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)} | KB:{a.kb}")

    grid = {}
    for lr, la in itertools.product(lams, lams):
        grid[(lr, la)] = topic_avg(vev, forward(vev, vbase, ms, emis, lr, la), topics)

    def pick(cands):
        return max(cands, key=lambda c: (grid[c], -(c[0] + c[1])))   # 同分取更小的 λ

    choices = {
        "只用相对知识": pick([c for c in grid if c[1] == 0]),
        "只用绝对知识": pick([c for c in grid if c[0] == 0]),
        "相对 + 绝对（组合）": pick(list(grid)),
    }
    g = [n["y"] for n in ev]
    keys = [(n["topic"], n["node_id"]) for n in ev]
    b_lab = {k: base[k] for k in keys}
    mf, fa, _ = cd.scores_of(g, [b_lab[k] for k in keys])
    print(f"\n【test 结果】（λr、λa 均在 valid 上选择）")
    print(f"  {'基础预测':22s} 合并MacroF1 {mf:.4f}  合并F_avg {fa:.4f}  话题平均MacroF1 {topic_avg(ev, b_lab, topics):.4f}")
    preds = {}
    for name, (lr, la) in choices.items():
        lab = forward(ev, base, ms, emis, lr, la)
        mf, fa, _ = cd.scores_of(g, [lab[k] for k in keys])
        ta = topic_avg(ev, lab, topics)
        fx = sum(lab[k] == n["y"] != base[k] for k, n in zip(keys, ev))
        br = sum(base[k] == n["y"] != lab[k] for k, n in zip(keys, ev))
        print(f"  {name:22s} 合并MacroF1 {mf:.4f}  合并F_avg {fa:.4f}  话题平均MacroF1 {ta:.4f}  "
              f"改对 {fx:4d} 改错 {br:4d}  λr={lr:g} λa={la:g}（valid {grid[(lr, la)]:.4f}）")
        preds[name] = lab

    print("\n【valid 上的 λ 网格（话题平均 MacroF1）】行=λr，列=λa")
    print("  λr\\λa " + " ".join(f"{la:>7g}" for la in lams))
    for lr in lams:
        print(f"  {lr:>6g} " + " ".join(f"{grid[(lr, la)]:7.4f}" for la in lams))

    comb = preds["相对 + 绝对（组合）"]
    with open(out / "pred_combined.jsonl", "w", encoding="utf-8") as f:
        for k, n in zip(keys, ev):
            f.write(json.dumps({"topic": k[0], "node_id": k[1], "depth": n["depth"], "stance_label": n["y"],
                                "predicted_stance": comb[k], "base_prediction": base[k],
                                "relative_only_prediction": preds["只用相对知识"][k],
                                "absolute_only_prediction": preds["只用绝对知识"][k]}, ensure_ascii=False) + "\n")
    print(f"\n已保存 {out / 'pred_combined.jsonl'}")
    print("显著性：--pred_field_b 可用 base_prediction / relative_only_prediction / absolute_only_prediction")


if __name__ == "__main__":
    main()