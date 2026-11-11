# -*- coding: utf-8 -*-
"""
motivation_stats.py
论文 §3（动机分析）用：LLM 能否判断「当前回复的立场与父评论立场之间的关系」。不调用 API。

评估集与 analyze_retrieval_significance.py 相同：test 中父评论 gold 立场为 favor / against 的节点；
真实关系 ∈ {inherit, reverse, neutralize}（由父子 gold 立场决定）。比较以下几种「关系判断」的准确率：
  多数类                 ：train 中最常见的关系
  LLM 标注的传导类型      ：llm_semantic_propagation 映射到关系（与 error_analysis.py 的 SP2REL 相同）；
                           不可映射的类型（unclear / target_shift / direct_expression …）计为错，另报可映射子集上的准确率
  LLM 标注的立场映射      ：llm_stance_mapping_to_parent（same_as_parent→inherit，opposite_to_parent→reverse）
  基础预测隐含的关系      ：rel(gold 父立场, ŷ_子)，即在已知真实父立场时，基础模型对子节点的预测所隐含的关系
  基础预测隐含的关系（父子都用预测）：rel(ŷ_父, ŷ_子)，ŷ_父 为 none 时计为错

用法（make_rag_data 目录下）：
python spin/experiments/ablation/motivation_stats.py \
  --instances processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
  --pred GPT-4o-mini=baseline_outputs/predictions_knn_fewshot_test.jsonl
"""
import argparse
import json
from collections import Counter

LABELS = ("favor", "against", "none")
SP2REL = {"stance_reversal": "reverse", "sarcastic_reversal": "reverse",
          "stance_inheritance": "inherit", "stance_amplification": "inherit",
          "stance_softening": "neutralize"}
MAP2REL = {"same_as_parent": "inherit", "opposite_to_parent": "reverse"}


def rel(p, c):
    if p not in ("favor", "against") or c not in LABELS:
        return None
    if c == "none":
        return "neutralize"
    return "inherit" if c == p else "reverse"


def norm(x):
    x = str(x or "").strip().lower()
    return x if x in LABELS else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", nargs="*", default=[], help="名字=test 预测文件（predicted_stance），可给多个")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.instances, encoding="utf-8") if l.strip()]
    by = {(str(r.get("topic")), str(r.get("node_id"))): r for r in rows}
    train_rel = Counter(rel(norm(r.get("parent_stance_label")), norm(r.get("stance_label")))
                        for r in rows if str(r.get("split")).lower() == "train")
    train_rel.pop(None, None)
    major = train_rel.most_common(1)[0][0]

    ev = []
    for k, r in by.items():
        if str(r.get("split")).lower() != "test":
            continue
        g = rel(norm(r.get("parent_stance_label")), norm(r.get("stance_label")))
        if g:
            ev.append((k, r, g))
    n = len(ev)
    gd = Counter(g for _, _, g in ev)
    print(f"评估节点（test，父评论 gold 有立场）{n}；真实关系分布 " +
          "  ".join(f"{x} {gd[x] / n:.1%}" for x in ("inherit", "reverse", "neutralize")))
    print(f"train 关系分布 " + "  ".join(f"{x} {train_rel[x] / sum(train_rel.values()):.1%}" for x in ("inherit", "reverse", "neutralize")))

    def report(name, preds):
        acc = sum(p == g for p, (_, _, g) in zip(preds, ev)) / n
        cov = [(p, g) for p, (_, _, g) in zip(preds, ev) if p]
        sub = sum(p == g for p, g in cov) / max(len(cov), 1)
        print(f"  {name:34s} 准确率 {acc:.3f}   （可判断的 {len(cov) / n:5.1%} 节点上 {sub:.3f}）")

    print("\n【关系判断准确率】")
    report("多数类（" + major + "）", [major] * n)
    report("LLM 标注的传导类型", [SP2REL.get(str(r.get("llm_semantic_propagation"))) for _, r, _ in ev])
    report("LLM 标注的立场映射", [MAP2REL.get(str(r.get("llm_stance_mapping_to_parent"))) for _, r, _ in ev])

    for item in a.pred:
        name, path = item.split("=", 1)
        pr = {}
        for line in open(path, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                v = norm(x.get("predicted_stance"))
                if v:
                    pr[(str(x.get("topic")), str(x.get("node_id")))] = v
        implied, both = [], []
        for k, r, g in ev:
            yc = pr.get(k)
            implied.append(rel(norm(r.get("parent_stance_label")), yc) if yc else None)
            pk = (k[0], str(r.get("parent_id")))
            both.append(rel(pr.get(pk), yc) if yc and pk in pr else None)
        report(f"{name}：rel(gold 父, ŷ 子)", implied)
        report(f"{name}：rel(ŷ 父, ŷ 子)", both)


if __name__ == "__main__":
    main()