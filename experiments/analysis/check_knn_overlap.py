# -*- coding: utf-8 -*-
"""
check_knn_overlap.py
离线检查（不调用 API）：knn_fewshot 的优势是否来自 train 与 test 之间的重复 / 近似重复评论。

1) 精确重复：test 样本的 (父评论, 当前评论) 或较长的当前评论（≥15 字）是否原样出现在同话题 train 中，以及标签是否一致
2) 按 top-1 检索相似度分档：各档样本数、knn_fewshot 准确率、top-1 示例标签与 gold 一致率，以及其他方法在同一档上的准确率
3) 去掉高相似样本（top-1 相似度 ≥ --high_sim）后，各方法的 Macro-F1 / F_avg

用法（make_rag_data 目录下）：
python spin/experiments/analysis/check_knn_overlap.py \
  --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl \
  --knn baseline_outputs/predictions_knn_fewshot_test.jsonl \
  --other ours_KB_tree=kb_confidence_causal/pred_KB_完整组合_pred_plain_valid_shared.jsonl \
          dynamic_semantic=rag_prediction_outputs_causal/predictions_dynamic_semantic_test.jsonl \
          cot=baseline_outputs/predictions_cot_test.jsonl \
          full_path=baseline_outputs/predictions_full_path_test.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict

LABELS = ["favor", "against", "none"]


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def norm(x):
    x = str(x or "").strip().lower()
    return x if x in LABELS else None


def scores(gold, pred):
    f = {}
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f[l] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    acc = sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1)
    return acc, sum(f.values()) / 3, (f["favor"] + f["against"]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--knn", required=True)
    ap.add_argument("--other", nargs="*", default=[], help="名称=预测文件，在同一子集上对比")
    ap.add_argument("--high_sim", type=float, default=0.9)
    a = ap.parse_args()

    inst = load(a.instances)
    gold = {}
    tr_pair, tr_cur = defaultdict(Counter), defaultdict(Counter)
    for r in inst:
        k = (str(r.get("topic")), str(r.get("node_id")))
        y = norm(r.get("stance_label"))
        if r.get("split") == "train" and y:
            cur, par = str(r.get("current_text") or "").strip(), str(r.get("parent_text") or "").strip()
            tr_pair[(r.get("topic"), par, cur)][y] += 1
            if len(cur) >= 15:
                tr_cur[(r.get("topic"), cur)][y] += 1
        if r.get("split") == "test" and y:
            gold[k] = (y, r)

    # ---- 1) 精确重复
    n_pair = n_pair_same = n_cur = n_cur_same = 0
    by_topic = defaultdict(lambda: [0, 0, 0])
    for k, (y, r) in gold.items():
        cur, par = str(r.get("current_text") or "").strip(), str(r.get("parent_text") or "").strip()
        c = tr_pair.get((r.get("topic"), par, cur))
        if c:
            n_pair += 1; n_pair_same += c.most_common(1)[0][0] == y
            by_topic[k[0]][0] += 1
        c2 = tr_cur.get((r.get("topic"), cur)) if len(cur) >= 15 else None
        if c2:
            n_cur += 1; n_cur_same += c2.most_common(1)[0][0] == y
            by_topic[k[0]][1] += 1
        by_topic[k[0]][2] += 1
    print(f"test 有标签样本 {len(gold)}")
    print("\n【1. 与同话题 train 的精确重复】")
    print(f"  (父评论, 当前评论) 完全相同：{n_pair} 条（{n_pair / len(gold):.1%}），其中 train 多数标签与 gold 一致 {n_pair_same} 条")
    print(f"  当前评论（≥15 字）完全相同：{n_cur} 条（{n_cur / len(gold):.1%}），其中标签一致 {n_cur_same} 条")
    for t, (p, c, n) in sorted(by_topic.items()):
        print(f"    {t:24s} n={n:4d}  父+当前重复 {p:4d}（{p / n:.1%}）  长当前评论重复 {c:4d}（{c / n:.1%}）")

    # ---- 2) 按 top-1 相似度分档
    knn = {(str(r.get("topic")), str(r.get("node_id"))): r for r in load(a.knn)}
    others = {}
    for s in a.other:
        name, path = s.split("=", 1)
        others[name] = {(str(r.get("topic")), str(r.get("node_id"))): norm(r.get("predicted_stance")) for r in load(path)}
    keys = [k for k in gold if k in knn and all(k in o for o in others.values())]
    print(f"\n共同样本 {len(keys)} 条")

    def top1(k):
        ex = knn[k].get("fewshot_examples") or []
        return (ex[0]["sim"], ex[0]["label"]) if ex else (0.0, None)

    bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]
    print("\n【2. 按 top-1 示例相似度分档】准确率")
    head = f"  {'相似度':12s} {'n':>5s} {'top1标签=gold':>13s} {'knn':>7s}" + "".join(f" {nm[:12]:>12s}" for nm in others)
    print(head)
    for lo, hi in bins:
        ks = [k for k in keys if lo <= top1(k)[0] < hi]
        if not ks:
            continue
        g = [gold[k][0] for k in ks]
        t1 = sum(top1(k)[1] == gold[k][0] for k in ks) / len(ks)
        accs = [sum(norm(knn[k].get("predicted_stance")) == gold[k][0] for k in ks) / len(ks)]
        accs += [sum(o[k] == gold[k][0] for k in ks) / len(ks) for o in others.values()]
        print(f"  [{lo:.2f},{min(hi, 1):.2f}) {len(ks):5d} {t1:13.3f} {accs[0]:7.3f}" + "".join(f" {x:12.3f}" for x in accs[1:]))

    # ---- 3) 去掉高相似样本
    print(f"\n【3. 去掉 top-1 相似度 ≥ {a.high_sim} 的样本后】")
    for subset_name, ks in (("全部", keys), (f"相似度<{a.high_sim}", [k for k in keys if top1(k)[0] < a.high_sim])):
        g = [gold[k][0] for k in ks]
        print(f"  {subset_name}（n={len(ks)}）")
        acc, mf1, fav = scores(g, [norm(knn[k].get("predicted_stance")) for k in ks])
        print(f"    {'knn_fewshot':16s} Acc {acc:.4f}  MacroF1 {mf1:.4f}  F_avg {fav:.4f}")
        for nm, o in others.items():
            acc, mf1, fav = scores(g, [o[k] for k in ks])
            print(f"    {nm[:16]:16s} Acc {acc:.4f}  MacroF1 {mf1:.4f}  F_avg {fav:.4f}")


if __name__ == "__main__":
    main()