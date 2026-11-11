# -*- coding: utf-8 -*-
"""
dataset_stats.py
一次输出论文表 1（数据集统计）和实验设置里需要的数字，不调用 API。

对每个数据集文件统计：
  各划分的节点数（有立场标签的 / 全部）、话题数、目标数（及 train/test 目标重叠，用于确认零样本）
  最大深度、平均深度、深度分布
  各划分的标签分布（favor / against / none）
  树的数量、根节点数、有父节点的比例
  语义标注缺失的节点数（llm_annotation_source == missing，或传导类型为空 / unknown）

用法（make_rag_data 目录下）：
python spin/experiments/main/dataset_stats.py \
  --data C-MTCSD=processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
         ZS-CSD=zscsd_exp/data/zscsd_semantic_instances_llm.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict

LABELS = ("favor", "against", "none")
SPLITS = ("train", "valid", "test")


def norm(x):
    x = str(x or "").strip().lower()
    return x if x in LABELS else None


def missing_annotation(r):
    if str(r.get("llm_annotation_source", "")).lower() == "missing":
        return True
    sp = str(r.get("llm_semantic_propagation") or "").strip().lower()
    return sp in ("", "unknown", "none", "null")


def stats(name, path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[str(r.get("split", "")).lower()].append(r)

    print(f"\n==================== {name}  （{path}）")
    print(f"总行数 {len(rows)}")
    topics = {str(r.get("topic")) for r in rows}
    print(f"话题数 {len(topics)}：{', '.join(sorted(topics)[:20])}{' …' if len(topics) > 20 else ''}")

    targets = {s: {str(r.get("main_target")) for r in by[s]} for s in SPLITS}
    print("目标数  " + "  ".join(f"{s} {len(targets[s])}" for s in SPLITS)
          + f"  | train∩test {len(targets['train'] & targets['test'])}"
          + f"  train∩valid {len(targets['train'] & targets['valid'])}")

    print(f"\n{'划分':6s} {'全部':>7s} {'有标签':>7s} {'favor':>14s} {'against':>14s} {'none':>14s} "
          f"{'树':>5s} {'根':>5s} {'最大深度':>6s} {'平均深度':>6s} {'标注缺失':>6s}")
    tot = Counter()
    for s in SPLITS + ("all",):
        rs = rows if s == "all" else by[s]
        lab = [norm(r.get("stance_label")) for r in rs]
        lab = [x for x in lab if x]
        c = Counter(lab)
        n = len(lab)
        depths = [int(r["depth"]) for r in rs if str(r.get("depth", "")).lstrip("-").isdigit()]
        trees = {(str(r.get("topic")), str(r.get("node_id")).split("-")[0]) for r in rs}
        roots = sum(1 for r in rs if r.get("parent_id") in (None, "", "None"))
        miss = sum(missing_annotation(r) for r in rs)
        cell = lambda l: f"{c[l]:6d} ({100 * c[l] / max(n, 1):4.1f}%)"
        print(f"{s:6s} {len(rs):7d} {n:7d} {cell('favor'):>14s} {cell('against'):>14s} {cell('none'):>14s} "
              f"{len(trees):5d} {roots:5d} {max(depths) if depths else -1:6d} "
              f"{sum(depths) / max(len(depths), 1):6.2f} {miss:6d}")
        if s != "all":
            tot["n"] += n

    dc = Counter(int(r["depth"]) for r in rows if str(r.get("depth", "")).lstrip("-").isdigit())
    print("\n深度分布（全部划分）：" + "  ".join(f"{d}:{dc[d]}" for d in sorted(dc)))

    if len(topics) > 1:
        print("\n各话题节点数（有标签，train / valid / test）：")
        for t in sorted(topics):
            cnt = [sum(1 for r in by[s] if str(r.get("topic")) == t and norm(r.get("stance_label"))) for s in SPLITS]
            print(f"  {t:28s} {cnt[0]:6d} / {cnt[1]:5d} / {cnt[2]:5d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="名字=文件，可给多个")
    a = ap.parse_args()
    for item in a.data:
        name, path = item.split("=", 1)
        stats(name, path)


if __name__ == "__main__":
    main()