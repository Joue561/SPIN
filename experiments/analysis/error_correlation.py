# -*- coding: utf-8 -*-
"""
error_correlation.py
统计基础模型在父子节点上的预测误差是否相关（不调用 API）。

会话树推断假设：给定真实立场，各节点的预测误差相互独立（观测模型 E 逐节点作用）。
若父节点预测错时子节点也更容易出错、并且错成“与父节点一致”的方向，
沿树传播的先验会放大错误，而不是纠正错误。

对每个预测文件，在 test 上统计所有父子都有预测的节点对：
  P(子错 | 父错)、P(子错 | 父对)、两者之比（风险比，>1 表示误差正相关）
  父子都错时，子的错误预测与父的错误预测“同向”的比例（相对于真实关系应有的方向）
  phi 系数（父错、子错两个二值变量的相关系数）

用法（make_rag_data 目录下）：
python spin/experiments/analysis/error_correlation.py \
  --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --pred SITPCL=zscsd_exp/outputs/sitpcl/seed1234/predictions_sitpcl_test.jsonl \
         GPT-4o-mini=zscsd_exp/outputs/gpt4omini/predictions_knn_fewshot_test.jsonl
"""
import argparse
import json
import math


def load_nodes(path):
    nodes = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("split") == "test":
                nodes[(str(r["topic"]), str(r["node_id"]))] = r
    return nodes


def load_pred(path):
    p = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            v = r.get("predicted_stance")
            if v in ("favor", "against", "none"):
                p[(str(r["topic"]), str(r["node_id"]))] = v
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="名字=预测文件，可给多个")
    a = ap.parse_args()
    nodes = load_nodes(a.instances)
    print(f"{'模型':14s} {'节点对':>6s} {'P(子错|父错)':>12s} {'P(子错|父对)':>12s} {'风险比':>6s} {'phi':>6s} "
          f"{'父子都错时子与父预测相同':>22s} {'（gold 中父子相同）':>16s}")
    for item in a.pred:
        name, path = item.split("=", 1)
        pred = load_pred(path)
        n11 = n10 = n01 = n00 = 0
        same_wrong = both_wrong = 0
        gold_same = pairs = 0
        for k, r in nodes.items():
            pid = r.get("parent_id")
            if not pid:
                continue
            pk = (k[0], str(pid))
            if k not in pred or pk not in pred or pk not in nodes:
                continue
            yc, yp = r["stance_label"], nodes[pk]["stance_label"]
            pw, cw = pred[pk] != yp, pred[k] != yc
            pairs += 1
            gold_same += yc == yp
            if pw and cw:
                n11 += 1; both_wrong += 1; same_wrong += pred[k] == pred[pk]
            elif pw:
                n10 += 1
            elif cw:
                n01 += 1
            else:
                n00 += 1
        p_ww = n11 / max(n11 + n10, 1)
        p_rw = n01 / max(n01 + n00, 1)
        denom = math.sqrt(max((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00), 1))
        phi = (n11 * n00 - n10 * n01) / denom
        print(f"{name:14s} {pairs:6d} {p_ww:12.3f} {p_rw:12.3f} {p_ww / max(p_rw, 1e-9):6.2f} {phi:6.3f} "
              f"{same_wrong / max(both_wrong, 1):22.3f} {gold_same / max(pairs, 1):16.3f}")


if __name__ == "__main__":
    main()