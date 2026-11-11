# -*- coding: utf-8 -*-
"""
map_sitpcl_preds.py
把 run_sitpcl.py 导出的逐条预测（按 ZS-CSD 原始 id）映射到本文的节点 id，
生成 analyze_combined_prior.py 可直接读取的预测文件。

映射方式：原代码评测时不打乱顺序，预测与数据文件按位置对齐（逐条核对 id）；
每条原始实例 (target, 路径 sentences) → 实例文件中 (split, main_target, ancestor_texts) 相同的节点。
重复路径（同一节点对应多条原始实例）取第一条。

输出（--out_dir）：
  predictions_sitpcl_valid.jsonl / predictions_sitpcl_test.jsonl
  每行 {topic, node_id, predicted_stance, confidence, probs, stance_label}
同时打印：原始 test 上的 F1-macro（与原论文 43.81 对比）、映射后节点上的 F1-macro。

用法（make_rag_data 目录下）：
python spin/base_models/sitpcl/map_sitpcl_preds.py --data_dir ZS-CSD/data \
  --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --raw_dir zscsd_exp/outputs/sitpcl/seed1234 --out_dir zscsd_exp/outputs/sitpcl/seed1234
"""
import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = ["favor", "against", "none"]
RAW = {"dev": ("valid", "dev_data.json"), "test": ("test", "test_data.json")}


def f1s(g, p):
    out = []
    for c in LABELS:
        tp = sum(a == c == b for a, b in zip(g, p))
        fp = sum(a != c and b == c for a, b in zip(g, p))
        fn = sum(a == c and b != c for a, b in zip(g, p))
        out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(out) / 3, (out[0] + out[1]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    lm = json.load(open(Path(a.data_dir) / "label.json", encoding="utf-8"))
    i2l = {v: ("none" if k == "neutral" else k) for k, v in lm.items()}

    idx = {}
    for line in open(a.instances, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            idx[(r["split"], r["main_target"], tuple(str(s) for s in r["ancestor_texts"]))] = r
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    for mode, (sp, fn) in RAW.items():
        raw_list = json.load(open(Path(a.data_dir) / fn, encoding="utf-8"))
        preds = [json.loads(l) for l in open(Path(a.raw_dir) / f"sitpcl_{mode}_raw.jsonl", encoding="utf-8") if l.strip()]
        # 原代码评测时 shuffle=False，预测顺序与数据文件一致：按位置对齐，并逐条核对 id
        if len(raw_list) != len(preds):
            raise SystemExit(f"{mode}：预测 {len(preds)} 条与数据 {len(raw_list)} 条数量不一致（是否用了 --debug？）")
        bad = sum(p["zscsd_id"] != r["id"] for p, r in zip(preds, raw_list))
        if bad:
            raise SystemExit(f"{mode}：有 {bad} 条预测的 id 与数据文件同一位置的 id 不一致，无法对齐")
        g_raw = [i2l[p["true"]] for p in preds]
        p_raw = [i2l[p["pred"]] for p in preds]
        mf, fa = f1s(g_raw, p_raw)
        print(f"【{sp}】SITPCL 原始实例 {len(preds)} 条：F1-macro {100 * mf:.2f}  F_avg {100 * fa:.2f}")
        rows, seen, stat = [], set(), Counter()
        for p, r in zip(preds, raw_list):
            node = idx.get((sp, r["target"], tuple(str(s) for s in r["sentences"])))
            if node is None:
                stat["找不到节点"] += 1; continue
            if node["node_id"] in seen:
                stat["重复路径（跳过）"] += 1; continue
            seen.add(node["node_id"])
            if i2l[p["true"]] != node["stance_label"]:
                stat["标签不一致"] += 1
            rows.append({"topic": node["topic"], "node_id": node["node_id"], "predicted_stance": i2l[p["pred"]],
                         "confidence": max(p["probs"]), "probs": p["probs"], "stance_label": node["stance_label"]})
        n_nodes = sum(1 for k in idx if k[0] == sp)
        mf, fa = f1s([r["stance_label"] for r in rows], [r["predicted_stance"] for r in rows])
        print(f"  映射到节点 {len(rows)} / {n_nodes}；{dict(stat) or '无异常'}")
        print(f"  映射后节点上：F1-macro {100 * mf:.2f}  F_avg {100 * fa:.2f}  预测分布 {dict(Counter(r['predicted_stance'] for r in rows))}")
        with open(out / f"predictions_sitpcl_{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  → {out / f'predictions_sitpcl_{sp}.jsonl'}")


if __name__ == "__main__":
    main()