# -*- coding: utf-8 -*-
"""Map qwen_sft_zscsd.py predictions (StanceAgent-R1 data/default ids) to our ZS-CSD node ids.

Each record (target, path sentences) -> the node in our instances file with the same
(split, main_target, ancestor_texts), as in map_sitpcl_preds.py. Duplicate paths keep the first.

python map_qwen_preds.py --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --data_dir StanceAgent-R1-main/data/default --pred_dir zscsd_exp/outputs/qwen7b_sft \
  --out_dir zscsd_exp/outputs/qwen7b_sft --name qwen7b_sft
Outputs predictions_<name>_{valid,test}.jsonl with {topic, node_id, predicted_stance, confidence, probs, stance_label}.
"""
import argparse, json
from pathlib import Path

CN2EN = {"支持": "favor", "反对": "against", "中立": "none"}
LABELS = ["favor", "against", "none"]


def f1(g, p):
    out = []
    for c in LABELS:
        tp = sum(a == c == b for a, b in zip(g, p)); fp = sum(a != c and b == c for a, b in zip(g, p))
        fn = sum(a == c and b != c for a, b in zip(g, p))
        out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return 100 * sum(out) / 3, 100 * (out[0] + out[1]) / 2


ap = argparse.ArgumentParser()
for k in ["instances", "data_dir", "pred_dir", "out_dir", "name"]:
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
idx = {}
for line in open(a.instances, encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        idx.setdefault((r["split"], r["main_target"], tuple(str(s) for s in r["ancestor_texts"])), r)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for mode, sp in [("dev", "valid"), ("test", "test")]:
    raw = {ex["id"]: ex for ex in json.load(open(Path(a.data_dir) / f"{mode}_data.json", encoding="utf-8"))}
    preds = [json.loads(l) for l in open(Path(a.pred_dir) / f"preds_{mode}.jsonl", encoding="utf-8") if l.strip()]
    seen, rows, miss = set(), [], 0
    for p in preds:
        ex = raw[p["id"]]
        r = idx.get((sp, ex["target"], tuple(str(s) for s in ex["sentences"])))
        if r is None:
            miss += 1; continue
        if r["node_id"] in seen:
            continue
        seen.add(r["node_id"])
        probs = {CN2EN[k]: v for k, v in p["probs"].items()}
        lab = CN2EN[p["pred"]]
        rows.append({"topic": r.get("topic", r["main_target"]), "node_id": r["node_id"], "predicted_stance": lab,
                     "confidence": probs[lab], "probs": probs, "stance_label": r.get("stance_label")})
    n_nodes = sum(1 for k in idx if k[0] == sp)
    with open(out / f"predictions_{a.name}_{sp}.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    g = [x["stance_label"] for x in rows]; pr = [x["predicted_stance"] for x in rows]
    mf, fa = f1(g, pr)
    print(f"{sp}: raw {len(preds)}  unmatched {miss}  mapped nodes {len(rows)} / {n_nodes}  "
          f"macro-F1 {mf:.2f}  F_avg {fa:.2f}  -> {out / f'predictions_{a.name}_{sp}.jsonl'}")