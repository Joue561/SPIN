# -*- coding: utf-8 -*-
"""A1 helper: restrict the knowledge base to the training examples a low-resource SFT run saw.

Training nodes NOT in the subset get split = "train_unused", so analyze_combined_prior.py /
analyze_combined_ablation.py build the KB only from the subset (they use split == "train").
Validation and test nodes are unchanged.

python make_subset_instances.py --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --data_dir StanceAgent-R1-main/data/default --ids zscsd_exp/outputs/qwen7b_sft_f10/train_ids_used.json \
  --out zscsd_exp/outputs/qwen7b_sft_f10/instances_subset.jsonl
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ["instances", "data_dir", "ids", "out"]:
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()

keep_ids = set(json.load(open(a.ids)))
raw = json.load(open(Path(a.data_dir) / "train_data.json", encoding="utf-8"))
keep_keys = {(ex["target"], tuple(str(s) for s in ex["sentences"])) for ex in raw if ex["id"] in keep_ids}

rows, kept, dropped = [], 0, 0
for line in open(a.instances, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    if str(r.get("split", "")).lower() == "train":
        if (r["main_target"], tuple(str(s) for s in r["ancestor_texts"])) in keep_keys:
            kept += 1
        else:
            r["split"] = "train_unused"; dropped += 1
    rows.append(r)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"subset ids {len(keep_ids)} | train nodes kept {kept}, marked train_unused {dropped} -> {a.out}")