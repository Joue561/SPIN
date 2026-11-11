#!/usr/bin/env python3
"""
evaluate_stance_predictions.py

用法：
python evaluate_stance_predictions.py \
  --predictions rag_prediction_outputs/predictions_full_rag_valid.jsonl \
  --output_dir rag_evaluation_outputs \
  --group_by true
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, classification_report, confusion_matrix
    )
    SKLEARN_OK = True
except ImportError:
    print("⚠️  sklearn 未安装，请运行：pip install scikit-learn")
    SKLEARN_OK = False


# ============ IO ============

def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第 {i+1} 行解析失败：{e}")
    return records


def write_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath.name}")


def write_jsonl(records: list, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath.name}（{len(records)} 条）")


def safe_str(val) -> str:
    return str(val).strip() if val else ""


def safe_list(val) -> list:
    return val if isinstance(val, list) else []


# ============ 标签规范化 ============

def normalize_label(label: str) -> str:
    l = safe_str(label).lower()
    if l in ("favor", "against", "none"):
        return l
    return "invalid"


# ============ 指标计算 ============

def compute_metrics(gold: list, pred: list, label_name: str = "") -> dict:
    if not SKLEARN_OK or not gold:
        return {"error": "sklearn not available or empty data"}

    labels = ["favor", "against", "none"]
    valid_pairs = [(g, p) for g, p in zip(gold, pred)
                   if g in labels and p in labels]

    if not valid_pairs:
        return {"error": "no valid pairs"}

    gold_v = [g for g, _ in valid_pairs]
    pred_v = [p for _, p in valid_pairs]
    invalid_count = len(gold) - len(valid_pairs)

    acc     = accuracy_score(gold_v, pred_v)
    mac_f1  = f1_score(gold_v, pred_v, labels=labels, average="macro", zero_division=0)
    wgt_f1  = f1_score(gold_v, pred_v, labels=labels, average="weighted", zero_division=0)

    per_class = {}
    for lbl in labels:
        per_class[lbl] = {
            "precision": round(precision_score(gold_v, pred_v, labels=[lbl],
                                               average="micro", zero_division=0), 4),
            "recall":    round(recall_score(gold_v, pred_v, labels=[lbl],
                                            average="micro", zero_division=0), 4),
            "f1":        round(f1_score(gold_v, pred_v, labels=[lbl],
                                        average="micro", zero_division=0), 4),
            "support":   gold_v.count(lbl),
        }

    return {
        "group":            label_name,
        "total_samples":    len(gold),
        "valid_samples":    len(valid_pairs),
        "invalid_count":    invalid_count,
        "accuracy":         round(acc, 4),
        "macro_f1":         round(mac_f1, 4),
        "weighted_f1":      round(wgt_f1, 4),
        "per_class":        per_class,
        "gold_distribution": dict(Counter(gold_v)),
        "pred_distribution": dict(Counter(pred_v)),
    }


def compute_group_metrics(records: list, group_fn, group_label: str) -> dict:
    groups = defaultdict(list)
    for r in records:
        key = group_fn(r)
        if key is not None:
            groups[key].append(r)

    result = {}
    for key, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
        gold = [normalize_label(r.get("stance_label", "")) for r in grp]
        pred = [normalize_label(r.get("predicted_stance", "")) for r in grp]
        result[str(key)] = compute_metrics(gold, pred, f"{group_label}={key}")
    return result


def build_confusion_matrix(gold: list, pred: list) -> dict:
    if not SKLEARN_OK:
        return {}
    labels = ["favor", "against", "none"]
    valid = [(g, p) for g, p in zip(gold, pred)
             if g in labels and p in labels]
    if not valid:
        return {}
    gv = [g for g, _ in valid]
    pv = [p for _, p in valid]
    cm = confusion_matrix(gv, pv, labels=labels).tolist()
    return {
        "labels": labels,
        "matrix": cm,
        "description": "行=gold，列=predicted",
    }


# ============ 错误案例 ============

ERROR_FIELDS = [
    "topic", "node_id", "main_target", "current_text", "parent_text",
    "stance_label", "predicted_stance", "confidence", "reason_short",
    "llm_reply_relation", "llm_semantic_propagation", "llm_target_relation",
    "llm_candidate_patterns", "retrieved_memory_ids", "retrieved_memory_summaries",
    "depth", "is_ambiguous_current",
]


def collect_error_cases(records: list) -> list:
    errors = []
    for r in records:
        gold = normalize_label(r.get("stance_label", ""))
        pred = normalize_label(r.get("predicted_stance", ""))
        if gold in ("favor", "against", "none") and gold != pred:
            errors.append({k: r.get(k) for k in ERROR_FIELDS})
    return errors


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="立场预测结果评估")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--group_by",    default="true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    do_group = args.group_by.lower() == "true"

    print(f"📂 读取预测结果：{args.predictions}")
    records = load_jsonl(args.predictions)
    print(f"  共加载 {len(records)} 条")

    gold_all = [normalize_label(r.get("stance_label", ""))     for r in records]
    pred_all = [normalize_label(r.get("predicted_stance", "")) for r in records]

    # ---- 总体评估 ----
    overall = compute_metrics(gold_all, pred_all, "overall")
    overall["method"] = safe_str(records[0].get("method")) if records else ""
    overall["split"]  = safe_str(records[0].get("split"))  if records else ""
    write_json(overall, output_dir / "evaluation_report.json")

    # ---- 混淆矩阵 ----
    cm = build_confusion_matrix(gold_all, pred_all)
    write_json(cm, output_dir / "confusion_matrix.json")

    # ---- 错误案例 ----
    errors = collect_error_cases(records)
    write_jsonl(errors, output_dir / "error_cases.jsonl")

    if not do_group:
        print(f"\n📊 总体 Accuracy={overall.get('accuracy')}  Macro-F1={overall.get('macro_f1')}")
        return

    # ---- 按 topic ----
    write_json(
        compute_group_metrics(records, lambda r: r.get("topic"), "topic"),
        output_dir / "evaluation_by_topic.json"
    )

    # ---- 按 depth ----
    def depth_group(r):
        d = r.get("depth", 0) or 0
        if d == 1:   return "depth=1"
        if d == 2:   return "depth=2"
        return "depth>=3"
    write_json(
        compute_group_metrics(records, depth_group, "depth"),
        output_dir / "evaluation_by_depth.json"
    )

    # ---- 短文本 ----
    def short_text_group(r):
        ct = safe_str(r.get("current_text")).replace(" ", "")
        if len(ct) <= 4:
            return "short_text_le4"
        if r.get("is_ambiguous_current"):
            return "is_ambiguous"
        return None
    write_json(
        compute_group_metrics(records, short_text_group, "short_text"),
        output_dir / "evaluation_by_short_text.json"
    )

    # ---- 按语义传导 ----
    write_json(
        compute_group_metrics(
            records,
            lambda r: safe_str(r.get("llm_semantic_propagation")) or None,
            "semantic_propagation"
        ),
        output_dir / "evaluation_by_semantic_propagation.json"
    )

    # ---- 按目标关系 ----
    write_json(
        compute_group_metrics(
            records,
            lambda r: safe_str(r.get("llm_target_relation")) or None,
            "target_relation"
        ),
        output_dir / "evaluation_by_target_relation.json"
    )

    # ---- 按 RAG 使用情况 ----
    def rag_group(r):
        ids = safe_list(r.get("retrieved_memory_ids"))
        return "used_rag" if ids else "no_rag"
    write_json(
        compute_group_metrics(records, rag_group, "rag_usage"),
        output_dir / "evaluation_by_rag_usage.json"
    )

    # ---- 按子回复使用 ----
    def child_group(r):
        if r.get("llm_use_child_feedback"):
            return "child_feedback_used"
        if safe_str(r.get("method")) == "full_rag_posthoc":
            return "posthoc_method"
        return None
    write_json(
        compute_group_metrics(records, child_group, "child_feedback"),
        output_dir / "evaluation_by_child_feedback.json"
    )

    print(f"""
📊 评估摘要：
  总样本：    {overall['total_samples']}
  有效样本：  {overall['valid_samples']}
  Accuracy：  {overall['accuracy']}
  Macro-F1：  {overall['macro_f1']}
  Weighted-F1：{overall['weighted_f1']}
  错误案例：  {len(errors)} 条
  各类别：
    favor：   P={overall['per_class']['favor']['precision']}  R={overall['per_class']['favor']['recall']}  F1={overall['per_class']['favor']['f1']}
    against：P={overall['per_class']['against']['precision']}  R={overall['per_class']['against']['recall']}  F1={overall['per_class']['against']['f1']}
    none：    P={overall['per_class']['none']['precision']}  R={overall['per_class']['none']['recall']}  F1={overall['per_class']['none']['f1']}
    """)


if __name__ == "__main__":
    main()