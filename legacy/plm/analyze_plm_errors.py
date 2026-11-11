#!/usr/bin/env python3
"""
analyze_plm_errors.py

用法：
python analyze_plm_errors.py \
  --predictions plm_model_outputs/plm_semantic_rag_feature_macbert-base/predictions_test.jsonl \
  --output_dir plm_error_analysis/plm_semantic_rag_feature
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter


LABEL_LIST = ["favor", "against", "none"]


# ============ IO ============

def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath.name}（{len(records)} 条）")


def write_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath.name}")


def safe_str(val) -> str:
    return str(val).strip() if val else ""


def safe_list(val) -> list:
    return val if isinstance(val, list) else ([] if val is None else [val])


# ============ 指标计算 ============

def compute_metrics(gold: list, pred: list) -> dict:
    try:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        )
    except ImportError:
        return {"error": "sklearn not available"}

    if not gold:
        return {"error": "empty group"}

    acc    = accuracy_score(gold, pred)
    mac_f1 = f1_score(gold, pred, labels=LABEL_LIST, average="macro", zero_division=0)
    wgt_f1 = f1_score(gold, pred, labels=LABEL_LIST, average="weighted", zero_division=0)

    per_class = {}
    for lbl in LABEL_LIST:
        per_class[lbl] = {
            "precision": round(precision_score(gold, pred, labels=[lbl], average="micro", zero_division=0), 4),
            "recall":    round(recall_score(gold, pred, labels=[lbl], average="micro", zero_division=0), 4),
            "f1":        round(f1_score(gold, pred, labels=[lbl], average="micro", zero_division=0), 4),
            "support":   gold.count(lbl),
        }

    return {
        "total_samples": len(gold),
        "accuracy":      round(float(acc), 4),
        "macro_f1":      round(float(mac_f1), 4),
        "weighted_f1":   round(float(wgt_f1), 4),
        "per_class":     per_class,
        "gold_distribution": dict(Counter(gold)),
        "pred_distribution": dict(Counter(pred)),
    }


def compute_group_metrics(records: list, group_fn, group_label: str) -> dict:
    groups = defaultdict(list)
    for r in records:
        key = group_fn(r)
        if key is not None:
            groups[key].append(r)

    result = {}
    for key, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
        gold = [r.get("stance_label") for r in grp]
        pred = [r.get("predicted_label") for r in grp]
        result[str(key)] = compute_metrics(gold, pred)
        result[str(key)]["group_label"] = f"{group_label}={key}"
    return result


# ============ 分组函数 ============

def depth_group(rec):
    d = rec.get("depth", 0) or 0
    if d == 1: return "depth=1"
    if d == 2: return "depth=2"
    return "depth>=3"


def short_text_group(rec):
    ct = safe_str(rec.get("current_text")).replace(" ", "")
    if len(ct) <= 4:
        return "short_text_le4"
    if rec.get("is_ambiguous_current"):
        return "is_ambiguous"
    return None


def pattern_group_expand(records: list) -> dict:
    """candidate_patterns 是列表字段，需要展开后分组"""
    groups = defaultdict(list)
    for r in records:
        patterns = safe_list(r.get("llm_candidate_patterns"))
        if not patterns:
            groups["no_pattern"].append(r)
        for p in patterns:
            groups[str(p)].append(r)

    result = {}
    for key, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
        gold = [r.get("stance_label") for r in grp]
        pred = [r.get("predicted_label") for r in grp]
        result[key] = compute_metrics(gold, pred)
    return result


# ============ 错误案例收集 ============

ERROR_FIELDS = [
    "topic", "node_id", "current_text", "parent_text",
    "stance_label", "predicted_label", "predicted_probs",
    "depth", "is_ambiguous_current",
    "llm_reply_relation", "llm_semantic_propagation",
    "llm_target_relation", "llm_candidate_patterns",
]


def collect_error_cases(records: list) -> list:
    errors = []
    for r in records:
        if r.get("stance_label") != r.get("predicted_label"):
            errors.append({k: r.get(k) for k in ERROR_FIELDS})
    return errors


def collect_none_error_cases(records: list) -> list:
    """gold 或 pred 任一为 none 的错误案例（中立类是常见难点，单独分析）"""
    errors = []
    for r in records:
        gold, pred = r.get("stance_label"), r.get("predicted_label")
        if gold != pred and (gold == "none" or pred == "none"):
            errors.append({k: r.get(k) for k in ERROR_FIELDS})
    return errors


def collect_rag_help_cases(records: list, baseline_records: list = None) -> list:
    """
    若提供 baseline_records（如 plm_window 的预测结果，按 node_id 对齐），
    标记 RAG 方法纠正了 baseline 错误的样本。
    若未提供 baseline，则标记当前模型预测正确且使用了高匹配度模式的样本作为正向案例。
    """
    cases = []
    if baseline_records:
        baseline_map = {r.get("node_id"): r for r in baseline_records}
        for r in records:
            nid = r.get("node_id")
            base = baseline_map.get(nid)
            if not base:
                continue
            gold = r.get("stance_label")
            cur_correct  = r.get("predicted_label") == gold
            base_correct = base.get("predicted_label") == gold
            if cur_correct and not base_correct:
                case = {k: r.get(k) for k in ERROR_FIELDS}
                case["baseline_predicted_label"] = base.get("predicted_label")
                case["rag_corrected"] = True
                cases.append(case)
    else:
        # 无 baseline 时，退化为：当前预测正确 + 候选模式非空的样本，标记为潜在受益样本
        for r in records:
            if (r.get("predicted_label") == r.get("stance_label")
                    and safe_list(r.get("llm_candidate_patterns"))):
                cases.append({k: r.get(k) for k in ERROR_FIELDS})
    return cases


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="PLM 立场检测预测结果错误分析")
    parser.add_argument("--predictions", required=True,
                        help="predictions_test.jsonl 路径")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--baseline_predictions", default=None,
                        help="可选：baseline 方法（如 plm_window）的 predictions_test.jsonl，"
                             "用于识别 RAG 方法纠正的样本")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 读取预测结果：{args.predictions}")
    records = load_jsonl(args.predictions)
    print(f"  共加载 {len(records)} 条")

    gold_all = [r.get("stance_label") for r in records]
    pred_all = [r.get("predicted_label") for r in records]

    overall = compute_metrics(gold_all, pred_all)
    print(f"  总体：accuracy={overall.get('accuracy')}  macro_f1={overall.get('macro_f1')}")

    # ---- 分组指标 ----
    group_metrics_all = {
        "overall":               overall,
        "by_topic":              compute_group_metrics(records, lambda r: r.get("topic"), "topic"),
        "by_depth":              compute_group_metrics(records, depth_group, "depth"),
        "by_short_text":         compute_group_metrics(records, short_text_group, "short_text"),
        "by_semantic_propagation": compute_group_metrics(
            records, lambda r: safe_str(r.get("llm_semantic_propagation")) or None, "semantic_propagation"
        ),
        "by_target_relation":    compute_group_metrics(
            records, lambda r: safe_str(r.get("llm_target_relation")) or None, "target_relation"
        ),
        "by_reply_relation":     compute_group_metrics(
            records, lambda r: safe_str(r.get("llm_reply_relation")) or None, "reply_relation"
        ),
        "by_candidate_pattern":  pattern_group_expand(records),
    }
    write_json(group_metrics_all, output_dir / "group_metrics.json")

    # ---- 错误案例 ----
    errors = collect_error_cases(records)
    write_jsonl(errors, output_dir / "error_cases.jsonl")

    none_errors = collect_none_error_cases(records)
    write_jsonl(none_errors, output_dir / "none_error_cases.jsonl")

    # ---- RAG 受益案例 ----
    baseline_records = None
    if args.baseline_predictions:
        print(f"📂 读取 baseline 预测：{args.baseline_predictions}")
        baseline_records = load_jsonl(args.baseline_predictions)
    rag_help_cases = collect_rag_help_cases(records, baseline_records)
    write_jsonl(rag_help_cases, output_dir / "rag_help_cases.jsonl")

    # ---- 控制台摘要 ----
    print(f"""
📊 错误分析摘要：
  总样本：        {len(records)}
  错误案例：      {len(errors)}（错误率 {len(errors)/max(len(records),1):.1%}）
  none 相关错误： {len(none_errors)}
  RAG 受益案例：  {len(rag_help_cases)}{'（vs baseline 纠正）' if baseline_records else '（无baseline，为潜在受益样本）'}
    """)

    print("📊 各 candidate_pattern 上的表现（前5个最常见模式）：")
    pattern_metrics = group_metrics_all["by_candidate_pattern"]
    for pname, m in list(pattern_metrics.items())[:5]:
        if "error" in m:
            continue
        print(f"  {pname:<30} n={m['total_samples']:<5} "
              f"acc={m['accuracy']}  macro_f1={m['macro_f1']}")


if __name__ == "__main__":
    main()
