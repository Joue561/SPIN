#!/usr/bin/env python3
"""
analyze_human_review.py

用法：
python analyze_human_review.py \
  --reviewed_csv simplified_human_review_filled.csv \
  --output_dir ./ai_audit_output
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# ============ 工具函数 ============

def load_csv(filepath: str) -> list:
    records = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{filepath}")
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                records.append(dict(row))
            except Exception as e:
                print(f"  ⚠️  第 {i+1} 行读取失败：{e}")
    return records


def to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_div(a, b, default=0.0) -> float:
    return round(a / b, 4) if b > 0 else default


def write_json(data: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath}")


# ============ 统计函数 ============

def overall_review_stats(records: list) -> dict:
    total = len(records)
    ai_correct   = sum(1 for r in records if r.get("human_ai_judgment_correct") == "正确")
    ai_wrong     = sum(1 for r in records if r.get("human_ai_judgment_correct") == "错误")
    ai_uncertain = sum(1 for r in records if r.get("human_ai_judgment_correct") == "不确定")
    reviewed     = ai_correct + ai_wrong + ai_uncertain

    return {
        "total_records":        total,
        "total_reviewed":       reviewed,
        "ai_correct_count":     ai_correct,
        "ai_wrong_count":       ai_wrong,
        "ai_uncertain_count":   ai_uncertain,
        "ai_accuracy_estimated": safe_div(ai_correct, reviewed),
        "unreviewed_count":     total - reviewed,
    }


def original_label_quality(records: list) -> dict:
    error_types = [
        "false_neutral", "false_against", "false_favor",
        "target_mismatch", "context_missing",
        "irrelevant_mislabeled_as_neutral", "ambiguous", "other"
    ]
    counts = {et: 0 for et in error_types}
    correct = wrong = uncertain = 0

    for r in records:
        # 使用人工最终判断或 ai 判断
        ai_correct = r.get("ai_original_label_correct", "")
        human_judgment = r.get("human_ai_judgment_correct", "")
        error_type = r.get("human_error_type") or r.get("ai_error_type") or ""

        # 人工判断 AI 错 → 原始标签也可能错
        if human_judgment == "错误":
            wrong += 1
            if error_type in counts:
                counts[error_type] += 1
        elif human_judgment == "正确":
            if ai_correct == "true":
                correct += 1
            elif ai_correct == "false":
                wrong += 1
                if error_type in counts:
                    counts[error_type] += 1
            else:
                uncertain += 1
        else:
            if ai_correct == "true":
                correct += 1
            elif ai_correct == "false":
                wrong += 1
                if error_type in counts:
                    counts[error_type] += 1
            else:
                uncertain += 1

    return {
        "original_correct_count":   correct,
        "original_wrong_count":     wrong,
        "original_uncertain_count": uncertain,
        **counts,
    }


def neutral_audit_result(records: list) -> dict:
    neutral_records = [r for r in records if r.get("stance_label_raw") == "中立"]
    total_neutral = len(neutral_records)

    final_labels = []
    for r in neutral_records:
        # 人工最终标签优先，其次 ai 标签
        label = r.get("human_final_label") or r.get("ai_recheck_label") or ""
        if r.get("human_ai_judgment_correct") == "正确":
            label = r.get("ai_recheck_label") or ""
        final_labels.append(label)

    label_counter = Counter(final_labels)

    true_neutral  = label_counter.get("真中立", 0)
    false_neutral = total_neutral - true_neutral
    actually_against  = label_counter.get("反对", 0)
    actually_favor    = label_counter.get("支持", 0)
    irrelevant        = label_counter.get("无关", 0)
    unclear           = label_counter.get("目标不清", 0)
    insufficient      = label_counter.get("上下文不足", 0)

    return {
        "reviewed_neutral_count":      total_neutral,
        "true_neutral_count":          true_neutral,
        "false_neutral_count":         false_neutral,
        "false_neutral_rate":          safe_div(false_neutral, total_neutral),
        "actually_against_count":      actually_against,
        "actually_favor_count":        actually_favor,
        "irrelevant_count":            irrelevant,
        "unclear_count":               unclear,
        "insufficient_context_count":  insufficient,
        "final_label_distribution":    dict(label_counter),
    }


def favor_audit_result(records: list) -> dict:
    favor_records = [r for r in records if r.get("stance_label_raw") == "支持"]
    total = len(favor_records)
    correct = wrong = 0
    error_types = Counter()

    for r in favor_records:
        human_judgment = r.get("human_ai_judgment_correct", "")
        ai_correct = r.get("ai_original_label_correct", "")
        error_type = r.get("human_error_type") or r.get("ai_error_type") or ""

        if human_judgment == "正确" and ai_correct == "true":
            correct += 1
        elif human_judgment == "错误" or ai_correct == "false":
            wrong += 1
            if error_type:
                error_types[error_type] += 1
        else:
            correct += 1  # uncertain 偏向正确

    return {
        "reviewed_favor_count":       total,
        "correct_favor_count":        correct,
        "wrong_favor_count":          wrong,
        "favor_precision_estimated":  safe_div(correct, total),
        "main_error_types":           dict(error_types.most_common(5)),
    }


def pattern_reliability(records: list) -> list:
    pattern_groups = defaultdict(list)
    for r in records:
        pattern = r.get("ai_abstract_pattern") or r.get("candidate_pattern") or "unknown"
        if "|" in str(pattern):
            pattern = str(pattern).split("|")[0].strip()
        pattern_groups[pattern].append(r)

    results = []
    for pattern_name, group in sorted(pattern_groups.items()):
        reviewed = len(group)
        pattern_correct = sum(
            1 for r in group
            if r.get("human_pattern_correct") == "正确"
            or (not r.get("human_pattern_correct") and r.get("human_ai_judgment_correct") == "正确")
        )
        pattern_wrong = sum(
            1 for r in group
            if r.get("human_pattern_correct") == "错误"
        )
        correct_rate = safe_div(pattern_correct, reviewed)

        final_label_dist = Counter(
            r.get("human_final_label") or r.get("ai_recheck_label") or "未填写"
            for r in group
        )
        error_dist = Counter(
            r.get("human_error_type") or r.get("ai_error_type") or "无"
            for r in group
            if r.get("human_ai_judgment_correct") == "错误"
               or r.get("ai_original_label_correct") == "false"
        )

        if reviewed >= 20 and correct_rate >= 0.85:
            suggested_usage = "use_as_strong_pattern"
        elif reviewed >= 20 and correct_rate >= 0.65:
            suggested_usage = "use_as_weak_pattern"
        elif reviewed < 20:
            suggested_usage = "needs_more_review"
        else:
            suggested_usage = "do_not_use"

        results.append({
            "pattern_name":                    pattern_name,
            "reviewed_count":                  reviewed,
            "ai_pattern_correct_count":        pattern_correct,
            "ai_pattern_wrong_count":          pattern_wrong,
            "pattern_correct_rate":            correct_rate,
            "final_label_distribution":        dict(final_label_dist),
            "original_label_error_distribution": dict(error_dist),
            "suggested_usage":                 suggested_usage,
        })

    return results


# ============ 修正规则候选 ============

RULE_TEMPLATES = [
    {
        "rule_name": "rhetorical_question_false_neutral_to_against",
        "pattern_name": "rhetorical_question_against_TARGET",
        "conditions": {"rhetorical_question_candidate": True, "stance_label_raw": "中立"},
        "original_label_condition": "中立",
        "suggested_corrected_label": "反对",
        "main_error_type": "false_neutral",
        "caution": "反问有时可能指向父评论或其他 target，使用时需结合上下文。",
    },
    {
        "rule_name": "agreement_with_parent_inherits_stance",
        "pattern_name": "agreement_with_parent",
        "conditions": {"agreement_cue": True, "depth": 2, "stance_label_raw": "中立"},
        "original_label_condition": "中立",
        "suggested_corrected_label": "根据父评论立场推断",
        "main_error_type": "context_missing",
        "caution": "需要父评论立场已知且可信，不能盲目继承。",
    },
    {
        "rule_name": "emotion_cue_false_neutral_to_against",
        "pattern_name": "negative_judgment_toward_TARGET",
        "conditions": {"emotion_or_stance_cue": True, "stance_label_raw": "中立"},
        "original_label_condition": "中立",
        "suggested_corrected_label": "反对",
        "main_error_type": "false_neutral",
        "caution": "情绪词也可能出现在支持性语境中，需谨慎。",
    },
    {
        "rule_name": "sarcasm_false_neutral_to_against",
        "pattern_name": "sarcasm_or_mockery",
        "conditions": {"disagreement_cue": True, "stance_label_raw": "中立"},
        "original_label_condition": "中立",
        "suggested_corrected_label": "反对",
        "main_error_type": "false_neutral",
        "caution": "讽刺语气较难自动识别，建议人工二次确认。",
    },
]


def build_correction_rules(
    pattern_stats: list,
    records: list,
) -> list:
    pattern_stat_map = {p["pattern_name"]: p for p in pattern_stats}
    rules = []

    for template in RULE_TEMPLATES:
        pname = template["pattern_name"]
        stat = pattern_stat_map.get(pname)
        if not stat:
            continue

        reviewed_count    = stat["reviewed_count"]
        correct_rate      = stat["pattern_correct_rate"]
        suggested_usage   = stat["suggested_usage"]

        # 计算该规则命中的样本数和精度
        matched = [
            r for r in records
            if (r.get("ai_abstract_pattern") or "").strip() == pname
        ]
        corrected_correctly = sum(
            1 for r in matched
            if r.get("human_ai_judgment_correct") == "正确"
               or r.get("ai_original_label_correct") == "true"
        )
        estimated_precision = safe_div(corrected_correctly, len(matched)) if matched else correct_rate

        rule = {
            "rule_name":                template["rule_name"],
            "pattern_name":             pname,
            "conditions":               template["conditions"],
            "original_label_condition": template["original_label_condition"],
            "suggested_corrected_label": template["suggested_corrected_label"],
            "confidence":               round(correct_rate, 4),
            "reviewed_count":           reviewed_count,
            "estimated_precision":      round(estimated_precision, 4),
            "main_error_type":          template["main_error_type"],
            "suggested_usage":          suggested_usage,
            "caution":                  template["caution"],
        }
        rules.append(rule)

    # 从数据中动态发现高精度 pattern
    for stat in pattern_stats:
        pname = stat["pattern_name"]
        if any(t["pattern_name"] == pname for t in RULE_TEMPLATES):
            continue  # 已在模板中
        if stat["reviewed_count"] < 10:
            continue
        if stat["pattern_correct_rate"] >= 0.80:
            dominant_label = max(
                stat["final_label_distribution"],
                key=lambda k: stat["final_label_distribution"][k],
                default="未知"
            )
            rules.append({
                "rule_name":                f"auto_discovered_{pname}",
                "pattern_name":             pname,
                "conditions":               {"ai_abstract_pattern": pname},
                "original_label_condition": "任意",
                "suggested_corrected_label": dominant_label,
                "confidence":               stat["pattern_correct_rate"],
                "reviewed_count":           stat["reviewed_count"],
                "estimated_precision":      stat["pattern_correct_rate"],
                "main_error_type":          "auto_discovered",
                "suggested_usage":          stat["suggested_usage"],
                "caution":                  "自动发现规则，建议人工进一步确认后再使用。",
            })

    return rules


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="人工审核结果分析")
    parser.add_argument("--reviewed_csv", required=True, help="人工填写后的 CSV 路径")
    parser.add_argument("--output_dir",   required=True, help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 读取已审核 CSV：{args.reviewed_csv}")
    records = load_csv(args.reviewed_csv)
    print(f"  共加载 {len(records)} 条")

    print(f"\n📊 统计分析中...")

    overall    = overall_review_stats(records)
    orig_qual  = original_label_quality(records)
    neutral_r  = neutral_audit_result(records)
    favor_r    = favor_audit_result(records)
    pattern_r  = pattern_reliability(records)
    rules      = build_correction_rules(pattern_r, records)

    report = {
        "overall_review_stats":      overall,
        "original_label_quality":    orig_qual,
        "neutral_audit_result":      neutral_r,
        "favor_audit_result":        favor_r,
        "pattern_reliability":       pattern_r,
        "correction_rule_candidates": rules,
    }

    write_json(report, str(output_dir / "pattern_reliability_report.json"))
    write_json(rules,  str(output_dir / "correction_rules_candidates.json"))

    print(f"""
📊 分析摘要：
  总审核样本：      {overall['total_reviewed']} / {overall['total_records']}
  AI 准确率估计：   {overall['ai_accuracy_estimated']:.1%}
  原始标签错误数：  {orig_qual['original_wrong_count']}
  假中立数：        {orig_qual.get('false_neutral', 0)}
  中立样本中假中立率：{neutral_r['false_neutral_rate']:.1%}
  支持标签精度估计：{favor_r['favor_precision_estimated']:.1%}
  发现 pattern 数： {len(pattern_r)}
  生成修正规则数：  {len(rules)}
    """)


if __name__ == "__main__":
    main()