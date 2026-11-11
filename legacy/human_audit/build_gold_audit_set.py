#!/usr/bin/env python3
"""
build_gold_audit_set.py

用法：
python build_gold_audit_set.py \
  --stance_instances stance_instances.jsonl \
  --audit_candidates audit_candidates.jsonl \
  --output_dir ./audit_output \
  --sample_size 2000 \
  --seed 42
"""

import json
import csv
import random
import argparse
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional


# ============ 抽样配置 ============

SAMPLING_PLAN = [
    {
        "group": "neutral_suspicious",
        "target": 600,
        "filter": lambda r: r.get("stance_label_raw") == "中立" and r.get("neutral_suspicious") == True,
        "priority_keys": ["depth", "is_short_reply", "agreement_cue", "disagreement_cue",
                          "rhetorical_question_candidate", "emotion_or_stance_cue"],
    },
    {
        "group": "neutral_non_suspicious",
        "target": 250,
        "filter": lambda r: r.get("stance_label_raw") == "中立" and r.get("neutral_suspicious") == False,
        "priority_keys": [],
    },
    {
        "group": "favor",
        "target": 500,
        "filter": lambda r: r.get("stance_label_raw") == "支持",
        "priority_keys": ["audit_priority", "is_short_reply", "emotion_or_stance_cue"],
    },
    {
        "group": "against",
        "target": 350,
        "filter": lambda r: r.get("stance_label_raw") == "反对",
        "priority_keys": ["emotion_or_stance_cue", "rhetorical_question_candidate"],
    },
    {
        "group": "depth2_short_reply",
        "target": 300,
        "filter": lambda r: r.get("depth", 0) >= 2 and r.get("is_short_reply") == True,
        "priority_keys": ["agreement_cue", "disagreement_cue"],
    },
    {
        "group": "rhetorical_question",
        "target": 250,
        "filter": lambda r: r.get("rhetorical_question_candidate") == True,
        "priority_keys": [],
    },
    {
        "group": "agreement_with_parent",
        "target": 250,
        "filter": lambda r: r.get("agreement_cue") == True and r.get("depth", 0) >= 2,
        "priority_keys": [],
    },
]

MAX_PER_ROOT = 10

GOLD_ANNOTATION_FIELDS = {
    "gold_stance_label":           None,
    "gold_direct_target":          None,
    "gold_stance_to_direct_target": None,
    "gold_stance_to_main_target":  None,
    "gold_reply_relation":         None,
    "gold_abstract_pattern":       None,
    "gold_evidence_span":          None,
    "gold_notes":                  None,
    "original_label_correct":      None,
    "label_error_type":            None,
}

CSV_COLUMNS = [
    "sample_id", "instance_id", "matched_sampling_groups", "depth",
    "root_text", "parent_text", "current_text", "context_text",
    "main_target", "stance_label_raw", "candidate_pattern", "audit_priority",
    "neutral_suspicious", "is_short_reply", "agreement_cue", "disagreement_cue",
    "rhetorical_question_candidate", "emotion_or_stance_cue",
    "gold_stance_label", "gold_direct_target", "gold_stance_to_direct_target",
    "gold_stance_to_main_target", "gold_reply_relation", "gold_abstract_pattern",
    "gold_evidence_span", "gold_notes", "original_label_correct", "label_error_type",
]


# ============ 文件读取 ============

def load_jsonl(filepath: str) -> List[dict]:
    records = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第 {i+1} 行 JSON 解析失败，已跳过：{e}")
    return records


# ============ 分层抽样 ============

def priority_score(record: dict, priority_keys: List[str]) -> int:
    """命中越多优先字段，分数越高"""
    score = 0
    for key in priority_keys:
        val = record.get(key)
        if val is True:
            score += 1
        elif key == "depth" and isinstance(val, int) and val >= 2:
            score += 1
        elif key == "audit_priority" and val == "high":
            score += 1
    return score


def stratified_sample(
    records: List[dict],
    plan: List[dict],
    max_per_root: int,
    seed: int,
) -> tuple:
    """
    分层抽样，返回 (gold_set, group_counts, warnings)
    gold_set: [{record, matched_groups}]
    """
    rng = random.Random(seed)
    warnings = []

    # instance_id -> record（用于去重）
    selected: Dict[str, dict] = {}
    # instance_id -> matched groups
    matched_groups: Dict[str, List[str]] = defaultdict(list)
    # root_message_id -> count（控制每个原帖最多抽取数）
    root_counts: Dict[str, int] = defaultdict(int)

    group_counts = {}

    for plan_item in plan:
        group = plan_item["group"]
        target = plan_item["target"]
        filter_fn = plan_item["filter"]
        priority_keys = plan_item["priority_keys"]

        # 筛选候选
        try:
            candidates = [r for r in records if filter_fn(r)]
        except Exception as e:
            warnings.append(f"{group}：筛选时出错：{e}")
            candidates = []

        # 按优先级降序排列
        candidates.sort(key=lambda r: priority_score(r, priority_keys), reverse=True)

        # 随机打散同分的记录（保持同分内随机性）
        from itertools import groupby
        shuffled = []
        for _, group_iter in groupby(candidates, key=lambda r: priority_score(r, priority_keys)):
            grp = list(group_iter)
            rng.shuffle(grp)
            shuffled.extend(grp)
        candidates = shuffled

        picked = 0
        for rec in candidates:
            if picked >= target:
                break
            iid = str(rec.get("instance_id") or rec.get("message_id", ""))
            rid = str(rec.get("root_message_id") or "")

            # 控制每个原帖的最大抽取数
            if rid and root_counts[rid] >= max_per_root:
                continue

            if iid not in selected:
                selected[iid] = rec
                if rid:
                    root_counts[rid] += 1

            matched_groups[iid].append(group)
            picked += 1

        group_counts[group] = picked
        if picked < target:
            warnings.append(
                f"抽样组 '{group}' 目标 {target} 条，实际只抽到 {picked} 条（候选池不足）"
            )

    # 检查是否有 root_message_id 占比过高
    if root_counts:
        max_root_id = max(root_counts, key=lambda k: root_counts[k])
        max_count = root_counts[max_root_id]
        total = len(selected)
        ratio = max_count / total if total > 0 else 0
        if ratio > 0.05:
            warnings.append(
                f"root_message_id={max_root_id} 占比 {ratio:.1%}（{max_count}/{total}），"
                f"可能支配复核集"
            )

    # 组装结果
    gold_set = []
    for iid, rec in selected.items():
        rec = dict(rec)
        rec["matched_sampling_groups"] = matched_groups[iid]
        gold_set.append(rec)

    return gold_set, group_counts, warnings, root_counts


# ============ 添加标注字段 ============

def add_annotation_fields(records: List[dict]) -> List[dict]:
    for rec in records:
        for field, default in GOLD_ANNOTATION_FIELDS.items():
            if field not in rec:
                rec[field] = default
    return records


# ============ 输出 JSONL ============

def write_jsonl(records: List[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath}（{len(records)} 条）")


# ============ 输出 CSV ============

def write_csv(records: List[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for i, rec in enumerate(records):
            row = {col: rec.get(col, "") for col in CSV_COLUMNS}
            row["sample_id"] = f"A{i+1:06d}"
            # matched_sampling_groups 转为字符串
            groups = rec.get("matched_sampling_groups", [])
            row["matched_sampling_groups"] = "|".join(groups) if isinstance(groups, list) else str(groups)
            # candidate_pattern 转为字符串
            cp = rec.get("candidate_pattern", [])
            row["candidate_pattern"] = "|".join(cp) if isinstance(cp, list) else str(cp)
            writer.writerow(row)
    print(f"  ✅ {filepath}（{len(records)} 条）")


# ============ 抽样报告 ============

def generate_sampling_report(
    gold_set: List[dict],
    group_counts: dict,
    root_counts: dict,
    warnings: List[str],
) -> dict:
    from collections import Counter

    label_dist = Counter(r.get("stance_label_raw") for r in gold_set)
    depth_dist = Counter(
        "depth=1" if r.get("depth", 0) == 1 else "depth=2+"
        for r in gold_set
    )
    priority_dist = Counter(r.get("audit_priority") for r in gold_set)

    # candidate_pattern 分布（展开列表）
    pattern_dist: Dict[str, int] = defaultdict(int)
    for r in gold_set:
        cp = r.get("candidate_pattern", [])
        if isinstance(cp, list):
            for p in cp:
                pattern_dist[p] += 1
        else:
            pattern_dist[str(cp)] += 1

    root_sample_counts = list(root_counts.values())
    unique_root_count = len(root_counts)
    max_per_root = max(root_sample_counts) if root_sample_counts else 0
    avg_per_root = (sum(root_sample_counts) / unique_root_count) if unique_root_count else 0

    return {
        "total_gold_audit_samples": len(gold_set),
        "sampling_group_counts": group_counts,
        "raw_label_distribution": dict(label_dist),
        "depth_distribution": dict(depth_dist),
        "audit_priority_distribution": dict(priority_dist),
        "candidate_pattern_distribution": dict(pattern_dist),
        "root_message_id_distribution_summary": {
            "unique_root_count": unique_root_count,
            "max_samples_per_root": max_per_root,
            "average_samples_per_root": round(avg_per_root, 2),
        },
        "warnings": warnings,
    }


# ============ 标注规范文档 ============

ANNOTATION_GUIDELINES = """# 人工标注规范文档

## 1. 本次标注目标

本次标注旨在对模型生成的 `stance_label_raw` 进行人工校准，构建高质量的黄金标准集（gold audit set），用于评估原始标签质量。重点检查以下问题：

- **假中立**：被标注为"中立"但实际含有明确立场的样本
- **少数类支持标签**：检查"支持"标签是否准确，保护少数类表达
- **二级评论的上下文依赖**：depth=2 的评论需结合父评论判断立场
- **target mismatch**：评论实际指向的对象与主 target 不一致
- **反问、讽刺、赞同父评论等语言结构**：这些结构常被模型误判为中立

---

## 2. 主 Target 定义

本次标注的主 target 是：

> **高市早苗发表引发网络争议的涉台言论**

标注时，请判断当前评论相对于该主 target 的立场，**而不仅仅是**对"中国""日本""台湾""高市早苗本人"等相关对象的单独态度。

例如：
- "支持中国反制日本"→ 对主 target 通常是**反对**
- "高市早苗应该道歉"→ 对主 target 通常是**反对**
- "这件事影响中日关系"→ 可能是事件讨论，需结合上下文判断

---

## 3. 标签定义

| 标签 | 定义 |
|------|------|
| **支持** | 当前评论明确支持、认同高市早苗的涉台言论，或认为其言论合理、无需撤回 |
| **反对** | 当前评论明确批评、谴责高市早苗的涉台言论，或支持对日本进行反制 |
| **真中立** | 明确提及主 target，但没有表达支持或反对态度，例如纯事件陈述、中性分析 |
| **无关** | 当前评论与主 target 完全无关，例如纯粹的玩笑、与事件无关的话题 |
| **目标不清** | 无法判断当前评论的立场对象是否为主 target |
| **上下文不足** | 评论本身无法单独判断立场，需要更多上下文 |
| **混合/条件立场** | 评论同时包含支持和反对的成分，或立场有条件限制 |
| **无法判断** | 即使结合上下文，仍无法确定立场 |

---

## 4. Direct Target 与 Main Target 的区别

- **direct_target**：当前评论直接评价的具体对象
- **main_target**：本次研究的主 target（高市早苗涉台言论）

**示例：**

| 评论 | direct_target | 对 main_target 的立场 |
|------|---------------|----------------------|
| "支持中国强硬反制" | 中国反制 | 反对（主 target） |
| "高市早苗下台" | 高市早苗本人 | 反对（其涉台言论） |
| "这件事影响中日关系" | 中日关系 | 目标不清 |
| "日本经济会变差" | 日本经济 | 可能是反对，需结合上下文 |

---

## 5. 二级评论标注规则

depth=2 的评论**必须同时参考**：
1. `root_text`（原帖）
2. `parent_text`（直接父评论）
3. `current_text`（当前评论）

**示例：**

> 原帖：高市早苗的言论引发中国强烈批评  
> 父评论：这明显是故意挑衅，中国必须强硬回应  
> 当前评论：确实

→ `gold_reply_relation`：赞同父评论  
→ `gold_stance_to_main_target`：反对（继承父评论立场）  
→ `gold_abstract_pattern`：agreement_with_parent  
→ `original_label_correct`：若原标签为"中立"则填 false，`label_error_type`：false_neutral

---

## 6. 中立标签判断规则

请注意以下区分：

- ✅ **真中立** = 明确谈到主 target，但没有支持或反对态度
- ❌ **无关 ≠ 中立**：与事件无关的评论应标为"无关"
- ❌ **上下文不足 ≠ 中立**：短句缺乏上下文时应标为"上下文不足"
- ❌ **目标不清 ≠ 中立**：无法判断立场对象时应标为"目标不清"
- ❌ **模型不确定 ≠ 中立**：模型打了中立不代表真的中立

---

## 7. 常见假中立类型

| 类型 | 示例 | 实际立场 |
|------|------|---------|
| 短回复赞同 | "确实""说得对""就是" | 继承父评论立场 |
| 短回复讽刺 | "呵呵""笑死""典" | 通常为反对 |
| 反问 | "这也能洗？""谁信啊" | 通常为反对 |
| 讽刺 | "真是太棒了，又来搞事" | 反对 |
| 赞同父评论 | "说得对""确实如此" | 继承父评论立场 |
| 攻击支持者 | "还在洗的人真可怕" | 反对 |
| 支持反制 | "支持中国强硬反制" | 反对主 target |

---

## 8. 输出字段填写说明

| 字段 | 说明 |
|------|------|
| `gold_stance_label` | 参考第3节标签定义，填写最终人工判断的主立场标签 |
| `gold_direct_target` | 填写评论直接评价的对象，参考第4节示例 |
| `gold_stance_to_direct_target` | 对 direct_target 的态度：支持/反对/中立/不明确 |
| `gold_stance_to_main_target` | 对主 target 的立场，参考第3节标签 |
| `gold_reply_relation` | 仅 depth=2 填写，描述与父评论的关系 |
| `gold_abstract_pattern` | 填写语言结构模式，可参考 candidate_pattern 字段的提示 |
| `gold_evidence_span` | 从 current_text 中摘录支持判断的关键片段，直接复制原文 |
| `gold_notes` | 记录标注时遇到的歧义、上下文问题、特殊情况 |
| `original_label_correct` | true/false/uncertain，判断原始 stance_label_raw 是否正确 |
| `label_error_type` | 仅当 original_label_correct=false 时填写，选择错误类型 |

---

## 9. 注意事项

1. 不要修改 `stance_label_raw`，只填写 gold_ 开头的字段
2. 遇到无法判断的样本，优先标注为"上下文不足"或"无法判断"，并在 `gold_notes` 中说明原因
3. depth=2 样本务必先阅读完整的 `context_text`，不要只看 `current_text`
4. emoji、话题标签、@用户可能携带立场信息，请不要忽略
5. 如有疑问请在 `gold_notes` 中记录，供后续讨论
"""


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="构建人工复核黄金标准集")
    parser.add_argument("--stance_instances", required=True, help="stance_instances.jsonl 路径")
    parser.add_argument("--audit_candidates", required=True, help="audit_candidates.jsonl 路径")
    parser.add_argument("--output_dir",       required=True, help="输出目录路径")
    parser.add_argument("--sample_size",      type=int, default=2000, help="目标抽样总量（默认2000）")
    parser.add_argument("--seed",             type=int, default=42,   help="随机种子（默认42）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据
    print(f"📂 读取 stance_instances：{args.stance_instances}")
    stance_records = load_jsonl(args.stance_instances)
    print(f"  共加载 {len(stance_records)} 条")

    print(f"📂 读取 audit_candidates：{args.audit_candidates}")
    audit_records = load_jsonl(args.audit_candidates)
    print(f"  共加载 {len(audit_records)} 条")

    # 合并去重（以 instance_id 为主键，优先保留 audit_candidates）
    merged: Dict[str, dict] = {}
    for r in stance_records:
        iid = str(r.get("instance_id") or r.get("message_id", ""))
        merged[iid] = r
    for r in audit_records:
        iid = str(r.get("instance_id") or r.get("message_id", ""))
        merged[iid] = r  # audit_candidates 覆盖

    all_records = list(merged.values())
    print(f"  合并去重后共 {len(all_records)} 条")

    # 2. 分层抽样
    print(f"\n🎲 开始分层抽样（seed={args.seed}，max_per_root={MAX_PER_ROOT}）...")
    gold_set, group_counts, warnings, root_counts = stratified_sample(
        all_records, SAMPLING_PLAN, MAX_PER_ROOT, args.seed
    )
    print(f"  抽样完成，共 {len(gold_set)} 条")
    for group, count in group_counts.items():
        print(f"    {group}：{count} 条")

    # 3. 添加标注字段
    gold_set = add_annotation_fields(gold_set)

    # 4. 写出文件
    print(f"\n📝 写出文件到 {output_dir}/")
    write_jsonl(gold_set, str(output_dir / "gold_audit_set.jsonl"))
    write_csv(gold_set,   str(output_dir / "gold_audit_set_for_annotation.csv"))

    report = generate_sampling_report(gold_set, group_counts, root_counts, warnings)
    report_path = output_dir / "audit_sampling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {report_path}")

    guidelines_path = output_dir / "annotation_guidelines.md"
    with open(guidelines_path, "w", encoding="utf-8") as f:
        f.write(ANNOTATION_GUIDELINES)
    print(f"  ✅ {guidelines_path}")

    # 5. 控制台摘要
    if warnings:
        print(f"\n⚠️  警告：")
        for w in warnings:
            print(f"  - {w}")

    print(f"""
📊 抽样摘要：
  总计抽取：       {len(gold_set)} 条
  原始标签分布：   {report['raw_label_distribution']}
  深度分布：       {report['depth_distribution']}
  审计优先级：     {report['audit_priority_distribution']}
  唯一原帖数：     {report['root_message_id_distribution_summary']['unique_root_count']}
  每原帖最多：     {report['root_message_id_distribution_summary']['max_samples_per_root']}
  每原帖平均：     {report['root_message_id_distribution_summary']['average_samples_per_root']}
    """)


if __name__ == "__main__":
    main()