#!/usr/bin/env python3
"""
process_stance_tree.py
用法：python process_stance_tree.py --input data.json --output_dir ./processed/
"""

import json
import argparse
import os
from pathlib import Path
from typing import Optional


# ============ 词表 ============

AGREEMENT_CUES = [
    "确实", "对", "对的", "说得对", "没错", "就是", "同意", "赞同",
    "支持", "+1", "顶", "认可", "是的", "可不是", "太对了"
]

DISAGREEMENT_CUES = [
    "不是", "不对", "你错了", "胡说", "造谣", "别洗了", "洗什么",
    "洗地", "你可真懂", "笑死", "呵呵", "典", "绷", "离谱", "扯淡", "瞎说"
]

RHETORICAL_CUES = [
    "？", "?", "这也能", "谁信", "凭什么", "怎么可能",
    "难道", "不会真有人", "还真有人", "你觉得合理吗"
]

EMOTION_STANCE_CUES = [
    "反制", "干涉", "挑衅", "作妖", "军国主义", "下台", "滚", "活该",
    "洗", "汉奸", "卖国", "恶心", "离谱", "无耻", "荒唐", "可笑",
    "霸权", "侵略", "支持", "反对", "抵制", "谴责", "制裁", "强硬",
    "领土", "主权", "台湾", "琉球", "日本", "高市"
]


# ============ 字段标准化 ============

def standardize_record(r: dict) -> dict:
    """字段重命名与标准化"""
    rec = {
        # 重命名字段
        "instance_id":          str(r.get("message_id", "")),
        "message_id":           r.get("message_id"),
        "content_id":           r.get("ods_content_id"),
        "parent_content_id":    r.get("parent_id"),
        "root_content_id":      r.get("root_id"),
        "current_text":         _clean_text(r.get("text")),
        "stance_label_raw":     r.get("stance_label"),
        "main_target":          r.get("target"),
        "platform":             r.get("platform_code"),
        "sentiment_raw":        r.get("sentiment"),
        # 保留字段
        "user_id":              r.get("user_id"),
        "created_at":           r.get("created_at"),
        "root_message_id":      r.get("root_message_id"),
        "parent_message_id":    r.get("parent_message_id"),
        "depth":                r.get("depth"),
        "ancestor_chain":       r.get("ancestor_chain"),
        "parent_text":          _clean_text(r.get("parent_text")),
        "root_text":            _clean_text(r.get("root_text")),
        "reply_children_count": r.get("reply_children_count"),
        "message_type":         r.get("message_type"),
    }
    return rec


def _clean_text(text) -> Optional[str]:
    """去除首尾空格，空字符串转 None"""
    if text is None:
        return None
    cleaned = str(text).strip()
    return cleaned if cleaned else None


# ============ 上下文修正（Step 3） ============

def fix_context(rec: dict) -> dict:
    depth = rec.get("depth", -1)
    current = rec.get("current_text")
    parent = rec.get("parent_text")
    root = rec.get("root_text")

    # 标记空 current_text
    rec["empty_current_text"] = (current is None)

    if depth == 0:
        rec["root_text"] = current
        rec["parent_text"] = None
        rec["context_type"] = "root_only"
        # root_message_id / root_content_id 兜底
        if not rec.get("root_message_id"):
            rec["root_message_id"] = rec.get("message_id")
        if not rec.get("root_content_id"):
            rec["root_content_id"] = rec.get("content_id")
        rec["context_text"] = current or ""
        rec["parent_is_root"] = None
        rec["missing_parent_text"] = None
        rec["missing_root_text"] = None

    elif depth == 1:
        # root_text 兜底
        if not root and parent:
            rec["root_text"] = parent
            root = parent
        rec["context_type"] = "root_current"
        # parent_text 与 root_text 相同则置空
        if rec.get("parent_text") and rec.get("parent_text") == rec.get("root_text"):
            rec["parent_text"] = None
            rec["parent_is_root"] = True
        else:
            rec["parent_is_root"] = False
        rec["context_text"] = f"原帖：{rec.get('root_text') or ''}\n当前评论：{current or ''}"
        rec["missing_parent_text"] = None
        rec["missing_root_text"] = (rec.get("root_text") is None)

    elif depth >= 2:
        rec["context_type"] = "root_parent_current"
        rec["parent_is_root"] = False
        rec["missing_parent_text"] = (rec.get("parent_text") is None)
        rec["missing_root_text"] = (rec.get("root_text") is None)
        rec["context_text"] = (
            f"原帖：{rec.get('root_text') or ''}\n"
            f"父评论：{rec.get('parent_text') or ''}\n"
            f"当前评论：{current or ''}"
        )

    return rec


# ============ Audit Flags（Step 4） ============

def contains_any(text: str, keywords: list) -> bool:
    if not text:
        return False
    return any(kw in text for kw in keywords)


def build_audit_flags(rec: dict) -> dict:
    current = rec.get("current_text") or ""
    root = rec.get("root_text") or ""
    parent = rec.get("parent_text") or ""
    depth = rec.get("depth", -1)
    stance = rec.get("stance_label_raw")
    main_target = rec.get("main_target") or ""

    current_stripped = current.replace(" ", "")

    is_neutral = stance == "中立"
    is_against = stance == "反对"
    is_favor   = stance == "支持"

    current_text_length = len(current)
    is_short_reply = len(current_stripped) <= 4

    has_parent_text = bool(rec.get("parent_text"))
    has_root_text   = bool(rec.get("root_text"))

    missing_parent = rec.get("missing_parent_text", False) or False
    missing_root   = rec.get("missing_root_text", False) or False

    agreement_cue             = contains_any(current, AGREEMENT_CUES)
    disagreement_cue          = contains_any(current, DISAGREEMENT_CUES)
    rhetorical_question_cand  = contains_any(current, RHETORICAL_CUES)
    emotion_or_stance_cue     = contains_any(current, EMOTION_STANCE_CUES)

    # needs_context_review
    target_in_current = main_target and main_target in current
    needs_context_review = any([
        depth >= 1 and is_short_reply,
        depth >= 2,
        (not target_in_current and (root or parent)),
        is_neutral and is_short_reply,
    ])

    # neutral_suspicious
    neutral_suspicious = is_neutral and any([
        is_short_reply,
        agreement_cue,
        disagreement_cue,
        rhetorical_question_cand,
        emotion_or_stance_cue,
        depth >= 2 and has_parent_text,
        missing_parent,
        missing_root,
    ])

    # candidate_pattern（多命中输出列表）
    patterns = []
    if agreement_cue and depth >= 2:
        patterns.append("agreement_with_parent")
    if disagreement_cue and depth >= 2:
        patterns.append("disagreement_with_parent")
    if rhetorical_question_cand:
        patterns.append("rhetorical_question_candidate")
    if emotion_or_stance_cue and stance == "反对":
        patterns.append("explicit_or_implicit_against_candidate")
    if emotion_or_stance_cue and stance == "支持":
        patterns.append("explicit_or_implicit_favor_candidate")
    if is_short_reply and depth >= 2:
        patterns.append("short_context_dependent_reply")
    candidate_pattern = patterns if patterns else ["unknown"]

    # audit_priority
    if (neutral_suspicious or is_favor or (depth >= 2 and is_short_reply)):
        audit_priority = "high"
    elif emotion_or_stance_cue or rhetorical_question_cand:
        audit_priority = "medium"
    else:
        audit_priority = "low"

    rec.update({
        "is_neutral":                   is_neutral,
        "is_against":                   is_against,
        "is_favor":                     is_favor,
        "current_text_length":          current_text_length,
        "is_short_reply":               is_short_reply,
        "has_parent_text":              has_parent_text,
        "has_root_text":                has_root_text,
        "needs_context_review":         needs_context_review,
        "agreement_cue":                agreement_cue,
        "disagreement_cue":             disagreement_cue,
        "rhetorical_question_candidate": rhetorical_question_cand,
        "emotion_or_stance_cue":        emotion_or_stance_cue,
        "neutral_suspicious":           neutral_suspicious,
        "candidate_pattern":            candidate_pattern,
        "audit_priority":               audit_priority,
    })
    return rec


# ============ 质量报告 ============

def generate_quality_report(roots: list, instances: list) -> dict:
    depth1 = [r for r in instances if r.get("depth") == 1]
    depth2 = [r for r in instances if r.get("depth") >= 2]

    def count(lst, key): return sum(1 for r in lst if r.get(key))

    return {
        "total_records":                    len(roots) + len(instances),
        "root_count":                       len(roots),
        "depth1_count":                     len(depth1),
        "depth2_count":                     len(depth2),
        "stance_instances_count":           len(instances),
        "empty_current_text_count":         count(roots + instances, "empty_current_text"),
        "missing_root_text_count":          count(instances, "missing_root_text"),
        "missing_parent_text_count":        count(instances, "missing_parent_text"),
        "neutral_count":                    count(instances, "is_neutral"),
        "against_count":                    count(instances, "is_against"),
        "favor_count":                      count(instances, "is_favor"),
        "neutral_suspicious_count":         count(instances, "neutral_suspicious"),
        "short_reply_count":                count(instances, "is_short_reply"),
        "depth2_short_reply_count":         sum(1 for r in depth2 if r.get("is_short_reply")),
        "agreement_cue_count":              count(instances, "agreement_cue"),
        "disagreement_cue_count":           count(instances, "disagreement_cue"),
        "rhetorical_question_candidate_count": count(instances, "rhetorical_question_candidate"),
        "emotion_or_stance_cue_count":      count(instances, "emotion_or_stance_cue"),
        "audit_high_count":                 sum(1 for r in instances if r.get("audit_priority") == "high"),
        "audit_medium_count":               sum(1 for r in instances if r.get("audit_priority") == "medium"),
        "audit_low_count":                  sum(1 for r in instances if r.get("audit_priority") == "low"),
    }


# ============ 写出 JSONL ============

def write_jsonl(records: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✅ 已写出 {len(records)} 条 → {filepath}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="微博会话树立场检测数据处理脚本")
    parser.add_argument("--input",      required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录路径")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取输入
    print(f"📂 读取输入文件：{args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  共加载 {len(raw_data)} 条记录")

    roots     = []
    instances = []

    for i, r in enumerate(raw_data):
        try:
            rec = standardize_record(r)   # Step 1
            rec = fix_context(rec)        # Step 2 & 3

            depth = rec.get("depth", -1)

            if depth == 0:
                roots.append(rec)
            else:
                rec = build_audit_flags(rec)  # Step 4
                instances.append(rec)

        except Exception as e:
            print(f"  ⚠️  第 {i} 条记录处理异常，已跳过：{e}")
            continue

    # 筛选 audit 候选
    audit_candidates = [r for r in instances if r.get("audit_priority") in ("high", "medium")]

    # 生成质量报告
    report = generate_quality_report(roots, instances)

    # 写出文件
    print(f"\n📝 写出文件到 {output_dir}/")
    write_jsonl(roots,           str(output_dir / "root_context.jsonl"))
    write_jsonl(instances,       str(output_dir / "stance_instances.jsonl"))
    write_jsonl(audit_candidates,str(output_dir / "audit_candidates.jsonl"))

    report_path = output_dir / "tree_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 质量报告 → {report_path}")

    # 控制台摘要
    print(f"""
📊 处理摘要：
  原帖（depth=0）：       {report['root_count']} 条
  一级评论（depth=1）：   {report['depth1_count']} 条
  二级评论（depth=2）：   {report['depth2_count']} 条
  空 current_text：       {report['empty_current_text_count']} 条
  中立可疑样本：          {report['neutral_suspicious_count']} 条
  audit high：            {report['audit_high_count']} 条
  audit medium：          {report['audit_medium_count']} 条
  audit low：             {report['audit_low_count']} 条
    """)


if __name__ == "__main__":
    main()