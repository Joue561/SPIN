#!/usr/bin/env python3
"""
build_semantic_propagation_annotations.py

用法：
python build_semantic_propagation_annotations.py \
  --windowed_instances processed_cmtscd_windows/cmtscd_windowed_instances.jsonl \
  --output_dir processed_cmtscd_semantic \
  --mode rule
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter


# ============ 词表 ============

AGREE_CUES = ["确实", "对", "对的", "没错", "说得对", "同意", "支持", "赞同",
              "是的", "太对了", "+1", "顶", "可不是"]

DISAGREE_CUES = ["不是", "不对", "不同意", "胡说", "瞎说", "别扯", "别洗",
                 "你错了", "并不是", "没这回事", "谁说的"]

MOCK_CUES = ["呵呵", "笑死", "典", "绷", "666", "懂的都懂", "你可真懂",
             "阅读理解满分", "真会说", "老板爱听"]

SARCASTIC_SURFACE_AGREE = ["确实", "是挺", "真是", "太好了", "效率高", "挺好"]
SARCASTIC_COUNTER_CUES  = ["一天开六个会", "谁受得了", "真文明", "真先进",
                            "又来了", "坐得久", "太辛苦", "累死", "没时间"]

QUESTION_CUES = ["？", "?", "你是说", "什么意思", "怎么就", "凭什么",
                 "这也能", "不会真有人", "难道"]

CLARIFY_CUES = ["我的意思是", "他说的是", "不是说", "重点是", "我说的是",
                "不是这个意思", "你理解错了"]

SUPPLEMENT_CUES = ["而且", "另外", "还有", "关键是", "主要是", "问题是", "补充一点"]

TOPIC_SHIFT_CUES = ["你", "你们", "楼上", "层主", "杠精", "阅读理解",
                    "你这逻辑", "别杠", "别扣帽子"]

ATTRIBUTE_CUES = ["价格", "质量", "体验", "性能", "效率", "成本", "管理",
                  "执行", "风险", "味道", "功能", "服务", "文化", "通勤",
                  "沟通", "会议", "工作效率", "方便", "安全"]

CONSEQUENCE_CUES = ["会导致", "最后", "结果", "影响", "后果", "翻车",
                    "麻烦", "吵起来", "浪费时间", "增加成本"]

CHILD_QUESTION_CUES  = ["你什么意思", "你是说", "你是在支持还是反对",
                         "你这是夸还是骂", "啥意思"]
CHILD_SARCASM_CUES   = ["明显反讽", "阴阳", "讽刺", "反讽", "不是夸", "是在骂"]
CHILD_CORRECT_CUES   = ["你理解错了", "不是这个意思", "他说的是", "他意思是", "别偷换"]
CHILD_DISAGREE_CUES  = ["不是", "不对", "不同意", "胡说", "别洗", "你错了"]
CHILD_CONFIRM_CUES   = ["对", "确实", "没错", "就是这个意思", "说得对", "同意"]

REPLY_RELATION_PRIORITY = [
    "sarcastic_agreement", "mock_parent", "disagree_with_parent",
    "agree_with_parent", "clarify_parent", "question_parent",
    "supplement_parent", "topic_shift", "unknown",
]

TARGET_RELATION_PRIORITY = [
    "reply_author_only", "same_target", "consequence_of_target",
    "attribute_of_target", "topic_shift", "unclear",
]


# ============ IO ============

def load_jsonl(filepath: str) -> list:
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
                print(f"  ⚠️  第 {i+1} 行解析失败，已跳过：{e}")
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


# ============ 工具 ============

def normalize_text(text) -> str:
    return str(text).strip() if text else ""


def contains_any(text: str, keywords: list) -> bool:
    return any(kw in text for kw in keywords)


def safe_list(val) -> list:
    if isinstance(val, list):
        return val
    return []


# ============ 回复关系检测 ============

def detect_reply_relation(
    current_text: str,
    parent_text: str,
    target_drift_candidate: str,
    is_ambiguous: bool,
) -> str:
    ct = normalize_text(current_text)

    hits = set()

    # sarcastic_agreement（优先级最高）
    if contains_any(ct, SARCASTIC_SURFACE_AGREE) and contains_any(ct, SARCASTIC_COUNTER_CUES):
        hits.add("sarcastic_agreement")

    # mock_parent
    if contains_any(ct, MOCK_CUES) and (len(ct.replace(" ", "")) <= 10 or is_ambiguous):
        hits.add("mock_parent")

    # disagree_with_parent
    if contains_any(ct, DISAGREE_CUES):
        hits.add("disagree_with_parent")

    # agree_with_parent
    if contains_any(ct, AGREE_CUES):
        hits.add("agree_with_parent")

    # clarify_parent
    if contains_any(ct, CLARIFY_CUES):
        hits.add("clarify_parent")

    # question_parent
    if contains_any(ct, QUESTION_CUES):
        hits.add("question_parent")

    # supplement_parent
    if contains_any(ct, SUPPLEMENT_CUES):
        hits.add("supplement_parent")

    # topic_shift
    if target_drift_candidate == "reply_author_only" or (
        contains_any(ct, TOPIC_SHIFT_CUES) and not contains_any(ct, AGREE_CUES + DISAGREE_CUES)
    ):
        hits.add("topic_shift")

    # 按优先级返回
    for rel in REPLY_RELATION_PRIORITY:
        if rel in hits:
            return rel
    return "unknown"


# ============ 语义传导模式 ============

def detect_semantic_propagation(
    reply_relation: str,
    parent_text: str,
    depth: int,
) -> str:
    has_parent = bool(normalize_text(parent_text))

    if not has_parent or depth <= 0:
        return "no_parent_context"

    mapping = {
        "agree_with_parent":    "stance_inheritance",
        "supplement_parent":    "stance_amplification",
        "disagree_with_parent": "stance_reversal",
        "mock_parent":          "stance_reversal",
        "sarcastic_agreement":  "sarcastic_reversal",
        "clarify_parent":       "stance_softening",
        "question_parent":      "unclear_propagation",
        "topic_shift":          "target_shift",
        "unknown":              "unclear_propagation",
    }
    return mapping.get(reply_relation, "unclear_propagation")


# ============ 父子立场映射 ============

def detect_stance_mapping_to_parent(
    semantic_propagation: str,
    parent_text: str,
) -> str:
    if not normalize_text(parent_text):
        return "not_applicable"

    mapping = {
        "stance_inheritance":  "same_as_parent",
        "stance_amplification": "same_as_parent",
        "stance_reversal":     "opposite_to_parent",
        "sarcastic_reversal":  "opposite_to_parent",
        "stance_softening":    "partial_or_conditional",
        "target_shift":        "not_applicable",
        "unclear_propagation": "unclear",
        "no_parent_context":   "not_applicable",
    }
    return mapping.get(semantic_propagation, "unclear")


# ============ target 关系检测 ============

def detect_target_relation(
    current_text: str,
    main_target: str,
    topic: str,
    target_drift_candidate: str,
) -> str:
    ct = normalize_text(current_text)
    ct_stripped = ct.replace(" ", "")

    hits = set()

    # reply_author_only
    if contains_any(ct, TOPIC_SHIFT_CUES) and not contains_any(ct, ATTRIBUTE_CUES):
        hits.add("reply_author_only")

    # same_target
    if main_target in ct or topic in ct:
        hits.add("same_target")

    # consequence_of_target
    if contains_any(ct, CONSEQUENCE_CUES):
        hits.add("consequence_of_target")

    # attribute_of_target
    if contains_any(ct, ATTRIBUTE_CUES):
        hits.add("attribute_of_target")

    # topic_shift
    if target_drift_candidate == "possible_topic_shift" and not hits:
        hits.add("topic_shift")

    # unclear
    if len(ct_stripped) <= 4 and not hits:
        hits.add("unclear")

    for rel in TARGET_RELATION_PRIORITY:
        if rel in hits:
            return rel
    return "unclear"


# ============ 子回复反馈信号 ============

def detect_child_feedback_signal(
    feedback_texts: list,
) -> str:
    if not feedback_texts:
        return "no_child"

    signals = set()
    for ftxt in feedback_texts:
        ft = normalize_text(ftxt)
        if contains_any(ft, CHILD_QUESTION_CUES):
            signals.add("child_questions_current_meaning")
        if contains_any(ft, CHILD_SARCASM_CUES):
            signals.add("child_indicates_sarcasm")
        if contains_any(ft, CHILD_CORRECT_CUES):
            signals.add("child_corrects_misreading")
        if contains_any(ft, CHILD_DISAGREE_CUES):
            signals.add("child_indicates_disagreement")
        if contains_any(ft, CHILD_CONFIRM_CUES):
            signals.add("child_confirms_current_meaning")

    if len(signals) > 1:
        # 冲突：既有确认又有质疑
        if ("child_confirms_current_meaning" in signals and
                any(s in signals for s in [
                    "child_questions_current_meaning",
                    "child_indicates_sarcasm",
                    "child_corrects_misreading",
                    "child_indicates_disagreement",
                ])):
            return "child_conflict"

    # 按严重程度返回最重要信号
    priority = [
        "child_indicates_sarcasm",
        "child_corrects_misreading",
        "child_questions_current_meaning",
        "child_indicates_disagreement",
        "child_confirms_current_meaning",
        "child_conflict",
    ]
    for sig in priority:
        if sig in signals:
            return sig

    return "no_useful_child_feedback"


# ============ candidate_pattern 生成 ============

def build_candidate_patterns(
    reply_relation: str,
    target_relation: str,
    child_feedback: str,
    is_ambiguous: bool,
) -> list:
    patterns = []

    # 回复关系 → pattern
    rel_map = {
        "agree_with_parent":    "agreement_with_parent",
        "disagree_with_parent": "disagreement_with_parent",
        "mock_parent":          "mocking_disagreement",
        "sarcastic_agreement":  "sarcastic_agreement",
    }
    if reply_relation in rel_map:
        patterns.append(rel_map[reply_relation])

    # 模糊短文本
    if is_ambiguous:
        patterns.append("short_ambiguous_reply")

    # 子回复消歧
    if child_feedback not in ("no_child", "no_useful_child_feedback"):
        patterns.append("child_feedback_disambiguation")

    # target 关系
    if target_relation in ("reply_author_only", "topic_shift"):
        patterns.append("target_drift")
    elif target_relation == "attribute_of_target":
        patterns.append("attribute_based_stance")
    elif target_relation == "consequence_of_target":
        patterns.append("consequence_based_stance")

    return patterns if patterns else ["unknown_pattern"]


# ============ 置信度计算 ============

def compute_semantic_confidence(
    reply_relation: str,
    target_relation: str,
    child_feedback: str,
    current_text: str,
) -> float:
    ct_stripped = normalize_text(current_text).replace(" ", "")
    base = 0.45

    # 有明确 cue 命中
    if reply_relation in ("agree_with_parent", "disagree_with_parent",
                           "sarcastic_agreement", "mock_parent", "question_parent"):
        base = 0.75

    # 子回复支持
    if child_feedback not in ("no_child", "no_useful_child_feedback", "child_conflict"):
        base += 0.10

    # target 关系加分
    if target_relation in ("same_target", "attribute_of_target", "consequence_of_target"):
        base += 0.05

    # 短文本无子回复扣分
    if len(ct_stripped) <= 4 and child_feedback in ("no_child", "no_useful_child_feedback"):
        base -= 0.10

    # 未知关系扣分
    if reply_relation == "unknown":
        base = 0.45

    # target 不明确扣分
    if target_relation in ("topic_shift", "unclear"):
        base -= 0.10

    return round(max(0.10, min(0.95, base)), 3)


# ============ 单条标注 ============

def annotate_instance(rec: dict) -> dict:
    current_text     = normalize_text(rec.get("current_text"))
    parent_text      = normalize_text(rec.get("parent_text"))
    main_target      = normalize_text(rec.get("main_target"))
    topic            = normalize_text(rec.get("topic"))
    depth            = rec.get("depth", 1)
    is_ambiguous     = bool(rec.get("is_ambiguous_current", False))
    target_drift     = normalize_text(rec.get("target_drift_candidate", "unclear"))
    feedback_texts   = safe_list(rec.get("downward_feedback_window_texts"))

    # 1. 回复关系
    reply_relation = detect_reply_relation(
        current_text, parent_text, target_drift, is_ambiguous
    )

    # 2. 语义传导
    semantic_prop = detect_semantic_propagation(reply_relation, parent_text, depth)

    # 3. 父子立场映射
    stance_mapping = detect_stance_mapping_to_parent(semantic_prop, parent_text)

    # 4. target 关系
    target_relation = detect_target_relation(
        current_text, main_target, topic, target_drift
    )

    # 5. 子回复信号
    child_signal = detect_child_feedback_signal(feedback_texts)

    # 6. candidate_pattern
    patterns = build_candidate_patterns(
        reply_relation, target_relation, child_signal, is_ambiguous
    )

    # 7. 置信度
    confidence = compute_semantic_confidence(
        reply_relation, target_relation, child_signal, current_text
    )

    out = dict(rec)
    out.update({
        "reply_relation_candidate":    reply_relation,
        "semantic_propagation_pattern": semantic_prop,
        "stance_mapping_to_parent":    stance_mapping,
        "target_relation_candidate":   target_relation,
        "child_feedback_signal":       child_signal,
        "candidate_pattern":           patterns,
        "semantic_confidence":         confidence,
        "semantic_annotation_source":  "rule_based",
    })
    return out


# ============ 质量报告 ============

def generate_report(annotated: list) -> dict:
    total = len(annotated)

    def dist(key):
        return dict(Counter(r.get(key) for r in annotated))

    def list_dist(key):
        c = Counter()
        for r in annotated:
            for p in safe_list(r.get(key)):
                c[p] += 1
        return dict(c)

    confidences = [r.get("semantic_confidence", 0) for r in annotated]
    avg_conf    = round(sum(confidences) / total, 4) if total else 0
    high_conf   = sum(1 for c in confidences if c >= 0.75)
    low_conf    = sum(1 for c in confidences if c < 0.5)

    short_ambiguous = sum(1 for r in annotated
                          if "short_ambiguous_reply" in safe_list(r.get("candidate_pattern")))
    child_disam     = sum(1 for r in annotated
                          if "child_feedback_disambiguation" in safe_list(r.get("candidate_pattern")))
    target_drift    = sum(1 for r in annotated
                          if "target_drift" in safe_list(r.get("candidate_pattern")))

    return {
        "total_instances":                   total,
        "annotated_instances":               total,
        "reply_relation_distribution":       dist("reply_relation_candidate"),
        "semantic_propagation_distribution": dist("semantic_propagation_pattern"),
        "stance_mapping_to_parent_distribution": dist("stance_mapping_to_parent"),
        "target_relation_distribution":      dist("target_relation_candidate"),
        "child_feedback_signal_distribution": dist("child_feedback_signal"),
        "candidate_pattern_distribution":    list_dist("candidate_pattern"),
        "average_semantic_confidence":       avg_conf,
        "high_confidence_count":             high_conf,
        "low_confidence_count":              low_conf,
        "split_distribution":                dist("split"),
        "topic_distribution":                dist("topic"),
        "depth_distribution":                dict(Counter(r.get("depth") for r in annotated)),
        "short_ambiguous_reply_count":       short_ambiguous,
        "child_feedback_disambiguation_count": child_disam,
        "target_drift_count":                target_drift,
    }


# ============ 示例文件 ============

EXAMPLE_FIELDS = [
    "topic", "node_id", "main_target", "current_text", "parent_text",
    "upward_window_texts", "downward_feedback_window_texts",
    "reply_relation_candidate", "semantic_propagation_pattern",
    "stance_mapping_to_parent", "target_relation_candidate",
    "child_feedback_signal", "candidate_pattern",
    "semantic_confidence", "causal_context_text", "posthoc_context_text",
    "stance_label",
]


def generate_examples(annotated: list) -> list:
    seen = set()
    examples = []

    def add_batch(pool, max_n=100):
        count = 0
        for r in pool:
            nid = r.get("node_id", "")
            if nid in seen or count >= max_n:
                continue
            seen.add(nid)
            examples.append({k: r.get(k) for k in EXAMPLE_FIELDS})
            count += 1

    def filter_pattern(pat):
        return [r for r in annotated
                if pat in safe_list(r.get("candidate_pattern"))]

    add_batch(filter_pattern("agreement_with_parent"))
    add_batch(filter_pattern("mocking_disagreement"))
    add_batch(filter_pattern("sarcastic_agreement"))
    add_batch(filter_pattern("child_feedback_disambiguation"))
    add_batch(filter_pattern("target_drift"))
    add_batch(filter_pattern("consequence_based_stance"))
    add_batch(sorted(annotated, key=lambda r: r.get("semantic_confidence", 1))[:200])

    return examples


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="语义传导标注模块")
    parser.add_argument("--windowed_instances", required=True)
    parser.add_argument("--output_dir",         required=True)
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 读取：{args.windowed_instances}")
    instances = load_jsonl(args.windowed_instances)
    print(f"  共加载 {len(instances)} 条")

    if args.mode == "llm":
        print("⚠️  LLM 模式未实现，回退到 rule 模式")

    print("\n🔍 规则标注中...")
    annotated = []
    for i, rec in enumerate(instances):
        try:
            annotated.append(annotate_instance(rec))
        except Exception as e:
            print(f"  ⚠️  第 {i+1} 条标注失败，已跳过：{e}")

    report   = generate_report(annotated)
    examples = generate_examples(annotated)

    print(f"\n📝 写出文件到 {output_dir}/")
    write_jsonl(annotated, output_dir / "cmtscd_semantic_instances.jsonl")
    write_jsonl(examples,  output_dir / "cmtscd_semantic_examples.jsonl")
    write_json(report,     output_dir / "cmtscd_semantic_quality_report.json")

    print(f"""
📊 标注摘要：
  总计：          {report['total_instances']}
  高置信（≥0.75）：{report['high_confidence_count']}
  低置信（<0.50）：{report['low_confidence_count']}
  平均置信度：    {report['average_semantic_confidence']}
  回复关系分布：  {report['reply_relation_distribution']}
  语义传导分布：  {report['semantic_propagation_distribution']}
  短模糊回复：    {report['short_ambiguous_reply_count']}
  子回复消歧：    {report['child_feedback_disambiguation_count']}
  目标漂移：      {report['target_drift_count']}
    """)


if __name__ == "__main__":
    main()