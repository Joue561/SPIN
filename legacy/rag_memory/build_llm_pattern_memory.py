#!/usr/bin/env python3
"""
build_llm_pattern_memory.py

用法：
python build_llm_pattern_memory.py \
  --input processed_cmtscd_semantic_llm/cmtscd_semantic_instances_llm.jsonl \
  --output_dir processed_cmtscd_rag_memory \
  --min_confidence 0.70 \
  --min_pattern_count 10 \
  --train_only true
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional
import math


# ============ 常量 ============

VALID_STANCE_LABELS = {"favor", "against", "none"}
STANCE_ZH = {"favor": "支持", "against": "反对", "none": "中立"}

VALID_LANGUAGE_PATTERNS = {
    "agreement_with_parent", "disagreement_with_parent", "mocking_disagreement",
    "sarcastic_agreement", "short_ambiguous_reply", "child_feedback_disambiguation",
    "direct_positive_evaluation", "direct_negative_evaluation",
    "conditional_or_mixed_stance", "attribute_based_stance",
    "consequence_based_stance", "target_drift", "topic_shift",
    "question_or_challenge", "clarification",
}

PATTERN_DESCRIPTIONS = {
    "agreement_with_parent":
        "当前评论对父评论表达赞同，例如'确实''说得对''同意'等。通常继承父评论语义方向。",
    "disagreement_with_parent":
        "当前评论对父评论提出明确反驳或质疑，立场通常与父评论相反。",
    "mocking_disagreement":
        "当前评论以嘲讽或讽刺方式否定父评论，常见'呵呵''笑死''典'等。",
    "sarcastic_agreement":
        "当前评论表面附和但实为讽刺，例如'真文明''效率高'等反语表达。",
    "short_ambiguous_reply":
        "当前评论文本极短或语义模糊，立场需结合父评论和子回复判断。",
    "child_feedback_disambiguation":
        "子回复帮助澄清当前评论的真实语义，例如子回复指出当前评论是讽刺或被误读。",
    "direct_positive_evaluation":
        "当前评论直接对主 target 或相关话题给出正面评价。",
    "direct_negative_evaluation":
        "当前评论直接对主 target 或相关话题给出负面评价。",
    "conditional_or_mixed_stance":
        "当前评论立场包含条件或混合态度，不能简单归为支持或反对。",
    "attribute_based_stance":
        "当前评论通过讨论 target 的属性（价格、质量、安全等）间接表达立场。",
    "consequence_based_stance":
        "当前评论通过讨论 target 的后果或影响间接表达立场。",
    "target_drift":
        "当前评论目标偏离主 target，转向回复作者或相关对象。",
    "topic_shift":
        "当前评论话题发生明显偏移，与主 target 关联性弱。",
    "question_or_challenge":
        "当前评论以反问或质疑方式表达立场，常含隐式反对或不信任。",
    "clarification":
        "当前评论澄清自身或父评论的语义，不一定含明确立场。",
}

PROPAGATION_DESCRIPTIONS = {
    "stance_inheritance":
        "当前评论继承父评论的立场，方向一致。",
    "stance_reversal":
        "当前评论反转父评论的立场，方向相反。",
    "sarcastic_reversal":
        "当前评论以讽刺方式反转父评论立场，表面赞同实则反对。",
    "stance_amplification":
        "当前评论强化父评论的立场，语气或证据更强。",
    "stance_softening":
        "当前评论弱化或附带条件地接受父评论立场。",
    "direct_stance_expression":
        "当前评论直接表达对主 target 的立场，不依赖父评论传导。",
    "target_shift":
        "当前评论的立场目标发生偏移，不再直接针对主 target。",
    "unclear_propagation":
        "语义传导关系不清晰，需结合更多上下文判断。",
    "no_parent_context":
        "缺少父评论上下文，无法判断传导关系。",
}

TARGET_RELATION_DESCRIPTIONS = {
    "same_target":
        "当前评论直接讨论主 target，立场指向明确。",
    "attribute_of_target":
        "当前评论讨论 target 的某项属性，如价格、性能、安全、体验等。",
    "consequence_of_target":
        "当前评论讨论 target 带来的后果或影响。",
    "reason_for_target":
        "当前评论讨论对 target 持某种立场的原因。",
    "opposition_to_related_target":
        "当前评论反对与主 target 相关的实体，间接表达对主 target 的立场。",
    "support_to_related_target":
        "当前评论支持与主 target 相关的实体，间接表达对主 target 的立场。",
    "reply_author_only":
        "当前评论主要针对回复对象本人，不应直接作为主 target 立场证据。",
    "topic_shift":
        "当前评论话题发生偏移，与主 target 关联性弱，谨慎使用。",
    "irrelevant":
        "当前评论与主 target 无关。",
    "unclear":
        "当前评论目标关系不明确。",
}

APPLICABLE_CONDITIONS = {
    "agreement_with_parent": [
        "当前评论回复父评论",
        "当前评论包含赞同或确认语气",
        "父评论立场较明确",
        "当前评论未转移到其他 target",
    ],
    "mocking_disagreement": [
        "当前评论包含'呵呵''笑死''典'等讽刺词",
        "当前评论较短，语气明显",
        "父评论立场较明确",
    ],
    "attribute_based_stance": [
        "当前评论讨论 target 的价格、质量、安全、体验等属性",
        "未引入新的评论目标",
    ],
    "consequence_based_stance": [
        "当前评论明确讨论 target 带来的后果或结果",
        "因果关系较明确",
    ],
}

CAUTIONS = {
    "agreement_with_parent":
        "父评论立场不明确或当前评论具反讽语气时不应直接套用。",
    "short_ambiguous_reply":
        "必须结合父评论和子回复才能判断立场，单独使用风险高。",
    "sarcastic_agreement":
        "讽刺识别难度高，建议结合子回复信号和 LLM 置信度使用。",
    "child_feedback_disambiguation":
        "子回复信号为离线 post-hoc 信息，实时预测不可使用。",
    "target_drift":
        "该模式下立场指向不明确，不建议直接用于主 target 立场推断。",
    "reply_author_only":
        "评论针对回复者本人，不应作为主 target 立场的直接证据。",
    "topic_shift":
        "话题偏移，知识库使用时需额外过滤确认。",
}


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


def safe_str(val) -> str:
    return str(val).strip() if val else ""


def safe_list(val) -> list:
    if isinstance(val, list):
        return val
    return []


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ============ 规范化 ============

def normalize_candidate_patterns(raw) -> list:
    """规范化 llm_candidate_patterns，处理各种格式"""
    if raw is None:
        return ["unknown_pattern"]

    # 拆解为字符串列表
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return ["unknown_pattern"]

    # 拆分 "|" 连接的字符串
    expanded = []
    for item in items:
        for p in str(item).split("|"):
            p = p.strip()
            if p:
                expanded.append(p)

    # 过滤：若有有效 pattern，移除 unknown_pattern
    valid = [p for p in expanded if p in VALID_LANGUAGE_PATTERNS]
    if valid:
        return list(dict.fromkeys(valid))  # 去重保序

    # 全为无效
    unknown = [p for p in expanded if p == "unknown_pattern"]
    return unknown if unknown else ["unknown_pattern"]


def normalize_label(stance_label: str) -> str:
    mapping = {"favor": "favor", "against": "against", "none": "none"}
    return mapping.get(safe_str(stance_label).lower(), "none")


def compute_llm_label_agreement(
    stance_norm: str,
    llm_inferred: str,
) -> str:
    non_comparable = {"mixed_or_conditional", "unclear", "irrelevant", "unknown"}
    if llm_inferred in non_comparable:
        return "not_comparable"
    if stance_norm == llm_inferred:
        return "agree"
    return "disagree"


# ============ 质量计算 ============

def compute_memory_quality(
    rec: dict,
    min_confidence: float,
    train_only: bool,
) -> dict:
    split          = safe_str(rec.get("split"))
    confidence     = safe_float(rec.get("llm_confidence", 0.0))
    reply_relation = safe_str(rec.get("llm_reply_relation"))
    target_relation = safe_str(rec.get("llm_target_relation"))
    patterns       = rec.get("_norm_patterns", ["unknown_pattern"])
    current_text   = safe_str(rec.get("current_text"))
    stance_norm    = rec.get("_stance_norm", "none")
    llm_agreement  = rec.get("_llm_label_agreement", "not_comparable")
    llm_inferred   = safe_str(rec.get("llm_inferred_stance_to_main_target", "unclear"))

    exclusion_reasons = []

    # 基本过滤
    if train_only and split != "train":
        exclusion_reasons.append(f"non_train_split:{split}")
    if confidence < min_confidence:
        exclusion_reasons.append(f"low_confidence:{confidence:.2f}")
    if reply_relation == "unknown":
        exclusion_reasons.append("unknown_reply_relation")
    if target_relation in ("irrelevant", "unclear"):
        exclusion_reasons.append(f"invalid_target_relation:{target_relation}")
    if patterns == ["unknown_pattern"]:
        exclusion_reasons.append("unknown_pattern_only")
    if not current_text:
        exclusion_reasons.append("empty_current_text")
    if stance_norm not in VALID_STANCE_LABELS:
        exclusion_reasons.append(f"invalid_stance:{stance_norm}")

    use_for_memory = len(exclusion_reasons) == 0

    # 质量等级
    if not use_for_memory:
        quality = "excluded_memory"
        base_weight = 0.0
    elif (confidence >= 0.85
          and target_relation in ("same_target", "attribute_of_target",
                                  "consequence_of_target", "reason_for_target")
          and reply_relation != "unknown"
          and patterns != ["unknown_pattern"]):
        quality = "strong_memory"
        base_weight = 1.0
    elif confidence >= 0.70:
        quality = "medium_memory"
        base_weight = 0.7
    else:
        quality = "weak_memory"
        base_weight = 0.3

    # 权重调整
    weight = base_weight
    if use_for_memory:
        if llm_agreement == "agree":
            weight = min(1.0, weight + 0.05)
        elif llm_agreement == "disagree":
            weight = max(0.1, weight - 0.15)
        if llm_inferred == "mixed_or_conditional":
            weight = max(0.1, weight - 0.10)

    return {
        "memory_quality_level": quality,
        "memory_label_weight":  round(weight, 4),
        "use_for_memory":       use_for_memory,
        "exclusion_reason":     "; ".join(exclusion_reasons) if exclusion_reasons else None,
        "child_feedback_used":  bool(rec.get("llm_use_child_feedback")),
    }


# ============ 过滤 ============

def filter_memory_instances(records: list) -> tuple:
    usable   = [r for r in records if r.get("use_for_memory")]
    excluded = [r for r in records if not r.get("use_for_memory")]
    return usable, excluded


# ============ 分布计算 ============

def compute_label_distribution(
    group: list,
    weight_field: str = "memory_label_weight",
) -> tuple:
    raw_dist      = Counter()
    weighted_dist = defaultdict(float)

    for r in group:
        label  = r.get("_stance_norm", "none")
        weight = safe_float(r.get(weight_field, 1.0))
        raw_dist[label] += 1
        weighted_dist[label] += weight

    total_w = sum(weighted_dist.values()) or 1
    w_norm  = {lbl: round(v / total_w, 4) for lbl, v in weighted_dist.items()}

    return dict(raw_dist), w_norm


def compute_label_lift(
    pattern_w_dist: dict,
    global_prior:   dict,
) -> dict:
    lifts = {}
    for lbl in ("favor", "against", "none"):
        p_label_given_pattern = pattern_w_dist.get(lbl, 0.0)
        p_label = global_prior.get(lbl, 1e-6)
        lifts[f"{lbl}_lift"] = round(p_label_given_pattern / p_label, 4)
    return lifts


def generic_dist(group: list, field: str) -> dict:
    return dict(Counter(safe_str(r.get(field)) for r in group))


def generic_list_dist(group: list, field: str) -> dict:
    c = Counter()
    for r in group:
        for p in safe_list(r.get(field)):
            c[str(p)] += 1
    return dict(c)


# ============ 典型 / 反例 ============

EXAMPLE_FIELDS = [
    "topic", "node_id", "current_text", "parent_text",
    "upward_window_texts", "downward_feedback_window_texts",
    "stance_label", "llm_inferred_stance_to_main_target",
    "llm_reply_relation", "llm_semantic_propagation",
    "llm_target_relation", "llm_confidence", "llm_reason_short",
]


def select_typical_examples(
    group: list,
    dominant_label: str,
    max_n: int = 5,
) -> list:
    candidates = sorted(
        [r for r in group if r.get("_stance_norm") == dominant_label
         and safe_str(r.get("current_text"))],
        key=lambda r: (
            -safe_float(r.get("memory_label_weight")),
            -safe_float(r.get("llm_confidence")),
        )
    )
    seen_topics = set()
    examples = []
    for r in candidates:
        if len(examples) >= max_n:
            break
        topic = r.get("topic", "")
        seen_topics.add(topic)
        examples.append({k: r.get(k) for k in EXAMPLE_FIELDS})
    return examples


def select_counter_examples(
    group: list,
    dominant_label: str,
    max_n: int = 3,
) -> list:
    candidates = sorted(
        [r for r in group
         if r.get("_stance_norm") != dominant_label
         and safe_str(r.get("current_text"))],
        key=lambda r: -safe_float(r.get("llm_confidence"))
    )
    return [{k: r.get(k) for k in EXAMPLE_FIELDS} for r in candidates[:max_n]]


# ============ 可靠性 ============

def compute_reliability(
    weighted_count: float,
    dominant_prob:  float,
    avg_confidence: float,
) -> tuple:
    if (weighted_count >= 100 and dominant_prob >= 0.70 and avg_confidence >= 0.80):
        level = "high"
        usage = "use_as_strong_rag_memory"
    elif (weighted_count >= 30 and dominant_prob >= 0.55 and avg_confidence >= 0.70):
        level = "medium"
        usage = "use_as_weak_rag_memory"
    elif weighted_count >= 10:
        level = "low"
        usage = "retrieval_only"
    else:
        level = "excluded"
        usage = "do_not_use"
    return level, usage


def build_memory_unit(
    group: list,
    global_prior: dict,
    key_name: str,
    key_value: str,
    memory_type: str,
    extra_fields: Optional[dict] = None,
) -> dict:
    """通用 memory unit 构建器"""
    raw_dist, w_dist = compute_label_distribution(group)
    lifts = compute_label_lift(w_dist, global_prior)

    total_raw    = sum(raw_dist.values())
    total_weight = sum(safe_float(r.get("memory_label_weight")) for r in group)
    avg_conf     = round(
        sum(safe_float(r.get("llm_confidence")) for r in group) / max(len(group), 1), 4
    )
    avg_weight   = round(total_weight / max(len(group), 1), 4)

    dominant_label = max(w_dist, key=w_dist.get) if w_dist else "none"
    dominant_prob  = w_dist.get(dominant_label, 0.0)
    reliability, usage = compute_reliability(total_weight, dominant_prob, avg_conf)

    typical  = select_typical_examples(group, dominant_label)
    counters = select_counter_examples(group, dominant_label)

    mem = {
        "memory_id":                   f"{memory_type}_{key_value.replace(' ', '_')}",
        "memory_type":                 memory_type,
        key_name:                      key_value,
        "source_sample_count_raw":     total_raw,
        "source_sample_count_weighted": round(total_weight, 2),
        "label_distribution_raw":      raw_dist,
        "label_distribution_weighted": w_dist,
        "llm_inferred_label_distribution": generic_dist(group, "llm_inferred_stance_to_main_target"),
        "topic_distribution":          generic_dist(group, "topic"),
        "depth_distribution":          dict(Counter(r.get("depth") for r in group)),
        "average_confidence":          avg_conf,
        "average_weight":              avg_weight,
        "dominant_label":              dominant_label,
        "dominant_label_probability":  dominant_prob,
        "reliability_level":           reliability,
        "suggested_usage":             usage,
        "typical_examples":            typical,
        "counter_examples":            counters,
        "applicable_conditions":       APPLICABLE_CONDITIONS.get(key_value, ["无特定条件限制"]),
        "cautions":                    CAUTIONS.get(key_value, "请结合具体上下文谨慎使用 AI-Silver 标签。"),
        **lifts,
    }
    if extra_fields:
        mem.update(extra_fields)
    return mem


# ============ 语言结构模式记忆库 ============

def build_language_pattern_memory(
    usable: list,
    global_prior: dict,
    min_count: int,
) -> list:
    groups = defaultdict(list)
    for r in usable:
        for pat in r.get("_norm_patterns", []):
            groups[pat].append(r)

    memories = []
    for pat, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < min_count:
            continue
        mem = build_memory_unit(
            group, global_prior, "pattern_name", pat, "language_pattern",
            extra_fields={
                "pattern_description":            PATTERN_DESCRIPTIONS.get(pat, pat),
                "reply_relation_distribution":    generic_dist(group, "llm_reply_relation"),
                "semantic_propagation_distribution": generic_dist(group, "llm_semantic_propagation"),
                "target_relation_distribution":   generic_dist(group, "llm_target_relation"),
                "stance_mapping_to_parent_distribution": generic_dist(group, "llm_stance_mapping_to_parent"),
            }
        )
        memories.append(mem)
    return memories


# ============ 语义传导模式记忆库 ============

def build_semantic_propagation_memory(
    usable: list,
    global_prior: dict,
    min_count: int,
) -> list:
    groups = defaultdict(list)
    for r in usable:
        prop = safe_str(r.get("llm_semantic_propagation"))
        if prop:
            groups[prop].append(r)

    memories = []
    for prop, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < min_count:
            continue
        child_used = sum(1 for r in group if r.get("llm_use_child_feedback"))
        mem = build_memory_unit(
            group, global_prior, "semantic_propagation", prop, "semantic_propagation",
            extra_fields={
                "propagation_description":   PROPAGATION_DESCRIPTIONS.get(prop, prop),
                "parent_mapping_distribution": generic_dist(group, "llm_stance_mapping_to_parent"),
                "common_reply_relations":    generic_dist(group, "llm_reply_relation"),
                "common_candidate_patterns": generic_list_dist(group, "_norm_patterns"),
                "target_relation_distribution": generic_dist(group, "llm_target_relation"),
                "child_feedback_usage_rate": round(child_used / max(len(group), 1), 4),
            }
        )
        memories.append(mem)
    return memories


# ============ 目标关系模式记忆库 ============

def build_target_relation_memory(
    usable: list,
    global_prior: dict,
    min_count: int,
) -> list:
    groups = defaultdict(list)
    for r in usable:
        tr = safe_str(r.get("llm_target_relation"))
        if tr:
            groups[tr].append(r)

    memories = []
    for tr, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < min_count:
            continue
        mem = build_memory_unit(
            group, global_prior, "target_relation", tr, "target_relation",
            extra_fields={
                "relation_description":       TARGET_RELATION_DESCRIPTIONS.get(tr, tr),
                "semantic_propagation_distribution": generic_dist(group, "llm_semantic_propagation"),
                "reply_relation_distribution": generic_dist(group, "llm_reply_relation"),
            }
        )
        memories.append(mem)
    return memories


# ============ 组合模式记忆库 ============

def build_composite_pattern_memory(
    usable: list,
    global_prior: dict,
    min_count: int,
) -> list:
    groups = defaultdict(list)
    for r in usable:
        for pat in r.get("_norm_patterns", []):
            reply = safe_str(r.get("llm_reply_relation"))
            prop  = safe_str(r.get("llm_semantic_propagation"))
            tr    = safe_str(r.get("llm_target_relation"))
            key   = f"{pat}+{reply}+{prop}+{tr}"
            groups[key].append(r)

    memories = []
    for key, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(group) < min_count:
            continue
        parts = key.split("+")
        pat, reply, prop, tr = parts if len(parts) == 4 else (key, "", "", "")
        raw_dist, w_dist = compute_label_distribution(group)
        lifts = compute_label_lift(w_dist, global_prior)
        total_weight = sum(safe_float(r.get("memory_label_weight")) for r in group)
        avg_conf = round(
            sum(safe_float(r.get("llm_confidence")) for r in group) / max(len(group), 1), 4
        )
        dominant = max(w_dist, key=w_dist.get) if w_dist else "none"
        dom_prob = w_dist.get(dominant, 0.0)
        reliability, usage = compute_reliability(total_weight, dom_prob, avg_conf)

        mem = {
            "memory_id":             f"composite_{key.replace('+', '_')}",
            "memory_type":           "composite_pattern",
            "composite_key":         key,
            "candidate_pattern":     pat,
            "reply_relation":        reply,
            "semantic_propagation":  prop,
            "target_relation":       tr,
            "source_sample_count_raw":     sum(raw_dist.values()),
            "source_sample_count_weighted": round(total_weight, 2),
            "label_distribution_raw":      raw_dist,
            "label_distribution_weighted": w_dist,
            "dominant_label":              dominant,
            "dominant_label_probability":  dom_prob,
            "average_confidence":          avg_conf,
            "reliability_level":           reliability,
            "suggested_usage":             usage,
            "typical_examples":            select_typical_examples(group, dominant),
            "counter_examples":            select_counter_examples(group, dominant),
            "applicable_conditions":       APPLICABLE_CONDITIONS.get(pat, ["无特定条件"]),
            "cautions":                    CAUTIONS.get(pat, "请结合上下文谨慎使用。"),
            **lifts,
        }
        memories.append(mem)
    return memories


# ============ RAG 文档生成 ============

def build_rag_document(mem: dict) -> dict:
    mtype    = mem.get("memory_type", "unknown")
    pname    = (mem.get("pattern_name") or mem.get("semantic_propagation")
                or mem.get("target_relation") or mem.get("composite_key") or "unknown")
    dominant = mem.get("dominant_label", "none")
    dist_w   = mem.get("label_distribution_weighted", {})
    dist_str = "，".join(
        f"{STANCE_ZH.get(lbl, lbl)} {round(v*100, 1)}%"
        for lbl, v in sorted(dist_w.items(), key=lambda x: -x[1])
    )
    reliability = mem.get("reliability_level", "low")
    usage       = mem.get("suggested_usage", "retrieval_only")

    desc = (mem.get("pattern_description") or mem.get("propagation_description")
            or mem.get("relation_description") or "")
    conds = mem.get("applicable_conditions", [])
    conds_str = "\n".join(f"- {c}" for c in conds)
    caution   = mem.get("cautions", "")

    examples = mem.get("typical_examples", [])
    ex_lines = []
    for i, ex in enumerate(examples[:3], 1):
        pt = f"\n   父评论：{str(ex.get('parent_text',''))[:80]}" if ex.get("parent_text") else ""
        ex_lines.append(
            f"{i}. 当前评论：{str(ex.get('current_text',''))[:100]}{pt}\n"
            f"   立场：{STANCE_ZH.get(ex.get('stance_label',''),'未知')}  "
            f"置信度：{ex.get('llm_confidence','')}"
        )
    ex_str = "\n".join(ex_lines) if ex_lines else "（暂无典型例子）"

    retrieval_text = f"""Pattern: {pname}

【Memory 类型】
{mtype}

【描述】
{desc}

【立场倾向（LLM-Silver，非人工 gold label）】
主导标签：{STANCE_ZH.get(dominant, dominant)}
加权立场分布：{dist_str}
数据质量：{reliability}（AI-Silver 伪标签，仅供参考）

【适用条件】
{conds_str}

【注意事项】
{caution}

【典型例子】
{ex_str}

【Lift 指标】
支持 lift={mem.get('favor_lift', 'N/A')}，反对 lift={mem.get('against_lift', 'N/A')}，中立 lift={mem.get('none_lift', 'N/A')}

【数据说明】
本文档基于 LLM-Silver 标注生成，非人工 gold label。建议用于检索参考，不直接用于标签决策。
"""

    doc_id = f"rag_{mtype}_{pname.replace('+', '_').replace(' ', '_')}"

    return {
        "doc_id":         doc_id,
        "doc_type":       "stance_pattern_memory",
        "memory_type":    mtype,
        "pattern_name":   pname,
        "retrieval_text": retrieval_text.strip(),
        "structured_metadata": {
            "memory_id":                      mem.get("memory_id"),
            "memory_type":                    mtype,
            "pattern_name":                   pname,
            "dominant_label":                 dominant,
            "dominant_label_probability":     mem.get("dominant_label_probability"),
            "label_distribution_weighted":    dist_w,
            "reliability_level":              reliability,
            "suggested_usage":                usage,
            "source_sample_count_raw":        mem.get("source_sample_count_raw"),
            "source_sample_count_weighted":   mem.get("source_sample_count_weighted"),
            "average_confidence":             mem.get("average_confidence"),
            "favor_lift":                     mem.get("favor_lift"),
            "against_lift":                   mem.get("against_lift"),
            "none_lift":                      mem.get("none_lift"),
            "evidence_source":                "llm_silver",
            "caution": "LLM-Silver pseudo-label，非人工 gold label，请谨慎使用。",
        },
    }


def build_rag_documents(
    lang_mems: list,
    prop_mems: list,
    trel_mems: list,
    comp_mems: list,
) -> list:
    docs = []
    for mem_list in (lang_mems, prop_mems, trel_mems, comp_mems):
        for mem in mem_list:
            if mem.get("suggested_usage") != "do_not_use":
                docs.append(build_rag_document(mem))
    return docs


# ============ 报告 ============

def generate_reports(
    all_records: list,
    normalized: list,
    usable: list,
    excluded: list,
    lang_mems: list,
    prop_mems: list,
    trel_mems: list,
    comp_mems: list,
    rag_docs: list,
    global_prior_raw: dict,
    global_prior_weighted: dict,
    warnings: list,
) -> tuple:
    def quality_count(level):
        return sum(1 for r in normalized if r.get("memory_quality_level") == level)

    pattern_rel_dist = Counter()
    usage_dist = Counter()
    for m in lang_mems + prop_mems + trel_mems + comp_mems:
        pattern_rel_dist[m.get("reliability_level", "unknown")] += 1
        usage_dist[m.get("suggested_usage", "unknown")] += 1

    top_lang = sorted(lang_mems, key=lambda m: -m.get("source_sample_count_weighted", 0))[:10]
    top_comp = sorted(comp_mems, key=lambda m: -m.get("source_sample_count_weighted", 0))[:10]
    top_favor_lift  = sorted(lang_mems, key=lambda m: -m.get("favor_lift", 0))[:5]
    top_against_lift = sorted(lang_mems, key=lambda m: -m.get("against_lift", 0))[:5]
    top_none_lift   = sorted(lang_mems, key=lambda m: -m.get("none_lift", 0))[:5]

    mem_report = {
        "data_note":                "LLM-Silver 数据，非人工 gold label",
        "total_input_instances":    len(all_records),
        "normalized_instances":     len(normalized),
        "use_for_memory_count":     len(usable),
        "excluded_count":           len(excluded),
        "strong_memory_count":      quality_count("strong_memory"),
        "medium_memory_count":      quality_count("medium_memory"),
        "weak_memory_count":        quality_count("weak_memory"),
        "language_pattern_count":   len(lang_mems),
        "semantic_propagation_memory_count": len(prop_mems),
        "target_relation_memory_count":  len(trel_mems),
        "composite_pattern_memory_count": len(comp_mems),
        "rag_document_count":       len(rag_docs),
        "global_label_prior_raw":   global_prior_raw,
        "global_label_prior_weighted": global_prior_weighted,
        "pattern_reliability_distribution": dict(pattern_rel_dist),
        "suggested_usage_distribution":     dict(usage_dist),
        "top_language_patterns_by_weighted_count": [
            {"pattern": m["pattern_name"],
             "weighted_count": m["source_sample_count_weighted"],
             "dominant_label": m["dominant_label"]}
            for m in top_lang
        ],
        "top_composite_patterns_by_weighted_count": [
            {"composite_key": m["composite_key"],
             "weighted_count": m["source_sample_count_weighted"],
             "dominant_label": m["dominant_label"]}
            for m in top_comp
        ],
        "top_patterns_by_favor_lift":   [
            {"pattern": m.get("pattern_name", m.get("composite_key")),
             "favor_lift": m.get("favor_lift")} for m in top_favor_lift
        ],
        "top_patterns_by_against_lift": [
            {"pattern": m.get("pattern_name", m.get("composite_key")),
             "against_lift": m.get("against_lift")} for m in top_against_lift
        ],
        "top_patterns_by_none_lift":    [
            {"pattern": m.get("pattern_name", m.get("composite_key")),
             "none_lift": m.get("none_lift")} for m in top_none_lift
        ],
        "warnings": warnings,
    }

    lengths = [len(d.get("retrieval_text", "")) for d in rag_docs]
    rag_report = {
        "total_documents":    len(rag_docs),
        "documents_by_memory_type": dict(Counter(d.get("memory_type") for d in rag_docs)),
        "documents_by_reliability": dict(Counter(
            d["structured_metadata"].get("reliability_level") for d in rag_docs
        )),
        "documents_by_suggested_usage": dict(Counter(
            d["structured_metadata"].get("suggested_usage") for d in rag_docs
        )),
        "average_retrieval_text_length": round(sum(lengths) / max(len(lengths), 1)),
        "max_retrieval_text_length":     max(lengths, default=0),
        "min_retrieval_text_length":     min(lengths, default=0),
    }

    return mem_report, rag_report


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="构建 LLM-Silver Pattern Memory 与 RAG 知识库")
    parser.add_argument("--input",              required=True)
    parser.add_argument("--output_dir",         required=True)
    parser.add_argument("--min_confidence",     type=float, default=0.70)
    parser.add_argument("--min_pattern_count",  type=int,   default=10)
    parser.add_argument("--train_only",         default="true")
    args = parser.parse_args()

    train_only = args.train_only.lower() == "true"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据
    print(f"📂 读取：{args.input}")
    all_records = load_jsonl(args.input)
    print(f"  共加载 {len(all_records)} 条")

    warnings = []

    # 2. 规范化
    print("\n🔄 规范化字段...")
    normalized = []
    for rec in all_records:
        r = dict(rec)
        r["_norm_patterns"]        = normalize_candidate_patterns(r.get("llm_candidate_patterns"))
        r["_stance_norm"]          = normalize_label(r.get("stance_label", "none"))
        r["_stance_zh_norm"]       = STANCE_ZH.get(r["_stance_norm"], "中立")
        r["_llm_inferred"]         = safe_str(r.get("llm_inferred_stance_to_main_target", "unclear"))
        r["_llm_label_agreement"]  = compute_llm_label_agreement(
            r["_stance_norm"], r["_llm_inferred"]
        )
        r["label_for_memory"]      = r["_stance_norm"]

        # 质量
        quality = compute_memory_quality(r, args.min_confidence, train_only)
        r.update(quality)
        normalized.append(r)

    # 3. 过滤
    usable, excluded = filter_memory_instances(normalized)
    print(f"  可用：{len(usable)}，排除：{len(excluded)}")

    # valid/test 检查
    non_train = [r for r in usable if r.get("split") != "train"]
    if non_train and train_only:
        warnings.append(f"⚠️  发现 {len(non_train)} 条非 train 样本被纳入可用集，请检查过滤逻辑")

    # 4. 全局先验
    raw_prior_counter = Counter(r["_stance_norm"] for r in usable)
    total_raw = sum(raw_prior_counter.values()) or 1
    global_prior_raw = {lbl: cnt / total_raw for lbl, cnt in raw_prior_counter.items()}

    weighted_prior = defaultdict(float)
    for r in usable:
        weighted_prior[r["_stance_norm"]] += safe_float(r.get("memory_label_weight"))
    total_w = sum(weighted_prior.values()) or 1
    global_prior_weighted = {lbl: v / total_w for lbl, v in weighted_prior.items()}

    # 5. 构建各类 memory
    print("\n🧩 构建 language_pattern_memory...")
    lang_mems = build_language_pattern_memory(usable, global_prior_weighted, args.min_pattern_count)

    print("🧩 构建 semantic_propagation_memory...")
    prop_mems = build_semantic_propagation_memory(usable, global_prior_weighted, args.min_pattern_count)

    print("🧩 构建 target_relation_memory...")
    trel_mems = build_target_relation_memory(usable, global_prior_weighted, args.min_pattern_count)

    print("🧩 构建 composite_pattern_memory...")
    comp_mems = build_composite_pattern_memory(usable, global_prior_weighted, args.min_pattern_count)

    # 6. RAG 文档
    print("📄 生成 RAG 知识文档...")
    rag_docs = build_rag_documents(lang_mems, prop_mems, trel_mems, comp_mems)

    # 7. 警告检查
    unknown_cnt = sum(1 for r in usable if r.get("_norm_patterns") == ["unknown_pattern"])
    if unknown_cnt > len(usable) * 0.1:
        warnings.append(f"⚠️  unknown_pattern 占可用样本 {unknown_cnt/max(len(usable),1):.1%}，pattern 覆盖不足")
    small_mems = [m for m in lang_mems if m["source_sample_count_raw"] < 20]
    if len(small_mems) > len(lang_mems) * 0.5:
        warnings.append(f"⚠️  超过半数 language pattern 样本数 < 20，模式过碎")
    topic_dist = Counter(r.get("topic") for r in usable)
    if topic_dist:
        top_topic, top_cnt = topic_dist.most_common(1)[0]
        if top_cnt / max(len(usable), 1) > 0.5:
            warnings.append(f"⚠️  话题 '{top_topic}' 占比 {top_cnt/len(usable):.1%}，可能支配知识库")

    # 8. 报告
    mem_report, rag_report = generate_reports(
        all_records, normalized, usable, excluded,
        lang_mems, prop_mems, trel_mems, comp_mems,
        rag_docs, global_prior_raw, global_prior_weighted, warnings
    )

    # 9. 写出
    print(f"\n📝 写出文件到 {output_dir}/")
    write_jsonl(normalized,  output_dir / "normalized_llm_semantic_instances.jsonl")
    write_jsonl(excluded,    output_dir / "excluded_low_quality_instances.jsonl")
    write_jsonl(lang_mems,   output_dir / "language_pattern_memory.jsonl")
    write_jsonl(prop_mems,   output_dir / "semantic_propagation_memory.jsonl")
    write_jsonl(trel_mems,   output_dir / "target_relation_memory.jsonl")
    write_jsonl(comp_mems,   output_dir / "composite_pattern_memory.jsonl")
    write_jsonl(rag_docs,    output_dir / "rag_knowledge_documents_llm.jsonl")
    write_json(mem_report,   output_dir / "pattern_memory_report_llm.json")
    write_json(rag_report,   output_dir / "rag_document_report_llm.json")

    print(f"""
📊 构建摘要：
  总输入：          {len(all_records)}
  可用样本：        {len(usable)}
  排除样本：        {len(excluded)}
  language pattern：{len(lang_mems)}
  semantic prop：   {len(prop_mems)}
  target relation：{len(trel_mems)}
  composite：       {len(comp_mems)}
  RAG 文档：        {len(rag_docs)}
  全局先验（加权）：{dict([(k, round(v,3)) for k,v in global_prior_weighted.items()])}
    """)
    if warnings:
        print("⚠️  警告：")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()