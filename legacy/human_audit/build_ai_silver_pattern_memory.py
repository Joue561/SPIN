#!/usr/bin/env python3
"""
build_ai_silver_pattern_memory.py

用法：
python build_ai_silver_pattern_memory.py \
  --stance_instances stance_instances.jsonl \
  --ai_preannotated ai_preannotated_instances.csv \
  --output_dir ./ai_silver_output \
  --min_pattern_count 10
"""

import json
import csv
import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional
import warnings


# ============ 常量 ============

MAIN_TARGET = "高市早苗发表引发网络争议的涉台言论"
EVIDENCE_SOURCE = "ai_silver"

EXCLUDED_LABELS = {"目标不清", "上下文不足", "无关", "无法判断"}

LABEL_NORMALIZE_MAP = {
    "支持": "支持", "favor": "支持", "support": "支持",
    "反对": "反对", "against": "反对",
    "中立": "真中立", "true_neutral": "真中立", "neutral": "真中立",
    "真中立": "真中立",
    "无关": "无关", "irrelevant": "无关",
    "目标不清": "目标不清", "unclear_target": "目标不清",
    "上下文不足": "上下文不足", "insufficient_context": "上下文不足",
    "混合/条件立场": "混合/条件立场", "mixed": "混合/条件立场",
    "无法判断": "无法判断", "undecidable": "无法判断",
}

CORE_LABELS = {"支持", "反对", "真中立"}


# ============ IO 工具 ============

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
                print(f"  ⚠️  JSONL 第 {i+1} 行解析失败，已跳过：{e}")
    return records


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
                print(f"  ⚠️  CSV 第 {i+1} 行读取失败，已跳过：{e}")
    return records


def write_jsonl(records: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath}（{len(records)} 条）")


def write_report(data: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath}")


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


# ============ 标签规范化 ============

def normalize_label(label: str) -> str:
    if not label:
        return "无法判断"
    label = str(label).strip()
    return LABEL_NORMALIZE_MAP.get(label, label)


def normalize_raw_stance(stance_raw: str) -> str:
    """原始 stance_label_raw 规范化：中立 → 真中立"""
    if not stance_raw:
        return "无法判断"
    m = {"支持": "支持", "反对": "反对", "中立": "真中立"}
    return m.get(str(stance_raw).strip(), str(stance_raw).strip())


# ============ 合并 AI 预标注 ============

def merge_ai_annotations(
    stance_records: list,
    ai_records: list,
) -> list:
    ai_map = {}
    for r in ai_records:
        iid = str(r.get("instance_id", "")).strip()
        if iid:
            ai_map[iid] = r

    merged = []
    for rec in stance_records:
        iid = str(rec.get("instance_id", "")).strip()
        ai = ai_map.get(iid, {})
        rec["_ai"] = ai
        merged.append(rec)

    print(f"  合并：stance={len(stance_records)}，ai_map={len(ai_map)}，"
          f"命中={sum(1 for r in merged if r['_ai'])}")
    return merged


# ============ Silver 标签生成 ============

def assign_silver_label(rec: dict) -> dict:
    ai = rec.get("_ai", {})
    stance_raw = rec.get("stance_label_raw", "") or ""
    current_text = rec.get("current_text", "") or ""
    audit_priority = rec.get("audit_priority", "low") or "low"
    neutral_suspicious = to_bool(rec.get("neutral_suspicious", False))

    norm_raw = normalize_raw_stance(stance_raw)

    # AI 字段
    ai_recheck_label     = normalize_label(ai.get("ai_recheck_label", ""))
    ai_stance_to_main    = normalize_label(ai.get("ai_stance_to_main_target", ""))
    ai_confidence        = safe_float(ai.get("ai_recheck_confidence", 0.0))
    ai_correct           = str(ai.get("ai_original_label_correct", "")).strip().lower()
    ai_abstract_pattern  = str(ai.get("ai_abstract_pattern", "")).strip()

    has_ai = bool(ai)

    # ---- 空文本直接排除 ----
    if not current_text.strip():
        return _make_silver(rec, "无法判断", "uncertain_excluded",
                            0.0, 0.0, "excluded_uncertain")

    # ---- 有 AI 标注 ----
    if has_ai:
        # 排除类标签
        if ai_recheck_label in EXCLUDED_LABELS or ai_confidence < 0.65:
            return _make_silver(rec, ai_recheck_label or "无法判断",
                                "uncertain_excluded", ai_confidence, 0.0,
                                "excluded_uncertain")

        # AI 与原标签语义一致
        labels_consistent = (ai_recheck_label == norm_raw or
                              ai_stance_to_main == norm_raw)

        if labels_consistent and ai_correct == "true":
            if ai_confidence >= 0.85:
                w = 1.0
            elif ai_confidence >= 0.75:
                w = 0.8
            else:
                w = 0.6
            return _make_silver(rec, norm_raw,
                                "ai_recheck_consistent", ai_confidence, w,
                                "strong_silver" if ai_confidence >= 0.85 else "weak_silver")

        # AI 修正原标签
        if ai_correct == "false" and ai_recheck_label not in EXCLUDED_LABELS:
            corrected_label = ai_recheck_label if ai_recheck_label else ai_stance_to_main
            if ai_confidence >= 0.85:
                w = 0.7
                quality = "corrected_silver"
            elif ai_confidence >= 0.70:
                w = 0.5
                quality = "corrected_silver"
            else:
                w = 0.0
                quality = "excluded_uncertain"
            src = "ai_corrected_silver" if quality != "excluded_uncertain" else "uncertain_excluded"
            return _make_silver(rec, corrected_label, src, ai_confidence, w, quality)

        # uncertain：AI 不确定
        if ai_confidence >= 0.75:
            w = 0.6
            return _make_silver(rec, ai_recheck_label or norm_raw,
                                "ai_recheck_consistent", ai_confidence, w, "weak_silver")
        else:
            return _make_silver(rec, ai_recheck_label or norm_raw,
                                "uncertain_excluded", ai_confidence, 0.0,
                                "excluded_uncertain")

    # ---- 无 AI 标注，仅凭原标签 + audit flags ----
    if audit_priority == "low" and not neutral_suspicious:
        w = 0.7
        quality = "strong_silver"
    elif audit_priority == "medium":
        w = 0.4
        quality = "weak_silver"
    else:
        w = 0.2
        quality = "weak_silver"

    return _make_silver(rec, norm_raw, "raw_model_label",
                        0.5, w, quality)


def _make_silver(rec, label, source, confidence, weight, quality) -> dict:
    rec["final_silver_label"] = label
    rec["label_source"]       = source
    rec["label_confidence"]   = round(confidence, 4)
    rec["label_weight"]       = round(weight, 4)
    rec["silver_quality_level"] = quality
    rec["_is_excluded"] = (quality == "excluded_uncertain")
    return rec


# ============ Pattern 推断 ============

def infer_final_pattern(rec: dict) -> dict:
    ai = rec.get("_ai", {})
    ai_pattern = str(ai.get("ai_abstract_pattern", "")).strip()
    cand_pattern = rec.get("candidate_pattern", "") or ""

    # candidate_pattern 可能是 "|" 分隔字符串
    if isinstance(cand_pattern, list):
        cand_list = cand_pattern
    elif "|" in str(cand_pattern):
        cand_list = [p.strip() for p in str(cand_pattern).split("|") if p.strip()]
    else:
        cand_list = [str(cand_pattern).strip()] if cand_pattern else []

    if ai_pattern and ai_pattern not in ("", "null", "None"):
        rec["final_pattern"]   = ai_pattern
        rec["primary_pattern"] = ai_pattern
        rec["pattern_source"]  = "ai_abstract_pattern"
    elif cand_list:
        rec["final_pattern"]   = cand_list
        rec["primary_pattern"] = cand_list[0]
        rec["pattern_source"]  = "candidate_pattern"
    else:
        rec["final_pattern"]   = "unknown_pattern"
        rec["primary_pattern"] = "unknown_pattern"
        rec["pattern_source"]  = "unknown"

    return rec


# ============ Pattern Memory 聚合 ============

def build_pattern_memory(
    silver_records: list,
    min_pattern_count: int = 10,
) -> tuple:
    # 全局标签先验（只用未排除样本）
    included = [r for r in silver_records if not r.get("_is_excluded")]
    global_label_counts: Counter = Counter()
    for r in included:
        lbl = r.get("final_silver_label", "无法判断")
        if lbl in CORE_LABELS:
            global_label_counts[lbl] += 1
    global_total = sum(global_label_counts.values()) or 1
    global_prior = {lbl: cnt / global_total
                    for lbl, cnt in global_label_counts.items()}

    # 按 primary_pattern 分组
    pattern_groups: dict = defaultdict(list)
    for r in silver_records:
        pp = r.get("primary_pattern", "unknown_pattern") or "unknown_pattern"
        pattern_groups[pp].append(r)

    memories = []
    pattern_report = []

    for pattern_name, group in sorted(
            pattern_groups.items(),
            key=lambda x: -len(x[1])):

        raw_count = len(group)
        included_group = [r for r in group if not r.get("_is_excluded")]

        # 原始标签分布
        label_dist_raw: Counter = Counter()
        for r in group:
            lbl = r.get("final_silver_label", "无法判断")
            label_dist_raw[lbl] += 1

        # 加权标签分布
        label_dist_weighted: dict = defaultdict(float)
        total_weight = 0.0
        for r in included_group:
            lbl = r.get("final_silver_label", "无法判断")
            w = safe_float(r.get("label_weight", 0.0))
            label_dist_weighted[lbl] += w
            total_weight += w

        weighted_count = round(total_weight, 2)
        avg_weight = round(total_weight / len(included_group), 4) \
            if included_group else 0.0
        avg_confidence = round(
            sum(safe_float(r.get("label_confidence", 0)) for r in included_group)
            / len(included_group), 4) if included_group else 0.0

        # 归一化加权分布
        label_dist_w_norm: dict = {}
        if total_weight > 0:
            label_dist_w_norm = {
                lbl: round(w / total_weight, 4)
                for lbl, w in label_dist_weighted.items()
            }

        # dominant label
        dominant_raw = label_dist_raw.most_common(1)[0][0] \
            if label_dist_raw else "无法判断"
        dominant_weighted = max(label_dist_w_norm, key=label_dist_w_norm.get) \
            if label_dist_w_norm else "无法判断"
        dominant_w_ratio = label_dist_w_norm.get(dominant_weighted, 0.0)

        # label lift
        def lift(label):
            p_label_given_pattern = label_dist_w_norm.get(label, 0.0)
            p_label = global_prior.get(label, 1e-6)
            return round(p_label_given_pattern / p_label, 4)

        favor_lift   = lift("支持")
        against_lift = lift("反对")
        neutral_lift = lift("真中立")

        # reliability_level
        mostly_excluded = (len(included_group) == 0 or
                           pattern_name == "unknown_pattern")
        if mostly_excluded:
            reliability = "excluded"
        elif weighted_count >= 100 and dominant_w_ratio >= 0.75 and avg_weight >= 0.75:
            reliability = "high"
        elif weighted_count >= 30 and dominant_w_ratio >= 0.60 and avg_weight >= 0.50:
            reliability = "medium"
        else:
            reliability = "low"

        # suggested_usage
        usage_map = {
            "high":     "use_as_strong_pattern",
            "medium":   "use_as_weak_pattern",
            "excluded": "do_not_use",
        }
        if reliability == "low":
            suggested_usage = "retrieval_only" if raw_count >= 30 else "needs_review"
        elif pattern_name == "unknown_pattern":
            suggested_usage = "needs_review"
        else:
            suggested_usage = usage_map.get(reliability, "needs_review")

        # 典型例子
        sortable = sorted(
            [r for r in included_group if r.get("current_text")],
            key=lambda r: (
                -safe_float(r.get("label_weight", 0)),
                -safe_float(r.get("label_confidence", 0)),
            )
        )
        typical_examples = []
        for r in sortable[:5]:
            typical_examples.append({
                "instance_id":       r.get("instance_id"),
                "current_text":      (r.get("current_text") or "")[:200],
                "parent_text":       (r.get("parent_text") or "")[:200],
                "root_text":         (r.get("root_text") or "")[:200],
                "final_silver_label": r.get("final_silver_label"),
                "label_confidence":  r.get("label_confidence"),
                "ai_reason":         (r.get("_ai") or {}).get("ai_reason", ""),
            })

        # 反例
        counter_examples = []
        for r in sorted(
            [r for r in included_group
             if r.get("final_silver_label") != dominant_weighted
             and r.get("current_text")],
            key=lambda r: -safe_float(r.get("label_weight", 0))
        )[:3]:
            counter_examples.append({
                "instance_id":       r.get("instance_id"),
                "current_text":      (r.get("current_text") or "")[:200],
                "final_silver_label": r.get("final_silver_label"),
                "label_confidence":  r.get("label_confidence"),
            })

        # pattern description
        pattern_description = _make_pattern_description(pattern_name)
        applicable_conditions = _make_applicable_conditions(pattern_name)
        caution = _make_caution(pattern_name)

        memory = {
            "pattern_id":                   f"pat_{pattern_name}",
            "pattern_name":                 pattern_name,
            "pattern_description":          pattern_description,
            "source_sample_count_raw":      raw_count,
            "source_sample_count_weighted": weighted_count,
            "label_distribution_raw":       dict(label_dist_raw),
            "label_distribution_weighted":  label_dist_w_norm,
            "label_lift_favor":             favor_lift,
            "label_lift_against":           against_lift,
            "label_lift_neutral":           neutral_lift,
            "dominant_label_raw":           dominant_raw,
            "dominant_label_weighted":      dominant_weighted,
            "dominant_label_weighted_ratio": dominant_w_ratio,
            "average_confidence":           avg_confidence,
            "average_weight":               avg_weight,
            "reliability_level":            reliability,
            "suggested_usage":              suggested_usage,
            "typical_examples":             typical_examples,
            "counter_examples":             counter_examples,
            "applicable_conditions":        applicable_conditions,
            "caution":                      caution,
            "evidence_source":              EVIDENCE_SOURCE,
        }
        memories.append(memory)

        pattern_report.append({
            "pattern_name":                 pattern_name,
            "raw_count":                    raw_count,
            "weighted_count":               weighted_count,
            "dominant_label":               dominant_weighted,
            "label_distribution_weighted":  label_dist_w_norm,
            "average_confidence":           avg_confidence,
            "average_weight":               avg_weight,
            "reliability_level":            reliability,
            "suggested_usage":              suggested_usage,
            "favor_lift":                   favor_lift,
            "against_lift":                 against_lift,
            "neutral_lift":                 neutral_lift,
        })

    # 按 weighted_count 排序
    memories.sort(key=lambda x: -x["source_sample_count_weighted"])
    pattern_report.sort(key=lambda x: -x["weighted_count"])

    return memories, pattern_report


def _make_pattern_description(pattern_name: str) -> str:
    descs = {
        "agreement_with_parent":
            "当前评论表达对父评论的赞同，例如'确实'、'说得对'、'没错'等短句，立场通常继承父评论。",
        "disagreement_with_parent":
            "当前评论对父评论提出反驳或质疑，立场通常与父评论相反。",
        "negative_judgment_toward_TARGET":
            "当前评论对主 target 表达直接的负面判断或批评。",
        "negative_judgment_toward_supporters_of_TARGET":
            "当前评论批评支持主 target 的群体，间接表达对主 target 的反对立场。",
        "rhetorical_question_against_TARGET":
            "当前评论以反问句式表达对主 target 的质疑或反对。",
        "sarcasm_or_mockery":
            "当前评论使用讽刺、嘲讽语气，表面上可能是中立，实际含有明显立场。",
        "support_countermeasure_against_TARGET":
            "当前评论支持针对主 target 相关方采取的反制措施，间接表达反对立场。",
        "criticize_Japan_or_Takaichi":
            "当前评论直接批评日本或高市早苗，通常对应反对主 target 的立场。",
        "discuss_event_consequence":
            "当前评论讨论事件的影响或后果，立场分布较分散。",
        "true_neutral_information":
            "当前评论仅陈述事实或提供信息，不含明显立场倾向。",
        "irrelevant_comment":
            "当前评论与主 target 无关。",
        "unclear_or_context_dependent":
            "当前评论立场不明确，或高度依赖上下文。",
        "unknown_pattern":
            "无法识别语言结构 pattern。",
    }
    return descs.get(pattern_name, f"语言结构：{pattern_name}")


def _make_applicable_conditions(pattern_name: str) -> list:
    conds = {
        "agreement_with_parent": [
            "depth = 2",
            "当前评论为简短赞同表达",
            "父评论立场明确",
            "当前评论未引入新 target",
        ],
        "rhetorical_question_against_TARGET": [
            "当前评论包含'？'或反问词",
            "评论针对主 target 相关内容",
        ],
        "sarcasm_or_mockery": [
            "包含讽刺性词汇或语气",
            "表面中立但含情绪色彩",
        ],
    }
    return conds.get(pattern_name, ["无特定条件限制"])


def _make_caution(pattern_name: str) -> str:
    cautions = {
        "agreement_with_parent":
            "父评论立场不明确时不适用；当前评论为反讽赞同时需人工判断。",
        "rhetorical_question_against_TARGET":
            "反问有时指向父评论或其他 target，需结合 root_text 判断。",
        "sarcasm_or_mockery":
            "讽刺识别难度较高，建议人工二次确认。",
        "unclear_or_context_dependent":
            "该 pattern 下样本立场混乱，不建议直接用于标签推断。",
        "unknown_pattern":
            "无法识别 pattern，不建议进入强知识库。",
    }
    return cautions.get(pattern_name,
                         "请结合具体样本和上下文谨慎使用 AI-silver 标签。")


# ============ RAG 文档生成 ============

def build_rag_documents(memories: list) -> list:
    docs = []
    for i, mem in enumerate(memories):
        if mem["suggested_usage"] == "do_not_use":
            continue

        pname = mem["pattern_name"]
        dist_w = mem["label_distribution_weighted"]
        dominant = mem["dominant_label_weighted"]
        reliability = mem["reliability_level"]
        examples = mem["typical_examples"]

        # 构造 retrieval_text
        dist_str = "，".join(
            f"{lbl} {round(v*100,1)}%"
            for lbl, v in sorted(dist_w.items(), key=lambda x: -x[1])
        )

        example_lines = []
        for j, ex in enumerate(examples[:3], 1):
            pt = f"\n   父评论：{ex['parent_text'][:80]}" if ex.get("parent_text") else ""
            example_lines.append(
                f"{j}. 当前评论：{ex['current_text'][:100]}{pt}\n"
                f"   标签：{ex['final_silver_label']}  "
                f"置信度：{ex.get('label_confidence', '')}"
            )
        examples_str = "\n".join(example_lines) if example_lines else "（暂无典型例子）"

        counter_lines = []
        for ex in mem.get("counter_examples", [])[:2]:
            counter_lines.append(
                f"- 当前评论：{ex['current_text'][:80]}  "
                f"标签：{ex['final_silver_label']}"
            )
        counter_str = "\n".join(counter_lines) if counter_lines else "（暂无反例）"

        conds_str = "\n".join(f"- {c}" for c in mem["applicable_conditions"])

        retrieval_text = f"""Pattern: {pname}

【Pattern 描述】
{mem['pattern_description']}

【立场倾向（AI-silver 数据，非人工 gold label）】
主导标签：{dominant}
加权标签分布：{dist_str}
数据质量：{reliability}（来源：AI-silver pseudo-label，仅供参考）

【适用条件】
{conds_str}

【注意事项】
{mem['caution']}

【典型例子】
{examples_str}

【反例】
{counter_str}

【Lift 指标】
支持 lift={mem['label_lift_favor']}，反对 lift={mem['label_lift_against']}，真中立 lift={mem['label_lift_neutral']}

【数据说明】
本文档基于 AI-silver 伪标签生成，非人工标注真值。建议用于检索参考和 stance hint，而非直接标签决策。
"""

        doc = {
            "doc_id":    f"rag_pat_{i+1:04d}_{pname}",
            "doc_type":  "stance_pattern_memory",
            "pattern_name": pname,
            "retrieval_text": retrieval_text.strip(),
            "structured_metadata": {
                "pattern_name":                pname,
                "dominant_label":              dominant,
                "label_distribution_weighted": dist_w,
                "reliability_level":           reliability,
                "suggested_usage":             mem["suggested_usage"],
                "source_sample_count_raw":     mem["source_sample_count_raw"],
                "source_sample_count_weighted": mem["source_sample_count_weighted"],
                "label_lift_favor":            mem["label_lift_favor"],
                "label_lift_against":          mem["label_lift_against"],
                "label_lift_neutral":          mem["label_lift_neutral"],
                "evidence_source":             EVIDENCE_SOURCE,
                "caution": "AI-silver pseudo-label，非人工 gold label，请谨慎使用。",
            },
        }
        docs.append(doc)
    return docs


# ============ 质量报告 ============

def build_quality_report(
    all_records: list,
    silver_records: list,
    memories: list,
    pattern_report: list,
) -> dict:
    excluded = [r for r in silver_records if r.get("_is_excluded")]
    included = [r for r in silver_records if not r.get("_is_excluded")]

    raw_dist = Counter(
        normalize_raw_stance(r.get("stance_label_raw", "")) for r in all_records
    )
    silver_dist = Counter(r.get("final_silver_label") for r in silver_records)
    source_dist = Counter(r.get("label_source") for r in silver_records)
    quality_dist = Counter(r.get("silver_quality_level") for r in silver_records)

    avg_conf = round(
        sum(safe_float(r.get("label_confidence", 0)) for r in included)
        / len(included), 4) if included else 0.0
    avg_weight = round(
        sum(safe_float(r.get("label_weight", 0)) for r in included)
        / len(included), 4) if included else 0.0

    neutral_raw = [r for r in all_records
                   if normalize_raw_stance(r.get("stance_label_raw", "")) == "真中立"]
    neutral_to_against = sum(
        1 for r in silver_records
        if normalize_raw_stance(r.get("stance_label_raw", "")) == "真中立"
        and r.get("final_silver_label") == "反对"
    )
    neutral_to_favor = sum(
        1 for r in silver_records
        if normalize_raw_stance(r.get("stance_label_raw", "")) == "真中立"
        and r.get("final_silver_label") == "支持"
    )

    favor_raw = [r for r in all_records
                 if normalize_raw_stance(r.get("stance_label_raw", "")) == "支持"]
    favor_kept = sum(
        1 for r in silver_records
        if normalize_raw_stance(r.get("stance_label_raw", "")) == "支持"
        and r.get("final_silver_label") == "支持"
    )

    usable_patterns = [m for m in memories
                       if m["suggested_usage"] not in ("do_not_use", "needs_review")]
    high_r = [m for m in memories if m["reliability_level"] == "high"]
    med_r  = [m for m in memories if m["reliability_level"] == "medium"]
    low_r  = [m for m in memories if m["reliability_level"] == "low"]

    # warnings
    warn = []
    if neutral_to_against > len(neutral_raw) * 0.3:
        warn.append(f"⚠️  {neutral_to_against} 条原始中立被 AI 修正为反对"
                    f"（占原始中立 {len(neutral_raw)} 的 "
                    f"{neutral_to_against/max(len(neutral_raw),1):.1%}），请人工抽查。")
    if silver_dist.get("支持", 0) < 100:
        warn.append(f"⚠️  支持类样本仅 {silver_dist.get('支持',0)} 条，仍严重不足。")
    unknown_count = sum(
        1 for r in silver_records if r.get("primary_pattern") == "unknown_pattern"
    )
    if unknown_count > len(silver_records) * 0.2:
        warn.append(f"⚠️  unknown_pattern 占比 "
                    f"{unknown_count/max(len(silver_records),1):.1%}，pattern 覆盖不足。")
    excl_ratio = len(excluded) / max(len(silver_records), 1)
    if excl_ratio > 0.3:
        warn.append(f"⚠️  excluded 样本占比 {excl_ratio:.1%}，知识库可用样本较少。")

    return {
        "data_note": "AI-silver 数据，非人工 gold label，请谨慎使用。",
        "total_instances":              len(all_records),
        "silver_instances_count":       len(included),
        "excluded_count":               len(excluded),
        "strong_silver_count":          quality_dist.get("strong_silver", 0),
        "corrected_silver_count":       quality_dist.get("corrected_silver", 0),
        "weak_silver_count":            quality_dist.get("weak_silver", 0),
        "raw_label_distribution":       dict(raw_dist),
        "final_silver_label_distribution": dict(silver_dist),
        "label_source_distribution":    dict(source_dist),
        "silver_quality_level_distribution": dict(quality_dist),
        "average_label_confidence":     avg_conf,
        "average_label_weight":         avg_weight,
        "neutral_raw_count":            len(neutral_raw),
        "neutral_corrected_to_against_count": neutral_to_against,
        "neutral_corrected_to_favor_count":   neutral_to_favor,
        "favor_raw_count":              len(favor_raw),
        "favor_kept_count":             favor_kept,
        "favor_changed_count":          len(favor_raw) - favor_kept,
        "pattern_count":                len(memories),
        "usable_pattern_count":         len(usable_patterns),
        "high_reliability_pattern_count":   len(high_r),
        "medium_reliability_pattern_count": len(med_r),
        "low_reliability_pattern_count":    len(low_r),
        "warnings":                     warn,
    }


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="构建 AI-silver pattern memory 与 RAG 知识库")
    parser.add_argument("--stance_instances", required=True)
    parser.add_argument("--ai_preannotated",  default=None)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--min_pattern_count", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据
    print(f"📂 读取 stance_instances：{args.stance_instances}")
    stance_records = load_jsonl(args.stance_instances)
    print(f"  共加载 {len(stance_records)} 条")

    ai_records = []
    if args.ai_preannotated:
        print(f"📂 读取 AI 预标注：{args.ai_preannotated}")
        if args.ai_preannotated.endswith(".csv"):
            ai_records = load_csv(args.ai_preannotated)
        else:
            ai_records = load_jsonl(args.ai_preannotated)
        print(f"  共加载 {len(ai_records)} 条")

    # 2. 合并
    print("\n🔗 合并 AI 标注...")
    records = merge_ai_annotations(stance_records, ai_records)

    # 3. 生成 silver 标签
    print("🏷️  生成 silver 标签...")
    for rec in records:
        assign_silver_label(rec)
        infer_final_pattern(rec)

    # 4. 分拣
    silver_included = [r for r in records if not r.get("_is_excluded")]
    silver_excluded = [r for r in records if r.get("_is_excluded")]
    print(f"  可用样本：{len(silver_included)}，排除样本：{len(silver_excluded)}")

    # 5. 构建 pattern memory
    print("\n🧩 聚合 pattern memory...")
    memories, pattern_report = build_pattern_memory(records, args.min_pattern_count)

    # 6. 构建 RAG 文档
    print("📄 生成 RAG 知识文档...")
    rag_docs = build_rag_documents(memories)

    # 7. 质量报告
    quality_report = build_quality_report(records, records, memories, pattern_report)

    # 8. 写出
    print(f"\n📝 写出文件到 {output_dir}/")

    # 清理内部字段
    def clean(r):
        r.pop("_ai", None)
        r.pop("_is_excluded", None)
        return r

    write_jsonl([clean(r) for r in silver_included],
                str(output_dir / "ai_silver_instances.jsonl"))
    write_jsonl([clean(r) for r in silver_excluded],
                str(output_dir / "excluded_uncertain_instances.jsonl"))
    write_jsonl(memories,
                str(output_dir / "pattern_memory_ai_silver.jsonl"))
    write_jsonl(rag_docs,
                str(output_dir / "rag_knowledge_documents.jsonl"))
    write_report(quality_report,
                 str(output_dir / "ai_silver_quality_report.json"))
    write_report({"patterns": pattern_report},
                 str(output_dir / "pattern_memory_report.json"))

    # 摘要
    print(f"""
📊 构建摘要：
  总样本：         {len(records)}
  可用 silver：    {len(silver_included)}
  排除样本：       {len(silver_excluded)}
  pattern 数：     {len(memories)}
  RAG 文档数：     {len(rag_docs)}
  strong_silver：  {quality_report['strong_silver_count']}
  corrected_silver：{quality_report['corrected_silver_count']}
  weak_silver：    {quality_report['weak_silver_count']}
    """)

    if quality_report["warnings"]:
        print("⚠️  警告：")
        for w in quality_report["warnings"]:
            print(f"  {w}")


if __name__ == "__main__":
    main()