#!/usr/bin/env python3
"""
ai_preannotation_pipeline.py

用法：
python ai_preannotation_pipeline.py \
  --input gold_audit_set_for_annotation.csv \
  --output_dir ./ai_audit_output \
  --seed 42 \
  --pattern_sample_size 50 \
  --mode rule
"""

import csv
import json
import random
import argparse
import os
import dashscope
import time
import openai
from dashscope import Generation
from pathlib import Path
from typing import Optional
from collections import defaultdict


# ============ 常量 ============

OPENAI_API_KEY  = "sk-zk2255c8dda4cacf8ba6dacd89d28ddf46169ae793cae699"  # 替换为你的 key
OPENAI_BASE_URL = "https://api.zhizengzeng.com/v1/chat/completions"    # 如使用代理请修改此处
OPENAI_MODEL    = "qwen-max"                        # 可改为 gpt-4-turbo / gpt-3.5-turbo

_openai_client = None


MAIN_TARGET = "高市早苗发表引发网络争议的涉台言论"


VALID_RECHECK_LABELS = [
    "支持", "反对", "真中立", "无关", "目标不清",
    "上下文不足", "混合/条件立场", "无法判断"
]

VALID_REPLY_RELATIONS = [
    "赞同父评论", "反对父评论", "嘲讽父评论", "补充父评论",
    "转移话题", "无法判断", "不适用"
]

VALID_ABSTRACT_PATTERNS = [
    "agreement_with_parent",
    "disagreement_with_parent",
    "negative_judgment_toward_TARGET",
    "negative_judgment_toward_supporters_of_TARGET",
    "rhetorical_question_against_TARGET",
    "sarcasm_or_mockery",
    "support_countermeasure_against_TARGET",
    "criticize_Japan_or_Takaichi",
    "discuss_event_consequence",
    "true_neutral_information",
    "irrelevant_comment",
    "unclear_or_context_dependent",
]

AI_PREANNOTATED_COLUMNS = [
    "sample_id", "instance_id", "depth",
    "root_text", "parent_text", "current_text",
    "main_target", "stance_label_raw",
    "ai_recheck_label", "ai_recheck_confidence",
    "ai_original_label_correct", "ai_error_type",
    "ai_direct_target", "ai_stance_to_direct_target",
    "ai_stance_to_main_target", "ai_reply_relation",
    "candidate_pattern", "ai_abstract_pattern",
    "ai_evidence_span", "ai_reason",
    "ai_needs_human_review", "ai_review_priority",
    "neutral_suspicious", "is_short_reply",
    "agreement_cue", "disagreement_cue",
    "rhetorical_question_candidate", "emotion_or_stance_cue",
    "context_text",
]

SIMPLIFIED_REVIEW_COLUMNS = [
    "sample_id", "depth",
    "root_text", "parent_text", "current_text",
    "main_target", "stance_label_raw",
    "ai_recheck_label", "ai_recheck_confidence",
    "ai_original_label_correct", "ai_reason",
    "ai_abstract_pattern", "ai_review_priority",
    "human_ai_judgment_correct",
    "human_final_label",
    "human_error_type",
    "human_pattern_correct",
    "human_notes",
]

PATTERN_REVIEW_COLUMNS = [
    "sample_id", "pattern_name",
    "root_text", "parent_text", "current_text",
    "stance_label_raw", "ai_recheck_label",
    "ai_reason", "ai_recheck_confidence",
    "human_ai_judgment_correct",
    "human_final_label",
    "human_pattern_correct",
    "human_notes",
]


# ============ 工具函数 ============

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return _openai_client

def to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def safe_int(val, default=0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


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
                print(f"  ⚠️  第 {i+1} 行读取失败，已跳过：{e}")
    return records


def write_csv(records: list, filepath: str, columns: list):
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for rec in records:
            row = {col: rec.get(col, "") for col in columns}
            # 列表类型转字符串
            for col in columns:
                if isinstance(row[col], list):
                    row[col] = "|".join(str(x) for x in row[col])
            writer.writerow(row)
    print(f"  ✅ {filepath}（{len(records)} 条）")


# ============ LLM Prompt 设计 ============

def build_prompt(row: dict) -> str:
    depth = safe_int(row.get("depth", 1))
    root_text    = row.get("root_text", "") or ""
    parent_text  = row.get("parent_text", "") or ""
    current_text = row.get("current_text", "") or ""
    stance_raw   = row.get("stance_label_raw", "") or ""
    candidate_pattern = row.get("candidate_pattern", "") or ""
    neutral_suspicious        = row.get("neutral_suspicious", "")
    is_short_reply            = row.get("is_short_reply", "")
    agreement_cue             = row.get("agreement_cue", "")
    disagreement_cue          = row.get("disagreement_cue", "")
    rhetorical_question_cand  = row.get("rhetorical_question_candidate", "")
    emotion_or_stance_cue     = row.get("emotion_or_stance_cue", "")

    parent_section = (
        f"【父评论（depth=2 时必须参考）】\n{parent_text}\n"
        if depth >= 2 else "【父评论】（本条为一级评论，无父评论）\n"
    )

    prompt = f"""你是一名社交媒体立场检测研究员。请对以下微博评论进行立场分析，并严格按照 JSON 格式输出结果。

【主 target】
{MAIN_TARGET}

【原帖】
{root_text}

{parent_section}
【当前评论】
{current_text}

【当前评论层级】depth = {depth}

【原始模型标签】
stance_label_raw = {stance_raw}

【Audit Flags（供参考）】
- candidate_pattern: {candidate_pattern}
- neutral_suspicious: {neutral_suspicious}
- is_short_reply: {is_short_reply}
- agreement_cue: {agreement_cue}
- disagreement_cue: {disagreement_cue}
- rhetorical_question_candidate: {rhetorical_question_cand}
- emotion_or_stance_cue: {emotion_or_stance_cue}

【判断要求】
1. 你的判断目标是当前评论相对于主 target（{MAIN_TARGET}）的立场。
2. 不要把对"中国""日本""台湾""高市早苗本人"的态度直接等同于对主 target 的立场，除非可以明确推断。
3. 严格区分：真中立 / 无关 / 目标不清 / 上下文不足 / 无法判断，不要混用。
4. depth=2 时，必须结合父评论判断，短句"确实""对""就是"等视情况继承父评论立场。
5. 反问、讽刺、嘲讽通常不是中立，请仔细判断。
6. 不要把模型不确定直接标为中立。

【输出格式】
只输出合法 JSON，不要有任何额外文字、代码块标记或说明。

{{
  "ai_recheck_label": "支持|反对|真中立|无关|目标不清|上下文不足|混合/条件立场|无法判断",
  "ai_recheck_confidence": 0.0到1.0之间的浮点数,
  "ai_original_label_correct": "true|false|uncertain",
  "ai_error_type": "false_neutral|false_against|false_favor|target_mismatch|context_missing|irrelevant_mislabeled_as_neutral|ambiguous|other|null",
  "ai_direct_target": "高市早苗涉台言论|高市早苗本人|日本政府|日本|中国反制|台湾问题|父评论作者|其他|不明确",
  "ai_stance_to_direct_target": "支持|反对|中立|不明确",
  "ai_stance_to_main_target": "支持|反对|真中立|无关|目标不清|上下文不足|混合/条件立场|无法判断",
  "ai_reply_relation": "赞同父评论|反对父评论|嘲讽父评论|补充父评论|转移话题|无法判断|不适用",
  "ai_abstract_pattern": "agreement_with_parent|disagreement_with_parent|negative_judgment_toward_TARGET|negative_judgment_toward_supporters_of_TARGET|rhetorical_question_against_TARGET|sarcasm_or_mockery|support_countermeasure_against_TARGET|criticize_Japan_or_Takaichi|discuss_event_consequence|true_neutral_information|irrelevant_comment|unclear_or_context_dependent",
  "ai_evidence_span": "从 current_text 中提取的关键片段，如依赖 parent_text 请注明",
  "ai_reason": "一句话解释，不超过80字"
}}"""
    return prompt


def call_llm_for_preannotation(row: dict) -> dict:
    """调用 OpenAI API 进行预标注，含重试机制"""
    prompt = build_prompt(row)
    client = get_openai_client()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名社交媒体立场检测研究员，专注于中文微博立场分析。"
                            "你只输出合法 JSON，不输出任何额外文字、代码块标记或解释。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
                timeout=30,
            )

            raw_text = response.choices[0].message.content.strip()

            # 去除可能的 markdown 代码块标记
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

            result = json.loads(raw_text)

            # ---- 字段校验与兜底 ----
            depth = safe_int(row.get("depth", 1))
            stance_raw = row.get("stance_label_raw", "") or ""

            # 确保所有必要字段存在
            defaults = {
                "ai_recheck_label":           "无法判断",
                "ai_recheck_confidence":      0.5,
                "ai_original_label_correct":  "uncertain",
                "ai_error_type":              None,
                "ai_direct_target":           "不明确",
                "ai_stance_to_direct_target": "不明确",
                "ai_stance_to_main_target":   "无法判断",
                "ai_reply_relation":          "不适用" if depth < 2 else "无法判断",
                "ai_abstract_pattern":        "unclear_or_context_dependent",
                "ai_evidence_span":           "",
                "ai_reason":                  "",
            }
            for key, default_val in defaults.items():
                if key not in result or result[key] is None:
                    result[key] = default_val

            # 置信度强制为 float
            result["ai_recheck_confidence"] = safe_float(
                result.get("ai_recheck_confidence"), 0.5
            )

            # ---- 计算 ai_needs_human_review ----
            neutral_suspicious = to_bool(row.get("neutral_suspicious", False))
            is_short_reply     = to_bool(row.get("is_short_reply", False))
            recheck_label      = result.get("ai_recheck_label", "")
            ai_correct         = result.get("ai_original_label_correct", "uncertain")
            confidence         = result["ai_recheck_confidence"]

            needs_review = any([
                ai_correct in ("false", "uncertain"),
                confidence < 0.75,
                stance_raw == "中立" and neutral_suspicious,
                stance_raw == "支持",
                depth >= 2 and is_short_reply,
                recheck_label in ("目标不清", "上下文不足", "无法判断", "混合/条件立场"),
            ])
            result["ai_needs_human_review"] = needs_review

            # ---- 计算 ai_review_priority ----
            is_favor   = stance_raw == "支持"
            is_neutral = stance_raw == "中立"
            if (
                ai_correct == "false"
                or is_favor
                or (is_neutral and neutral_suspicious)
                or confidence < 0.6
                or (depth >= 2 and is_short_reply)
            ):
                result["ai_review_priority"] = "high"
            elif (
                confidence < 0.75
                or recheck_label in ("目标不清", "上下文不足", "混合/条件立场")
            ):
                result["ai_review_priority"] = "medium"
            else:
                result["ai_review_priority"] = "low"

            return result

        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON 解析失败（第 {attempt+1} 次），sample_id={row.get('sample_id','')}：{e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise RuntimeError(f"JSON 解析失败，已重试 {max_retries} 次：{e}")

        except openai.RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"  ⏳ 触发限速，等待 {wait}s 后重试...")
            time.sleep(wait)

        except openai.APITimeoutError:
            print(f"  ⏳ 请求超时（第 {attempt+1} 次），重试中...")
            time.sleep(3)

        except openai.AuthenticationError:
            raise RuntimeError("API Key 无效，请检查 OPENAI_API_KEY 配置。")

        except Exception as e:
            print(f"  ⚠️  未知错误（第 {attempt+1} 次）：{e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise


# ============ 规则预标注 ============

def rule_based_preannotation(row: dict) -> dict:
    stance_raw  = row.get("stance_label_raw", "") or ""
    depth       = safe_int(row.get("depth", 1))
    neutral_suspicious       = to_bool(row.get("neutral_suspicious", False))
    is_short_reply           = to_bool(row.get("is_short_reply", False))
    agreement_cue            = to_bool(row.get("agreement_cue", False))
    disagreement_cue         = to_bool(row.get("disagreement_cue", False))
    rhetorical_question_cand = to_bool(row.get("rhetorical_question_candidate", False))
    emotion_or_stance_cue    = to_bool(row.get("emotion_or_stance_cue", False))
    parent_text  = row.get("parent_text", "") or ""
    current_text = row.get("current_text", "") or ""

    result = {
        "ai_recheck_label":           stance_raw,  # 默认与原标签一致
        "ai_recheck_confidence":      0.5,
        "ai_original_label_correct":  "uncertain",
        "ai_error_type":              None,
        "ai_direct_target":           "不明确",
        "ai_stance_to_direct_target": "不明确",
        "ai_stance_to_main_target":   stance_raw,
        "ai_reply_relation":          "不适用" if depth < 2 else "无法判断",
        "ai_abstract_pattern":        "unclear_or_context_dependent",
        "ai_evidence_span":           current_text[:50] if current_text else "",
        "ai_reason":                  "规则预标注，未接入大模型",
        "ai_needs_human_review":      False,
        "ai_review_priority":         "low",
    }

    needs_review  = False
    review_priority = "low"

    # ---- 规则 1：中立 + 赞同线索 + depth=2 ----
    if stance_raw == "中立" and agreement_cue and depth >= 2:
        result["ai_recheck_label"]          = "上下文不足"
        result["ai_abstract_pattern"]       = "agreement_with_parent"
        result["ai_original_label_correct"] = "uncertain"
        result["ai_reason"]                 = "短句赞同父评论，立场依赖父评论内容，需人工结合上下文判断"
        result["ai_recheck_confidence"]     = 0.55
        result["ai_reply_relation"]         = "赞同父评论"
        needs_review  = True
        review_priority = "high"

    # ---- 规则 2：中立 + 反问 ----
    elif stance_raw == "中立" and rhetorical_question_cand:
        result["ai_recheck_label"]          = "反对"
        result["ai_abstract_pattern"]       = "rhetorical_question_against_TARGET"
        result["ai_original_label_correct"] = "false"
        result["ai_error_type"]             = "false_neutral"
        result["ai_stance_to_main_target"]  = "反对"
        result["ai_reason"]                 = "含反问结构，通常隐含反对立场，疑似假中立"
        result["ai_recheck_confidence"]     = 0.72
        needs_review  = True
        review_priority = "high"

    # ---- 规则 3：中立 + 情绪词 ----
    elif stance_raw == "中立" and emotion_or_stance_cue:
        result["ai_recheck_label"]          = "无法判断"
        result["ai_abstract_pattern"]       = "unclear_or_context_dependent"
        result["ai_original_label_correct"] = "uncertain"
        result["ai_error_type"]             = "false_neutral"
        result["ai_reason"]                 = "含情绪或立场词，中立标签存疑，需人工核查"
        result["ai_recheck_confidence"]     = 0.5
        needs_review  = True
        review_priority = "high"

    # ---- 规则 4：中立 + 反对线索 ----
    elif stance_raw == "中立" and disagreement_cue:
        result["ai_recheck_label"]          = "反对"
        result["ai_abstract_pattern"]       = "disagreement_with_parent" if depth >= 2 else "negative_judgment_toward_TARGET"
        result["ai_original_label_correct"] = "false"
        result["ai_error_type"]             = "false_neutral"
        result["ai_stance_to_main_target"]  = "反对"
        result["ai_reason"]                 = "含反对/质疑线索，疑似假中立"
        result["ai_recheck_confidence"]     = 0.68
        needs_review  = True
        review_priority = "high"

    # ---- 规则 5：支持标签 ----
    elif stance_raw == "支持":
        result["ai_recheck_label"]          = "支持"
        result["ai_original_label_correct"] = "uncertain"
        result["ai_stance_to_main_target"]  = "支持"
        result["ai_reason"]                 = "支持标签为少数类，需人工确认是否真实支持主 target"
        result["ai_recheck_confidence"]     = 0.6
        needs_review  = True
        review_priority = "high"

    # ---- 规则 6：反对 + 情绪词 ----
    elif stance_raw == "反对" and emotion_or_stance_cue:
        result["ai_recheck_label"]          = "反对"
        result["ai_original_label_correct"] = "true"
        result["ai_stance_to_main_target"]  = "反对"
        result["ai_abstract_pattern"]       = "negative_judgment_toward_TARGET"
        result["ai_reason"]                 = "含情绪词，反对立场与原标签一致，置信度较高"
        result["ai_recheck_confidence"]     = 0.82
        review_priority = "medium"

    # ---- 规则 7：中立 + 短回复 ----
    elif stance_raw == "中立" and is_short_reply:
        result["ai_recheck_label"]          = "上下文不足"
        result["ai_original_label_correct"] = "uncertain"
        result["ai_reason"]                 = "短回复，无法从文本本身判断立场，需结合上下文"
        result["ai_recheck_confidence"]     = 0.45
        needs_review  = True
        review_priority = "medium"

    # ---- 规则 8：反对 + 反问 ----
    elif stance_raw == "反对" and rhetorical_question_cand:
        result["ai_recheck_label"]          = "反对"
        result["ai_original_label_correct"] = "true"
        result["ai_abstract_pattern"]       = "rhetorical_question_against_TARGET"
        result["ai_stance_to_main_target"]  = "反对"
        result["ai_reason"]                 = "反问结构 + 反对标签一致，置信度较高"
        result["ai_recheck_confidence"]     = 0.80
        review_priority = "medium"

    # ---- 规则 9：depth=2 短回复 ----
    elif depth >= 2 and is_short_reply:
        result["ai_recheck_label"]          = "上下文不足"
        result["ai_original_label_correct"] = "uncertain"
        result["ai_reply_relation"]         = "无法判断"
        result["ai_reason"]                 = "二级短回复，立场高度依赖父评论，需人工核查"
        result["ai_recheck_confidence"]     = 0.4
        needs_review  = True
        review_priority = "high"

    # ---- 规则 10：无高风险线索，默认与原标签一致 ----
    else:
        result["ai_recheck_label"]          = stance_raw if stance_raw in VALID_RECHECK_LABELS else "无法判断"
        result["ai_original_label_correct"] = "uncertain"
        result["ai_stance_to_main_target"]  = stance_raw if stance_raw in VALID_RECHECK_LABELS else "无法判断"
        result["ai_reason"]                 = "无明显高风险线索，暂时沿用原标签，置信度中等"
        result["ai_recheck_confidence"]     = 0.65
        review_priority = "low"

    # ---- 补充 needs_human_review 强制规则 ----
    if (
        result["ai_original_label_correct"] == "false"
        or result["ai_original_label_correct"] == "uncertain"
        or result["ai_recheck_confidence"] < 0.75
        or (stance_raw == "中立" and neutral_suspicious)
        or stance_raw == "支持"
        or (depth >= 2 and is_short_reply)
        or result["ai_recheck_label"] in ("目标不清", "上下文不足", "无法判断", "混合/条件立场")
    ):
        needs_review = True

    result["ai_needs_human_review"] = needs_review

    # ---- 补充 ai_review_priority 强制规则 ----
    if (
        result["ai_original_label_correct"] == "false"
        or stance_raw == "支持"
        or (stance_raw == "中立" and neutral_suspicious)
        or result["ai_recheck_confidence"] < 0.6
    ):
        review_priority = "high"
    elif review_priority not in ("high", "medium"):
        review_priority = "low"

    result["ai_review_priority"] = review_priority
    return result


# ============ 预标注主流程 ============

def run_preannotation(records: list, mode: str) -> list:
    results = []
    total = len(records)
    for i, row in enumerate(records):
        if (i + 1) % 200 == 0:
            print(f"  进度：{i+1}/{total}")
        try:
            if mode == "llm":
                ai_fields = call_llm_for_preannotation(row)
            else:
                ai_fields = rule_based_preannotation(row)
            row.update(ai_fields)
        except NotImplementedError:
            raise
        except Exception as e:
            print(f"  ⚠️  sample_id={row.get('sample_id','')} 预标注失败：{e}")
            row.update({
                "ai_recheck_label": "无法判断",
                "ai_recheck_confidence": 0.0,
                "ai_original_label_correct": "uncertain",
                "ai_error_type": None,
                "ai_direct_target": "不明确",
                "ai_stance_to_direct_target": "不明确",
                "ai_stance_to_main_target": "无法判断",
                "ai_reply_relation": "无法判断",
                "ai_abstract_pattern": "unclear_or_context_dependent",
                "ai_evidence_span": "",
                "ai_reason": f"预标注异常：{e}",
                "ai_needs_human_review": True,
                "ai_review_priority": "high",
            })
        results.append(row)
    return results


# ============ 简化审核文件 ============

def build_simplified_review(records: list) -> list:
    simplified = []
    for rec in records:
        row = {col: rec.get(col, "") for col in SIMPLIFIED_REVIEW_COLUMNS}
        # 添加空白人工填写字段
        for field in ["human_ai_judgment_correct", "human_final_label",
                      "human_error_type", "human_pattern_correct", "human_notes"]:
            if field not in row or not row[field]:
                row[field] = ""
        simplified.append(row)
    return simplified


# ============ Pattern 审核批次 ============

def build_pattern_review_batches(
    records: list,
    pattern_sample_size: int,
    seed: int,
) -> list:
    rng = random.Random(seed)
    pattern_groups = defaultdict(list)

    for rec in records:
        # 优先使用 ai_abstract_pattern，回退到 candidate_pattern
        pattern = rec.get("ai_abstract_pattern") or rec.get("candidate_pattern") or "unknown"
        # candidate_pattern 可能是 "|" 连接的多值，取第一个
        if "|" in str(pattern):
            pattern = str(pattern).split("|")[0].strip()
        pattern_groups[pattern].append(rec)

    batches = []
    for pattern_name, group in sorted(pattern_groups.items()):
        # 优先级排序：high > medium > low，confidence 低的优先
        def priority_key(r):
            p = {"high": 0, "medium": 1, "low": 2}.get(r.get("ai_review_priority", "low"), 2)
            c = safe_float(r.get("ai_recheck_confidence", 1.0))
            is_neutral = 1 if r.get("stance_label_raw") == "中立" else 0
            is_wrong = 1 if r.get("ai_original_label_correct") == "false" else 0
            return (p, -is_wrong, -is_neutral, c)

        sorted_group = sorted(group, key=priority_key)
        sampled = sorted_group[:pattern_sample_size]
        rng.shuffle(sampled)  # 同一 pattern 内随机化顺序

        for rec in sampled:
            row = {
                "sample_id":    rec.get("sample_id", ""),
                "pattern_name": pattern_name,
                "root_text":    rec.get("root_text", ""),
                "parent_text":  rec.get("parent_text", ""),
                "current_text": rec.get("current_text", ""),
                "stance_label_raw":       rec.get("stance_label_raw", ""),
                "ai_recheck_label":       rec.get("ai_recheck_label", ""),
                "ai_reason":              rec.get("ai_reason", ""),
                "ai_recheck_confidence":  rec.get("ai_recheck_confidence", ""),
                "human_ai_judgment_correct": "",
                "human_final_label":         "",
                "human_pattern_correct":     "",
                "human_notes":               "",
            }
            batches.append(row)

    return batches


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="AI 预标注流水线")
    parser.add_argument("--input",               required=True, help="输入 CSV 路径")
    parser.add_argument("--output_dir",          required=True, help="输出目录")
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--pattern_sample_size", type=int, default=50)
    parser.add_argument("--mode",                choices=["rule", "llm"], default="rule")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 读取输入：{args.input}")
    records = load_csv(args.input)
    print(f"  共加载 {len(records)} 条")

    print(f"\n🤖 执行预标注（模式：{args.mode}）...")
    annotated = run_preannotation(records, args.mode)

    print(f"\n📝 写出文件...")

    # 确保所有列都存在
    for rec in annotated:
        for col in AI_PREANNOTATED_COLUMNS:
            if col not in rec:
                rec[col] = ""

    write_csv(annotated, str(output_dir / "ai_preannotated_gold_audit.csv"),
              AI_PREANNOTATED_COLUMNS)

    simplified = build_simplified_review(annotated)
    write_csv(simplified, str(output_dir / "simplified_human_review.csv"),
              SIMPLIFIED_REVIEW_COLUMNS)

    batches = build_pattern_review_batches(annotated, args.pattern_sample_size, args.seed)
    write_csv(batches, str(output_dir / "pattern_review_batches.csv"),
              PATTERN_REVIEW_COLUMNS)

    # 统计摘要
    high   = sum(1 for r in annotated if r.get("ai_review_priority") == "high")
    medium = sum(1 for r in annotated if r.get("ai_review_priority") == "medium")
    low    = sum(1 for r in annotated if r.get("ai_review_priority") == "low")
    needs  = sum(1 for r in annotated if to_bool(r.get("ai_needs_human_review", False)))
    patterns_found = len(set(
        (r.get("ai_abstract_pattern") or "unknown") for r in annotated
    ))

    print(f"""
📊 预标注摘要：
  总计：         {len(annotated)} 条
  需人工复核：   {needs} 条
  审核优先级：   high={high}  medium={medium}  low={low}
  发现 pattern： {patterns_found} 种
  pattern 批次： {len(batches)} 条（每 pattern 最多 {args.pattern_sample_size} 条）
    """)


if __name__ == "__main__":
    main()