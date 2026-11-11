#!/usr/bin/env python3
"""
build_semantic_propagation_annotations_llm.py

用法（第一步小样本测试）：
python build_semantic_propagation_annotations_llm.py \
  --windowed_instances processed_cmtscd_windows/cmtscd_windowed_instances.jsonl \
  --output_dir processed_cmtscd_semantic_llm_test \
  --provider openai \
  --model gpt-4o-mini \
  --sample_limit 50 \
  --only_split train \
  --context_mode both \
  --resume true
"""

import json
import os
import re
import time
import hashlib
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# ============ Prompt 模板 ============

SYSTEM_PROMPT = """You are an expert in conversational stance detection and semantic propagation analysis for social media threads. Your task is not simple sentiment classification, but analyzing the semantic function of the current comment within the conversation chain: its reply relation, target orientation, and how stance may propagate from the parent comment. You must strictly distinguish between the main_target, the parent comment, the reply author, and topic shift. You may only output JSON, nothing else."""

USER_PROMPT_TEMPLATE = """Please analyze the following social media conversation sample.

main_target:
{main_target}

topic:
{topic}

target_anchor:
{target_anchor}

upward_window (upstream conversation):
{upward_window_texts}

current_text (current comment):
{current_text}

downward_feedback (child-reply feedback):
{downward_feedback_window_texts}

Full causal_context_text:
{causal_context_text}

Full posthoc_context_text:
{posthoc_context_text}

Please complete the following analysis:

1. Determine the reply relation between the current comment and its parent comment.
2. Determine whether stance propagates from the parent comment to the current comment: inheritance, reversal, amplification, softening, sarcastic reversal, target shift, or unclear.
3. Determine the relation between what the current comment is directly discussing and the main_target.
4. If the current comment is short or semantically ambiguous, determine whether the child replies help disambiguate its meaning.
5. Infer the likely stance of the current comment toward the main_target (this inference is AI-silver only, not human ground truth).
6. Provide a list of language-pattern labels.
7. Output a confidence score (0-1) and a one-sentence justification (no more than 40 words).

Important requirements:
- Do not simply label "irrelevant" / "unclear target" / "insufficient context" as none.
- If the current comment is a short backchannel token such as "true", "lol", "lmao", "facts", "same", you must judge its pragmatic function together with the parent comment.
- If a child reply contains content such as "are you praising or mocking", "clearly sarcastic", "you misunderstood", it can be used to infer the true meaning of the current comment, and llm_use_child_feedback should be marked true.
- If the current comment only attacks the reply author personally rather than discussing the main_target, label llm_target_relation as reply_author_only.
- If the current comment discusses an attribute, effect, consequence, or execution detail of the main_target, label it as attribute_of_target or consequence_of_target rather than topic_shift.
- Output valid JSON only, nothing else.

Please strictly follow this JSON schema in your output (all fields are required):
{{
  "llm_reply_relation": "agree_with_parent|disagree_with_parent|mock_parent|sarcastic_agreement|clarify_parent|supplement_parent|question_parent|answer_parent|topic_shift|direct_comment_to_target|unknown",
  "llm_semantic_propagation": "stance_inheritance|stance_reversal|sarcastic_reversal|stance_amplification|stance_softening|direct_stance_expression|target_shift|unclear_propagation|no_parent_context",
  "llm_stance_mapping_to_parent": "same_as_parent|opposite_to_parent|partial_or_conditional|independent_from_parent|unclear|not_applicable",
  "llm_target_relation": "same_target|attribute_of_target|consequence_of_target|reason_for_target|opposition_to_related_target|support_to_related_target|reply_author_only|topic_shift|irrelevant|unclear",
  "llm_child_feedback_signal": "child_confirms_current_meaning|child_questions_current_meaning|child_corrects_misreading|child_indicates_sarcasm|child_indicates_disagreement|child_conflict|no_useful_child_feedback|no_child",
  "llm_candidate_patterns": ["agreement_with_parent|disagreement_with_parent|mocking_disagreement|sarcastic_agreement|short_ambiguous_reply|child_feedback_disambiguation|direct_positive_evaluation|direct_negative_evaluation|conditional_or_mixed_stance|attribute_based_stance|consequence_based_stance|target_drift|topic_shift|question_or_challenge|clarification|unknown_pattern"],
  "llm_direct_target": "main_target|parent_comment|reply_author|target_attribute|target_consequence|related_entity|unclear",
  "llm_inferred_stance_to_main_target": "favor|against|none|mixed_or_conditional|unclear|irrelevant",
  "llm_confidence": 0.0,
  "llm_reason_short": "a justification of no more than 40 words",
  "llm_use_child_feedback": false,
  "llm_context_setting": "causal|posthoc|both"
}}"""

VALID_FIELDS = {
    "llm_reply_relation": [
        "agree_with_parent", "disagree_with_parent", "mock_parent",
        "sarcastic_agreement", "clarify_parent", "supplement_parent",
        "question_parent", "answer_parent", "topic_shift",
        "direct_comment_to_target", "unknown"
    ],
    "llm_semantic_propagation": [
        "stance_inheritance", "stance_reversal", "sarcastic_reversal",
        "stance_amplification", "stance_softening", "direct_stance_expression",
        "target_shift", "unclear_propagation", "no_parent_context"
    ],
    "llm_stance_mapping_to_parent": [
        "same_as_parent", "opposite_to_parent", "partial_or_conditional",
        "independent_from_parent", "unclear", "not_applicable"
    ],
    "llm_target_relation": [
        "same_target", "attribute_of_target", "consequence_of_target",
        "reason_for_target", "opposition_to_related_target",
        "support_to_related_target", "reply_author_only",
        "topic_shift", "irrelevant", "unclear"
    ],
    "llm_child_feedback_signal": [
        "child_confirms_current_meaning", "child_questions_current_meaning",
        "child_corrects_misreading", "child_indicates_sarcasm",
        "child_indicates_disagreement", "child_conflict",
        "no_useful_child_feedback", "no_child"
    ],
    "llm_direct_target": [
        "main_target", "parent_comment", "reply_author",
        "target_attribute", "target_consequence", "related_entity", "unclear"
    ],
    "llm_inferred_stance_to_main_target": [
        "favor", "against", "none", "mixed_or_conditional", "unclear", "irrelevant"
    ],
    "llm_context_setting": ["causal", "posthoc", "both"],
}

EXAMPLE_FIELDS = [
    "topic", "node_id", "main_target", "current_text", "parent_text",
    "upward_window_texts", "downward_feedback_window_texts",
    "llm_reply_relation", "llm_semantic_propagation",
    "llm_stance_mapping_to_parent", "llm_target_relation",
    "llm_child_feedback_signal", "llm_candidate_patterns",
    "llm_inferred_stance_to_main_target", "llm_confidence",
    "llm_reason_short", "causal_context_text", "posthoc_context_text",
]


# ============ 在此处填写你的配置 ============
# 安全提示：请不要把真实 API Key 硬编码进代码文件（尤其是要提交/分享的脚本）。
# 这里改为从环境变量读取，运行前 export OPENAI_API_KEY=... 等即可，
# 也可以沿用你原脚本里"直接把 key 写进 API_CONFIGS"的方式，但不建议对外分享该文件。

import os as _os

API_CONFIGS = {
    "openai": {
        "api_key":  _os.environ.get("OPENAI_API_KEY", ""),
        "base_url": _os.environ.get("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1/chat/completions"),
        "model":    "gpt-4o-mini",
    },
    "deepseek": {
        "api_key":  _os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.zhizengzeng.com/v1/chat/completions",
        "model":    "deepseek-chat",
    },
    "qwen": {
        "api_key":  _os.environ.get("QWEN_API_KEY", ""),
        "base_url": "https://api.zhizengzeng.com/v1/chat/completions",
        "model":    "qwen-max",
    },
}

# ============ IO ============

def load_jsonl(filepath: str) -> list:
    records = []
    path = Path(filepath)
    if not path.exists():
        return []
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


def write_jsonl_append(rec: dict, filepath: Path):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath.name}")


def load_existing_ids(filepath: Path) -> set:
    """读取已处理的 node_id 集合，用于断点续跑"""
    if not filepath.exists():
        return set()
    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                nid = rec.get("node_id", "")
                if nid:
                    ids.add(nid)
            except Exception:
                pass
    return ids


def safe_str(val) -> str:
    return str(val).strip() if val else ""


def safe_list(val) -> list:
    if isinstance(val, list):
        return val
    return []


# ============ Prompt 构建 ============

def build_llm_prompt(rec: dict, context_mode: str) -> str:
    main_target   = safe_str(rec.get("main_target"))
    topic         = safe_str(rec.get("topic"))
    target_anchor = safe_str(rec.get("target_anchor"))
    current_text  = safe_str(rec.get("current_text"))
    causal_ctx    = safe_str(rec.get("causal_context_text"))
    posthoc_ctx   = safe_str(rec.get("posthoc_context_text"))

    up_texts = safe_list(rec.get("upward_window_texts"))
    upward_str = "\n".join(f"- {t}" for t in up_texts) if up_texts else "(no upstream comments)"

    down_texts = safe_list(rec.get("downward_feedback_window_texts"))
    if context_mode == "causal":
        downward_str = "(causal mode: child replies not used)"
        posthoc_ctx  = "(causal mode: not provided)"
    else:
        downward_str = "\n".join(f"- {t}" for t in down_texts) if down_texts else "(no child replies)"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        main_target=main_target,
        topic=topic,
        target_anchor=target_anchor[:300],
        upward_window_texts=upward_str,
        current_text=current_text,
        downward_feedback_window_texts=downward_str,
        causal_context_text=causal_ctx[:600],
        posthoc_context_text=posthoc_ctx[:600],
    )
    return user_prompt


def prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


# ============ LLM 调用 ============

def call_llm(prompt, provider, model, api_base, temperature, max_retries, sleep_seconds):
    if provider in ("openai", "deepseek", "qwen"):
        try:
            import openai
        except ImportError:
            raise ImportError("请先安装：pip install openai")

        # 优先读配置块，没有再读环境变量
        config  = API_CONFIGS.get(provider, {})
        api_key = config.get("api_key") or os.environ.get({
            "openai":   "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen":     "DASHSCOPE_API_KEY",
        }[provider])

        if not api_key or api_key.startswith("sk-xxx"):
            raise EnvironmentError(
                f"请在脚本顶部 API_CONFIGS['{provider}']['api_key'] 填写真实 API Key"
            )

        base_url = api_base or config.get("base_url")
        # 如果命令行没传 --model，也用配置块里的默认值
        actual_model = model or config.get("model", "gpt-4o-mini")

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        # ... 后续 retry 逻辑不变，把 model 换成 actual_model ...

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=800,
                    timeout=60,
                )
                return resp.choices[0].message.content.strip()

            except openai.RateLimitError:
                wait = sleep_seconds * (attempt + 1) * 3
                print(f"  ⏳ 限速，等待 {wait:.1f}s（第 {attempt+1} 次）...")
                time.sleep(wait)

            except openai.APITimeoutError:
                print(f"  ⏳ 超时，重试（第 {attempt+1} 次）...")
                time.sleep(sleep_seconds)

            except openai.AuthenticationError:
                raise EnvironmentError(f"API Key 无效（{provider}），请检查环境变量。")

            except Exception as e:
                print(f"  ⚠️  调用异常（第 {attempt+1} 次）：{e}")
                if attempt < max_retries - 1:
                    time.sleep(sleep_seconds)
                else:
                    raise

    elif provider == "local":
        raise NotImplementedError(
            "local 模式未实现，请在 call_llm() 中接入本地推理接口（如 Ollama）。"
        )
    else:
        raise ValueError(f"不支持的 provider：{provider}")

    raise RuntimeError(f"LLM 调用失败，已重试 {max_retries} 次")


# ============ JSON 解析与修复 ============

def parse_llm_json(raw_text: str) -> dict:
    """尝试解析 LLM 返回文本为 JSON，自动修复 markdown 包裹"""
    text = raw_text.strip()

    # 去除 ```json ... ``` 包裹
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个 {...} 块
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"JSON 解析失败，原始内容：{raw_text[:200]}")


# ============ 字段校验 ============

def validate_llm_output(result: dict) -> dict:
    """校验并填充 LLM 输出的必要字段"""
    defaults = {
        "llm_reply_relation":              "unknown",
        "llm_semantic_propagation":        "unclear_propagation",
        "llm_stance_mapping_to_parent":    "unclear",
        "llm_target_relation":             "unclear",
        "llm_child_feedback_signal":       "no_child",
        "llm_candidate_patterns":          ["unknown_pattern"],
        "llm_direct_target":               "unclear",
        "llm_inferred_stance_to_main_target": "unclear",
        "llm_confidence":                  0.5,
        "llm_reason_short":                "",
        "llm_use_child_feedback":          False,
        "llm_context_setting":             "both",
        "llm_annotation_source":           "llm",
    }

    for key, default_val in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_val

    # 值域校验
    for field, valid_vals in VALID_FIELDS.items():
        val = result.get(field)
        if isinstance(val, str) and val not in valid_vals:
            result[field] = defaults.get(field, val)

    # confidence 范围
    result["llm_confidence"] = float(
        max(0.1, min(0.95, result.get("llm_confidence", 0.5)))
    )

    # candidate_patterns 必须是列表
    cp = result.get("llm_candidate_patterns")
    if not isinstance(cp, list):
        result["llm_candidate_patterns"] = [str(cp)] if cp else ["unknown_pattern"]

    # reason_short 截断
    result["llm_reason_short"] = str(result.get("llm_reason_short", ""))[:120]

    result["llm_annotation_source"] = "llm"
    return result


# ============ 单条标注 ============

def annotate_one_instance(
    rec: dict,
    provider: str,
    model: str,
    api_base: Optional[str],
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
    context_mode: str,
    cache: dict,
    save_raw: bool = False,
) -> tuple:
    """
    返回 (annotated_rec, error_msg)
    error_msg 为 None 表示成功
    """
    prompt = build_llm_prompt(rec, context_mode)
    phash  = prompt_hash(prompt)

    # 命中缓存
    if phash in cache:
        cached = cache[phash]
        out = dict(rec)
        out.update(cached)
        out["llm_prompt_hash"] = phash
        if not save_raw:
            out.pop("llm_raw_response", None)
        return out, None

    # 调用 LLM
    for attempt in range(max_retries):
        try:
            raw = call_llm(
                prompt, provider, model, api_base,
                temperature, max_retries, sleep_seconds
            )
            result = parse_llm_json(raw)
            result = validate_llm_output(result)
            result["llm_raw_response"] = raw if save_raw else None
            result["llm_prompt_hash"]  = phash

            cache[phash] = {k: v for k, v in result.items()
                            if k.startswith("llm_")}

            out = dict(rec)
            out.update(result)
            if not save_raw:
                out.pop("llm_raw_response", None)
            return out, None

        except (ValueError, RuntimeError) as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                return rec, str(e)

        except Exception as e:
            return rec, str(e)

    return rec, "max_retries_exceeded"


# ============ 样本筛选 ============

def select_instances(
    instances: list,
    only_split:              Optional[str],
    only_ambiguous:          bool,
    only_with_child_feedback: bool,
    only_depth_ge:           Optional[int],
    sample_limit:            Optional[int],
    rule_semantic:           Optional[list],
) -> list:
    selected = instances

    if only_split:
        selected = [r for r in selected if r.get("split") == only_split]

    if only_ambiguous:
        selected = [r for r in selected if r.get("is_ambiguous_current")]

    if only_with_child_feedback:
        selected = [r for r in selected
                    if safe_list(r.get("downward_feedback_window_texts"))]

    if only_depth_ge is not None:
        selected = [r for r in selected if r.get("depth", 0) >= only_depth_ge]

    if rule_semantic:
        rule_map = {r.get("node_id"): r for r in rule_semantic}
        low_quality_ids = set()
        for nid, r in rule_map.items():
            if (
                r.get("semantic_confidence", 1.0) < 0.5
                or r.get("reply_relation_candidate") == "unknown"
                or r.get("target_relation_candidate") in ("topic_shift", "unclear")
                or "unknown_pattern" in safe_list(r.get("candidate_pattern"))
            ):
                low_quality_ids.add(nid)
        selected = [r for r in selected if r.get("node_id") in low_quality_ids]

    if sample_limit:
        selected = selected[:sample_limit]

    return selected


# ============ 质量报告 ============

def generate_report(
    all_input: list,
    selected: list,
    annotated: list,
    failed: list,
    skipped: int,
    rule_semantic: Optional[list],
) -> dict:
    total_annotated = len(annotated)

    def dist(key):
        return dict(Counter(r.get(key) for r in annotated))

    def list_dist(key):
        c = Counter()
        for r in annotated:
            for p in safe_list(r.get(key)):
                c[p] += 1
        return dict(c)

    confs = [r.get("llm_confidence", 0) for r in annotated]
    avg_conf = round(sum(confs) / total_annotated, 4) if total_annotated else 0

    report = {
        "total_input_instances":       len(all_input),
        "selected_instances":          len(selected),
        "annotated_instances":         total_annotated,
        "failed_instances":            len(failed),
        "skipped_existing_instances":  skipped,
        "split_distribution":          dist("split"),
        "topic_distribution":          dist("topic"),
        "depth_distribution":          dict(Counter(r.get("depth") for r in annotated)),
        "llm_reply_relation_distribution":       dist("llm_reply_relation"),
        "llm_semantic_propagation_distribution": dist("llm_semantic_propagation"),
        "llm_stance_mapping_to_parent_distribution": dist("llm_stance_mapping_to_parent"),
        "llm_target_relation_distribution":      dist("llm_target_relation"),
        "llm_child_feedback_signal_distribution": dist("llm_child_feedback_signal"),
        "llm_candidate_pattern_distribution":    list_dist("llm_candidate_patterns"),
        "llm_inferred_stance_distribution":      dist("llm_inferred_stance_to_main_target"),
        "average_llm_confidence":      avg_conf,
        "high_confidence_count":       sum(1 for c in confs if c >= 0.75),
        "low_confidence_count":        sum(1 for c in confs if c < 0.5),
        "child_feedback_used_count":   sum(1 for r in annotated
                                           if r.get("llm_use_child_feedback")),
        "posthoc_context_used_count":  sum(1 for r in annotated
                                           if r.get("llm_context_setting") in ("posthoc", "both")),
        "causal_context_used_count":   sum(1 for r in annotated
                                           if r.get("llm_context_setting") == "causal"),
    }

    # 与规则版对比
    if rule_semantic:
        rule_map = {r.get("node_id"): r for r in rule_semantic}
        matched_rule = [rule_map[r.get("node_id")]
                        for r in annotated if r.get("node_id") in rule_map]
        matched_llm  = [r for r in annotated if r.get("node_id") in rule_map]

        rule_unknown = sum(1 for r in matched_rule
                           if r.get("reply_relation_candidate") == "unknown")
        llm_unknown  = sum(1 for r in matched_llm
                           if r.get("llm_reply_relation") == "unknown")
        rule_confs = [r.get("semantic_confidence", 0) for r in matched_rule]
        llm_confs  = [r.get("llm_confidence", 0)      for r in matched_llm]

        report.update({
            "rule_unknown_count_in_selected":     rule_unknown,
            "llm_unknown_count":                  llm_unknown,
            "unknown_reduced_by":                 rule_unknown - llm_unknown,
            "average_confidence_rule":
                round(sum(rule_confs) / len(rule_confs), 4) if rule_confs else 0,
            "average_confidence_llm":
                round(sum(llm_confs)  / len(llm_confs),  4) if llm_confs  else 0,
        })

    return report


# ============ 示例文件 ============

def write_examples(annotated: list, filepath: Path):
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
                if pat in safe_list(r.get("llm_candidate_patterns"))]

    add_batch(filter_pattern("agreement_with_parent"))
    add_batch(filter_pattern("disagreement_with_parent"))
    add_batch(filter_pattern("mocking_disagreement"))
    add_batch(filter_pattern("sarcastic_agreement"))
    add_batch(filter_pattern("child_feedback_disambiguation"))
    add_batch(filter_pattern("attribute_based_stance"))
    add_batch(filter_pattern("consequence_based_stance"))
    add_batch(filter_pattern("target_drift"))
    add_batch(sorted(annotated, key=lambda r: r.get("llm_confidence", 1))[:200])
    add_batch(sorted(annotated, key=lambda r: -r.get("llm_confidence", 0))[:200])

    with open(filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath.name}（{len(examples)} 条）")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="LLM 语义传导标注")
    parser.add_argument("--windowed_instances", required=True)
    parser.add_argument("--output_dir",         required=True)
    parser.add_argument("--mode",               default="llm", choices=["llm", "rule"])
    parser.add_argument("--provider",           default="openai",
                        choices=["openai", "deepseek", "qwen", "local"])
    parser.add_argument("--model",              default="gpt-4o-mini")
    parser.add_argument("--api_base",           default=None)
    parser.add_argument("--temperature",        type=float, default=0.1)
    parser.add_argument("--max_retries",        type=int,   default=3)
    parser.add_argument("--sleep_seconds",      type=float, default=1.5)
    parser.add_argument("--sample_limit",       type=int,   default=None)
    parser.add_argument("--only_split",         default=None)
    parser.add_argument("--only_ambiguous",     default="false")
    parser.add_argument("--only_with_child_feedback", default="false")
    parser.add_argument("--only_depth_ge",      type=int, default=None)
    parser.add_argument("--context_mode",       default="both",
                        choices=["causal", "posthoc", "both"])
    parser.add_argument("--resume",             default="true")
    parser.add_argument("--save_raw_response",  default="false")
    parser.add_argument("--input_rule_semantic", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    only_ambiguous          = args.only_ambiguous.lower() == "true"
    only_with_child_feedback = args.only_with_child_feedback.lower() == "true"
    resume                  = args.resume.lower() == "true"
    save_raw                = args.save_raw_response.lower() == "true"

    out_main    = output_dir / "mtcsd_semantic_instances_llm.jsonl"
    out_failed  = output_dir / "mtcsd_semantic_llm_failed.jsonl"
    out_report  = output_dir / "mtcsd_semantic_quality_report_llm.json"
    out_examples = output_dir / "mtcsd_semantic_llm_examples.jsonl"
    out_prompt  = output_dir / "llm_prompt_template.txt"
    cache_path  = output_dir / ".llm_prompt_cache.json"

    # 保存 prompt 模板
    with open(out_prompt, "w", encoding="utf-8") as f:
        f.write("=== SYSTEM PROMPT ===\n")
        f.write(SYSTEM_PROMPT + "\n\n")
        f.write("=== USER PROMPT TEMPLATE ===\n")
        f.write(USER_PROMPT_TEMPLATE)
    print(f"  ✅ {out_prompt.name}")

    # 加载数据
    print(f"📂 读取 windowed instances：{args.windowed_instances}")
    all_instances = load_jsonl(args.windowed_instances)
    print(f"  共加载 {len(all_instances)} 条")

    rule_semantic = None
    if args.input_rule_semantic:
        print(f"📂 读取规则版语义标注：{args.input_rule_semantic}")
        rule_semantic = load_jsonl(args.input_rule_semantic)
        print(f"  共加载 {len(rule_semantic)} 条")

    # 筛选
    selected = select_instances(
        all_instances,
        only_split=args.only_split,
        only_ambiguous=only_ambiguous,
        only_with_child_feedback=only_with_child_feedback,
        only_depth_ge=args.only_depth_ge,
        sample_limit=args.sample_limit,
        rule_semantic=rule_semantic,
    )
    print(f"  筛选后：{len(selected)} 条")

    # 断点续跑
    existing_ids = set()
    skipped = 0
    if resume and out_main.exists():
        existing_ids = load_existing_ids(out_main)
        skipped = len(existing_ids)
        print(f"  断点续跑：跳过已处理 {skipped} 条")

    # 加载 prompt 缓存
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"  加载缓存：{len(cache)} 条")
        except Exception:
            cache = {}

    # 处理
    annotated_list = []
    failed_list    = []
    to_process     = [r for r in selected if r.get("node_id") not in existing_ids]
    total_todo     = len(to_process)

    print(f"\n🤖 LLM 标注（provider={args.provider}，model={args.model}，"
          f"共 {total_todo} 条）...")

    for i, rec in enumerate(to_process):
        nid = rec.get("node_id", f"idx_{i}")
        if (i + 1) % 50 == 0:
            print(f"  进度：{i+1}/{total_todo}，成功={len(annotated_list)}，失败={len(failed_list)}")
            # 每 50 条保存一次缓存
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

        annotated_rec, error = annotate_one_instance(
            rec,
            provider=args.provider,
            model=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
            context_mode=args.context_mode,
            cache=cache,
            save_raw=save_raw,
        )

        if error:
            print(f"  ❌ {nid}：{error}")
            failed_rec = dict(rec)
            failed_rec["_error"] = error
            failed_list.append(failed_rec)
            write_jsonl_append(failed_rec, out_failed)
        else:
            annotated_list.append(annotated_rec)
            write_jsonl_append(annotated_rec, out_main)

        time.sleep(args.sleep_seconds)

    # 最终保存缓存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    # 报告和示例（基于本次新标注）
    print(f"\n📝 生成报告和示例...")
    report = generate_report(
        all_instances, selected, annotated_list,
        failed_list, skipped, rule_semantic
    )
    write_json(report, out_report)
    write_examples(annotated_list, out_examples)

    print(f"""
📊 标注摘要：
  总输入：        {len(all_instances)}
  本次筛选：      {len(selected)}
  跳过（已有）：  {skipped}
  成功标注：      {len(annotated_list)}
  失败：          {len(failed_list)}
  平均置信度：    {report['average_llm_confidence']}
  高置信（≥0.75）：{report['high_confidence_count']}
  低置信（<0.50）：{report['low_confidence_count']}
  使用子回复：    {report['child_feedback_used_count']}
    """)


if __name__ == "__main__":
    main()