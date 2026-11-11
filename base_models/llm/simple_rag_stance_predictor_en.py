#!/usr/bin/env python3
"""
simple_rag_stance_predictor.py

用法（小样本测试）：
python simple_rag_stance_predictor.py \
  --instances processed_cmtscd_semantic_llm/cmtscd_semantic_instances_llm.jsonl \
  --knowledge processed_cmtscd_rag_memory/rag_knowledge_documents_llm.jsonl \
  --output_dir rag_prediction_outputs \
  --split valid \
  --method full_rag \
  --top_k 5 \
  --provider openai \
  --model gpt-4o-mini \
  --sample_limit 300 \
  --resume true
"""

import json
import os
import re
import time
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# ============ 常量 ============

VALID_STANCES   = {"favor", "against", "none"}
VALID_METHODS   = [
    "current_only", "root_current", "parent_current", "full_path",
    "dynamic_window", "dynamic_semantic", "dynamic_rag",
    "full_rag", "full_rag_posthoc",
]

SYSTEM_PROMPT = """You are an expert in social media stance detection. Your task is to determine the stance of the current comment toward the main_target. You must distinguish between the current comment's stance toward the main_target, its attitude toward the parent comment, its attitude toward the reply author, and topic shift. You may only output JSON, nothing else."""

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
        raise FileNotFoundError(f"文件不存在：{filepath}")
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


def prompt_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ============ 加载知识库 ============

def load_knowledge_documents(
    filepath: str,
    include_low_reliability: bool = False,
    exclude_memory_types: list = None,
) -> list:
    docs = load_jsonl(filepath)
    exclude_memory_types = set(exclude_memory_types or [])
    filtered = []
    for doc in docs:
        meta = doc.get("structured_metadata", {})
        usage       = meta.get("suggested_usage", "")
        reliability = meta.get("reliability_level", "")
        weighted    = safe_float(meta.get("source_sample_count_weighted", 0))
        mtype       = doc.get("memory_type", "")

        if mtype in exclude_memory_types:
            continue

        if not include_low_reliability:
            if usage == "do_not_use":
                continue
            if reliability == "excluded":
                continue
            if weighted < 10:
                continue
        filtered.append(doc)
    print(f"  加载知识文档：{len(docs)} 条 → 过滤后 {len(filtered)} 条"
          f"{f'（已排除 memory_type={sorted(exclude_memory_types)}）' if exclude_memory_types else ''}")
    return filtered


# ============ 规范化 patterns ============

def normalize_patterns(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    result = []
    for item in items:
        for p in str(item).split("|"):
            p = p.strip().lower().replace(" ", "_")
            if p and p != "unknown_pattern":
                result.append(p)
    return list(dict.fromkeys(result))


# ============ 检索 ============

def build_query_metadata(rec: dict) -> dict:
    return {
        "main_target":             safe_str(rec.get("main_target")),
        "topic":                   safe_str(rec.get("topic")),
        "current_text":            safe_str(rec.get("current_text")),
        "parent_text":             safe_str(rec.get("parent_text")),
        "llm_reply_relation":      safe_str(rec.get("llm_reply_relation")),
        "llm_semantic_propagation": safe_str(rec.get("llm_semantic_propagation")),
        "llm_target_relation":     safe_str(rec.get("llm_target_relation")),
        "llm_candidate_patterns":  normalize_patterns(rec.get("llm_candidate_patterns")),
        "llm_stance_mapping":      safe_str(rec.get("llm_stance_mapping_to_parent")),
    }


def retrieve_memory_rule_hybrid(
    query: dict,
    docs: list,
    top_k: int,
) -> list:
    scored = []
    query_patterns  = set(query["llm_candidate_patterns"])
    query_prop      = query["llm_semantic_propagation"]
    query_reply     = query["llm_reply_relation"]
    query_target    = query["llm_target_relation"]
    query_words     = set(
        (query["current_text"] + " " + query["parent_text"]).split()
    )

    for doc in docs:
        meta  = doc.get("structured_metadata", {})
        pname = safe_str(doc.get("pattern_name") or doc.get("composite_key", ""))
        retrieval_text = safe_str(doc.get("retrieval_text", ""))

        # ---- 第一步：先算"内容相关性"分数，完全不看可靠性 ----
        relevance = 0.0

        # candidate pattern 匹配
        for qp in query_patterns:
            if qp in pname:
                relevance += 3

        # semantic_propagation 匹配
        if query_prop and query_prop in retrieval_text:
            relevance += 2

        # reply_relation 匹配
        if query_reply and query_reply in retrieval_text:
            relevance += 2

        # target_relation 匹配
        if query_target and query_target in retrieval_text:
            relevance += 2

        # 词面重合
        doc_words = set(retrieval_text.split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            relevance += min(overlap * 0.1, 0.5)

        # ---- 第二步：只有 relevance > 0（确实有内容关联）才允许可靠性
        # 参与调节分数——避免"可靠性高但跟当前评论毫无关系"的文档
        # 单靠可靠性加分就挤进 top_k，把真正主题相关的记忆挤出去。
        # 这是之前那版的 bug：可靠性被当成了相关性的替代品。
        score = relevance
        if relevance > 0:
            reliability = meta.get("reliability_level", "")
            if reliability == "high":
                score += 1.5
            elif reliability == "medium":
                score += 0.5
            elif reliability == "low":
                score -= 1
            if meta.get("suggested_usage") == "use_as_strong_rag_memory":
                score += 1
            elif meta.get("suggested_usage") == "do_not_use":
                score -= 5

        scored.append((score, doc))

    scored.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored[:top_k] if _ > 0] or [doc for _, doc in scored[:top_k]]


def retrieve_memory_tfidf(
    query: dict,
    docs: list,
    top_k: int,
    tfidf_index: Optional[dict] = None,
) -> list:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("  ⚠️  sklearn 未安装，回退到 rule_hybrid")
        return retrieve_memory_rule_hybrid(query, docs, top_k)

    query_text = (
        f"target: {query['main_target']} "
        f"current_comment: {query['current_text']} "
        f"parent_comment: {query['parent_text']} "
        f"reply_relation: {query['llm_reply_relation']} "
        f"semantic_propagation: {query['llm_semantic_propagation']} "
        f"target_relation: {query['llm_target_relation']} "
        f"candidate_patterns: {' '.join(query['llm_candidate_patterns'])}"
    )

    corpus = [safe_str(doc.get("retrieval_text", "")) for doc in docs]
    if not corpus:
        return []

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    try:
        tfidf_matrix = vec.fit_transform(corpus)
        q_vec = vec.transform([query_text])
        scores = cosine_similarity(q_vec, tfidf_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [docs[i] for i in top_indices]
    except Exception as e:
        print(f"  ⚠️  TF-IDF 检索失败：{e}，回退到 rule_hybrid")
        return retrieve_memory_rule_hybrid(query, docs, top_k)


def retrieve_memory_hybrid(
    query: dict,
    docs: list,
    top_k: int,
) -> list:
    rule_docs  = retrieve_memory_rule_hybrid(query, docs, top_k * 2)
    tfidf_docs = retrieve_memory_tfidf(query, docs, top_k * 2)

    seen = {}
    for doc in rule_docs:
        did = doc.get("doc_id", "")
        seen[did] = seen.get(did, 0) + 2
    for doc in tfidf_docs:
        did = doc.get("doc_id", "")
        seen[did] = seen.get(did, 0) + 1

    all_docs = {doc.get("doc_id", ""): doc for doc in rule_docs + tfidf_docs}
    ranked = sorted(seen.items(), key=lambda x: -x[1])
    return [all_docs[did] for did, _ in ranked[:top_k] if did in all_docs]


def retrieve_documents(
    query: dict,
    docs: list,
    top_k: int,
    mode: str = "rule_hybrid",
) -> list:
    if mode == "tfidf":
        return retrieve_memory_tfidf(query, docs, top_k)
    elif mode == "hybrid":
        return retrieve_memory_hybrid(query, docs, top_k)
    return retrieve_memory_rule_hybrid(query, docs, top_k)


# ============ Prompt 构建 ============

def _format_memory_block(retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "(no relevant pattern memory)"
    lines = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta  = doc.get("structured_metadata", {})
        pname = doc.get("pattern_name") or doc.get("composite_key", "")
        dom   = meta.get("dominant_label", "")
        dom_p = meta.get("dominant_label_probability", "")
        dist  = meta.get("label_distribution_weighted", {})
        rel   = meta.get("reliability_level", "")
        dist_str = "  ".join(f"{k}:{round(v*100,1)}%" for k, v in dist.items())
        lines.append(
            f"{i}. [{doc.get('doc_id','')}] pattern: {pname}\n"
            f"   reliability: {rel}  dominant_label: {dom}({round(safe_float(dom_p)*100,1)}%)\n"
            f"   distribution: {dist_str}"
        )
    return "\n".join(lines)


def build_prediction_prompt(
    rec: dict,
    method: str,
    retrieved_docs: list,
) -> str:
    main_target  = safe_str(rec.get("main_target"))
    current_text = safe_str(rec.get("current_text"))
    parent_text  = safe_str(rec.get("parent_text"))
    root_text    = safe_str(rec.get("root_text"))
    target_anchor = safe_str(rec.get("target_anchor"))
    causal_ctx   = safe_str(rec.get("causal_context_text"))
    posthoc_ctx  = safe_str(rec.get("posthoc_context_text"))
    ancestor_texts = safe_list(rec.get("ancestor_texts", []))

    # ---- context block ----
    if method == "current_only":
        context_block = f"Current comment: {current_text}"

    elif method == "root_current":
        context_block = (
            f"Target anchor: {target_anchor}\n"
            f"Post: {root_text}\n"
            f"Current comment: {current_text}"
        )

    elif method == "parent_current":
        context_block = (
            f"Parent comment: {parent_text}\n"
            f"Current comment: {current_text}"
        )

    elif method == "full_path":
        path_str = "\n".join(
            f"[depth {i}] {safe_str(t)}"
            for i, t in enumerate(ancestor_texts)
        )
        context_block = f"Full conversation path:\n{path_str}\nCurrent comment: {current_text}"

    elif method in ("dynamic_window", "dynamic_semantic", "dynamic_rag", "full_rag"):
        context_block = f"Dynamic conversation window:\n{causal_ctx}"

    elif method == "full_rag_posthoc":
        down_texts = safe_list(rec.get("downward_feedback_window_texts"))
        down_str   = "\n".join(f"- {t}" for t in down_texts) if down_texts else "(no child replies)"
        context_block = (
            f"Dynamic conversation window (with child replies):\n{posthoc_ctx}\n\n"
            f"Child-reply feedback (offline):\n{down_str}"
        )
    else:
        context_block = f"Current comment: {current_text}"

    # ---- semantic propagation block ----
    SEMANTIC_METHODS = {
        "dynamic_semantic", "full_rag", "full_rag_posthoc"
    }
    if method in SEMANTIC_METHODS:
        patterns_str = ", ".join(normalize_patterns(rec.get("llm_candidate_patterns"))) or "unknown"
        semantic_block = (
            f"reply_relation: {safe_str(rec.get('llm_reply_relation'))}\n"
            f"semantic_propagation: {safe_str(rec.get('llm_semantic_propagation'))}\n"
            f"stance_mapping_to_parent: {safe_str(rec.get('llm_stance_mapping_to_parent'))}\n"
            f"target_relation: {safe_str(rec.get('llm_target_relation'))}\n"
            f"candidate_patterns: {patterns_str}"
        )
        if method == "full_rag_posthoc":
            semantic_block += f"\nchild_feedback_signal: {safe_str(rec.get('llm_child_feedback_signal'))}"
    else:
        semantic_block = "(this method does not use semantic propagation analysis)"

    # ---- RAG block ----
    RAG_METHODS = {"dynamic_rag", "full_rag", "full_rag_posthoc"}
    if method in RAG_METHODS and retrieved_docs:
        memory_block = _format_memory_block(retrieved_docs)
    else:
        memory_block = "(this method does not use RAG memory)"

    user_prompt = f"""Determine the stance of the current comment below toward main_target.

main_target:
{main_target}

Input context:
{context_block}

Semantic propagation analysis:
{semantic_block}

Structured language-pattern memory retrieved by RAG:
{memory_block}

Important requirements:
1. Choose exactly one of favor, against, none.
2. Do not lazily default to none just because the target isn't explicitly repeated or the context feels thin. But the reverse matters just as much: if, after careful consideration, the comment still does not clearly express a favor or against stance toward main_target, output none. semantic_propagation and RAG memory below are aids for interpreting an existing stance, not a mandate to always produce a directional label.
3. If the current comment is a short backchannel token (e.g. "true", "lol", "lmao", "facts"), you must judge it together with the parent comment or the conversation window.
4. semantic_propagation describes how the current comment's tone relates to the parent comment, not a rule that forces a directional stance. If semantic_propagation is stance_inheritance, this only means that IF the current comment expresses a stance at all, it likely aligns with the parent's stance direction — it does NOT mean you should assign favor/against when the comment itself expresses no clear stance toward main_target.
5. Likewise, if semantic_propagation is stance_reversal or sarcastic_reversal, this only means that IF the comment expresses a stance, it likely opposes the parent's — it does not license inventing a stance where none is clearly expressed.
6. When target_relation is reply_author_only or topic_shift, be cautious when inferring the stance toward main_target; this is often genuine evidence for none, not just a reason for caution.
7. RAG memory is historical reference, not an absolute rule; if a counter-example or caution applies, use it cautiously.
8. Do not assume any existing label is correct; judge independently.
9. semantic_propagation and RAG memory can help you interpret an ambiguous but stance-bearing comment; they must never be the sole basis for converting a comment with no clear stance expression into favor or against.

Please output strictly in the following JSON format:
{{
  "predicted_stance": "favor|against|none",
  "confidence": 0.0,
  "reason_short": "no more than 40 words",
  "used_memory_ids": [],
  "prediction_basis": {{
    "used_context": true,
    "used_semantic_propagation": false,
    "used_rag_memory": false,
    "used_child_feedback": false
  }}
}}"""
    return user_prompt


# ============ LLM 调用 ============

def call_llm(
    prompt: str,
    provider: str,
    model: str,
    api_base: Optional[str] = None,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
    temperature: float = 0.1,
) -> str:
    if provider in ("openai", "deepseek", "qwen"):
        try:
            import openai
        except ImportError:
            raise ImportError("请先安装：pip install openai")

        cfg     = API_CONFIGS.get(provider, {})
        api_key = cfg.get("api_key") or os.environ.get({
            "openai":   "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen":     "DASHSCOPE_API_KEY",
        }.get(provider, ""))

        if not api_key:
            raise EnvironmentError(
                f"未找到 API Key，请设置环境变量或在脚本顶部 API_CONFIGS 中填写。"
            )

        base_url = api_base or cfg.get("base_url")
        actual_model = model or cfg.get("model", "gpt-4o-mini")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=actual_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=400,
                    timeout=45,
                )
                return resp.choices[0].message.content.strip()
            except openai.RateLimitError:
                wait = sleep_seconds * (attempt + 1) * 3
                print(f"  ⏳ 限速，等待 {wait:.1f}s...")
                time.sleep(wait)
            except openai.APITimeoutError:
                time.sleep(sleep_seconds)
            except openai.AuthenticationError:
                raise EnvironmentError("API Key 无效，请检查配置。")
            except Exception as e:
                print(f"  ⚠️  调用异常（第 {attempt+1} 次）：{e}")
                if attempt < max_retries - 1:
                    time.sleep(sleep_seconds)
                else:
                    raise
    elif provider == "local":
        raise NotImplementedError("local 模式请自行实现 call_llm() 中的本地推理接口。")
    else:
        raise ValueError(f"不支持的 provider：{provider}")
    raise RuntimeError("LLM 调用失败")


# ============ 解析预测 JSON ============

def parse_prediction_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"JSON 解析失败：{raw_text[:200]}")


# ============ 单条预测 ============

def predict_one_instance(
    rec: dict,
    docs: list,
    method: str,
    top_k: int,
    retrieval_mode: str,
    provider: str,
    model: str,
    api_base: Optional[str],
    max_retries: int,
    sleep_seconds: float,
    save_raw: bool = False,
) -> tuple:
    RAG_METHODS = {"dynamic_rag", "full_rag", "full_rag_posthoc"}
    retrieved_docs = []

    if method in RAG_METHODS and docs:
        query = build_query_metadata(rec)
        retrieved_docs = retrieve_documents(query, docs, top_k, retrieval_mode)

    prompt = build_prediction_prompt(rec, method, retrieved_docs)
    phash  = prompt_hash(prompt)

    for attempt in range(max_retries):
        try:
            raw = call_llm(prompt, provider, model, api_base,
                           max_retries=1, sleep_seconds=sleep_seconds)
            result = parse_prediction_json(raw)

            predicted = safe_str(result.get("predicted_stance", "")).lower()
            if predicted not in VALID_STANCES:
                predicted = "none"

            out = {
                "dataset":          safe_str(rec.get("dataset")),
                "topic":            safe_str(rec.get("topic")),
                "split":            safe_str(rec.get("split")),
                "node_id":          safe_str(rec.get("node_id")),
                "main_target":      safe_str(rec.get("main_target")),
                "current_text":     safe_str(rec.get("current_text")),
                "parent_text":      safe_str(rec.get("parent_text")),
                "depth":            rec.get("depth"),
                "stance_label":     safe_str(rec.get("stance_label")),
                "stance_label_zh":  safe_str(rec.get("stance_label_zh")),
                "is_ambiguous_current": rec.get("is_ambiguous_current", False),
                "llm_reply_relation":   safe_str(rec.get("llm_reply_relation")),
                "llm_semantic_propagation": safe_str(rec.get("llm_semantic_propagation")),
                "llm_target_relation":  safe_str(rec.get("llm_target_relation")),
                "llm_candidate_patterns": safe_list(rec.get("llm_candidate_patterns")),
                "llm_use_child_feedback": rec.get("llm_use_child_feedback", False),
                "method":           method,
                "retrieved_memory_ids": [d.get("doc_id") for d in retrieved_docs],
                "retrieved_memory_summaries": [
                    {
                        "doc_id":       d.get("doc_id"),
                        "pattern_name": d.get("pattern_name") or d.get("composite_key"),
                        "dominant_label": d.get("structured_metadata", {}).get("dominant_label"),
                        "reliability":    d.get("structured_metadata", {}).get("reliability_level"),
                    }
                    for d in retrieved_docs
                ],
                "predicted_stance": predicted,
                "confidence":       safe_float(result.get("confidence", 0.5)),
                "reason_short":     safe_str(result.get("reason_short", ""))[:120],
                "prediction_basis": result.get("prediction_basis", {}),
                "prompt_hash":      phash,
            }
            if save_raw:
                out["raw_llm_response"] = raw
            return out, None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                return rec, str(e)

    return rec, "max_retries_exceeded"


# ============ 样本筛选 ============

def select_instances(
    records: list,
    split: str,
    sample_limit: Optional[int],
    only_topic: Optional[str],
    only_depth_ge: Optional[int],
    only_ambiguous: bool,
    only_child_feedback: bool,
) -> list:
    sel = [r for r in records if r.get("split") == split]

    if only_topic:
        sel = [r for r in sel if r.get("topic") == only_topic]
    if only_depth_ge is not None:
        sel = [r for r in sel if (r.get("depth") or 0) >= only_depth_ge]
    if only_ambiguous:
        sel = [r for r in sel if r.get("is_ambiguous_current")]
    if only_child_feedback:
        sel = [r for r in sel if r.get("llm_use_child_feedback")]
    if sample_limit:
        sel = sel[:sample_limit]

    return sel


# ============ 运行报告 ============

def generate_prediction_report(
    predictions: list,
    failed: list,
    selected_count: int,
    method: str,
    split: str,
) -> dict:
    total = len(predictions)
    stance_dist = Counter(r.get("predicted_stance") for r in predictions)
    topic_dist  = Counter(r.get("topic") for r in predictions)
    rag_used    = sum(1 for r in predictions if r.get("retrieved_memory_ids"))

    return {
        "method":           method,
        "split":            split,
        "total_selected":   selected_count,
        "total_predicted":  total,
        "total_failed":     len(failed),
        "predicted_stance_distribution": dict(stance_dist),
        "topic_distribution":            dict(topic_dist),
        "rag_memory_used_count":         rag_used,
        "average_confidence": round(
            sum(safe_float(r.get("confidence")) for r in predictions) / max(total, 1), 4
        ),
    }


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="RAG 增强立场预测")
    parser.add_argument("--instances",         required=True)
    parser.add_argument("--knowledge",         required=True)
    parser.add_argument("--output_dir",        required=True)
    parser.add_argument("--split",             default="valid")
    parser.add_argument("--method",            default="full_rag", choices=VALID_METHODS)
    parser.add_argument("--top_k",             type=int,   default=5)
    parser.add_argument("--provider",          default="openai")
    parser.add_argument("--model",             default="gpt-4o-mini")
    parser.add_argument("--api_base",          default=None)
    parser.add_argument("--temperature",       type=float, default=0.1)
    parser.add_argument("--max_retries",       type=int,   default=3)
    parser.add_argument("--sleep_seconds",     type=float, default=1.0)
    parser.add_argument("--sample_limit",      type=int,   default=None)
    parser.add_argument("--retrieval_mode",    default="rule_hybrid",
                        choices=["rule_hybrid", "tfidf", "hybrid"])
    parser.add_argument("--include_low_reliability", default="false")
    parser.add_argument("--exclude_memory_types", default="",
                         help="逗号分隔，比如 semantic_propagation，用于做消融测试排除某类记忆")
    parser.add_argument("--only_topic",        default=None)
    parser.add_argument("--only_depth_ge",     type=int, default=None)
    parser.add_argument("--only_ambiguous",    default="false")
    parser.add_argument("--only_child_feedback", default="false")
    parser.add_argument("--resume",            default="true")
    parser.add_argument("--save_raw",          default="false")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_low = args.include_low_reliability.lower() == "true"
    resume      = args.resume.lower() == "true"
    save_raw    = args.save_raw.lower() == "true"
    only_ambig  = args.only_ambiguous.lower() == "true"
    only_child  = args.only_child_feedback.lower() == "true"

    suffix = f"{args.method}_{args.split}"
    out_pred   = output_dir / f"predictions_{suffix}.jsonl"
    out_failed = output_dir / f"failed_predictions_{suffix}.jsonl"
    out_report = output_dir / f"prediction_run_report_{suffix}.json"
    out_prompt = output_dir / f"prompt_template_{args.method}.txt"

    # 保存 prompt 模板示例
    with open(out_prompt, "w", encoding="utf-8") as f:
        f.write(SYSTEM_PROMPT + "\n\n---\n")
        f.write(build_prediction_prompt(
            {"main_target": "{main_target}", "current_text": "{current_text}",
             "parent_text": "{parent_text}", "root_text": "{root_text}",
             "target_anchor": "{target_anchor}", "causal_context_text": "{causal_context_text}",
             "posthoc_context_text": "{posthoc_context_text}"},
            args.method, []
        ))

    print(f"📂 读取样本：{args.instances}")
    all_records = load_jsonl(args.instances)

    print(f"📂 读取知识库：{args.knowledge}")
    exclude_types = [t.strip() for t in args.exclude_memory_types.split(",") if t.strip()]
    docs = load_knowledge_documents(args.knowledge, include_low, exclude_types)

    selected = select_instances(
        all_records, args.split, args.sample_limit,
        args.only_topic, args.only_depth_ge, only_ambig, only_child
    )
    print(f"  筛选后：{len(selected)} 条（split={args.split}）")

    # 断点续跑
    existing_ids = set()
    skipped = 0
    if resume and out_pred.exists():
        existing = load_jsonl(str(out_pred))
        existing_ids = {r.get("node_id") for r in existing}
        skipped = len(existing_ids)
        print(f"  断点续跑：跳过 {skipped} 条")

    predictions, failed_list = [], []
    to_process = [r for r in selected if r.get("node_id") not in existing_ids]
    total = len(to_process)

    print(f"\n🤖 开始预测（method={args.method}，共 {total} 条）...")

    for i, rec in enumerate(to_process):
        if (i + 1) % 50 == 0:
            print(f"  进度：{i+1}/{total}，成功={len(predictions)}，失败={len(failed_list)}")

        result, error = predict_one_instance(
            rec, docs, args.method, args.top_k, args.retrieval_mode,
            args.provider, args.model, args.api_base,
            args.max_retries, args.sleep_seconds, save_raw
        )

        if error:
            print(f"  ❌ {rec.get('node_id','?')}：{error}")
            fail_rec = dict(rec)
            fail_rec["_error"] = error
            failed_list.append(fail_rec)
            write_jsonl_append(fail_rec, out_failed)
        else:
            predictions.append(result)
            write_jsonl_append(result, out_pred)

        time.sleep(args.sleep_seconds)

    report = generate_prediction_report(
        predictions, failed_list, len(selected), args.method, args.split
    )
    write_json(report, out_report)

    print(f"""
✅ 预测完成：
  成功：{len(predictions)}  失败：{len(failed_list)}  跳过：{skipped}
  输出：{out_pred}
    """)


if __name__ == "__main__":
    main()