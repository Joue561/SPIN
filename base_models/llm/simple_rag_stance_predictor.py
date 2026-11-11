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
    # 新增：相对记忆（只存“相对父评论的立场变换”，不含话题立场先验）
    "relative_reasoning",   # 两步推理（先判父评论立场，再判变换），不使用记忆 —— 用于分离“推理方式”的效果
    "full_rag_relative",    # 两步推理 + 相对记忆（knowledge 需为 rag_knowledge_documents_relative.jsonl）
    # 新增基线
    "knn_fewshot",          # 通用 RAG：检索同话题 train 中最相似的已标注样本作为 few-shot 示例（检索样本，而非模式）
    "cot",                  # 零样本思维链：动态窗口 + 先逐步分析再给标签，不用语义传导和记忆
    # 冲突感知闭环：对知识库推断后仍不确定的节点做定向复核（配合 --id_file 只跑挑中的节点）
    "relation_verify",      # 问"与父评论的立场关系"，并给出同一传导模式下结果相反的训练示例
]
RELATIVE_METHODS = {"relative_reasoning", "full_rag_relative"}
REL_SHOW_EXAMPLES = False   # 由 --relative_memory_examples 控制；默认与 full_rag 一样不展示例子，保证对比干净
RAG_METHOD_SET = {"dynamic_rag", "full_rag", "full_rag_posthoc", "full_rag_relative"}
FEWSHOT_INDEX = {}      # knn_fewshot：{topic: (vectorizer, matrix, train_records)}，在 main 中构建
FEWSHOT_K = 5
LLM_MAX_TOKENS = 400     # 由 --max_tokens 设置
LLM_TIMEOUT = 45         # 由 --timeout 设置（秒）
CONTRAST_INDEX = {}     # relation_verify：{(topic, level_key): {relation: [train 记录]}}
CONTRAST_TFIDF = {}     # relation_verify：{topic: (vectorizer, {id(rec): row})}
REL_ZH = {"inherit": "继承父评论立场", "reverse": "反转父评论立场", "neutralize": "不表态（中立化）"}

SYSTEM_PROMPT = """你是一个社交媒体立场检测专家。你的任务是判断当前评论相对于 main_target 的立场。你必须区分当前评论对 main_target 的立场、对父评论的态度、对回复作者的态度以及话题转移。你只能输出 JSON，不要输出 JSON 以外的内容。"""

API_CONFIGS = {
    "openai": {
        "api_key":  None,   # 填你的 OpenAI Key
        "base_url": "https://api.zhizengzeng.com/v1/chat/completions",     # 国内代理改这里
        "model":    "gpt-4o-mini",                   # 默认模型
    },
    "deepseek": {
        "api_key":  None,   # 填你的 DeepSeek Key
        "base_url": "https://api.zhizengzeng.com/v1/chat/completions",
        "model":    "deepseek-chat",
    },
    "qwen": {
        "api_key":  None,   # 填你的通义千问 Key
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
) -> list:
    docs = load_jsonl(filepath)
    filtered = []
    for doc in docs:
        meta = doc.get("structured_metadata", {})
        usage       = meta.get("suggested_usage", "")
        reliability = meta.get("reliability_level", "")
        weighted    = safe_float(meta.get("source_sample_count_weighted", 0))

        if not include_low_reliability:
            if usage == "do_not_use":
                continue
            if reliability == "excluded":
                continue
            if weighted < 10:
                continue
        filtered.append(doc)
    print(f"  加载知识文档：{len(docs)} 条 → 过滤后 {len(filtered)} 条")
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
        score = 0.0
        meta  = doc.get("structured_metadata", {})
        pname = safe_str(doc.get("pattern_name") or doc.get("composite_key", ""))
        retrieval_text = safe_str(doc.get("retrieval_text", ""))

        # candidate pattern 匹配
        for qp in query_patterns:
            if qp in pname:
                score += 3

        # semantic_propagation 匹配
        if query_prop and query_prop in retrieval_text:
            score += 2

        # reply_relation 匹配
        if query_reply and query_reply in retrieval_text:
            score += 2

        # target_relation 匹配
        if query_target and query_target in retrieval_text:
            score += 2

        # reliability bonus
        if meta.get("reliability_level") == "high":
            score += 1
        if meta.get("suggested_usage") == "use_as_strong_rag_memory":
            score += 1

        # 词面重合
        doc_words = set(retrieval_text.split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            score += min(overlap * 0.1, 0.5)

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
        f"目标：{query['main_target']} "
        f"当前评论：{query['current_text']} "
        f"父评论：{query['parent_text']} "
        f"回复关系：{query['llm_reply_relation']} "
        f"语义传导：{query['llm_semantic_propagation']} "
        f"目标关系：{query['llm_target_relation']} "
        f"候选模式：{' '.join(query['llm_candidate_patterns'])}"
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
        return "（无相关模式记忆）"
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
            f"{i}. [{doc.get('doc_id','')}] 模式：{pname}\n"
            f"   可靠性：{rel}  主导标签：{dom}({round(safe_float(dom_p)*100,1)}%)\n"
            f"   分布：{dist_str}"
        )
    return "\n".join(lines)


def _format_relative_memory_block(retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "（无相关模式记忆）"
    lines = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.get("structured_metadata", {})
        pname = doc.get("pattern_name", "")
        rd = meta.get("relation_distribution_parent_polar", {}) or {}
        nd = meta.get("relation_distribution_parent_none", {}) or {}
        rel_str = "  ".join(f"{REL_ZH.get(k, k)}:{round(safe_float(v)*100,1)}%" for k, v in rd.items())
        nd_str = (f"自行表态:{round(safe_float(nd.get('express'))*100,1)}%  "
                  f"也不表态:{round(safe_float(nd.get('stay_none'))*100,1)}%")
        ex = (meta.get("examples") or [])[:1] if REL_SHOW_EXAMPLES else []
        ex_str = ""
        if ex:
            e = ex[0]
            ex_str = (f"\n   例：父「{safe_str(e.get('parent_text'))[:50]}」→ 当前「{safe_str(e.get('current_text'))[:50]}」"
                      f"（{REL_ZH.get(e.get('relation'), e.get('relation'))}）")
        lines.append(
            f"{i}. [{doc.get('doc_id','')}] 模式：{pname}\n"
            f"   可靠性：{meta.get('reliability_level','')}  样本：父有立场 {meta.get('n_parent_polar',0)} / 父无立场 {meta.get('n_parent_none',0)}\n"
            f"   父评论有立场时：{rel_str}\n"
            f"   父评论无立场时：{nd_str}{ex_str}"
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

    # ---- 上下文块 ----
    if method == "current_only":
        context_block = f"当前评论：{current_text}"

    elif method == "root_current":
        context_block = (
            f"目标锚点：{target_anchor}\n"
            f"原帖：{root_text}\n"
            f"当前评论：{current_text}"
        )

    elif method == "parent_current":
        context_block = (
            f"父评论：{parent_text}\n"
            f"当前评论：{current_text}"
        )

    elif method == "full_path":
        path_str = "\n".join(
            f"[层级{i}] {safe_str(t)}"
            for i, t in enumerate(ancestor_texts)
        )
        context_block = f"完整会话路径：\n{path_str}\n当前评论：{current_text}"

    elif method in ("dynamic_window", "dynamic_semantic", "dynamic_rag", "full_rag",
                    "relative_reasoning", "full_rag_relative", "knn_fewshot", "cot", "relation_verify"):
        context_block = f"动态会话窗口：\n{causal_ctx}"

    elif method == "full_rag_posthoc":
        down_texts = safe_list(rec.get("downward_feedback_window_texts"))
        down_str   = "\n".join(f"- {t}" for t in down_texts) if down_texts else "（无子回复）"
        context_block = (
            f"动态会话窗口（含子回复）：\n{posthoc_ctx}\n\n"
            f"子回复反馈（离线）：\n{down_str}"
        )
    else:
        context_block = f"当前评论：{current_text}"

    # ---- 语义传导块 ----
    SEMANTIC_METHODS = {
        "dynamic_semantic", "full_rag", "full_rag_posthoc",
        "relative_reasoning", "full_rag_relative",
    }
    if method in SEMANTIC_METHODS:
        patterns_str = ", ".join(normalize_patterns(rec.get("llm_candidate_patterns"))) or "未知"
        semantic_block = (
            f"回复关系：{safe_str(rec.get('llm_reply_relation'))}\n"
            f"语义传导：{safe_str(rec.get('llm_semantic_propagation'))}\n"
            f"父子立场映射：{safe_str(rec.get('llm_stance_mapping_to_parent'))}\n"
            f"目标关系：{safe_str(rec.get('llm_target_relation'))}\n"
            f"候选模式：{patterns_str}"
        )
        if method == "full_rag_posthoc":
            semantic_block += f"\n子回复信号：{safe_str(rec.get('llm_child_feedback_signal'))}"
    else:
        semantic_block = "（本方法不使用语义传导分析）"

    # ---- RAG 块 ----
    if method == "full_rag_relative" and retrieved_docs:
        memory_block = _format_relative_memory_block(retrieved_docs)
    elif method in RAG_METHOD_SET and retrieved_docs:
        memory_block = _format_memory_block(retrieved_docs)
    else:
        memory_block = "（本方法不使用 RAG 记忆）"

    if method in RELATIVE_METHODS:
        return _build_relative_prompt(main_target, context_block, semantic_block, memory_block,
                                      use_memory=(method == "full_rag_relative"))
    if method == "knn_fewshot":
        return _build_fewshot_prompt(main_target, context_block, retrieved_docs)
    if method == "cot":
        return _build_cot_prompt(main_target, context_block)
    if method == "relation_verify":
        return _build_verify_prompt(main_target, rec, context_block, retrieved_docs)

    user_prompt = f"""请判断下面当前评论相对于 main_target 的立场。

main_target：
{main_target}

输入上下文：
{context_block}

语义传导分析：
{semantic_block}

RAG 检索到的结构化语言模式记忆：
{memory_block}

重要要求：
1. 只能在 favor、against、none 中选择一个。
2. 不要把"无关""目标不清""上下文不足"直接当成 none；根据是否能映射回 main_target 判断。
3. 如果当前评论是短文本（如"确实""呵呵""666""笑死"），必须结合父评论或会话窗口判断。
4. 语义传导为 stance_inheritance 时，通常倾向继承父评论立场，但要结合 target_relation 判断。
5. 语义传导为 stance_reversal 或 sarcastic_reversal 时，通常倾向反转父评论立场。
6. target_relation 为 reply_author_only 或 topic_shift 时，应谨慎判断对 main_target 的立场。
7. RAG 记忆是历史参考，不是绝对规则；如反例或 caution 适用，请谨慎使用。
8. 不要假设已有标签正确，独立判断。

请严格按以下 JSON 输出：
{{
  "predicted_stance": "favor|against|none",
  "confidence": 0.0,
  "reason_short": "不超过80字",
  "used_memory_ids": [],
  "prediction_basis": {{
    "used_context": true,
    "used_semantic_propagation": false,
    "used_rag_memory": false,
    "used_child_feedback": false
  }}
}}"""
    return user_prompt


def _build_relative_prompt(main_target, context_block, semantic_block, memory_block, use_memory):
    mem_title = "RAG 检索到的相对立场变换记忆" if use_memory else "RAG 检索到的结构化语言模式记忆"
    mem_rule = ("7. 相对变换记忆只描述“当前评论相对父评论”的历史变换比例，来自其他评论，不含任何话题的立场倾向；"
                "它是参考，不是绝对规则，与当前语境不符时应以语境为准。\n8. 不要假设已有标签正确，独立判断。"
                if use_memory else "7. 不要假设已有标签正确，独立判断。")
    return f"""请判断下面当前评论相对于 main_target 的立场。

main_target：
{main_target}

输入上下文：
{context_block}

语义传导分析：
{semantic_block}

{mem_title}：
{memory_block}

请按以下步骤推理：
步骤 1：判断父评论对 main_target 的立场（favor / against / none）。如果没有父评论或父评论就是原帖，判断原帖对 main_target 的立场。
步骤 2：判断当前评论相对父评论的立场变换：
  - inherit：当前评论对 main_target 的立场与父评论相同；
  - reverse：当前评论对 main_target 的立场与父评论相反；
  - neutralize：当前评论对 main_target 不表态；
  - independent：父评论无立场，当前评论自行表达立场。
步骤 3：由父评论立场和变换得到当前评论的立场；如果当前评论直接、明确地表达了对 main_target 的立场，以直接表达为准。

重要要求：
1. 只能在 favor、against、none 中选择一个。
2. 不要把"无关""目标不清""上下文不足"直接当成 none；根据是否能映射回 main_target 判断。
3. 如果当前评论是短文本（如"确实""呵呵""666""笑死"），必须结合父评论或会话窗口判断。
4. 注意：反驳父评论的观点、论据或细节，不等于对 main_target 的立场相反。两人可能对 main_target 持相同立场，只是在原因或细节上争论；此时应为 inherit。
5. 讽刺、反问、表面赞同实际否定时，要识别真实语用含义后再判断变换。
6. target_relation 为 reply_author_only 或 topic_shift 时，应谨慎判断对 main_target 的立场。
{mem_rule}

请严格按以下 JSON 输出：
{{
  "parent_stance": "favor|against|none",
  "relation_to_parent": "inherit|reverse|neutralize|independent",
  "predicted_stance": "favor|against|none",
  "confidence": 0.0,
  "reason_short": "不超过80字",
  "used_memory_ids": [],
  "prediction_basis": {{
    "used_context": true,
    "used_semantic_propagation": false,
    "used_rag_memory": false,
    "used_child_feedback": false
  }}
}}"""


# ============ 新增基线：通用 RAG（相似样本 few-shot）与零样本 CoT ============

def _fewshot_text(r: dict) -> str:
    return f"{safe_str(r.get('parent_text'))[:150]} || {safe_str(r.get('current_text'))[:200]}"


def build_fewshot_index(all_records: list, topics: set):
    """每个话题只用该话题 train 中有标签的样本建索引；字符 1–2 gram TF-IDF（适用于中文）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    for t in topics:
        tr = [r for r in all_records if r.get("split") == "train" and r.get("topic") == t
              and safe_str(r.get("stance_label")).lower() in VALID_STANCES]
        if not tr:
            continue
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        mat = vec.fit_transform([_fewshot_text(r) for r in tr])
        FEWSHOT_INDEX[t] = (vec, mat, tr)
        print(f"  few-shot 索引：{t} train {len(tr)} 条")


def retrieve_fewshot(rec: dict, k: int) -> list:
    idx = FEWSHOT_INDEX.get(rec.get("topic"))
    if not idx:
        return []
    vec, mat, tr = idx
    sims = (mat @ vec.transform([_fewshot_text(rec)]).T).toarray().ravel()
    order = sims.argsort()[::-1]
    out, seen = [], set()
    for i in order:
        r = tr[i]
        key = (safe_str(r.get("node_id")), safe_str(r.get("current_text")))
        if key in seen:
            continue
        seen.add(key)
        out.append({"doc_id": f"train::{r.get('node_id')}", "parent_text": r.get("parent_text"),
                    "current_text": r.get("current_text"), "stance_label": safe_str(r.get("stance_label")).lower(),
                    "similarity": round(float(sims[i]), 4)})
        if len(out) >= k:
            break
    return out


def _build_fewshot_prompt(main_target, context_block, examples):
    if examples:
        demo = "\n\n".join(
            f"示例{i + 1}：\n父评论：{safe_str(e.get('parent_text'))[:150] or '（无）'}\n"
            f"当前评论：{safe_str(e.get('current_text'))[:200]}\n立场：{e.get('stance_label')}"
            for i, e in enumerate(examples))
    else:
        demo = "（无）"
    return f"""请判断下面当前评论相对于 main_target 的立场。

main_target：
{main_target}

以下是同一话题下与当前评论最相似的已标注评论，供参考：
{demo}

输入上下文：
{context_block}

重要要求：
1. 只能在 favor、against、none 中选择一个。
2. 示例只是参考，相似的字面表达在不同语境下立场可能不同，请结合当前上下文独立判断。
3. 如果当前评论是短文本（如"确实""呵呵""666""笑死"），必须结合父评论或会话窗口判断。

请严格按以下 JSON 输出：
{{
  "predicted_stance": "favor|against|none",
  "confidence": 0.0,
  "reason_short": "不超过80字"
}}"""


def _build_cot_prompt(main_target, context_block):
    return f"""请判断下面当前评论相对于 main_target 的立场。

main_target：
{main_target}

输入上下文：
{context_block}

请一步一步分析：先理解会话中各条评论在讨论什么、当前评论在回应谁、表达了什么态度，再推断当前评论对 main_target 的立场。

重要要求：
1. 只能在 favor、against、none 中选择一个。
2. 不要把"无关""目标不清""上下文不足"直接当成 none；根据是否能映射回 main_target 判断。
3. 如果当前评论是短文本（如"确实""呵呵""666""笑死"），必须结合父评论或会话窗口判断。

请严格按以下 JSON 输出（先写分析，再写结论）：
{{
  "analysis": "逐步分析，不超过200字",
  "predicted_stance": "favor|against|none",
  "confidence": 0.0,
  "reason_short": "不超过80字"
}}"""


# ============ 冲突感知闭环：定向复核（relation_verify） ============

def _nv(x) -> str:
    """与 analyze_depth_dynamics.nv 一致的类别字段规范化（保证检索键与知识库相同）。"""
    if x is None:
        return "unknown"
    if isinstance(x, (list, tuple)):
        x = "|".join(map(str, x))
    s = re.sub(r"[\s\-]+", "_", str(x).strip().lower())
    return s or "unknown"


def _depth_bucket(rec: dict) -> str:
    try:
        d = int(rec.get("depth"))
    except (TypeError, ValueError):
        d = len(str(rec.get("node_id", "")).split("-")) - 1
    return str(d) if d < 4 else "4+"


def _key_levels(rec: dict) -> list:
    """与 KB:完整组合 相同的从细到粗的检索键。"""
    rr, sp, tr = _nv(rec.get("llm_reply_relation")), _nv(rec.get("llm_semantic_propagation")), _nv(rec.get("llm_target_relation"))
    db = _depth_bucket(rec)
    return [("c", rr, sp, tr, db), ("rs", rr, sp, db), ("spd", sp, db), ("sp", sp), ("d", db), ("g",)]


def _gold_relation(rec: dict) -> Optional[str]:
    pa = safe_str(rec.get("parent_stance_label")).lower()
    y = safe_str(rec.get("stance_label")).lower()
    if pa not in ("favor", "against") or y not in VALID_STANCES:
        return None
    return "neutralize" if y == "none" else ("inherit" if y == pa else "reverse")


def build_contrast_index(all_records: list, topics: set):
    """每个话题只用自己的 train：按知识库检索键分组，记录每条样本相对父评论的真实传导关系。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    for t in topics:
        tr = [r for r in all_records if r.get("split") == "train" and r.get("topic") == t and _gold_relation(r)]
        if not tr:
            continue
        for r in tr:
            for k in _key_levels(r):
                CONTRAST_INDEX.setdefault((t, k), defaultdict(list))[_gold_relation(r)].append(r)
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        mat = vec.fit_transform([_fewshot_text(r) for r in tr])
        CONTRAST_TFIDF[t] = (vec, mat, {id(r): i for i, r in enumerate(tr)})
        print(f"  对比示例索引：{t} train（父评论有立场）{len(tr)} 条")


def retrieve_contrast(rec: dict) -> list:
    """在当前节点所属的最细一级知识单元里，各取一条"继承""反转"（若有则再取"中立化"）的最相似训练样本。"""
    t = rec.get("topic")
    if t not in CONTRAST_TFIDF:
        return []
    vec, mat, row = CONTRAST_TFIDF[t]
    qv = vec.transform([_fewshot_text(rec)])
    for k in _key_levels(rec):
        groups = CONTRAST_INDEX.get((t, k))
        if not groups or not groups.get("inherit") or not groups.get("reverse"):
            continue
        out = []
        for rel in ("inherit", "reverse", "neutralize"):
            cands = groups.get(rel) or []
            if not cands:
                continue
            idx = [row[id(r)] for r in cands]
            sims = (mat[idx] @ qv.T).toarray().ravel()
            best = cands[int(sims.argmax())]
            out.append({"doc_id": f"train::{best.get('node_id')}", "relation": rel, "level": k[0],
                        "parent_text": best.get("parent_text"), "current_text": best.get("current_text"),
                        "parent_stance": safe_str(best.get("parent_stance_label")).lower(),
                        "stance_label": safe_str(best.get("stance_label")).lower(),
                        "similarity": round(float(sims.max()), 4)})
        return out
    return []


REL_DESC = {"inherit": "立场与父评论相同", "reverse": "立场与父评论相反", "neutralize": "对主目标不表态"}


def _build_verify_prompt(main_target, rec, context_block, examples):
    parent_text = safe_str(rec.get("parent_text"))
    if examples:
        demo = "\n\n".join(
            f"示例{i + 1}（同一类回复，{REL_DESC[e['relation']]}）：\n"
            f"父评论：{safe_str(e.get('parent_text'))[:150]}\n当前评论：{safe_str(e.get('current_text'))[:200]}\n"
            f"父评论对 main_target 的立场：{e.get('parent_stance')}；当前评论的立场：{e.get('stance_label')}"
            for i, e in enumerate(examples))
    else:
        demo = "（无）"
    return f"""请仔细分析当前评论与父评论之间的立场关系，并判断当前评论对 main_target 的立场。

main_target：
{main_target}

输入上下文：
{context_block}

父评论：
{parent_text or "（无）"}

以下是同一话题中、与当前评论属于同一类回复方式的已标注例子。它们的回复方式相似，但相对父评论的立场关系不同，请注意区分：
{demo}

请按以下步骤判断：
步骤 1：父评论对 main_target 的立场是什么（favor / against / none）？
步骤 2：当前评论在回应父评论的什么？
  - 反驳父评论对 main_target 的整体态度 → 通常与父评论立场相反；
  - 只反驳父评论的某个细节、论据、说法或针对父评论作者本人，但对 main_target 的态度并不相反 → 可能与父评论立场相同；
  - 附和、补充、加强父评论 → 通常与父评论立场相同；
  - 转移话题或只是玩笑、不涉及 main_target → 不表态。
  注意讽刺、反问和表面赞同实际否定的说法。
步骤 3：得出当前评论相对父评论的关系，以及当前评论对 main_target 的立场。如果当前评论直接、明确地表达了对 main_target 的立场，以直接表达为准。

请严格按以下 JSON 输出：
{{
  "parent_stance": "favor|against|none",
  "relation_to_parent": "inherit|reverse|neutralize|independent",
  "predicted_stance": "favor|against|none",
  "confidence": 0.0,
  "reason_short": "不超过80字"
}}"""


# ============ LLM 调用 ============

def _normalize_base_url(url: Optional[str]) -> Optional[str]:
    """openai SDK 会自动在 base_url 后拼接 /chat/completions，这里去掉重复的后缀。"""
    if not url:
        return url
    url = url.rstrip("/")
    for suf in ("/chat/completions", "/completions"):
        if url.endswith(suf):
            url = url[: -len(suf)]
    return url


_LOGPROB_WARNED = {"done": False}


def call_llm(
    prompt: str,
    provider: str,
    model: str,
    api_base: Optional[str] = None,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
    temperature: float = 0.1,
    return_logprobs: bool = False,
    top_logprobs: int = 10,
):
    """返回模型输出文本；return_logprobs=True 时返回 (文本, 逐 token 的 logprobs 列表或 None)。"""
    if provider in ("openai", "deepseek", "qwen", "local"):
        try:
            import openai
        except ImportError:
            raise ImportError("请先安装：pip install openai")

        cfg     = API_CONFIGS.get(provider, {})
        if provider == "local":     # 本地 OpenAI 兼容服务（如 vLLM），默认不校验 key
            cfg = {"base_url": "http://localhost:8000/v1"}
            api_key = os.environ.get("LOCAL_API_KEY", "EMPTY")
        else:
            api_key = cfg.get("api_key") or os.environ.get({
                "openai":   "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "qwen":     "DASHSCOPE_API_KEY",
            }.get(provider, ""))

        if not api_key:
            raise EnvironmentError(
                f"未找到 API Key，请设置环境变量或在脚本顶部 API_CONFIGS 中填写。"
            )

        base_url = _normalize_base_url(api_base or cfg.get("base_url"))
        actual_model = model or cfg.get("model", "gpt-4o-mini")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=actual_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=LLM_TIMEOUT,
                )
                if return_logprobs:
                    kwargs.update(logprobs=True, top_logprobs=top_logprobs)
                resp = client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("模型返回空内容")
                if not return_logprobs:
                    return text
                lp = getattr(resp.choices[0], "logprobs", None)
                content = getattr(lp, "content", None) if lp is not None else None
                if not content:
                    if not _LOGPROB_WARNED["done"]:
                        print("  ⚠️  接口没有返回 logprobs（代理可能不支持），该样本的概率将退化为 one-hot")
                        _LOGPROB_WARNED["done"] = True
                    return text, None
                toks = [{"token": t.token, "logprob": t.logprob,
                         "top": [(x.token, x.logprob) for x in (t.top_logprobs or [])]} for t in content]
                return text, toks
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
    # 兜底：JSON 被截断或格式不完整时，只要已给出 predicted_stance 就逐字段提取
    m = re.search(r'"predicted_stance"\s*:\s*"(favor|against|none)"', text, re.I)
    if m:
        out = {"predicted_stance": m.group(1).lower(), "_parse_fallback": True}
        for f in ("parent_stance", "relation_to_parent"):
            mm = re.search(rf'"{f}"\s*:\s*"([a-z_]+)"', text, re.I)
            if mm:
                out[f] = mm.group(1).lower()
        mc = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if mc:
            out["confidence"] = float(mc.group(1))
        return out
    raise ValueError(f"JSON 解析失败：{raw_text[:200]}")


# ============ 三类概率 ============

STANCE_ORDER = ["favor", "against", "none"]


def extract_label_probs(toks: list) -> Optional[dict]:
    """从逐 token 的 logprobs 中取出 "predicted_stance" 取值第一个 token 位置的候选分布，
    映射到 favor / against / none 并归一化。返回 {"favor":p, "against":p, "none":p, "_mass":覆盖概率} 或 None。"""
    import math
    if not toks:
        return None
    full, starts = "", []
    for t in toks:
        starts.append(len(full))
        full += t["token"]
    k = full.find('"predicted_stance"')
    if k < 0:
        return None
    colon = full.find(":", k + len('"predicted_stance"'))
    q = full.find('"', colon + 1) if colon >= 0 else -1
    if q < 0:
        return None
    vstart = q + 1
    idx = None
    for i, s in enumerate(starts):
        e = s + len(toks[i]["token"])
        if s <= vstart < e:
            idx = i
            break
    if idx is None:
        return None
    cands = list(toks[idx]["top"]) or [(toks[idx]["token"], toks[idx]["logprob"])]
    mass = {l: 0.0 for l in STANCE_ORDER}
    for tok, lp in cands:
        t = tok.strip().lstrip('"').strip().lower()
        if not t:
            continue
        for l in STANCE_ORDER:
            if l.startswith(t) or t.startswith(l):
                mass[l] += math.exp(lp)
                break
    tot = sum(mass.values())
    if tot <= 0:
        return None
    out = {l: round(mass[l] / tot, 6) for l in STANCE_ORDER}
    out["_mass"] = round(tot, 6)
    return out


def onehot(label: str) -> dict:
    return {l: (1.0 if l == label else 0.0) for l in STANCE_ORDER}


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
    prob_mode: str = "none",
    prob_samples: int = 5,
    sample_temperature: float = 0.7,
    top_logprobs: int = 10,
    temperature: float = 0.1,
) -> tuple:
    retrieved_docs = []

    if method in RAG_METHOD_SET and docs:
        query = build_query_metadata(rec)
        retrieved_docs = retrieve_documents(query, docs, top_k, retrieval_mode)
    elif method == "knn_fewshot":
        retrieved_docs = retrieve_fewshot(rec, FEWSHOT_K)
    elif method == "relation_verify":
        retrieved_docs = retrieve_contrast(rec)

    prompt = build_prediction_prompt(rec, method, retrieved_docs)
    phash  = prompt_hash(prompt)

    for attempt in range(max_retries):
        try:
            toks = None
            if prob_mode == "logprobs":
                raw, toks = call_llm(prompt, provider, model, api_base, max_retries=1,
                                     sleep_seconds=sleep_seconds, temperature=temperature,
                                     return_logprobs=True, top_logprobs=top_logprobs)
            else:
                raw = call_llm(prompt, provider, model, api_base, max_retries=1,
                               sleep_seconds=sleep_seconds, temperature=temperature)
            result = parse_prediction_json(raw)

            predicted = safe_str(result.get("predicted_stance", "")).lower()
            if predicted not in VALID_STANCES:
                predicted = "none"

            stance_probs, probs_source = None, None
            if prob_mode == "logprobs":
                p = extract_label_probs(toks) if toks else None
                if p:
                    stance_probs, probs_source = p, "logprobs"
                else:
                    stance_probs, probs_source = onehot(predicted), "fallback_onehot"
            elif prob_mode == "sample":
                votes = Counter([predicted])
                for _ in range(max(prob_samples - 1, 0)):
                    try:
                        r2 = parse_prediction_json(call_llm(
                            prompt, provider, model, api_base, max_retries=1,
                            sleep_seconds=sleep_seconds, temperature=sample_temperature))
                        p2 = safe_str(r2.get("predicted_stance", "")).lower()
                        votes[p2 if p2 in VALID_STANCES else "none"] += 1
                    except Exception:
                        continue
                n = sum(votes.values())
                stance_probs = {l: round(votes[l] / n, 6) for l in STANCE_ORDER}
                stance_probs["_mass"] = n
                probs_source = f"sample{n}"

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
                "fewshot_examples": ([{"doc_id": d["doc_id"], "label": d["stance_label"], "sim": d["similarity"]}
                                      for d in retrieved_docs] if method == "knn_fewshot" else None),
                "cot_analysis": (safe_str(result.get("analysis"))[:300] if method == "cot" else None),
                "contrast_examples": ([{"doc_id": d["doc_id"], "relation": d["relation"], "level": d["level"],
                                        "sim": d["similarity"]} for d in retrieved_docs]
                                      if method == "relation_verify" else None),
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
                "parent_stance_pred": safe_str(result.get("parent_stance", "")).lower() or None,
                "relation_to_parent_pred": safe_str(result.get("relation_to_parent", "")).lower() or None,
                "confidence":       safe_float(result.get("confidence", 0.5)),
                "reason_short":     safe_str(result.get("reason_short", ""))[:120],
                "prediction_basis": result.get("prediction_basis", {}),
                "parse_fallback": bool(result.get("_parse_fallback")),
                "prompt_hash":      phash,
            }
            if prob_mode != "none":
                out["stance_probs"] = stance_probs
                out["probs_source"] = probs_source
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
    parser.add_argument("--provider",          default="openai", help="openai / deepseek / qwen / local（本地 vLLM 等 OpenAI 兼容服务）")
    parser.add_argument("--model",             default="gpt-4o-mini")
    parser.add_argument("--api_base",          default=None)
    parser.add_argument("--temperature",       type=float, default=0.1)
    parser.add_argument("--max_retries",       type=int,   default=3)
    parser.add_argument("--sleep_seconds",     type=float, default=1.0)
    parser.add_argument("--sample_limit",      type=int,   default=None)
    parser.add_argument("--retrieval_mode",    default="rule_hybrid",
                        choices=["rule_hybrid", "tfidf", "hybrid"])
    parser.add_argument("--include_low_reliability", default="false")
    parser.add_argument("--only_topic",        default=None)
    parser.add_argument("--only_depth_ge",     type=int, default=None)
    parser.add_argument("--only_ambiguous",    default="false")
    parser.add_argument("--only_child_feedback", default="false")
    parser.add_argument("--resume",            default="true")
    parser.add_argument("--save_raw",          default="false")
    parser.add_argument("--relative_memory_examples", default="false",
                        help="full_rag_relative 是否在记忆中展示 1 条例子（默认 false，与 full_rag 的记忆格式保持一致）")
    parser.add_argument("--return_probs", default="none", choices=["none", "logprobs", "sample"],
                        help="同时记录 favor/against/none 三类概率：logprobs = 读取标签 token 的 top_logprobs（每条 1 次调用）；"
                             "sample = 额外采样 prob_samples-1 次投票（每条 prob_samples 次调用）")
    parser.add_argument("--top_logprobs", type=int, default=10)
    parser.add_argument("--prob_samples", type=int, default=5)
    parser.add_argument("--sample_temperature", type=float, default=0.7)
    parser.add_argument("--fewshot_k", type=int, default=5, help="knn_fewshot 检索的示例数")
    parser.add_argument("--max_tokens", type=int, default=400, help="单次输出的最大 token 数")
    parser.add_argument("--timeout", type=float, default=45, help="单次请求超时（秒）；较慢的模型可设 90–120")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发请求数（本地 vLLM 建议 16–32；API 代理建议 ≤4，避免限速）")
    parser.add_argument("--id_file", default=None,
                        help="只预测这些节点：jsonl，每行含 topic 与 node_id（如 verification_pipeline.py select 的输出）")
    args = parser.parse_args()

    global REL_SHOW_EXAMPLES, LLM_MAX_TOKENS, LLM_TIMEOUT
    REL_SHOW_EXAMPLES = args.relative_memory_examples.lower() == "true"
    LLM_MAX_TOKENS, LLM_TIMEOUT = args.max_tokens, args.timeout
    if args.method == "full_rag_relative" and "relative" not in Path(args.knowledge).name:
        print("  ⚠️  full_rag_relative 应使用 build_relative_pattern_memory.py 生成的 rag_knowledge_documents_relative.jsonl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_low = args.include_low_reliability.lower() == "true"
    resume      = args.resume.lower() == "true"
    save_raw    = args.save_raw.lower() == "true"
    only_ambig  = args.only_ambiguous.lower() == "true"
    only_child  = args.only_child_feedback.lower() == "true"

    suffix = f"{args.method}_{args.split}" + ("" if args.return_probs == "none" else f"_probs_{args.return_probs}")
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
    docs = load_knowledge_documents(args.knowledge, include_low)

    selected = select_instances(
        all_records, args.split, args.sample_limit,
        args.only_topic, args.only_depth_ge, only_ambig, only_child
    )
    print(f"  筛选后：{len(selected)} 条（split={args.split}）")

    if args.id_file:
        want = {(safe_str(r.get("topic")), safe_str(r.get("node_id"))) for r in load_jsonl(args.id_file)}
        selected = [r for r in selected if (safe_str(r.get("topic")), safe_str(r.get("node_id"))) in want]
        print(f"  按 --id_file 过滤后：{len(selected)} 条（文件中 {len(want)} 条）")
    if args.method == "relation_verify":
        build_contrast_index(all_records, {r.get("topic") for r in selected})
    if args.method == "knn_fewshot":
        global FEWSHOT_K
        FEWSHOT_K = args.fewshot_k
        build_fewshot_index(all_records, {r.get("topic") for r in selected})

    # 断点续跑
    existing_ids = set()
    skipped = 0
    if resume and out_pred.exists():
        existing = load_jsonl(str(out_pred))
        existing_ids = {(safe_str(r.get("topic")), safe_str(r.get("node_id"))) for r in existing}
        skipped = len(existing_ids)
        print(f"  断点续跑：跳过 {skipped} 条")

    predictions, failed_list = [], []
    to_process = [r for r in selected if (safe_str(r.get("topic")), safe_str(r.get("node_id"))) not in existing_ids]
    total = len(to_process)

    print(f"\n🤖 开始预测（method={args.method}，共 {total} 条）...")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def _run(rec):
        return rec, predict_one_instance(
            rec, docs, args.method, args.top_k, args.retrieval_mode,
            args.provider, args.model, args.api_base,
            args.max_retries, args.sleep_seconds, save_raw,
            prob_mode=args.return_probs, prob_samples=args.prob_samples,
            sample_temperature=args.sample_temperature, top_logprobs=args.top_logprobs,
            temperature=args.temperature,
        )

    def _handle(i, rec, result, error):
        with lock:
            if (i + 1) % 50 == 0:
                print(f"  进度：{i+1}/{total}，成功={len(predictions)}，失败={len(failed_list)}")
            if (not error and args.return_probs != "none" and i < 3):
                print(f"  概率示例 {rec.get('node_id')}: 预测={result['predicted_stance']} "
                      f"概率={result.get('stance_probs')} 来源={result.get('probs_source')}")
            if error:
                print(f"  ❌ {rec.get('node_id','?')}：{error}")
                fail_rec = dict(rec)
                fail_rec["_error"] = error
                failed_list.append(fail_rec)
                write_jsonl_append(fail_rec, out_failed)
            else:
                predictions.append(result)
                write_jsonl_append(result, out_pred)

    if args.workers <= 1:
        for i, rec in enumerate(to_process):
            _, (result, error) = _run(rec)
            _handle(i, rec, result, error)
            time.sleep(args.sleep_seconds)
    else:
        print(f"  并发数：{args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run, rec) for rec in to_process]
            for i, fu in enumerate(as_completed(futs)):
                rec, (result, error) = fu.result()
                _handle(i, rec, result, error)

    report = generate_prediction_report(
        predictions, failed_list, len(selected), args.method, args.split
    )
    if args.return_probs != "none":
        report["probs_source_distribution"] = dict(Counter(r.get("probs_source") for r in predictions))
        print(f"  概率来源分布：{report['probs_source_distribution']}")
    write_json(report, out_report)

    print(f"""
✅ 预测完成：
  成功：{len(predictions)}  失败：{len(failed_list)}  跳过：{skipped}
  输出：{out_pred}
    """)


if __name__ == "__main__":
    main()