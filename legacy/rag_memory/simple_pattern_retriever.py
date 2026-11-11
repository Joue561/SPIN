#!/usr/bin/env python3
"""
simple_pattern_retriever.py

用法：
python simple_pattern_retriever.py \
  --knowledge rag_knowledge_documents.jsonl \
  --query_text "确实" \
  --candidate_pattern agreement_with_parent \
  --top_k 5
"""

import json
import argparse
import math
from pathlib import Path
from collections import Counter


# ============ 加载知识库 ============

def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ============ TF-IDF 简单实现 ============

def tokenize(text: str) -> list:
    """简单按字符切分，适合中文"""
    if not text:
        return []
    tokens = []
    for ch in text:
        if ch.strip() and ch not in "，。！？、「」【】《》…—":
            tokens.append(ch)
    # 加入 2-gram
    for i in range(len(text) - 1):
        bi = text[i:i+2].strip()
        if bi:
            tokens.append(bi)
    return tokens


def build_tfidf_index(docs: list) -> tuple:
    """返回 (doc_tokens_list, idf_map)"""
    doc_tokens = []
    df: Counter = Counter()

    for doc in docs:
        text = doc.get("retrieval_text", "")
        tokens = set(tokenize(text))
        doc_tokens.append(list(tokenize(text)))
        for t in tokens:
            df[t] += 1

    N = len(docs)
    idf = {t: math.log((N + 1) / (cnt + 1)) + 1
           for t, cnt in df.items()}
    return doc_tokens, idf


def tfidf_score(query_tokens: list, doc_tokens: list, idf: dict) -> float:
    """余弦相似度（TF-IDF）"""
    query_tf = Counter(query_tokens)
    doc_tf   = Counter(doc_tokens)
    all_terms = set(query_tokens) | set(doc_tokens)

    dot, q_norm, d_norm = 0.0, 0.0, 0.0
    for t in all_terms:
        idf_val = idf.get(t, 1.0)
        qv = query_tf.get(t, 0) * idf_val
        dv = doc_tf.get(t, 0) * idf_val
        dot   += qv * dv
        q_norm += qv ** 2
        d_norm += dv ** 2

    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))


def pattern_exact_bonus(candidate_pattern: str, doc_pattern: str) -> float:
    """若 candidate_pattern 与文档 pattern 完全匹配，给额外加分"""
    if not candidate_pattern or not doc_pattern:
        return 0.0
    if candidate_pattern.strip() == doc_pattern.strip():
        return 0.5
    if candidate_pattern.strip() in doc_pattern.strip():
        return 0.2
    return 0.0


# ============ 检索 ============

def retrieve(
    docs: list,
    query_text: str,
    candidate_pattern: str,
    top_k: int,
) -> list:
    doc_tokens_list, idf = build_tfidf_index(docs)
    query_tokens = tokenize(query_text + " " + (candidate_pattern or ""))

    scored = []
    for i, doc in enumerate(docs):
        tfidf = tfidf_score(query_tokens, doc_tokens_list[i], idf)
        bonus = pattern_exact_bonus(candidate_pattern,
                                    doc.get("pattern_name", ""))
        score = round(tfidf + bonus, 6)
        scored.append((score, doc))

    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="RAG Pattern 简单检索原型")
    parser.add_argument("--knowledge",         required=True, help="rag_knowledge_documents.jsonl 路径")
    parser.add_argument("--query_text",        required=True, help="当前评论文本")
    parser.add_argument("--candidate_pattern", default="",   help="候选 pattern（可选）")
    parser.add_argument("--top_k",             type=int, default=5)
    args = parser.parse_args()

    print(f"📂 加载知识库：{args.knowledge}")
    docs = load_jsonl(args.knowledge)
    print(f"  共加载 {len(docs)} 个文档\n")

    print(f"🔍 检索：query='{args.query_text}'，pattern='{args.candidate_pattern}'")
    results = retrieve(docs, args.query_text, args.candidate_pattern, args.top_k)

    print(f"\n{'='*60}")
    print(f"Top-{args.top_k} 检索结果")
    print(f"{'='*60}")

    for rank, (score, doc) in enumerate(results, 1):
        meta = doc.get("structured_metadata", {})
        dist = meta.get("label_distribution_weighted", {})
        dist_str = "  ".join(
            f"{lbl}:{round(v*100,1)}%"
            for lbl, v in sorted(dist.items(), key=lambda x: -x[1])
        )
        dominant = meta.get("dominant_label", "未知")
        reliability = meta.get("reliability_level", "未知")
        usage = meta.get("suggested_usage", "未知")

        print(f"\n【Rank {rank}】检索分数：{score:.4f}")
        print(f"  Pattern：      {doc.get('pattern_name')}")
        print(f"  主导标签：     {dominant}")
        print(f"  标签分布：     {dist_str}")
        print(f"  可靠性等级：   {reliability}")
        print(f"  建议用途：     {usage}")
        print(f"  样本数（原始）：{meta.get('source_sample_count_raw')}")
        print(f"  样本数（加权）：{meta.get('source_sample_count_weighted')}")
        print(f"  Lift：支持={meta.get('label_lift_favor')}  "
              f"反对={meta.get('label_lift_against')}  "
              f"真中立={meta.get('label_lift_neutral')}")
        print(f"  ⚠️   数据来源：AI-silver pseudo-label，非人工 gold label")

        # 给出 stance hint
        hint = _stance_hint(dominant, reliability, usage, score)
        print(f"  💡 Stance hint：{hint}")

    print(f"\n{'='*60}")


def _stance_hint(dominant: str, reliability: str, usage: str, score: float) -> str:
    if score < 0.05:
        return "检索分数过低，建议不使用该 pattern 作为参考。"
    if usage == "do_not_use":
        return "该 pattern 不建议用于标签推断。"
    if reliability == "high":
        return f"高可靠 pattern，主导立场为【{dominant}】，可作为强参考。"
    if reliability == "medium":
        return f"中等可靠 pattern，主导立场为【{dominant}】，建议结合上下文判断。"
    return f"低可靠 pattern，主导立场为【{dominant}】，仅供参考，请勿直接采用。"


if __name__ == "__main__":
    main()