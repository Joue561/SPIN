#!/usr/bin/env python3
"""
prepare_plm_stance_data.py

用法：
python prepare_plm_stance_data.py \
  --instances processed_cmtscd_semantic_llm/cmtscd_semantic_instances_llm.jsonl \
  --knowledge processed_cmtscd_rag_memory/rag_knowledge_documents_llm.jsonl \
  --output_dir plm_experiment_data \
  --method plm_semantic_rag_feature \
  --top_k 5
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# ============ 常量 ============

LABEL_MAP = {"favor": 0, "against": 1, "none": 2}
LABEL_NAMES = {0: "favor", 1: "against", 2: "none"}

TEXT_ONLY_METHODS = {
    "plm_current", "plm_parent", "plm_window",
    "plm_semantic_text", "plm_rag_text", "plm_semantic_rag_text",
}
RAG_TEXT_METHODS = {"plm_rag_text", "plm_semantic_rag_text"}
SEMANTIC_METHODS = {
    "plm_semantic_text", "plm_semantic_rag_text",
    "plm_rag_feature", "plm_semantic_rag_feature",
}
FEATURE_METHODS = {"plm_rag_feature", "plm_semantic_rag_feature"}
RAG_METHODS = RAG_TEXT_METHODS | FEATURE_METHODS

ALL_METHODS = [
    "plm_current", "plm_parent", "plm_window",
    "plm_semantic_text", "plm_rag_text", "plm_semantic_rag_text",
    "plm_rag_feature", "plm_semantic_rag_feature",
]

# input_text 截断时各部分的最大字符预算（粗略按字符近似 token，安全边界）
MAX_TARGET_ANCHOR_CHARS = 150
MAX_CONTEXT_CHARS       = 800
MAX_MEMORY_SUMMARY_CHARS = 400  # 全部 retrieved memory summary 加起来的预算


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
    return val if isinstance(val, list) else ([] if val is None else [val])


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def truncate(text: str, max_chars: int) -> str:
    text = safe_str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


# ============ 加载知识库（仅 train 构建，此处只负责加载与过滤） ============

def load_knowledge_documents(filepath: str) -> list:
    docs = load_jsonl(filepath)
    filtered = []
    for doc in docs:
        meta = doc.get("structured_metadata", {})
        usage       = meta.get("suggested_usage", "")
        reliability = meta.get("reliability_level", "")
        weighted    = safe_float(meta.get("source_sample_count_weighted", 0))

        if usage == "do_not_use":
            continue
        if reliability == "excluded":
            continue
        if weighted < 10:
            continue
        filtered.append(doc)
    print(f"  知识文档：{len(docs)} 条 → 过滤后可用 {len(filtered)} 条")
    return filtered


# ============ candidate_pattern 规范化 ============

def normalize_patterns(raw) -> list:
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
    result = []
    for item in items:
        for p in str(item).split("|"):
            p = p.strip().lower().replace(" ", "_")
            if p and p != "unknown_pattern":
                result.append(p)
    return list(dict.fromkeys(result))


# ============ RAG 检索（rule_hybrid） ============

def build_query_metadata(rec: dict) -> dict:
    return {
        "current_text":  safe_str(rec.get("current_text")),
        "parent_text":   safe_str(rec.get("parent_text")),
        "reply_relation": safe_str(rec.get("llm_reply_relation")),
        "semantic_prop": safe_str(rec.get("llm_semantic_propagation")),
        "target_relation": safe_str(rec.get("llm_target_relation")),
        "patterns":      set(normalize_patterns(rec.get("llm_candidate_patterns"))),
    }


def retrieve_memory_rule_hybrid(query: dict, docs: list, top_k: int) -> list:
    if not docs:
        return []

    query_words = set(
        (query["current_text"] + " " + query["parent_text"]).split()
    )
    # 中文没有天然分词空格，补充字符级重合判断
    query_chars = set(query["current_text"] + query["parent_text"])

    scored = []
    for doc in docs:
        score = 0.0
        meta = doc.get("structured_metadata", {})
        pname = safe_str(doc.get("pattern_name") or doc.get("composite_key", ""))
        retrieval_text = safe_str(doc.get("retrieval_text", ""))

        for qp in query["patterns"]:
            if qp and qp in pname:
                score += 3
        if query["semantic_prop"] and query["semantic_prop"] in retrieval_text:
            score += 2
        if query["reply_relation"] and query["reply_relation"] in retrieval_text:
            score += 2
        if query["target_relation"] and query["target_relation"] in retrieval_text:
            score += 2
        if meta.get("reliability_level") == "high":
            score += 1
        if meta.get("suggested_usage") == "use_as_strong_rag_memory":
            score += 1

        doc_chars = set(retrieval_text)
        overlap = len(query_chars & doc_chars)
        if overlap > 0:
            score += min(overlap * 0.01, 0.5)

        scored.append((score, doc))

    scored.sort(key=lambda x: -x[0])
    top = [d for s, d in scored[:top_k] if s > 0]
    return top if top else [d for _, d in scored[:top_k]]


# ============ retrieved_memory_summaries 构建 ============

def build_memory_summary_text(doc: dict) -> str:
    meta = doc.get("structured_metadata", {})
    pname = safe_str(doc.get("pattern_name") or doc.get("composite_key", ""))
    dist = meta.get("label_distribution_weighted", {})
    dist_str = ", ".join(f"{k}={round(safe_float(v), 2)}" for k, v in dist.items())
    caution = ""
    # retrieval_text 中通常包含 caution 描述，尝试提取简短片段
    # 注意：英文版 RAG 文档模板用的是 "[Cautions]"（见 build_llm_pattern_memory_en.py），
    # 这里必须同步改成匹配英文标记，否则这个字符串永远匹配不到，caution 会一直是空。
    rtext = safe_str(doc.get("retrieval_text", ""))
    if "[Cautions]" in rtext:
        try:
            caution = rtext.split("[Cautions]")[1].split("[")[0].strip("\n: ").strip()
            caution = truncate(caution, 60)
        except Exception:
            caution = ""

    lines = [
        f"Pattern: {pname}",
        f"Weighted historical distribution: {dist_str}" if dist_str else "Weighted historical distribution: unknown",
    ]
    if caution:
        lines.append(f"Caution: {caution}")
    return "\n".join(lines)


def build_retrieved_summaries(retrieved_docs: list) -> tuple:
    """返回 (memory_ids, summaries_text_list)"""
    ids = [d.get("doc_id", "") for d in retrieved_docs]
    summaries = [build_memory_summary_text(d) for d in retrieved_docs]
    return ids, summaries


def join_summaries_truncated(summaries: list, max_chars: int) -> str:
    """把多条 summary 拼接，整体预算不超过 max_chars"""
    if not summaries:
        return "(no relevant historical pattern memory)"
    out_parts = []
    remaining = max_chars
    for i, s in enumerate(summaries, 1):
        block = f"{i}. {s}"
        if len(block) > remaining:
            block = block[:max(remaining, 0)] + "…"
        out_parts.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return "\n".join(out_parts)


# ============ RAG 特征构建 ============

RAG_FEATURE_KEYS = [
    "rag_favor_prob", "rag_against_prob", "rag_none_prob",
    "rag_dominant_label_id", "rag_dominant_label_probability",
    "rag_avg_memory_confidence", "rag_avg_memory_weighted_count",
    "rag_high_reliability_count", "rag_medium_reliability_count",
    "rag_low_reliability_count", "rag_memory_count",
    "rag_pattern_match_score", "rag_semantic_match_score",
    "rag_target_relation_match_score", "rag_reply_relation_match_score",
    "rag_strong_usage_count", "rag_weak_usage_count",
]


def build_rag_features(
    rec: dict,
    retrieved_docs: list,
    norm_stats: Optional[dict] = None,
) -> dict:
    """构造 RAG 结构化特征向量（原始数值，标准化在 normalize_rag_features 中统一处理）"""
    query = build_query_metadata(rec)

    if not retrieved_docs:
        feats = {k: 0.0 for k in RAG_FEATURE_KEYS}
        feats["rag_favor_prob"]   = 1 / 3
        feats["rag_against_prob"] = 1 / 3
        feats["rag_none_prob"]    = 1 / 3
        feats["rag_dominant_label_id"] = 2.0  # 默认 none
        return feats

    favor_sum, against_sum, none_sum = 0.0, 0.0, 0.0
    conf_sum, weighted_sum = 0.0, 0.0
    high_cnt, med_cnt, low_cnt = 0, 0, 0
    strong_cnt, weak_cnt = 0, 0
    pattern_score, semantic_score, target_score, reply_score = 0.0, 0.0, 0.0, 0.0

    n = len(retrieved_docs)
    for doc in retrieved_docs:
        meta = doc.get("structured_metadata", {})
        dist = meta.get("label_distribution_weighted", {})
        favor_sum   += safe_float(dist.get("favor", 0))
        against_sum += safe_float(dist.get("against", 0))
        none_sum    += safe_float(dist.get("none", 0))

        conf_sum     += safe_float(meta.get("average_confidence", 0))
        weighted_sum += safe_float(meta.get("source_sample_count_weighted", 0))

        rel = meta.get("reliability_level", "")
        if rel == "high":
            high_cnt += 1
        elif rel == "medium":
            med_cnt += 1
        elif rel == "low":
            low_cnt += 1

        usage = meta.get("suggested_usage", "")
        if usage == "use_as_strong_rag_memory":
            strong_cnt += 1
        elif usage == "use_as_weak_rag_memory":
            weak_cnt += 1

        pname = safe_str(doc.get("pattern_name") or doc.get("composite_key", ""))
        rtext = safe_str(doc.get("retrieval_text", ""))
        if query["patterns"] and any(p in pname for p in query["patterns"]):
            pattern_score += 1
        if query["semantic_prop"] and query["semantic_prop"] in rtext:
            semantic_score += 1
        if query["target_relation"] and query["target_relation"] in rtext:
            target_score += 1
        if query["reply_relation"] and query["reply_relation"] in rtext:
            reply_score += 1

    total = favor_sum + against_sum + none_sum
    if total > 0:
        favor_prob, against_prob, none_prob = (
            favor_sum / total, against_sum / total, none_sum / total
        )
    else:
        favor_prob = against_prob = none_prob = 1 / 3

    probs = {"favor": favor_prob, "against": against_prob, "none": none_prob}
    dominant_label = max(probs, key=probs.get)
    dominant_id = LABEL_MAP[dominant_label]
    dominant_prob = probs[dominant_label]

    feats = {
        "rag_favor_prob":                  round(favor_prob, 4),
        "rag_against_prob":                round(against_prob, 4),
        "rag_none_prob":                   round(none_prob, 4),
        "rag_dominant_label_id":           float(dominant_id),
        "rag_dominant_label_probability":  round(dominant_prob, 4),
        "rag_avg_memory_confidence":       round(conf_sum / n, 4),
        "rag_avg_memory_weighted_count":   round(weighted_sum / n, 4),
        "rag_high_reliability_count":      float(high_cnt),
        "rag_medium_reliability_count":    float(med_cnt),
        "rag_low_reliability_count":       float(low_cnt),
        "rag_memory_count":                float(n),
        "rag_pattern_match_score":         round(pattern_score / n, 4),
        "rag_semantic_match_score":        round(semantic_score / n, 4),
        "rag_target_relation_match_score": round(target_score / n, 4),
        "rag_reply_relation_match_score":  round(reply_score / n, 4),
        "rag_strong_usage_count":          float(strong_cnt),
        "rag_weak_usage_count":            float(weak_cnt),
    }
    return feats


def compute_normalization_stats(all_features: list) -> dict:
    """对需要标准化的计数类特征计算 train 集上的 min-max，概率类特征本身已在[0,1]不处理"""
    count_keys = [
        "rag_avg_memory_weighted_count", "rag_memory_count",
        "rag_high_reliability_count", "rag_medium_reliability_count",
        "rag_low_reliability_count", "rag_strong_usage_count", "rag_weak_usage_count",
    ]
    stats = {}
    for k in count_keys:
        vals = [f.get(k, 0.0) for f in all_features]
        stats[k] = {"min": min(vals, default=0.0), "max": max(vals, default=1.0)}
    return stats


def normalize_rag_features(feats: dict, norm_stats: dict) -> dict:
    """对计数类特征做 min-max 归一化到 [0,1]，概率/匹配分数类已天然在合理范围"""
    out = dict(feats)
    for k, rng in norm_stats.items():
        v = feats.get(k, 0.0)
        lo, hi = rng["min"], rng["max"]
        out[k] = round((v - lo) / (hi - lo), 4) if hi > lo else 0.0
    return out


# ============ input_text 构建（按 method） ============

def build_input_text(rec: dict, method: str, memory_summary_text: str = "") -> str:
    main_target  = safe_str(rec.get("main_target"))
    current_text = safe_str(rec.get("current_text"))
    parent_text  = safe_str(rec.get("parent_text"))
    causal_ctx   = truncate(rec.get("causal_context_text"), MAX_CONTEXT_CHARS)

    semantic_block = (
        f"Reply relation: {safe_str(rec.get('llm_reply_relation'))}\n"
        f"Semantic propagation: {safe_str(rec.get('llm_semantic_propagation'))}\n"
        f"Target relation: {safe_str(rec.get('llm_target_relation'))}\n"
        f"Stance mapping to parent: {safe_str(rec.get('llm_stance_mapping_to_parent'))}\n"
        f"Candidate patterns: {', '.join(normalize_patterns(rec.get('llm_candidate_patterns'))) or 'none'}"
    )

    if method == "plm_current":
        return f"Target: {main_target}\nCurrent comment: {current_text}"

    if method == "plm_parent":
        return f"Target: {main_target}\nCurrent comment: {current_text}\nParent comment: {parent_text}"

    if method == "plm_window":
        return (
            f"Target: {main_target}\nCurrent comment: {current_text}\n\n"
            f"Conversation context:\n{causal_ctx}"
        )

    if method == "plm_semantic_text":
        return (
            f"Target: {main_target}\nCurrent comment: {current_text}\n\n"
            f"Semantic propagation info:\n{semantic_block}\n\n"
            f"Conversation context:\n{causal_ctx}"
        )

    if method == "plm_rag_text":
        return (
            f"Target: {main_target}\nCurrent comment: {current_text}\n\n"
            f"Retrieved historical language-pattern memory:\n{memory_summary_text}\n\n"
            f"Conversation context:\n{causal_ctx}"
        )

    if method == "plm_semantic_rag_text":
        return (
            f"Target: {main_target}\nCurrent comment: {current_text}\n\n"
            f"Semantic propagation info:\n{semantic_block}\n\n"
            f"Retrieved historical language-pattern memory:\n{memory_summary_text}\n\n"
            f"Conversation context:\n{causal_ctx}"
        )

    if method == "plm_rag_feature":
        # 文本部分不含 RAG summary，RAG 信息走 feature
        return (
            f"Target: {main_target}\nCurrent comment: {current_text}\n\n"
            f"Conversation context:\n{causal_ctx}"
        )

    if method == "plm_semantic_rag_feature":
        return (
            f"Target: {main_target}\nCurrent comment: {current_text}\n\n"
            f"Semantic propagation info:\n{semantic_block}\n\n"
            f"Conversation context:\n{causal_ctx}"
        )

    raise ValueError(f"未知 method：{method}")


# ============ 主流程 ============

def process_split(
    records: list,
    split: str,
    method: str,
    docs: list,
    top_k: int,
    norm_stats: Optional[dict],
) -> tuple:
    """处理单个 split，返回 (output_records, norm_stats_used)"""
    use_rag = method in RAG_METHODS
    use_feature = method in FEATURE_METHODS

    outputs = []
    raw_features_for_norm = []

    for rec in records:
        stance = safe_str(rec.get("stance_label", "")).lower()
        if stance not in LABEL_MAP:
            continue  # 跳过无效标签样本
        current_text = safe_str(rec.get("current_text"))
        if not current_text:
            continue

        retrieved_ids, retrieved_summaries, memory_summary_joined = [], [], ""
        if use_rag:
            query = build_query_metadata(rec)
            retrieved_docs = retrieve_memory_rule_hybrid(query, docs, top_k)
            retrieved_ids, retrieved_summaries = build_retrieved_summaries(retrieved_docs)
            memory_summary_joined = join_summaries_truncated(
                retrieved_summaries, MAX_MEMORY_SUMMARY_CHARS
            )
        else:
            retrieved_docs = []

        input_text = build_input_text(rec, method, memory_summary_joined)

        out = {
            "node_id":         safe_str(rec.get("node_id")),
            "topic":           safe_str(rec.get("topic")),
            "split":           split,
            "input_text":      input_text,
            "label":           LABEL_MAP[stance],
            "label_name":      stance,
            "current_text":    current_text,
            "parent_text":     safe_str(rec.get("parent_text")),
            "depth":           rec.get("depth"),
            "is_ambiguous_current": rec.get("is_ambiguous_current", False),
            "llm_reply_relation":       safe_str(rec.get("llm_reply_relation")),
            "llm_semantic_propagation": safe_str(rec.get("llm_semantic_propagation")),
            "llm_target_relation":      safe_str(rec.get("llm_target_relation")),
            "llm_candidate_patterns":   normalize_patterns(rec.get("llm_candidate_patterns")),
            "retrieved_memory_ids":       retrieved_ids,
            "retrieved_memory_summaries": retrieved_summaries,
        }

        if use_feature:
            raw_feats = build_rag_features(rec, retrieved_docs)
            out["_raw_rag_features"] = raw_feats  # 临时字段，归一化后会替换
            raw_features_for_norm.append(raw_feats)
        else:
            out["rag_features"] = {}

        outputs.append(out)

    # ---- RAG feature 标准化 ----
    if use_feature:
        if norm_stats is None:
            # train split：在此计算归一化统计量
            norm_stats = compute_normalization_stats(raw_features_for_norm)
        for out in outputs:
            raw = out.pop("_raw_rag_features", {})
            out["rag_features"] = normalize_rag_features(raw, norm_stats)

    return outputs, norm_stats


def generate_data_report(
    method: str,
    outputs_by_split: dict,
    docs_count: int,
) -> dict:
    report = {
        "method": method,
        "knowledge_docs_used": docs_count,
        "splits": {},
    }
    for split, outs in outputs_by_split.items():
        label_dist = Counter(o["label_name"] for o in outs)
        topic_dist = Counter(o["topic"] for o in outs)
        avg_memory_count = round(
            sum(len(o.get("retrieved_memory_ids", [])) for o in outs) / max(len(outs), 1), 3
        )
        report["splits"][split] = {
            "total":             len(outs),
            "label_distribution": dict(label_dist),
            "topic_distribution":  dict(topic_dist),
            "avg_retrieved_memory_count": avg_memory_count,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="构建 PLM 立场检测实验数据")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--knowledge", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    method = args.method
    out_dir = Path(args.output_dir) / method
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 读取样本：{args.instances}")
    all_records = load_jsonl(args.instances)
    print(f"  共加载 {len(all_records)} 条")

    docs = []
    if method in RAG_METHODS:
        print(f"📂 读取知识库：{args.knowledge}")
        docs = load_knowledge_documents(args.knowledge)
    else:
        print("  本方法不使用 RAG 知识库")

    by_split = defaultdict(list)
    for rec in all_records:
        by_split[safe_str(rec.get("split"))].append(rec)

    outputs_by_split = {}
    norm_stats = None  # 在 train 上计算，valid/test 复用

    # 先处理 train，得到归一化统计量
    split_order = ["train", "valid", "test"]
    for split in split_order:
        recs = by_split.get(split, [])
        print(f"\n🔄 处理 {split} split（{len(recs)} 条原始记录）...")
        outs, stats = process_split(recs, split, method, docs, args.top_k, norm_stats)
        if split == "train":
            norm_stats = stats  # 仅 train 计算的统计量用于后续 split
        outputs_by_split[split] = outs
        write_jsonl(outs, out_dir / f"{split}.jsonl")

    report = generate_data_report(method, outputs_by_split, len(docs))
    if norm_stats:
        report["rag_feature_normalization_stats"] = norm_stats
    write_json(report, out_dir / "data_report.json")

    print(f"""
📊 数据准备完成（method={method}）：
  train：{len(outputs_by_split.get('train', []))} 条
  valid：{len(outputs_by_split.get('valid', []))} 条
  test： {len(outputs_by_split.get('test', []))} 条
  输出目录：{out_dir}
    """)


if __name__ == "__main__":
    main()
