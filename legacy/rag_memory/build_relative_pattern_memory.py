#!/usr/bin/env python3
"""
build_relative_pattern_memory.py

构建“相对记忆”：记忆中只保存当前评论相对父评论的立场变换（继承 / 反转 / 中立化），
不保存相对 main_target 的绝对立场分布，从而去掉记忆中的话题标签先验。

样本过滤、质量权重、记忆键（language_pattern / semantic_propagation / target_relation /
composite_pattern）、可靠性阈值全部复用 build_llm_pattern_memory.py，唯一的区别是记忆里存什么。

父评论立场取自 parent_stance_label（train 样本的人工标注，构建记忆时使用不构成泄漏）。
  - 父评论有立场（favor/against）：relation ∈ {inherit 继承, reverse 反转, neutralize 中立化}
  - 父评论无立场（none）：只记录当前评论“是否表达立场”（express / stay_none），不记录方向

用法（与原脚本参数一致）：
python build_relative_pattern_memory.py \
  --input topic_split_inputs/Apollo_Go.jsonl \
  --output_dir processed_cmtscd_rag_memory_relative_by_topic/Apollo_Go \
  --min_confidence 0.70 --min_pattern_count 10 --train_only true

输出：rag_knowledge_documents_relative.jsonl（供 simple_rag_stance_predictor.py --method full_rag_relative 使用）
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "orig_mem", Path(__file__).resolve().parent / "build_llm_pattern_memory.py")
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)

RELS = ["inherit", "reverse", "neutralize"]
REL_ZH = {"inherit": "继承父评论立场", "reverse": "反转父评论立场", "neutralize": "不表态（中立化）"}
NONE_PARENT = ["express", "stay_none"]
NONE_PARENT_ZH = {"express": "当前评论自行表达立场", "stay_none": "当前评论也不表态"}
STANCE_ZH = {"favor": "支持", "against": "反对", "none": "无立场"}


def relation_of(parent, child):
    if parent in ("favor", "against"):
        if child == "none":
            return "neutralize"
        return "inherit" if child == parent else "reverse"
    return "express" if child in ("favor", "against") else "stay_none"


def wdist(group, keys):
    w = defaultdict(float)
    for r in group:
        w[r["_relation"]] += orig.safe_float(r.get("memory_label_weight"))
    tot = sum(w[k] for k in keys)
    return {k: round(w[k] / tot, 4) if tot else 0.0 for k in keys}, round(tot, 2)


def pick_examples(group, max_n=3):
    """优先挑选不同话题的高权重例子，展示“父立场 → 当前立场 → 变换”。"""
    cands = sorted([r for r in group if orig.safe_str(r.get("current_text"))],
                   key=lambda r: (-orig.safe_float(r.get("memory_label_weight")),
                                  -orig.safe_float(r.get("llm_confidence"))))
    chosen, seen = [], set()
    for r in cands:                      # 第一轮：每个话题最多一个
        if r.get("topic") not in seen:
            chosen.append(r); seen.add(r.get("topic"))
        if len(chosen) >= max_n:
            break
    for r in cands:                      # 第二轮：不足时补齐
        if len(chosen) >= max_n:
            break
        if r not in chosen:
            chosen.append(r)
    out = []
    for r in chosen:
        out.append({"topic": r.get("topic"), "main_target": r.get("main_target"),
                    "parent_text": orig.safe_str(r.get("parent_text"))[:80],
                    "current_text": orig.safe_str(r.get("current_text"))[:100],
                    "parent_stance": r["_parent"], "current_stance": r["_stance_norm"],
                    "relation": r["_relation"]})
        if len(out) >= max_n:
            break
    return out


def build_unit(group, memory_type, name, desc):
    polar = [r for r in group if r["_parent"] in ("favor", "against")]
    nonep = [r for r in group if r["_parent"] not in ("favor", "against")]
    rel_dist, w_polar = wdist(polar, RELS)
    np_dist, w_none = wdist(nonep, NONE_PARENT)
    total_w = w_polar + w_none
    dom_rel = max(RELS, key=lambda k: rel_dist[k]) if polar else None
    dom_p = rel_dist[dom_rel] if dom_rel else 0.0
    avg_conf = round(sum(orig.safe_float(r.get("llm_confidence")) for r in group) / max(len(group), 1), 4)
    # 可靠性沿用原阈值，但以“父有立场”部分的变换分布为依据
    level, usage = orig.compute_reliability(w_polar, dom_p, avg_conf)
    ex = pick_examples(polar) + pick_examples(nonep, max_n=1)
    rel_str = "，".join(f"{REL_ZH[k]} {rel_dist[k]*100:.1f}%" for k in sorted(RELS, key=lambda k: -rel_dist[k]))
    np_str = "，".join(f"{NONE_PARENT_ZH[k]} {np_dist[k]*100:.1f}%" for k in NONE_PARENT)
    ex_lines = []
    for i, e in enumerate(ex, 1):
        ex_lines.append(
            f"{i}. [{e['main_target']}] 父评论：{e['parent_text']}\n"
            f"   当前评论：{e['current_text']}\n"
            f"   父评论对其主目标：{STANCE_ZH.get(e['parent_stance'])} → 当前评论对其主目标："
            f"{STANCE_ZH.get(e['current_stance'])}（{REL_ZH.get(e['relation'], NONE_PARENT_ZH.get(e['relation']))}）")
    retrieval_text = f"""Pattern: {name}

【Memory 类型】
{memory_type}（相对记忆）

【描述】
{desc}

【父评论有立场时，当前评论的立场变换（n={len(polar)}）】
{rel_str if polar else "（样本不足）"}

【父评论无立场时（n={len(nonep)}）】
{np_str if nonep else "（样本不足）"}

【典型例子（立场均相对各自的主目标）】
{chr(10).join(ex_lines) if ex_lines else "（暂无）"}

【说明】
本记忆只描述“当前评论相对父评论”的立场变换规律，不包含任何话题的立场先验。"""
    doc_id = f"relrag_{memory_type}_{name.replace('+', '_').replace(' ', '_')}"
    return {
        "doc_id": doc_id, "doc_type": "relative_stance_pattern_memory", "memory_type": memory_type,
        "pattern_name": name, "retrieval_text": retrieval_text.strip(),
        "structured_metadata": {
            "memory_type": memory_type, "pattern_name": name,
            "relation_distribution_parent_polar": rel_dist, "n_parent_polar": len(polar),
            "relation_distribution_parent_none": np_dist, "n_parent_none": len(nonep),
            "dominant_relation": dom_rel, "dominant_relation_probability": dom_p,
            "reliability_level": level, "suggested_usage": usage,
            "source_sample_count_raw": len(group), "source_sample_count_weighted": total_w,
            "average_confidence": avg_conf, "topic_distribution": dict(Counter(r.get("topic") for r in group)),
            "examples": ex,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="构建相对立场变换记忆")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_confidence", type=float, default=0.70)
    ap.add_argument("--min_pattern_count", type=int, default=10)
    ap.add_argument("--train_only", default="true")
    ap.add_argument("--parent_field", default="parent_stance_label")
    a = ap.parse_args()
    train_only = a.train_only.lower() == "true"
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    recs = orig.load_jsonl(a.input)
    print(f"📂 读取 {a.input}：{len(recs)} 条")
    normalized = []
    for rec in recs:
        r = dict(rec)
        r["_norm_patterns"] = orig.normalize_candidate_patterns(r.get("llm_candidate_patterns"))
        r["_stance_norm"] = orig.normalize_label(r.get("stance_label", "none"))
        r["_llm_inferred"] = orig.safe_str(r.get("llm_inferred_stance_to_main_target", "unclear"))
        r["_llm_label_agreement"] = orig.compute_llm_label_agreement(r["_stance_norm"], r["_llm_inferred"])
        r.update(orig.compute_memory_quality(r, a.min_confidence, train_only))
        p = orig.safe_str(r.get(a.parent_field)).lower()
        r["_parent"] = p if p in ("favor", "against", "none") else "none"
        r["_relation"] = relation_of(r["_parent"], r["_stance_norm"])
        normalized.append(r)
    usable = [r for r in normalized if r.get("use_for_memory")]
    print(f"  可用：{len(usable)}  | 父立场分布：{dict(Counter(r['_parent'] for r in usable))}")
    print(f"  变换分布：{dict(Counter(r['_relation'] for r in usable))}")

    groups = {"language_pattern": defaultdict(list), "semantic_propagation": defaultdict(list),
              "target_relation": defaultdict(list), "composite_pattern": defaultdict(list)}
    for r in usable:
        rr, sp, tr = (orig.safe_str(r.get(k)) for k in ("llm_reply_relation", "llm_semantic_propagation", "llm_target_relation"))
        for p in r["_norm_patterns"]:
            groups["language_pattern"][p].append(r)
            groups["composite_pattern"][f"{p}+{rr}+{sp}+{tr}"].append(r)
        if sp:
            groups["semantic_propagation"][sp].append(r)
        if tr:
            groups["target_relation"][tr].append(r)
    descs = {"language_pattern": orig.PATTERN_DESCRIPTIONS, "semantic_propagation": orig.PROPAGATION_DESCRIPTIONS,
             "target_relation": orig.TARGET_RELATION_DESCRIPTIONS, "composite_pattern": {}}

    docs, stats = [], Counter()
    for mtype, g in groups.items():
        for name, group in sorted(g.items(), key=lambda x: -len(x[1])):
            if len(group) < a.min_pattern_count:
                continue
            d = build_unit(group, mtype, name, descs[mtype].get(name.split("+")[0], name))
            stats[d["structured_metadata"]["suggested_usage"]] += 1
            if d["structured_metadata"]["suggested_usage"] != "do_not_use":
                docs.append(d)
    with open(out / "rag_knowledge_documents_relative.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    report = {"input": a.input, "total_input_instances": len(recs), "use_for_memory_count": len(usable),
              "rag_document_count": len(docs), "documents_by_type": dict(Counter(d["memory_type"] for d in docs)),
              "suggested_usage_distribution": dict(stats),
              "relation_distribution": dict(Counter(r["_relation"] for r in usable)),
              "topic_distribution": dict(Counter(r.get("topic") for r in usable)),
              "note": "相对记忆：不含相对 main_target 的绝对立场分布"}
    json.dump(report, open(out / "relative_memory_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  ✅ rag_knowledge_documents_relative.jsonl（{len(docs)} 条） → {out}")


if __name__ == "__main__":
    main()