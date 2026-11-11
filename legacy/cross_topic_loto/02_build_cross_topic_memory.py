# -*- coding: utf-8 -*-
"""
02_build_cross_topic_memory.py
对每个 held-out topic，仅用其 source_train.jsonl 构建结构化模式记忆：
  language_pattern_memory      键: candidate_pattern
  semantic_propagation_memory  键: (reply_relation, semantic_propagation)
  target_relation_memory       键: (target_relation, stance_mapping_to_parent)
  composite_pattern_memory     键: (candidate_pattern, reply_relation, semantic_propagation, target_relation)
并输出 RAG 文档：
  rag_knowledge_documents_source.jsonl        含源话题信息的普通文档
  rag_knowledge_documents_pattern_only.jsonl  去话题化文档（[TARGET]/[ATTRIBUTE]/[CONSEQUENCE]）
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ct_common import (F_CONF, F_MAP, F_PAT, F_PROP, F_REPLY, F_TREL, LABELS, UNKNOWN, WILDCARD, Detopicalizer,
                       add_common_args, clean_text, dist_from_counts, has_semantic, heldout_topic_of_dir,
                       list_topic_dirs, log_args, norm_value, normalize_label, normalize_patterns, parse_confidence,
                       read_json, read_jsonl, report_path, select_topics, setup_logger, short_text, stable_id,
                       stage_guard, str2bool, topic_dirname, write_json, write_jsonl)

MEMORY_TYPES = {
    "language_pattern": ("LP", "语言模式记忆"),
    "semantic_propagation": ("SP", "语义传导记忆"),
    "target_relation": ("TR", "目标关系记忆"),
    "composite_pattern": ("CP", "复合模式记忆"),
}


def parse_args():
    ap = argparse.ArgumentParser(description="基于 source_train 构建跨话题结构化模式记忆")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_confidence", type=float, default=0.70)
    ap.add_argument("--low_conf_mode", choices=["downweight", "exclude"], default="downweight")
    ap.add_argument("--low_conf_weight", type=float, default=0.5, help="低置信样本权重")
    ap.add_argument("--missing_conf_weight", type=float, default=0.8, help="缺失置信度样本权重")
    ap.add_argument("--min_pattern_count", type=int, default=5, help="memory 最少原始样本数")
    ap.add_argument("--min_composite_count", type=int, default=None, help="复合 memory 最少样本数（默认同上）")
    ap.add_argument("--weak_target_relations", nargs="*", default=["irrelevant", "unclear", "unknown"])
    ap.add_argument("--weak_reply_relations", nargs="*", default=["unknown"])
    ap.add_argument("--include_weak_in_all", type=str2bool, default=False,
                    help="弱结构样本（目标关系不明/回复关系 unknown）是否也进入 LP/SP/CP memory；默认只进入 TR memory 且标记为 weak")
    ap.add_argument("--high_min_count", type=int, default=20)
    ap.add_argument("--high_min_prob", type=float, default=0.70)
    ap.add_argument("--high_min_conf", type=float, default=0.75)
    ap.add_argument("--medium_min_prob", type=float, default=0.55)
    ap.add_argument("--n_typical", type=int, default=3)
    ap.add_argument("--n_counter", type=int, default=2)
    ap.add_argument("--detopic_lexicon", default=None,
                    help="可选 JSON: {topics:{topic:[词]}, attributes:[...], consequences:[...]} 用于补充去话题化词表")
    ap.add_argument("--intopic", type=str2bool, default=False,
                    help="同话题对照：记忆额外纳入 held-out 话题自身的 train 样本（排除全部 target_eval 样本），"
                         "输出到 --output_dir（建议 memories_intopic），memory_id 加 IN_ 前缀")
    ap.add_argument("--instances", default=None, help="--intopic 时必需：原始 instances 文件（用于取 held-out 的 train 样本）")
    add_common_args(ap)
    return ap.parse_args()


def load_heldout_train(instances, heldout, eval_uids, logger):
    """用与 01 完全相同的预处理（sample_uid 规则一致）读取 held-out 话题的 train 样本，并排除所有评估样本。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("split01", Path(__file__).resolve().parent / "01_build_cross_topic_splits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    recs = mod.prepare_records(read_jsonl(instances, logger), logger)
    out, excluded = [], 0
    for r in recs:
        if r["topic"] != heldout or r["split"] != "train" or r["stance_label"] is None:
            continue
        if r["sample_uid"] in eval_uids:
            excluded += 1
            continue
        out.append(r)
    return out, excluded


# ---------------------------------------------------------------------------
class Agg:
    def __init__(self):
        self.raw = Counter()
        self.w = Counter()
        self.topic_raw: Dict[str, Counter] = defaultdict(Counter)
        self.topic_w: Dict[str, Counter] = defaultdict(Counter)
        self.confs: List[float] = []
        self.samples: List[Dict[str, Any]] = []
        self.weak_struct = False

    def add(self, s):
        lab, w, t = s["label"], s["weight"], s["topic"]
        self.raw[lab] += 1
        self.w[lab] += w
        self.topic_raw[t][lab] += 1
        self.topic_w[t][lab] += w
        if s["conf"] is not None:
            self.confs.append(s["conf"])
        self.samples.append(s)

    @property
    def n_raw(self):
        return sum(self.raw.values())


def prepare_sample(r, args, stats):
    lab = normalize_label(r.get("stance_label"))
    if lab is None:
        stats["drop_invalid_label"] += 1
        return None
    if not has_semantic(r):
        stats["drop_missing_semantic"] += 1
        return None
    conf = parse_confidence(r.get(F_CONF))
    if conf is None:
        w = args.missing_conf_weight
        stats["missing_confidence"] += 1
    elif conf < args.min_confidence:
        if args.low_conf_mode == "exclude":
            stats["drop_low_confidence"] += 1
            return None
        w = args.low_conf_weight
        stats["downweighted_low_confidence"] += 1
    else:
        w = 1.0
    rr, sp, tr, sm = (norm_value(r.get(F_REPLY)), norm_value(r.get(F_PROP)),
                      norm_value(r.get(F_TREL)), norm_value(r.get(F_MAP)))
    strong_ok = tr not in set(args.weak_target_relations) and rr not in set(args.weak_reply_relations)
    if not strong_ok:
        stats["weak_structure_samples"] += 1
    return {
        "uid": r.get("sample_uid"), "topic": r.get("topic"), "label": lab, "weight": w, "conf": conf,
        "rr": rr, "sp": sp, "tr": tr, "sm": sm, "patterns": normalize_patterns(r.get(F_PAT)),
        "strong_ok": strong_ok, "current_text": clean_text(r.get("current_text")),
        "parent_text": clean_text(r.get("parent_text")), "main_target": clean_text(r.get("main_target")),
        "target_anchor": clean_text(r.get("target_anchor")),
    }


def keys_for(s, include_weak):
    """返回该样本贡献的 (memory_type, key_tuple, fields) 列表。"""
    out = []
    full = s["strong_ok"] or include_weak
    if full:
        for p in s["patterns"]:
            out.append(("language_pattern", (p,), {"candidate_pattern": p}))
            out.append(("composite_pattern", (p, s["rr"], s["sp"], s["tr"]),
                        {"candidate_pattern": p, "reply_relation": s["rr"], "semantic_propagation": s["sp"],
                         "target_relation": s["tr"]}))
        out.append(("semantic_propagation", (s["rr"], s["sp"]),
                    {"reply_relation": s["rr"], "semantic_propagation": s["sp"]}))
    out.append(("target_relation", (s["tr"], s["sm"]),
                {"target_relation": s["tr"], "stance_mapping_to_parent": s["sm"]}))
    return out


def reliability(n_raw, dom_p, avg_conf, args):
    if n_raw >= args.high_min_count and dom_p >= args.high_min_prob and avg_conf >= args.high_min_conf:
        return "high"
    if n_raw >= args.min_pattern_count and dom_p >= args.medium_min_prob:
        return "medium"
    return "low"


def make_example(s, detop=None):
    d = {"parent_text": short_text(s["parent_text"], 60), "current_text": short_text(s["current_text"], 80),
         "label": s["label"]}
    if detop is None:
        d.update({"sample_uid": s["uid"], "topic": s["topic"]})
    else:
        d = {k: (detop(v) if k in ("parent_text", "current_text") else v) for k, v in d.items()}
    return d


def dedup_examples(samples, detop, k):
    """去话题化后不同话题的例子可能变得完全相同，去重后保留 k 个。"""
    out, seen = [], set()
    for s in samples:
        e = make_example(s, detop)
        key = (e["parent_text"], e["current_text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= k:
            break
    return out


def pattern_name(mtype, f, detop=None):
    g = (lambda x: detop(x) if detop else x)
    if mtype == "language_pattern":
        return f"候选模式={g(f['candidate_pattern'])}"
    if mtype == "semantic_propagation":
        return f"回复关系={f['reply_relation']} → 语义传导={f['semantic_propagation']}"
    if mtype == "target_relation":
        return f"目标关系={f['target_relation']} / 父子立场映射={f['stance_mapping_to_parent']}"
    return (f"{g(f['candidate_pattern'])} | 回复={f['reply_relation']} | 传导={f['semantic_propagation']} "
            f"| 目标={f['target_relation']}")


def fmt_dist(d):
    return " / ".join(f"{l} {d[l]:.2f}" for l in LABELS)


def render_text(u, pattern_only: bool):
    zh = MEMORY_TYPES[u["memory_type"]][1]
    name = u["pattern_only_name"] if pattern_only else u["pattern_name"]
    lines = [f"【{zh}】{name}"]
    if not pattern_only:
        lines.append(f"来源话题: {', '.join(u['source_topics'])}")
    lines.append(f"回复关系: {u['reply_relation']}; 语义传导: {u['semantic_propagation']}; "
                 f"目标关系: {u['target_relation']}; 父子映射: {u['stance_mapping_to_parent']}; "
                 f"候选模式: {u['pattern_only_candidate'] if pattern_only else u['candidate_pattern']}")
    lines.append(f"样本数: {u['source_sample_count_raw']} (加权 {u['source_sample_count_weighted']:.1f}); "
                 f"立场分布: {fmt_dist(u['label_distribution_weighted'])}; "
                 f"主导立场: {u['dominant_label']} ({u['dominant_label_probability']:.2f})")
    lines.append(f"可靠性: {u['reliability_level']}; 建议用法: {u['suggested_usage']}; 注意: {u['caution']}")
    exs = u["pattern_only_examples"] if pattern_only else u["typical_examples"]
    for e in exs[:2]:
        lines.append(f"例: 父「{e['parent_text']}」→ 当前「{e['current_text']}」[{e['label']}]")
    return "\n".join(lines)


def build_units(samples, args, detop, source_topics, heldout):
    aggs: Dict[tuple, Agg] = {}
    fields: Dict[tuple, dict] = {}
    for s in samples:
        for mtype, key, f in keys_for(s, args.include_weak_in_all):
            k = (mtype,) + key
            if k not in aggs:
                aggs[k] = Agg()
                fields[k] = f
            aggs[k].add(s)
            if not s["strong_ok"]:
                aggs[k].weak_struct = True

    min_comp = args.min_composite_count or args.min_pattern_count
    units, dropped = [], Counter()
    weak_tr, weak_rr = set(args.weak_target_relations), set(args.weak_reply_relations)
    for k, a in aggs.items():
        mtype = k[0]
        thr = min_comp if mtype == "composite_pattern" else args.min_pattern_count
        if a.n_raw < thr:
            dropped[mtype] += 1
            continue
        f = fields[k]
        base = {"candidate_pattern": WILDCARD, "reply_relation": WILDCARD, "semantic_propagation": WILDCARD,
                "target_relation": WILDCARD, "stance_mapping_to_parent": WILDCARD}
        base.update(f)
        dist_raw = dict(zip(LABELS, dist_from_counts(a.raw)))
        dist_w = dict(zip(LABELS, dist_from_counts(a.w)))
        dom = max(LABELS, key=lambda l: dist_w[l])
        dom_p = dist_w[dom]
        avg_conf = float(np.mean(a.confs)) if a.confs else 0.0
        level = reliability(a.n_raw, dom_p, avg_conf, args)
        weak_struct = base["target_relation"] in weak_tr or base["reply_relation"] in weak_rr or (
            mtype == "target_relation" and a.weak_struct)
        if level == "high" and dom_p >= args.high_min_prob and not weak_struct:
            usage = "strong"
        elif level in ("high", "medium") and not weak_struct:
            usage = "moderate"
        else:
            usage = "weak"
        cautions = []
        if dom_p < 0.6:
            cautions.append("立场分布较分散，仅作弱参考")
        if weak_struct:
            cautions.append("目标关系不明确或与主目标无关，提示可能为 none/话题偏移，需结合语境")
        if a.n_raw < args.high_min_count:
            cautions.append("样本量较小")
        if len(a.topic_raw) <= 1:
            cautions.append("仅来自单一源话题，迁移性未知")
        cautions.append("历史模式仅作参考，需核验当前语境")
        typical_all = sorted([s for s in a.samples if s["label"] == dom],
                             key=lambda s: (-(s["conf"] or 0), len(s["current_text"])))
        counter_all = sorted([s for s in a.samples if s["label"] != dom],
                             key=lambda s: (-(s["conf"] or 0), len(s["current_text"])))
        typical, counter = typical_all[: args.n_typical], counter_all[: args.n_counter]
        prefix = MEMORY_TYPES[mtype][0]
        pattern_key = "||".join(str(x) for x in k[1:])
        topic_distribution = {
            t: {"count_raw": int(sum(a.topic_raw[t].values())),
                "count_weighted": float(sum(a.topic_w[t].values())),
                "label_counts_raw": {l: int(a.topic_raw[t][l]) for l in LABELS},
                "label_counts_weighted": {l: float(a.topic_w[t][l]) for l in LABELS}}
            for t in sorted(a.topic_raw)}
        u = {
            "memory_id": stable_id(prefix, mtype, pattern_key),
            "memory_type": mtype,
            "pattern_key": pattern_key,
            "pattern_name": pattern_name(mtype, base),
            "pattern_only_name": pattern_name(mtype, base, detop),
            "candidate_pattern": base["candidate_pattern"],
            "pattern_only_candidate": detop(base["candidate_pattern"]) if base["candidate_pattern"] != WILDCARD else WILDCARD,
            "reply_relation": base["reply_relation"],
            "semantic_propagation": base["semantic_propagation"],
            "target_relation": base["target_relation"],
            "stance_mapping_to_parent": base["stance_mapping_to_parent"],
            "heldout_topic": heldout,
            "source_topics": sorted(a.topic_raw.keys()),
            "fold_source_topics": source_topics,
            "topic_distribution": topic_distribution,
            "source_sample_count_raw": int(a.n_raw),
            "source_sample_count_weighted": float(sum(a.w.values())),
            "label_distribution_raw": dist_raw,
            "label_distribution_weighted": dist_w,
            "dominant_label": dom,
            "dominant_label_probability": float(dom_p),
            "average_confidence": avg_conf,
            "typical_examples": [make_example(s) for s in typical],
            "counter_examples": [make_example(s) for s in counter],
            "pattern_only_examples": dedup_examples(typical_all, detop, args.n_typical),
            "pattern_only_counter_examples": dedup_examples(counter_all, detop, args.n_counter),
            "reliability_level": level,
            "suggested_usage": usage,
            "weak_structure": bool(weak_struct),
            "caution": "；".join(cautions),
        }
        u["retrieval_text"] = render_text(u, pattern_only=False)
        u["pattern_only_retrieval_text"] = render_text(u, pattern_only=True)
        u["structured_metadata"] = {
            "memory_type": mtype, "candidate_pattern": u["candidate_pattern"],
            "reply_relation": u["reply_relation"], "semantic_propagation": u["semantic_propagation"],
            "target_relation": u["target_relation"], "stance_mapping_to_parent": u["stance_mapping_to_parent"],
            "dominant_label": dom, "dominant_label_probability": dom_p, "reliability_level": level,
            "suggested_usage": usage, "n_source_topics": len(a.topic_raw),
        }
        units.append(u)
    units.sort(key=lambda u: (u["memory_type"], -u["source_sample_count_raw"]))
    return units, dropped


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    logger = setup_logger("02_build_cross_topic_memory", out_root / "logs", args.debug)
    log_args(logger, args)
    splits_dir = Path(args.splits_dir)
    topic_dirs = select_topics(list_topic_dirs(splits_dir, "source_train.jsonl"), args.only_heldout_topic)
    if not topic_dirs:
        raise SystemExit(f"{splits_dir} 下未找到任何 held-out 划分目录")
    summary = {}
    for tdir in topic_dirs:
        split_rep = read_json(tdir / "split_report.json", {})
        heldout = split_rep.get("heldout_topic") or tdir.name
        source_topics = split_rep.get("source_topics", [])
        odir = out_root / tdir.name
        type_files = {m: odir / f"{m}_memory.jsonl" for m in MEMORY_TYPES}
        files = list(type_files.values()) + [odir / "memory_units.jsonl", odir / "rag_knowledge_documents_source.jsonl",
                                             odir / "rag_knowledge_documents_pattern_only.jsonl",
                                             odir / "memory_build_report.json"]
        if not stage_guard(files, args.overwrite, args.skip_existing, logger, f"02:{heldout}"):
            summary[heldout] = {"skipped": True}
            continue

        raw = read_jsonl(tdir / "source_train.jsonl", logger)
        n_source_train = len(raw)
        intopic_info = None
        if args.intopic:
            if not args.instances:
                raise SystemExit("--intopic true 需要同时提供 --instances")
            eval_uids = {r["sample_uid"] for r in read_jsonl(tdir / "target_eval.jsonl", logger)}
            ht, excluded = load_heldout_train(args.instances, heldout, eval_uids, logger)
            leak = [r["sample_uid"] for r in ht if r["sample_uid"] in eval_uids]
            if leak:
                raise RuntimeError(f"[{heldout}] intopic memory 混入评估样本 {len(leak)} 条，终止")
            if not ht:
                logger.warning("[%s] held-out 话题没有可用的 train 样本（可能评估集用了 all 回退），intopic 记忆等同于 source 记忆", heldout)
            intopic_info = {"n_heldout_train_added": len(ht), "n_heldout_train_excluded_as_eval": excluded,
                            "eval_uid_overlap": 0}
            logger.info("[%s] intopic：加入 held-out 话题 train 样本 %d 条（因属于评估集排除 %d 条）",
                        heldout, len(ht), excluded)
            src_part = raw
            raw = src_part + ht
        # ---- 泄漏硬检查 ----
        leaked = [r.get("sample_uid") for r in raw if r.get("topic") == heldout]
        if leaked and not args.intopic:
            raise RuntimeError(f"[{heldout}] source_train 中发现 held-out 样本 {len(leaked)} 条，终止以防泄漏")
        non_train = sum(str(r.get("split", "")).lower() != "train" for r in raw)
        if non_train:
            raise RuntimeError(f"[{heldout}] source_train 中存在 {non_train} 条非 train 样本")

        stats = Counter()
        samples = []
        for r in raw:
            try:
                s = prepare_sample(r, args, stats)
            except Exception as e:
                stats["drop_error"] += 1
                logger.debug("样本处理失败 %s: %s", r.get("sample_uid"), e)
                continue
            if s:
                samples.append(s)
        logger.info("[%s] source_train=%d，可用于 memory=%d，统计=%s", heldout, len(raw), len(samples), dict(stats))

        extra_terms = set()
        for s in samples:
            extra_terms.update([s["main_target"], s["target_anchor"], s["topic"]])
        detop = Detopicalizer.build(extra_terms, args.detopic_lexicon)
        mem_topics = source_topics + ([heldout] if args.intopic else [])
        units, dropped = build_units(samples, args, detop, mem_topics, heldout)
        if args.intopic:
            for u in units:
                u["memory_id"] = "IN_" + u["memory_id"]
                u["intopic"] = True
        else:
            assert all(heldout not in u["source_topics"] for u in units), "memory 中出现 held-out topic！"

        for m, fp in type_files.items():
            write_jsonl(fp, [u for u in units if u["memory_type"] == m])
        write_jsonl(odir / "memory_units.jsonl", units)
        src_docs, po_docs, residual = [], [], 0
        for u in units:
            src_docs.append({"doc_id": f"DOC_{u['memory_id']}", "memory_id": u["memory_id"],
                             "memory_type": u["memory_type"], "text": u["retrieval_text"],
                             "metadata": {**u["structured_metadata"], "source_topics": u["source_topics"],
                                          "label_distribution_weighted": u["label_distribution_weighted"]}})
            po_text = u["pattern_only_retrieval_text"]
            res = detop.residual_terms(po_text)
            if res:
                residual += 1
                logger.debug("pattern-only 残留话题词 %s: %s", u["memory_id"], res)
            po_docs.append({"doc_id": f"PODOC_{u['memory_id']}", "memory_id": u["memory_id"],
                            "memory_type": u["memory_type"], "text": po_text,
                            "metadata": {**u["structured_metadata"],
                                         "candidate_pattern": u["pattern_only_candidate"],
                                         "label_distribution_weighted": u["label_distribution_weighted"],
                                         "caution": u["caution"]}})
        write_jsonl(odir / "rag_knowledge_documents_source.jsonl", src_docs)
        write_jsonl(odir / "rag_knowledge_documents_pattern_only.jsonl", po_docs)

        rep = {
            "heldout_topic": heldout,
            "source_topics": mem_topics,
            "cross_topic_source_topics": source_topics,
            "intopic": bool(args.intopic),
            "intopic_info": intopic_info,
            "n_source_train": n_source_train,
            "n_memory_input_samples": len(raw),
            "n_samples_used": len(samples),
            "sample_filter_stats": dict(stats),
            "n_memory_units": len(units),
            "by_type": dict(Counter(u["memory_type"] for u in units)),
            "by_reliability": dict(Counter(u["reliability_level"] for u in units)),
            "by_usage": dict(Counter(u["suggested_usage"] for u in units)),
            "dropped_below_min_count": dict(dropped),
            "pattern_only_residual_topic_term_docs": residual,
            "pattern_only_residual_rate": residual / len(units) if units else 0.0,
            "params": {k: getattr(args, k) for k in ("min_confidence", "low_conf_mode", "low_conf_weight",
                                                       "min_pattern_count", "min_composite_count",
                                                       "weak_target_relations", "weak_reply_relations",
                                                       "include_weak_in_all")},
            "leakage_checks": ({"eval_samples_in_memory": 0, "non_train_samples": 0,
                                "note": "intopic 对照：含 held-out 话题 train，但不含任何评估样本"}
                               if args.intopic else
                               {"heldout_samples_in_source_train": 0, "non_train_samples": 0,
                                "heldout_in_memory_source_topics": False}),
        }
        write_json(odir / "memory_build_report.json", rep)
        summary[heldout] = {k: rep[k] for k in ("source_topics", "n_samples_used", "n_memory_units", "by_type",
                                                "by_reliability", "pattern_only_residual_rate")}
        logger.info("[%s] memory=%d %s | pattern-only 残留话题词文档比例=%.3f",
                    heldout, len(units), rep["by_type"], rep["pattern_only_residual_rate"])
    write_json(report_path(out_root / "all_memory_build_summary.json", args.overwrite), summary)
    logger.info("完成")


if __name__ == "__main__":
    main()