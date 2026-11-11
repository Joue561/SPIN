# -*- coding: utf-8 -*-
"""
04_retrieve_cross_topic_memory.py
对每个 held-out topic 的 target_eval 样本，在该 fold 的 source memory 中检索 top-k。

rule_hybrid 分数：
  candidate_pattern 匹配 +3 | semantic_propagation +2 | reply_relation +2 | target_relation +2
  reliability=high +1 | suggested_usage=strong +1 | 文本词汇重叠 +0.5×overlap
  （至少一个结构字段匹配才进入候选）
transferable_rag：
  rerank: S + λ·TransRel ；filter: 仅保留 TransRel ≥ threshold
random_memory_rag：从 source memory 中以 (seed, sample_uid) 为种子随机抽取 k 个。

注意：本脚本只读取 target_eval 的文本与 LLM 语义字段，不读取 stance_label；输出也不包含标签。
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ct_common import (F_CONF, F_MAP, F_PAT, F_PROP, F_REPLY, F_TREL, LABELS, RAG_METHODS, UNKNOWN, WILDCARD,
                       add_common_args, has_semantic, normalize_label, parse_confidence, short_text, stable_id,
                       char_ngrams, clean_text, list_topic_dirs, log_args, norm_entropy, norm_value,
                       normalize_patterns, overlap_ratio, read_json, read_jsonl, report_path, select_topics,
                       setup_logger, stage_guard, str2bool, write_json, write_jsonl)

TYPE_ZH = {"language_pattern": "语言模式", "semantic_propagation": "语义传导",
           "target_relation": "目标关系", "composite_pattern": "复合模式"}


def parse_args():
    ap = argparse.ArgumentParser(description="跨话题 memory 检索")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--memory_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--methods", nargs="+", default=RAG_METHODS, choices=RAG_METHODS)
    ap.add_argument("--intopic_memory_dir", default=None,
                    help="intopic_rag 的记忆目录（02 --intopic true 的输出），默认为 memory_dir 同级的 memories_intopic")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--transfer_lambda", type=float, default=0.5)
    ap.add_argument("--transfer_mode", choices=["rerank", "filter"], default="rerank")
    ap.add_argument("--transfer_threshold", type=float, default=None,
                    help="filter 模式阈值，默认读取 03 报告中的 high_transfer_threshold")
    ap.add_argument("--filter_fallback_rerank", type=str2bool, default=False,
                    help="filter 模式下若无高迁移 memory 命中，是否回退到 rerank")
    ap.add_argument("--transferable_text", choices=["source", "pattern_only"], default="source",
                    help="transferable_rag 的摘要文本风格（默认 source，以便与 source_rag 只差 TransRel）")
    ap.add_argument("--w_pattern", type=float, default=3.0)
    ap.add_argument("--w_semantic", type=float, default=2.0)
    ap.add_argument("--w_reply", type=float, default=2.0)
    ap.add_argument("--w_target", type=float, default=2.0)
    ap.add_argument("--w_reliability", type=float, default=1.0)
    ap.add_argument("--w_strong", type=float, default=1.0)
    ap.add_argument("--w_lexical", type=float, default=0.5)
    ap.add_argument("--composite_require_pattern", type=str2bool, default=True,
                    help="复合 memory 必须命中 candidate_pattern 才进入候选（否则高可靠复合 memory 会仅凭 target_relation 部分匹配被大量召回）")
    ap.add_argument("--min_structural_score", type=float, default=2.0,
                    help="结构字段匹配分（不含可靠性/词汇加分）至少达到该值才进入候选")
    ap.add_argument("--example_max_per_topic", type=int, default=2,
                    help="example_rag：每个源话题最多取几条示例（保证示例来自多个话题）")
    ap.add_argument("--example_candidates", type=int, default=300,
                    help="example_rag：按结构分取前 N 个候选再计算词汇重叠")
    ap.add_argument("--max_per_type", type=int, default=0, help=">0 时限制 top-k 中每种 memory_type 的最大数量")
    ap.add_argument("--include_examples_in_summary", type=str2bool, default=True)
    add_common_args(ap)
    return ap.parse_args()


# ---------------------------------------------------------------------------

def pattern_match(mp: str, qps: List[str]) -> bool:
    if mp == WILDCARD or not qps:
        return False
    for q in qps:
        if q == mp or (len(q) >= 2 and len(mp) >= 2 and (q in mp or mp in q)):
            return True
    return False


def build_query(r):
    return {
        "patterns": normalize_patterns(r.get(F_PAT)),
        "rr": norm_value(r.get(F_REPLY)),
        "sp": norm_value(r.get(F_PROP)),
        "tr": norm_value(r.get(F_TREL)),
        "grams": char_ngrams(clean_text(r.get("current_text"))),
    }


def components(m, q, args):
    pm = args.w_pattern if pattern_match(m["candidate_pattern"], q["patterns"]) else 0.0
    sm = args.w_semantic if m["semantic_propagation"] not in (WILDCARD, UNKNOWN) and m["semantic_propagation"] == q["sp"] else 0.0
    rm = args.w_reply if m["reply_relation"] not in (WILDCARD, UNKNOWN) and m["reply_relation"] == q["rr"] else 0.0
    tm = args.w_target if m["target_relation"] not in (WILDCARD, UNKNOWN) and m["target_relation"] == q["tr"] else 0.0
    structural = pm + sm + rm + tm
    rel = args.w_reliability if m["reliability_level"] == "high" else 0.0
    st = args.w_strong if m["suggested_usage"] == "strong" else 0.0
    lex = args.w_lexical * overlap_ratio(q["grams"], m["_grams"])
    return {"pattern": pm, "semantic": sm, "reply": rm, "target": tm, "structural": structural,
            "reliability": rel, "strong": st, "lexical": round(lex, 4),
            "base_score": structural + rel + st + lex}


def summary_of(m, style, tr_info, with_examples):
    po = style == "pattern_only"
    name = m["pattern_only_name"] if po else m["pattern_name"]
    d = m["label_distribution_weighted"]
    parts = [f"[{m['memory_id']}] {TYPE_ZH.get(m['memory_type'], m['memory_type'])}：{name}"]
    if not po:
        parts.append(f"来源话题：{', '.join(m['source_topics'])}")
    parts.append(f"结构：回复关系={m['reply_relation']}；语义传导={m['semantic_propagation']}；"
                 f"目标关系={m['target_relation']}；父子映射={m.get('stance_mapping_to_parent', WILDCARD)}")
    parts.append(f"历史立场分布：favor {d['favor']:.2f} / against {d['against']:.2f} / none {d['none']:.2f}"
                 f"（主导 {m['dominant_label']}，样本 {m['source_sample_count_raw']}）")
    parts.append(f"立场提示：该模式在源话题中有 {d['none']:.0%} 的评论未对主目标表态（none）；"
                 f"只有当前评论明确指向主目标时，favor/against 的历史分布才有参考意义")
    rel_line = f"可靠性：{m['reliability_level']}；建议用法：{m['suggested_usage']}"
    if tr_info is not None:
        rel_line += f"；迁移可靠性 TransRel={tr_info['TransRel']:.2f}（覆盖 {tr_info['n_covered_topics']} 个源话题）"
    parts.append(rel_line)
    parts.append(f"注意：{m['caution']}")
    if with_examples:
        exs = m["pattern_only_examples"] if po else m["typical_examples"]
        if exs:
            e = exs[0]
            parts.append(f"示例：父「{e['parent_text']}」→ 当前「{e['current_text']}」[{e['label']}]")
    return "\n".join(parts)


def rag_features(sel, weights, high_thr):
    k = len(sel)
    f: Dict[str, Any] = {}
    if k == 0:
        f.update({"rag_favor_prob": 0.0, "rag_against_prob": 0.0, "rag_none_prob": 0.0,
                  "rag_dominant_label_id": -1, "rag_dominant_label": None, "rag_dominant_label_probability": 0.0,
                  "rag_avg_reliability": 0.0, "rag_avg_transferability": 0.0, "rag_memory_count": 0,
                  "rag_high_reliability_count": 0, "rag_high_transfer_count": 0, "rag_htr": 0.0,
                  "rag_pattern_match_score": 0.0, "rag_semantic_match_score": 0.0,
                  "rag_target_relation_match_score": 0.0, "rag_reply_relation_match_score": 0.0,
                  "rag_evidence_entropy": 1.0, "rag_evidence_consistency": 0.0, "rag_structural_match_count": 0})
        return f
    w = np.array(weights, float)
    if w.sum() <= 0:
        w = np.ones(k)
    w = w / w.sum()
    P = np.array([[m["label_distribution_weighted"][l] for l in LABELS] for m, _, _ in sel])
    p = (w[:, None] * P).sum(0)
    p = p / p.sum()
    dom = int(np.argmax(p))
    doms = [LABELS.index(m["dominant_label"]) for m, _, _ in sel]
    trs = [t["TransRel"] if t else 0.0 for _, _, t in sel]
    rels = [t["Rel"] if t else 0.0 for _, _, t in sel]
    comps = [c for _, c, _ in sel]
    n_ht = sum(1 for t in trs if t >= high_thr)
    f.update({
        "rag_favor_prob": float(p[0]), "rag_against_prob": float(p[1]), "rag_none_prob": float(p[2]),
        "rag_dominant_label_id": dom, "rag_dominant_label": LABELS[dom],
        "rag_dominant_label_probability": float(p[dom]),
        "rag_avg_reliability": float(np.mean(rels)),
        "rag_avg_transferability": float(np.mean(trs)),
        "rag_memory_count": k,
        "rag_high_reliability_count": sum(1 for m, _, _ in sel if m["reliability_level"] == "high"),
        "rag_high_transfer_count": n_ht,
        "rag_htr": n_ht / k,
        "rag_pattern_match_score": float(np.mean([c["pattern"] for c in comps])),
        "rag_semantic_match_score": float(np.mean([c["semantic"] for c in comps])),
        "rag_target_relation_match_score": float(np.mean([c["target"] for c in comps])),
        "rag_reply_relation_match_score": float(np.mean([c["reply"] for c in comps])),
        "rag_evidence_entropy": norm_entropy(p),
        "rag_evidence_consistency": sum(1 for d in doms if d == dom) / k,
        "rag_structural_match_count": sum(1 for c in comps if c["structural"] > 0),
    })
    return f


def select_topk(cands, k, max_per_type):
    out, cnt, seen = [], Counter(), set()
    for item in cands:
        m = item[0]
        if m["memory_id"] in seen:
            continue
        if max_per_type and cnt[m["memory_type"]] >= max_per_type:
            continue
        out.append(item)
        seen.add(m["memory_id"])
        cnt[m["memory_type"]] += 1
        if len(out) >= k:
            break
    return out


def retrieve_one(r, method, mems, trmap, args, high_thr, thr):
    q = build_query(r)
    uid = r["sample_uid"]
    if method == "random_memory_rag":
        rng = random.Random(f"{args.seed}::{uid}")
        picked = rng.sample(mems, min(args.top_k, len(mems)))
        sel = []
        for m in picked:
            c = components(m, q, args)
            sel.append((m, c, trmap.get(m["memory_id"]), 0.0))
        return sel, "random", [1.0] * len(sel)
    pool = mems
    mode = "rule_hybrid"
    if method == "transferable_rag" and args.transfer_mode == "filter":
        pool = [m for m in mems if trmap.get(m["memory_id"], {}).get("TransRel", 0.0) >= thr]
        mode = f"rule_hybrid+transrel_filter(>={thr:.2f})"
    elif method == "transferable_rag":
        mode = f"rule_hybrid+transrel_rerank(lambda={args.transfer_lambda})"

    def score_pool(pl, add_tr):
        res = []
        for m in pl:
            c = components(m, q, args)
            if c["structural"] <= 0 or c["structural"] < args.min_structural_score:
                continue
            if args.composite_require_pattern and m["memory_type"] == "composite_pattern" and c["pattern"] <= 0:
                continue
            t = trmap.get(m["memory_id"])
            s = c["base_score"] + (args.transfer_lambda * (t["TransRel"] if t else 0.0) if add_tr else 0.0)
            res.append((m, c, t, s))
        # 稳定排序：分数降序，其次样本量降序，再按 id
        res.sort(key=lambda x: (-x[3], -x[0]["source_sample_count_raw"], x[0]["memory_id"]))
        return res

    add_tr = method == "transferable_rag" and args.transfer_mode == "rerank"
    cands = score_pool(pool, add_tr)
    if method == "transferable_rag" and args.transfer_mode == "filter" and not cands and args.filter_fallback_rerank:
        cands = score_pool(mems, True)
        mode += "→fallback_rerank"
    sel = select_topk([(m, c, t, s) for m, c, t, s in cands], args.top_k, args.max_per_type)
    return sel, mode, [s for _, _, _, s in sel]


class ExamplePool:
    """example_rag：从 source_train 真实样本中检索结构相似的示例（仅源话题，标签可用）。向量化打分。"""

    def __init__(self, source_train, heldout, args, logger):
        self.args = args
        recs = []
        for r in source_train:
            if r.get("topic") == heldout:
                raise RuntimeError(f"[{heldout}] example 池中出现 held-out 样本，终止")
            lab = normalize_label(r.get("stance_label"))
            if lab is None or not has_semantic(r):
                continue
            recs.append({
                "uid": r.get("sample_uid"), "topic": r.get("topic"), "label": lab,
                "main_target": clean_text(r.get("main_target")) or clean_text(r.get("topic")),
                "rr": norm_value(r.get(F_REPLY)), "sp": norm_value(r.get(F_PROP)), "tr": norm_value(r.get(F_TREL)),
                "sm": norm_value(r.get(F_MAP)), "patterns": normalize_patterns(r.get(F_PAT)),
                "conf": parse_confidence(r.get(F_CONF)) or 0.5,
                "parent_text": clean_text(r.get("parent_text")), "current_text": clean_text(r.get("current_text")),
            })
        self.recs = recs
        self.rr = np.array([x["rr"] for x in recs], dtype=object)
        self.sp = np.array([x["sp"] for x in recs], dtype=object)
        self.tr = np.array([x["tr"] for x in recs], dtype=object)
        self.conf = np.array([x["conf"] for x in recs], float)
        self.pat_index = {}
        for i, x in enumerate(recs):
            for p in x["patterns"]:
                self.pat_index.setdefault(p, []).append(i)
        self.grams = [None] * len(recs)
        logger.info("[%s] example 池：%d 条源话题训练样本", heldout, len(recs))

    def retrieve(self, r):
        a = self.args
        q = build_query(r)
        n = len(self.recs)
        if n == 0:
            return [], "example_structural", []
        pm = np.zeros(n)
        if q["patterns"]:
            hit = set()
            for p, idx in self.pat_index.items():
                if pattern_match(p, q["patterns"]):
                    hit.update(idx)
            if hit:
                pm[list(hit)] = a.w_pattern
        sm = (self.sp == q["sp"]) * a.w_semantic if q["sp"] != UNKNOWN else np.zeros(n)
        rm = (self.rr == q["rr"]) * a.w_reply if q["rr"] != UNKNOWN else np.zeros(n)
        tm = (self.tr == q["tr"]) * a.w_target if q["tr"] != UNKNOWN else np.zeros(n)
        struct = pm + sm + rm + tm
        ok = np.where((struct > 0) & (struct >= a.min_structural_score))[0]
        if ok.size == 0:
            return [], "example_structural", []
        # 结构分 + 置信度微小加成做初排，取前 N 个再算词汇重叠
        pre = struct[ok] + 0.01 * self.conf[ok]
        cand = ok[np.argsort(-pre, kind="stable")[: a.example_candidates]]
        scored = []
        for i in cand:
            if self.grams[i] is None:
                self.grams[i] = char_ngrams(self.recs[i]["current_text"])
            lex = a.w_lexical * overlap_ratio(q["grams"], self.grams[i])
            scored.append((float(struct[i] + lex + 0.01 * self.conf[i]), int(i),
                           {"pattern": float(pm[i]), "semantic": float(sm[i]), "reply": float(rm[i]),
                            "target": float(tm[i]), "structural": float(struct[i]), "reliability": 0.0,
                            "strong": 0.0, "lexical": round(lex, 4)}))
        scored.sort(key=lambda x: (-x[0], x[1]))
        sel, per_topic, seen = [], Counter(), set()
        for s, i, c in scored:
            x = self.recs[i]
            key = x["current_text"]
            if key in seen or per_topic[x["topic"]] >= a.example_max_per_topic:
                continue
            seen.add(key)
            per_topic[x["topic"]] += 1
            m = {"memory_id": stable_id("EX", x["uid"], length=8), "memory_type": "example",
                 "dominant_label": x["label"], "reliability_level": "n/a",
                 "label_distribution_weighted": {l: (1.0 if l == x["label"] else 0.0) for l in LABELS},
                 "_summary": (f"[{stable_id('EX', x['uid'], length=8)}] 源话题主目标：{x['main_target']}\n"
                              f"结构：回复关系={x['rr']}；语义传导={x['sp']}；目标关系={x['tr']}；父子映射={x['sm']}；"
                              f"候选模式={' | '.join(x['patterns']) or '无'}\n"
                              f"父评论「{short_text(x['parent_text'], 80)}」→ 当前评论「{short_text(x['current_text'], 100)}」\n"
                              f"该示例相对于其自身主目标（{x['main_target']}）的标注立场：{x['label']}"),
                 "source_uid": x["uid"], "source_topic": x["topic"]}
            sel.append((m, c, None, s))
            if len(sel) >= a.top_k:
                break
        return sel, "example_structural", [s for *_, s in sel]


def load_memory_set(mdir, heldout, eval_uids, intopic, logger):
    """读取一个 fold 的 memory + TransRel，并做泄漏检查。"""
    brep = read_json(mdir / "memory_build_report.json", {})
    if brep.get("heldout_topic") not in (None, heldout):
        raise RuntimeError(f"memory 目录 {mdir} 的 held-out 与划分不一致")
    if intopic and not brep.get("intopic"):
        raise RuntimeError(f"{mdir} 不是 intopic 记忆（请用 02 --intopic true 构建）")
    mems = read_jsonl(mdir / "memory_units.jsonl", logger)
    if intopic:
        # 同话题对照：允许含 held-out 话题，但绝不能含任何评估样本
        bad = [m["memory_id"] for m in mems for e in m.get("typical_examples", []) + m.get("counter_examples", [])
               if e.get("sample_uid") in eval_uids]
        if bad:
            raise RuntimeError(f"[{heldout}] intopic memory 的示例中出现评估样本（{len(bad)} 处），终止")
    else:
        bad = [m["memory_id"] for m in mems if heldout in m.get("source_topics", [])]
        if bad:
            raise RuntimeError(f"[{heldout}] memory 含 held-out 来源（{len(bad)} 条），终止")
    for m in mems:
        m["_grams"] = set()
        for e in m.get("typical_examples", []):
            m["_grams"] |= char_ngrams(e.get("current_text", ""))
        m["_grams"] |= char_ngrams(m["candidate_pattern"] if m["candidate_pattern"] != WILDCARD else "")
    trmap = {t["memory_id"]: t for t in read_jsonl(mdir / "memory_transferability.jsonl", logger)}
    tr_rep = read_json(mdir / "memory_transferability_report.json", {})
    return mems, trmap, tr_rep


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    logger = setup_logger("04_retrieve_cross_topic_memory", out_root / "logs", args.debug)
    log_args(logger, args)
    split_dirs = select_topics(list_topic_dirs(args.splits_dir, "target_eval.jsonl"), args.only_heldout_topic)
    summary = {}
    for sdir in split_dirs:
        srep = read_json(sdir / "split_report.json", {})
        heldout = srep.get("heldout_topic", sdir.name)
        mdir = Path(args.memory_dir) / sdir.name
        if not (mdir / "memory_units.jsonl").exists():
            logger.error("[%s] 缺少 memory_units.jsonl，跳过（请先运行 02）", heldout)
            continue
        evals = read_jsonl(sdir / "target_eval.jsonl", logger)
        eval_uids = {r["sample_uid"] for r in evals}
        mems, trmap, tr_rep = load_memory_set(mdir, heldout, eval_uids, False, logger)
        in_set = None
        if "intopic_rag" in args.methods:
            idir = Path(args.intopic_memory_dir or (Path(args.memory_dir).parent / "memories_intopic")) / sdir.name
            if not (idir / "memory_units.jsonl").exists():
                logger.error("[%s] 缺少 intopic 记忆 %s，跳过 intopic_rag（请先运行 02 --intopic true 和 03）", heldout, idir)
            else:
                in_set = load_memory_set(idir, heldout, eval_uids, True, logger)
        high_thr = tr_rep.get("params", {}).get("high_transfer_threshold", 0.45)
        thr = args.transfer_threshold if args.transfer_threshold is not None else high_thr
        if not trmap:
            logger.warning("[%s] 未找到 TransRel（请先运行 03）；transferable_rag 将退化，avg_transferability=0", heldout)
        expool = ExamplePool(read_jsonl(sdir / "source_train.jsonl", logger), heldout, args, logger) \
            if "example_rag" in args.methods else None
        odir = out_root / sdir.name
        fold_summary = {"heldout_topic": heldout, "source_topics": srep.get("source_topics"), "methods": {}}
        for method in args.methods:
            ofile = odir / f"retrieval_{method}.jsonl"
            if not stage_guard([ofile], args.overwrite, args.skip_existing, logger, f"04:{heldout}:{method}"):
                fold_summary["methods"][method] = {"skipped": True}
                continue
            cur_mems, cur_tr = mems, trmap
            if method == "intopic_rag":
                if in_set is None:
                    fold_summary["methods"][method] = {"error": "missing_intopic_memory"}
                    continue
                cur_mems, cur_tr = in_set[0], in_set[1]
            if method == "example_rag":
                style = "example"
            elif method == "pattern_only_rag":
                style = "pattern_only"
            elif method == "transferable_rag":
                style = args.transferable_text
            else:
                style = "source"
            if method == "pattern_only_rag":
                # pattern-only：词汇重叠只基于去话题化例句，避免话题词贡献分数
                mm = []
                for m in mems:
                    g = set()
                    for e in m.get("pattern_only_examples", []):
                        txt = e.get("current_text", "")
                        for ph in ("[TARGET]", "[ATTRIBUTE]", "[CONSEQUENCE]"):
                            txt = txt.replace(ph, "")
                        g |= char_ngrams(txt)
                    mm.append({**m, "_grams": g})
            else:
                mm = cur_mems
            show_tr = method == "transferable_rag"  # 只有 transferable_rag 的 prompt 中展示 TransRel，避免混淆对比
            out, n_err = [], 0
            type_cnt = Counter()
            for r in evals:
                try:
                    if method == "example_rag":
                        sel, mode, scores = expool.retrieve(r)
                    else:
                        sel, mode, scores = retrieve_one(r, method, mm, cur_tr, args, high_thr, thr)
                    weights = [max(s, 0.0) for s in scores] if method != "random_memory_rag" else [1.0] * len(sel)
                    feats = rag_features([(m, c, t) for m, c, t, _ in sel], weights, high_thr)
                    for m, *_ in sel:
                        type_cnt[m["memory_type"]] += 1
                    out.append({
                        "sample_uid": r["sample_uid"], "heldout_topic": heldout, "method": method,
                        "retrieval_method": mode, "summary_style": style,
                        "retrieved_memory_ids": [m["memory_id"] for m, *_ in sel],
                        "retrieved_memory_types": [m["memory_type"] for m, *_ in sel],
                        "retrieved_memory_dominant_labels": [m["dominant_label"] for m, *_ in sel],
                        "retrieved_memory_summaries": [m["_summary"] if "_summary" in m else
                                                       summary_of(m, style, t if show_tr else None,
                                                                  args.include_examples_in_summary)
                                                       for m, c, t, _ in sel],
                        "retrieval_scores": [round(float(s), 4) for s in scores],
                        "score_components": [c for _, c, _, _ in sel],
                        "retrieved_transrel": [round(t["TransRel"], 4) if t else 0.0 for _, _, t, _ in sel],
                        "rag_features": feats,
                    })
                except Exception as e:
                    n_err += 1
                    logger.warning("[%s/%s] 样本 %s 检索失败: %s", heldout, method, r.get("sample_uid"), e)
            write_jsonl(ofile, out)
            n = len(out)
            cov_any = sum(1 for o in out if o["rag_features"]["rag_memory_count"] > 0) / n if n else 0.0
            cov_struct = sum(1 for o in out if o["rag_features"]["rag_structural_match_count"] > 0) / n if n else 0.0
            ms = {
                "n": n, "errors": n_err, "retrieval_mode": out[0]["retrieval_method"] if out else None,
                "coverage_test": cov_any, "structural_coverage_test": cov_struct,
                "avg_memory_count": float(np.mean([o["rag_features"]["rag_memory_count"] for o in out])) if out else 0.0,
                "avg_htr": float(np.mean([o["rag_features"]["rag_htr"] for o in out])) if out else 0.0,
                "avg_transferability": float(np.mean([o["rag_features"]["rag_avg_transferability"] for o in out])) if out else 0.0,
                "avg_evidence_consistency": float(np.mean([o["rag_features"]["rag_evidence_consistency"] for o in out])) if out else 0.0,
                "retrieved_type_distribution": dict(type_cnt),
            }
            fold_summary["methods"][method] = ms
            logger.info("[%s][%s] n=%d coverage=%.3f struct_cov=%.3f avg_k=%.2f avg_HTR=%.3f",
                        heldout, method, n, cov_any, cov_struct, ms["avg_memory_count"], ms["avg_htr"])
        fold_summary["params"] = {k: getattr(args, k) for k in ("top_k", "transfer_lambda", "transfer_mode",
                                                               "transferable_text", "max_per_type", "seed",
                                                               "composite_require_pattern", "min_structural_score",
                                                               "w_pattern", "w_semantic", "w_reply", "w_target",
                                                               "w_reliability", "w_strong", "w_lexical")}
        fold_summary["transfer_threshold_used"] = thr
        fold_summary["high_transfer_threshold"] = high_thr
        write_json(report_path(odir / "retrieval_report.json", args.overwrite), fold_summary)
        summary[heldout] = fold_summary
    write_json(report_path(out_root / "all_retrieval_summary.json", args.overwrite), summary)
    logger.info("完成")


if __name__ == "__main__":
    main()