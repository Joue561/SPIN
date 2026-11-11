# -*- coding: utf-8 -*-
"""
01_build_cross_topic_splits.py
构建 Leave-One-Topic-Out 划分：
  source_train = 非 held-out topic 且 split=train 的样本（仅它们可用于构建 memory）
  target_eval  = held-out topic 的评估样本（默认 split=test；语义字段不足时可回退 all / train_as_eval）
held-out 中缺少 LLM 语义字段的样本输出到 to_annotate.jsonl。
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from ct_common import (F_CONF, REQUIRED_SEMANTIC_FIELDS, SEMANTIC_FIELDS, UNKNOWN, add_common_args, has_semantic,
                       log_args, missing_semantic_fields, normalize_label, read_jsonl, report_path, setup_logger, stage_guard,
                       topic_dirname, write_json, write_jsonl)


def parse_args():
    ap = argparse.ArgumentParser(description="构建跨话题 LOTO 划分")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_split", default="test", help="held-out topic 优先使用的 split（test/dev/all）")
    ap.add_argument("--fallback_eval_split", default="all", choices=["all", "train_as_eval", "none"],
                    help="eval_split 语义覆盖不足时的回退方式")
    ap.add_argument("--min_semantic_coverage", type=float, default=0.5,
                    help="eval_split 中具有语义字段的样本比例低于此值则回退")
    ap.add_argument("--min_eval_samples", type=int, default=20,
                    help="eval_split 中可用样本少于此值则回退")
    ap.add_argument("--keep_missing_semantic", type=bool_arg(), default=False,
                    help="是否保留缺语义字段的 held-out 样本（字段填 unknown）；默认排除并写入 to_annotate")
    ap.add_argument("--sample_limit", type=int, default=None, help="每个 held-out topic 的 target_eval 最多样本数（调试用，seed 抽样）")
    ap.add_argument("--topics", nargs="*", default=None, help="限定参与实验的 topic（默认自动识别全部）")
    add_common_args(ap)
    return ap.parse_args()


def bool_arg():
    from ct_common import str2bool
    return str2bool


def prepare_records(raw, logger):
    uid_counter = Counter()
    recs = []
    for i, r in enumerate(raw):
        try:
            topic = str(r.get("topic") or r.get("main_target") or "UNKNOWN").strip()
            r["topic"] = topic
            r["split"] = str(r.get("split") or UNKNOWN).strip().lower()
            raw_label = r.get("stance_label")
            lab = normalize_label(raw_label)
            if lab is not None and lab != raw_label:
                r["stance_label_raw"] = raw_label
            r["stance_label"] = lab
            nid = r.get("node_id")
            uid = r.get("sample_uid") or f"{topic}::{nid if nid is not None else 'row' + str(i)}"
            uid_counter[uid] += 1
            if uid_counter[uid] > 1:
                uid = f"{uid}::dup{uid_counter[uid]}"
            r["sample_uid"] = uid
            r["_row_index"] = i
            recs.append(r)
        except Exception as e:  # 单条坏数据不影响整体
            logger.warning("第 %d 条记录预处理失败，已跳过: %s", i, e)
    dups = sum(c - 1 for c in uid_counter.values() if c > 1)
    if dups:
        logger.warning("发现 %d 个重复 (topic,node_id)，已追加 ::dupN 后缀区分", dups)
    return recs


def select_eval(heldout_recs, args, logger, topic):
    if args.eval_split == "all":
        cand = list(heldout_recs)
    else:
        cand = [r for r in heldout_recs if r["split"] == args.eval_split]
    sem_ok = sum(has_semantic(r) for r in cand)
    cov = sem_ok / len(cand) if cand else 0.0
    info = {"requested_eval_split": args.eval_split, "requested_count": len(cand),
            "requested_semantic_coverage": round(cov, 4)}
    if cand and cov >= args.min_semantic_coverage and sem_ok >= args.min_eval_samples:
        info["used_eval_split"] = args.eval_split
        info["fallback_used"] = False
        return cand, info
    logger.warning("[%s] eval_split=%s 不足（n=%d, 语义覆盖=%.2f），回退方式=%s",
                   topic, args.eval_split, len(cand), cov, args.fallback_eval_split)
    if args.fallback_eval_split == "all":
        cand = list(heldout_recs)
    elif args.fallback_eval_split == "train_as_eval":
        cand = [r for r in heldout_recs if r["split"] == "train"]
    info["used_eval_split"] = args.fallback_eval_split if args.fallback_eval_split != "none" else args.eval_split
    info["fallback_used"] = args.fallback_eval_split != "none"
    info["fallback_note"] = ("held-out topic 的所有样本均不参与 memory 构建，"
                             "因此使用其 train/dev 样本做评估不构成泄漏；论文中需注明评估集口径。")
    return cand, info


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    logger = setup_logger("01_build_cross_topic_splits", out_dir / "logs", args.debug)
    log_args(logger, args)
    if args.debug and args.sample_limit is None:
        args.sample_limit = 30
        logger.info("debug 模式：sample_limit 默认设为 30")

    raw = read_jsonl(args.instances, logger)
    logger.info("读取样本 %d 条", len(raw))
    recs = prepare_records(raw, logger)

    all_topics = sorted({r["topic"] for r in recs})
    if args.topics:
        keep = set(args.topics)
        recs = [r for r in recs if r["topic"] in keep]
        all_topics = [t for t in all_topics if t in keep]
    logger.info("识别到 topic (%d): %s", len(all_topics), all_topics)
    if len(all_topics) < 2:
        raise SystemExit("topic 数量不足 2，无法进行 leave-one-topic-out 实验")

    # 全局数据画像
    profile = {}
    for t in all_topics:
        tr = [r for r in recs if r["topic"] == t]
        miss = Counter()
        for r in tr:
            for f in missing_semantic_fields(r):
                miss[f] += 1
        profile[t] = {
            "n": len(tr),
            "split_counts": dict(Counter(r["split"] for r in tr)),
            "label_counts": dict(Counter(str(r["stance_label"]) for r in tr)),
            "missing_semantic_by_field": dict(miss),
            "n_without_semantic": sum(not has_semantic(r) for r in tr),
            "n_missing_confidence": sum(r.get(F_CONF) in (None, "") for r in tr),
        }
        if profile[t]["n_without_semantic"]:
            logger.warning("[%s] %d 条样本缺少 LLM 语义字段 %s", t, profile[t]["n_without_semantic"], dict(miss))

    heldouts = all_topics
    if args.only_heldout_topic:
        wanted = set(args.only_heldout_topic) | {topic_dirname(x) for x in args.only_heldout_topic}
        heldouts = [t for t in all_topics if t in wanted or topic_dirname(t) in wanted]
        logger.info("仅处理 held-out: %s", heldouts)

    summary = {"instances": str(args.instances), "all_topics": all_topics, "topic_profile": profile, "folds": {}}
    rng_master = random.Random(args.seed)

    for heldout in heldouts:
        tdir = out_dir / topic_dirname(heldout)
        files = [tdir / "source_train.jsonl", tdir / "target_eval.jsonl", tdir / "to_annotate.jsonl",
                 tdir / "split_report.json"]
        if not stage_guard(files, args.overwrite, args.skip_existing, logger, f"01:{heldout}"):
            summary["folds"][heldout] = {"skipped": True}
            continue
        source_topics = [t for t in all_topics if t != heldout]

        # ---- source_train：只取 source topics 的 train，标签合法 ----
        source_train, src_drop = [], Counter()
        for r in recs:
            if r["topic"] == heldout:
                continue
            if r["split"] != "train":
                src_drop["not_train_split"] += 1
                continue
            if r["stance_label"] is None:
                src_drop["invalid_label"] += 1
                continue
            source_train.append(r)
        # 泄漏硬检查
        assert all(r["topic"] != heldout for r in source_train), "held-out topic 样本泄漏进 source_train!"

        # ---- target_eval ----
        heldout_recs = [r for r in recs if r["topic"] == heldout]
        cand, eval_info = select_eval(heldout_recs, args, logger, heldout)
        target_eval, to_annotate, tgt_drop = [], [], Counter()
        for r in cand:
            if r["stance_label"] is None:
                tgt_drop["invalid_label"] += 1
                continue
            miss = missing_semantic_fields(r)
            if miss:
                ann = {k: v for k, v in r.items() if k not in ("stance_label", "stance_label_zh", "stance_label_raw")}
                ann["missing_fields"] = miss
                to_annotate.append(ann)
                if not args.keep_missing_semantic:
                    tgt_drop["missing_semantic"] += 1
                    continue
                r = dict(r)
                for f in SEMANTIC_FIELDS:
                    if r.get(f) in (None, "", []):
                        r[f] = UNKNOWN
                r["semantic_missing"] = True
            target_eval.append(r)

        if args.sample_limit and len(target_eval) > args.sample_limit:
            rng = random.Random(f"{args.seed}::{heldout}")
            target_eval = sorted(rng.sample(target_eval, args.sample_limit), key=lambda x: x["_row_index"])
            logger.info("[%s] sample_limit=%d，已抽样", heldout, args.sample_limit)

        src_uids = {r["sample_uid"] for r in source_train}
        overlap = [r["sample_uid"] for r in target_eval if r["sample_uid"] in src_uids]
        assert not overlap, f"source_train 与 target_eval 存在重叠样本: {overlap[:5]}"

        write_jsonl(tdir / "source_train.jsonl", source_train)
        write_jsonl(tdir / "target_eval.jsonl", target_eval)
        write_jsonl(tdir / "to_annotate.jsonl", to_annotate)
        report = {
            "heldout_topic": heldout,
            "heldout_dirname": topic_dirname(heldout),
            "source_topics": source_topics,
            "n_source_train": len(source_train),
            "source_train_by_topic": dict(Counter(r["topic"] for r in source_train)),
            "source_train_label_counts": dict(Counter(r["stance_label"] for r in source_train)),
            "source_train_without_semantic": sum(not has_semantic(r) for r in source_train),
            "source_dropped": dict(src_drop),
            "n_target_eval": len(target_eval),
            "target_eval_split_counts": dict(Counter(r["split"] for r in target_eval)),
            "target_eval_label_counts": dict(Counter(r["stance_label"] for r in target_eval)),
            "target_dropped": dict(tgt_drop),
            "n_to_annotate": len(to_annotate),
            "eval_selection": eval_info,
            "leakage_checks": {
                "heldout_in_source_train": False,
                "uid_overlap_source_target": 0,
                "note": "memory 仅可由 source_train 构建；target_eval 的 stance_label 只在 07/08 评估阶段读取。",
            },
            "seed": args.seed,
        }
        write_json(tdir / "split_report.json", report)
        summary["folds"][heldout] = {k: report[k] for k in
                                     ("source_topics", "n_source_train", "n_target_eval", "n_to_annotate", "eval_selection")}
        logger.info("[held-out=%s] source=%s | source_train=%d | target_eval=%d | to_annotate=%d",
                    heldout, source_topics, len(source_train), len(target_eval), len(to_annotate))
        if to_annotate:
            logger.warning("[%s] %d 条 held-out 样本缺少语义字段，已写入 %s（请补标后重跑）",
                           heldout, len(to_annotate), tdir / "to_annotate.jsonl")

    summ_path = report_path(out_dir / "all_split_summary.json", args.overwrite)
    write_json(summ_path, summary)
    logger.info("完成，汇总写入 %s", summ_path)


if __name__ == "__main__":
    main()
