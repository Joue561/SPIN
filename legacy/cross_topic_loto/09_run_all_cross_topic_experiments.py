# -*- coding: utf-8 -*-
"""
09_run_all_cross_topic_experiments.py
一键运行 01→08 全流程（每一步以子进程调用对应脚本，日志与报告各自独立）。

  --run_prediction false : 只跑到 05（生成 prompts）
  --dry_run true         : 06 只生成待预测文件，跳过 07/08
  --skip_existing true   : 已完成的步骤/topic 自动跳过（断点续跑）
  --overwrite true       : 覆盖全部已有输出
  --debug true           : DEBUG 日志；未指定 --sample_limit 时每个 held-out 只取 30 条
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ct_common import ALL_METHODS, RAG_METHODS, report_path, setup_logger, str2bool, write_json

HERE = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description="跨目标泛化实验一键运行")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--output_dir", default="cross_topic_outputs")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    # 01
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--fallback_eval_split", default="all", choices=["all", "train_as_eval", "none"])
    ap.add_argument("--sample_limit", type=int, default=None)
    # 02
    ap.add_argument("--min_confidence", type=float, default=0.70)
    ap.add_argument("--min_pattern_count", type=int, default=5)
    ap.add_argument("--low_conf_mode", choices=["downweight", "exclude"], default="downweight")
    ap.add_argument("--detopic_lexicon", default=None)
    # 03
    ap.add_argument("--sample_order", choices=["file", "hash"], default="file", help="06 的 --sample_order")
    ap.add_argument("--pred_sample_limit", type=int, default=None,
                    help="06 中每个 held-out × method 最多预测多少条（不影响 01 划分）")
    ap.add_argument("--min_topic_count", type=int, default=2)
    ap.add_argument("--high_transfer_threshold", type=float, default=0.45)
    ap.add_argument("--jsd_form", choices=["divergence", "distance"], default="divergence")
    # 04
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--transfer_lambda", type=float, default=0.5)
    ap.add_argument("--transfer_mode", choices=["rerank", "filter"], default="rerank")
    # 05
    ap.add_argument("--context_mode", choices=["causal", "posthoc"], default="causal")
    # 06
    ap.add_argument("--run_prediction", type=str2bool, default=True)
    ap.add_argument("--provider", default="openai", choices=["openai", "deepseek", "qwen", "local", "mock"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", type=str2bool, default=True)
    ap.add_argument("--dry_run", type=str2bool, default=False)
    # 通用
    ap.add_argument("--overwrite", type=str2bool, default=False)
    ap.add_argument("--skip_existing", type=str2bool, default=False)
    ap.add_argument("--only_heldout_topic", nargs="*", default=None)
    ap.add_argument("--debug", type=str2bool, default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--continue_on_error", type=str2bool, default=False)
    ap.add_argument("--from_step", type=int, default=1, help="从第几步开始（1-8）")
    ap.add_argument("--to_step", type=int, default=8)
    return ap.parse_args()


def b(x):
    return "true" if x else "false"


def main():
    a = parse_args()
    # 子进程在脚本目录运行，路径一律转为绝对路径
    a.instances = str(Path(a.instances).resolve())
    if a.detopic_lexicon:
        a.detopic_lexicon = str(Path(a.detopic_lexicon).resolve())
    out = Path(a.output_dir).resolve()
    logger = setup_logger("09_run_all_cross_topic_experiments", out / "logs", a.debug)
    d = {k: out / k for k in ("splits", "memories", "memories_intopic", "retrieval", "prompts", "predictions",
                              "metrics", "analysis")}
    common = ["--overwrite", b(a.overwrite), "--skip_existing", b(a.skip_existing), "--debug", b(a.debug),
              "--seed", str(a.seed)]
    only = (["--only_heldout_topic", *a.only_heldout_topic] if a.only_heldout_topic else [])
    rag_methods = [m for m in a.methods if m in RAG_METHODS]
    if "context_memory" in a.methods and "source_rag" not in rag_methods:
        rag_methods.append("source_rag")  # context_memory 复用 source_rag 的检索结果
    sample_limit = a.sample_limit if a.sample_limit is not None else (30 if a.debug else None)

    steps = [
        (1, "01_build_cross_topic_splits.py",
         ["--instances", a.instances, "--output_dir", str(d["splits"]), "--eval_split", a.eval_split,
          "--fallback_eval_split", a.fallback_eval_split]
         + (["--sample_limit", str(sample_limit)] if sample_limit else []) + only),
        (2, "02_build_cross_topic_memory.py",
         ["--splits_dir", str(d["splits"]), "--output_dir", str(d["memories"]), "--min_confidence", str(a.min_confidence),
          "--min_pattern_count", str(a.min_pattern_count), "--low_conf_mode", a.low_conf_mode]
         + (["--detopic_lexicon", a.detopic_lexicon] if a.detopic_lexicon else []) + only),
        (3, "03_compute_memory_transferability.py",
         ["--memory_dir", str(d["memories"]), "--output_dir", str(d["memories"]), "--min_topic_count", str(a.min_topic_count),
          "--high_transfer_threshold", str(a.high_transfer_threshold), "--jsd_form", a.jsd_form] + only),
        (4, "04_retrieve_cross_topic_memory.py",
         ["--splits_dir", str(d["splits"]), "--memory_dir", str(d["memories"]), "--output_dir", str(d["retrieval"]),
          "--methods", *(rag_methods or RAG_METHODS), "--top_k", str(a.top_k), "--transfer_lambda", str(a.transfer_lambda),
          "--transfer_mode", a.transfer_mode] + only),
        (5, "05_build_cross_topic_prompts.py",
         ["--splits_dir", str(d["splits"]), "--retrieval_dir", str(d["retrieval"]), "--output_dir", str(d["prompts"]),
          "--methods", *a.methods, "--context_mode", a.context_mode] + only),
        (6, "06_run_cross_topic_llm_prediction.py",
         ["--prompts_dir", str(d["prompts"]), "--output_dir", str(d["predictions"]), "--provider", a.provider,
          "--model", a.model, "--temperature", str(a.temperature), "--max_tokens", str(a.max_tokens),
          "--workers", str(a.workers), "--resume", b(a.resume), "--dry_run", b(a.dry_run), "--methods", *a.methods]
         + (["--base_url", a.base_url] if a.base_url else [])
         + (["--sample_limit", str(a.pred_sample_limit), "--sample_order", a.sample_order] if a.pred_sample_limit else [])
         + only),
        (7, "07_evaluate_cross_topic_predictions.py",
         ["--predictions_dir", str(d["predictions"]), "--splits_dir", str(d["splits"]),
          "--retrieval_dir", str(d["retrieval"]), "--output_dir", str(d["metrics"]), "--methods", *a.methods] + only),
        (8, "08_analyze_transferability.py",
         ["--predictions_dir", str(d["predictions"]), "--retrieval_dir", str(d["retrieval"]),
          "--memory_dir", str(d["memories"]), "--splits_dir", str(d["splits"]), "--output_dir", str(d["analysis"]),
          "--rag_methods", *(rag_methods or RAG_METHODS), "--methods", *a.methods] + only),
    ]
    if "intopic_rag" in a.methods:
        # 同话题对照记忆：在第 2/3 步之后额外构建
        steps.insert(2, (2, "02_build_cross_topic_memory.py",
                         ["--splits_dir", str(d["splits"]), "--output_dir", str(d["memories_intopic"]),
                          "--intopic", "true", "--instances", a.instances,
                          "--min_confidence", str(a.min_confidence), "--min_pattern_count", str(a.min_pattern_count),
                          "--low_conf_mode", a.low_conf_mode]
                         + (["--detopic_lexicon", a.detopic_lexicon] if a.detopic_lexicon else []) + only))
        steps.insert(4, (3, "03_compute_memory_transferability.py",
                         ["--memory_dir", str(d["memories_intopic"]), "--output_dir", str(d["memories_intopic"]),
                          "--min_topic_count", str(a.min_topic_count),
                          "--high_transfer_threshold", str(a.high_transfer_threshold), "--jsd_form", a.jsd_form] + only))
    report = {"args": vars(a), "steps": []}
    t_all = time.time()
    for idx, script, extra in steps:
        if idx < a.from_step or idx > a.to_step:
            continue
        if idx >= 6 and not a.run_prediction:
            logger.info("--run_prediction=false，停止于第 5 步（prompts 已生成）")
            break
        if idx >= 7 and a.dry_run:
            logger.info("--dry_run=true，跳过评估与分析（尚无预测结果）")
            break
        if idx == 4 and not rag_methods:
            logger.info("methods 中无 RAG 方法，跳过检索")
            continue
        cmd = [sys.executable, str(HERE / script)] + extra + common
        logger.info("========== Step %d: %s ==========", idx, script)
        logger.debug("CMD: %s", " ".join(cmd))
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(HERE))
        dt = time.time() - t0
        report["steps"].append({"step": idx, "script": script, "returncode": rc, "elapsed_sec": round(dt, 1),
                                "cmd": cmd[1:]})
        if rc != 0:
            logger.error("Step %d 失败（returncode=%d）", idx, rc)
            if not a.continue_on_error:
                break
        else:
            logger.info("Step %d 完成，用时 %.1fs", idx, dt)
    report["total_elapsed_sec"] = round(time.time() - t_all, 1)
    report["outputs"] = {k: str(v) for k, v in d.items()}
    write_json(report_path(out / "pipeline_report.json", a.overwrite), report)
    failed = [s for s in report["steps"] if s["returncode"] != 0]
    if failed:
        logger.error("流水线存在失败步骤：%s", [s["script"] for s in failed])
        sys.exit(1)
    logger.info("全部完成。总表：%s ；分析报告：%s", d["metrics"] / "final_summary_table.md", d["analysis"] / "analysis_report.md")


if __name__ == "__main__":
    main()