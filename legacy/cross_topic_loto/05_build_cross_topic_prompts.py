# -*- coding: utf-8 -*-
"""
05_build_cross_topic_prompts.py
为 6 种 method 构建 LLM 立场预测 prompt。

防泄漏：
  - 采用字段白名单构建 prompt 视图，stance_label / stance_label_zh / llm_inferred_stance_to_main_target /
    llm_reason_short 等字段永远不会进入 prompt；
  - 构建后再次扫描 prompt，若出现禁用字段名则报错并跳过该样本。
上下文：默认 causal_context_text（主实验）；--context_mode posthoc 使用 posthoc_context_text（会在记录中标明）。
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ct_common import (ALL_METHODS, F_MAP, F_PAT, F_PROP, F_REPLY, F_TREL, FORBIDDEN_PROMPT_FIELDS, RAG_METHODS,
                       RAG_PROMPT_METHODS, NO_SEMANTIC_METHODS, RETRIEVAL_SOURCE, stable_id,
                       add_common_args, clean_text, list_topic_dirs, log_args, normalize_patterns, read_json,
                       read_jsonl, report_path, select_topics, setup_logger, stage_guard, write_json, write_jsonl)

PROMPT_WHITELIST = ["sample_uid", "main_target", "current_text", "parent_text", "root_text", "ancestor_texts",
                    "causal_context_text", "posthoc_context_text", F_REPLY, F_PROP, F_TREL, F_MAP, F_PAT, "topic"]

HEADER = """你是一名社交媒体多轮会话立场检测专家。
任务：判断【当前评论】相对于【主目标】的立场。
可选标签：favor, against, none。

请注意：
1. 不要只根据字面情感判断；
2. 需要结合父评论和会话上下文；
3. 如果当前评论是短文本，如“确实”“呵呵”“666”，需要判断其如何回应父评论；
4. 如果存在反讽、反问、表面赞同实际否定，需要识别真实语用含义；
5. 如果当前评论只是攻击回复者、话题转移或与主目标无关，应优先考虑 none；"""

HEADER_CURRENT_ONLY = """你是一名社交媒体立场检测专家。
任务：判断【当前评论】相对于【主目标】的立场。
可选标签：favor, against, none。

请注意：
1. 不要只根据字面情感判断；
2. 如果存在反讽、反问、表面赞同实际否定，需要识别真实语用含义；
3. 如果当前评论与主目标无关，应优先考虑 none；"""

RAG_RULES = """6. RAG memory 只是历史模式参考，不可机械照搬；
7. 如果 RAG memory 与当前语境不匹配，应忽略或降低其权重；
8. 历史模式记忆来自其他话题，只用于帮助理解当前评论的语用功能（例如：反讽通常表示否定父评论，简短附和通常继承父评论的立场），不能直接据此推断当前评论对主目标的立场；
9. 只有当当前评论本身、或经由父评论的立场传导后，明确指向主目标并表达态度时，才判为 favor 或 against；如果只是讨论与主目标相关的其他话题（如薪资、家庭、产品细节等）而没有对主目标本身表态，应判为 none。"""

EXAMPLE_RULES = """6. 下面的示例来自其他话题，示例的标注立场是相对于【示例自己的主目标】，不是当前主目标；
7. 示例只用于参考“相似的回复结构下，立场如何由父评论传导到当前评论”，不可照搬示例标签；与当前语境不匹配的示例应忽略；
8. 只有当当前评论本身、或经由父评论的立场传导后，明确指向当前主目标并表达态度时，才判为 favor 或 against；如果只是讨论与主目标相关的其他话题（如薪资、家庭、产品细节等）而没有对主目标本身表态，应判为 none。"""

EMPTY_MEMORY_TEXT = "（本设置不提供历史模式记忆，请仅依据会话上下文与语义传导信息判断）"

JSON_SPEC_RAG = """请只输出如下 JSON（不要输出任何其他文字）：
{
  "predicted_label": "favor/against/none",
  "confidence": 0.0-1.0,
  "reason_short": "简要说明判断依据",
  "used_rag": true/false,
  "used_memory_ids": [],
  "target_relation_warning": "none/topic_shift/reply_author_only/unclear"
}"""

JSON_SPEC_NORAG = """请只输出如下 JSON（不要输出任何其他文字；本设置没有检索记忆，used_rag 填 false，used_memory_ids 填 []）：
{
  "predicted_label": "favor/against/none",
  "confidence": 0.0-1.0,
  "reason_short": "简要说明判断依据",
  "used_rag": false,
  "used_memory_ids": [],
  "target_relation_warning": "none/topic_shift/reply_author_only/unclear"
}"""


def parse_args():
    ap = argparse.ArgumentParser(description="构建跨话题立场预测 prompt")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--retrieval_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--context_mode", choices=["causal", "posthoc"], default="causal")
    ap.add_argument("--max_context_chars", type=int, default=1500, help="上下文过长时保留尾部（离当前评论最近的部分）")
    ap.add_argument("--max_memory_chars", type=int, default=2500)
    add_common_args(ap)
    return ap.parse_args()


def view_of(r):
    """白名单视图：只有这些字段可能进入 prompt。"""
    return {k: r.get(k) for k in PROMPT_WHITELIST}


def build_context(v, mode, max_chars):
    key = "causal_context_text" if mode == "causal" else "posthoc_context_text"
    ctx = clean_text(v.get(key))
    if not ctx:
        parts = []
        anc = v.get("ancestor_texts")
        if isinstance(anc, list) and anc:
            parts.extend(f"[祖先{i + 1}] {clean_text(a)}" for i, a in enumerate(anc))
        elif v.get("root_text"):
            parts.append(f"[根帖] {clean_text(v['root_text'])}")
        if v.get("parent_text"):
            parts.append(f"[父评论] {clean_text(v['parent_text'])}")
        parts.append(f"[当前评论] {clean_text(v.get('current_text'))}")
        ctx = "\n".join(parts)
    if len(ctx) > max_chars:
        ctx = "…" + ctx[-max_chars:]
    return ctx


def semantic_block(v):
    pats = normalize_patterns(v.get(F_PAT))
    return ("【语义传导信息】\n"
            f"回复关系：{clean_text(v.get(F_REPLY)) or 'unknown'}\n"
            f"语义传导：{clean_text(v.get(F_PROP)) or 'unknown'}\n"
            f"父子立场映射：{clean_text(v.get(F_MAP)) or 'unknown'}\n"
            f"目标关系：{clean_text(v.get(F_TREL)) or 'unknown'}\n"
            f"候选语言模式：{' | '.join(pats) if pats else '无'}")


def memory_block(ret, max_chars, item="记忆"):
    sums = (ret or {}).get("retrieved_memory_summaries") or []
    if not sums:
        if item == "示例":
            return "（未检索到与当前样本结构相似的源话题示例，请仅依据会话上下文判断）"
        return "（未检索到与当前样本结构匹配的源话题历史模式记忆，请仅依据会话上下文判断）"
    out, total = [], 0
    for i, s in enumerate(sums, 1):
        block = f"{item}{i}：\n{s}"
        if total + len(block) > max_chars and out:
            break
        out.append(block)
        total += len(block)
    return "\n\n".join(out)


def build_prompt(v, method, ret, args):
    target = clean_text(v.get("main_target")) or clean_text(v.get("topic"))
    cur = clean_text(v.get("current_text"))
    if method == "current_only":
        return "\n\n".join([HEADER_CURRENT_ONLY, f"【主目标】\n{target}", f"【当前评论】\n{cur}", JSON_SPEC_NORAG])
    ctx_title = "【会话上下文】" if args.context_mode == "causal" else "【会话上下文（posthoc：含子回复反馈）】"
    rules = EXAMPLE_RULES if method == "example_rag" else RAG_RULES
    parts = [HEADER + ("\n" + rules if method in RAG_PROMPT_METHODS else ""),
             f"【主目标】\n{target}",
             f"{ctx_title}\n{build_context(v, args.context_mode, args.max_context_chars)}",
             f"【当前评论】\n{cur}"]
    if method not in NO_SEMANTIC_METHODS:
        parts.append(semantic_block(v))
    if method in RAG_PROMPT_METHODS:
        if method == "example_rag":
            title, mem = "【检索到的源话题相似结构示例】", memory_block(ret, args.max_memory_chars, item="示例")
        else:
            title = "【检索到的历史模式记忆（去话题化）】" if method == "pattern_only_rag" else "【检索到的源话题历史模式记忆】"
            mem = EMPTY_MEMORY_TEXT if method == "rag_empty" else memory_block(ret, args.max_memory_chars)
        parts.append(f"{title}\n{mem}")
        parts.append(JSON_SPEC_RAG)
    else:
        parts.append(JSON_SPEC_NORAG)
    return "\n\n".join(parts)


def leak_scan(prompt, rec):
    issues = [f for f in FORBIDDEN_PROMPT_FIELDS if f in prompt]
    return issues


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    logger = setup_logger("05_build_cross_topic_prompts", out_root / "logs", args.debug)
    log_args(logger, args)
    if args.context_mode == "posthoc":
        logger.warning("context_mode=posthoc：使用含子回复反馈的上下文，属于 posthoc setting，结果需单独报告")
    dirs = select_topics(list_topic_dirs(args.splits_dir, "target_eval.jsonl"), args.only_heldout_topic)
    summary = {}
    for sdir in dirs:
        srep = read_json(sdir / "split_report.json", {})
        heldout = srep.get("heldout_topic", sdir.name)
        evals = read_jsonl(sdir / "target_eval.jsonl", logger)
        odir = out_root / sdir.name
        fold = {"heldout_topic": heldout, "source_topics": srep.get("source_topics"), "context_mode": args.context_mode,
                "methods": {}}
        for method in args.methods:
            ofile = odir / f"prompts_{method}.jsonl"
            if not stage_guard([ofile], args.overwrite, args.skip_existing, logger, f"05:{heldout}:{method}"):
                fold["methods"][method] = {"skipped": True}
                continue
            retmap = {}
            needs_ret = method in RAG_METHODS or method in RETRIEVAL_SOURCE
            if needs_ret:
                rfile = Path(args.retrieval_dir) / sdir.name / f"retrieval_{RETRIEVAL_SOURCE.get(method, method)}.jsonl"
                if not rfile.exists():
                    logger.error("[%s] 缺少 %s，跳过 %s（请先运行 04）", heldout, rfile, method)
                    fold["methods"][method] = {"error": "missing_retrieval"}
                    continue
                retmap = {r["sample_uid"]: r for r in read_jsonl(rfile, logger)}
            rows, stats = [], Counter()
            for r in evals:
                try:
                    v = view_of(r)
                    ret = retmap.get(r["sample_uid"]) if needs_ret else None
                    if needs_ret and ret is None:
                        stats["missing_retrieval_record"] += 1
                    prompt = build_prompt(v, method, ret, args)
                    issues = leak_scan(prompt, r)
                    if issues:
                        stats["leak_blocked"] += 1
                        logger.error("[%s/%s] %s prompt 含禁用字段 %s，已跳过", heldout, method, r["sample_uid"], issues)
                        continue
                    rows.append({
                        "prompt_id": f"{sdir.name}::{method}::{r['sample_uid']}",
                        "sample_uid": r["sample_uid"], "heldout_topic": heldout, "heldout_dirname": sdir.name,
                        "method": method, "context_mode": args.context_mode,
                        "candidate_memory_ids": (ret or {}).get("retrieved_memory_ids", []),
                        "prompt_hash": stable_id("P", prompt, length=12),
                        "prompt": prompt,
                    })
                    stats["ok"] += 1
                except Exception as e:
                    stats["error"] += 1
                    logger.warning("[%s/%s] 样本 %s 构建失败: %s", heldout, method, r.get("sample_uid"), e)
            write_jsonl(ofile, rows)
            lens = [len(x["prompt"]) for x in rows] or [0]
            fold["methods"][method] = {**dict(stats), "n_prompts": len(rows),
                                       "avg_prompt_chars": sum(lens) / len(lens), "max_prompt_chars": max(lens)}
            logger.info("[%s][%s] prompts=%d avg_chars=%.0f %s", heldout, method, len(rows),
                        fold["methods"][method]["avg_prompt_chars"], dict(stats))
        write_json(report_path(odir / "prompt_report.json", args.overwrite), fold)
        summary[heldout] = fold
    write_json(report_path(out_root / "all_prompt_summary.json", args.overwrite), summary)
    logger.info("完成")


if __name__ == "__main__":
    main()