# -*- coding: utf-8 -*-
"""
06_run_cross_topic_llm_prediction.py
读取 prompts/{topic}/prompts_{method}.jsonl，调用 LLM 预测立场。

provider（均走 OpenAI 兼容 /chat/completions 接口，requests 实现）：
  openai   OPENAI_API_KEY      https://api.openai.com/v1
  deepseek DEEPSEEK_API_KEY    https://api.deepseek.com/v1
  qwen     DASHSCOPE_API_KEY   https://dashscope.aliyuncs.com/compatible-mode/v1
  local    LOCAL_API_KEY(可空) http://localhost:8000/v1   （vLLM / Ollama / LM Studio 等）
  mock     不调用 API，基于 prompt 的启发式规则输出（仅用于流程冒烟测试，严禁用于论文结果）
--base_url / --api_key 可覆盖默认。
特性：断点续跑（--resume）、指数退避重试、JSON 解析修复（规则修复 + 可选 LLM 重格式化）、
保存原始响应、并发（--workers）、--dry_run 只生成待预测文件。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ct_common import (ALL_METHODS, LABELS, add_common_args, append_jsonl, list_topic_dirs, log_args,
                       normalize_label, read_jsonl, report_path, select_topics, setup_logger, str2bool, write_json,
                       write_jsonl)

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "local": ("http://localhost:8000/v1", "LOCAL_API_KEY"),
    "mock": ("", ""),
}
WARNINGS = {"none", "topic_shift", "reply_author_only", "unclear"}
SYSTEM_PROMPT = "你是严谨的立场检测标注专家。只输出一个合法 JSON 对象，不要输出 Markdown 或其他文字。"


def parse_args():
    ap = argparse.ArgumentParser(description="跨话题 LLM 立场预测")
    ap.add_argument("--prompts_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--provider", choices=list(PROVIDERS), default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key", default=None, help="默认读取对应环境变量")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--json_mode", choices=["auto", "true", "false"], default="auto",
                    help="是否发送 response_format=json_object（auto: local 关闭，其余开启）")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--resume", type=str2bool, default=True)
    ap.add_argument("--dry_run", type=str2bool, default=False)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.0, help="每次请求后的休眠秒数（限速）")
    ap.add_argument("--repair_with_llm", type=str2bool, default=True, help="规则解析失败时让模型重格式化一次")
    ap.add_argument("--sample_limit", type=int, default=None, help="每个 prompts 文件最多预测多少条（调试）")
    ap.add_argument("--sample_order", choices=["file", "hash"], default="file",
                    help="配合 --sample_limit：file=取文件前 N 条；hash=按 (seed, sample_uid) 哈希排序后取前 N 条"
                         "（随机且各方法一致，适合正式实验抽样）")
    ap.add_argument("--preflight", type=str2bool, default=True, help="批量预测前先发一条测试请求检查配置")
    add_common_args(ap)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# JSON 解析与修复
# ---------------------------------------------------------------------------

def _extract_obj(text: str) -> Optional[str]:
    t = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return t[start:] + "}"  # 截断时尝试补全


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("：", ":").replace("，", ",")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    s = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)  # 未加引号的键
    return s


def normalize_pred(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lab = normalize_label(obj.get("predicted_label") or obj.get("label") or obj.get("stance"))
    if lab is None:
        raw = str(obj.get("predicted_label", "")).lower()
        hits = [l for l in LABELS if l in raw]
        lab = hits[0] if len(hits) == 1 else None
    if lab is None:
        return None
    try:
        conf = float(obj.get("confidence", 0.5))
        conf = conf / 100 if 1 < conf <= 100 else conf
        conf = min(max(conf, 0.0), 1.0)
    except (TypeError, ValueError):
        conf = None
    ids = obj.get("used_memory_ids") or []
    if isinstance(ids, str):
        ids = [x.strip() for x in re.split(r"[,，\s]+", ids) if x.strip()]
    warn = str(obj.get("target_relation_warning", "none")).strip().lower()
    if warn not in WARNINGS:
        warn = "unclear"
    used = obj.get("used_rag", False)
    used = used if isinstance(used, bool) else str(used).lower() in {"true", "1", "yes"}
    return {"predicted_label": lab, "confidence": conf, "reason_short": str(obj.get("reason_short", ""))[:300],
            "used_rag": used, "used_memory_ids": [str(i) for i in ids if i], "target_relation_warning": warn}


def parse_response(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if not text:
        return None, "empty"
    blob = _extract_obj(text)
    if blob:
        for status, cand in (("ok", blob), ("repaired", _repair(blob))):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    p = normalize_pred(obj)
                    if p:
                        return p, status
            except json.JSONDecodeError:
                pass
    m = re.search(r'predicted_label"?\s*[:：]\s*"?\s*(favor|against|none)', text, re.I)
    if m:
        return normalize_pred({"predicted_label": m.group(1)}), "regex_fallback"
    hits = {l for l in LABELS if re.search(rf"\b{l}\b", text.lower())}
    if len(hits) == 1:
        return normalize_pred({"predicted_label": hits.pop()}), "bare_label_fallback"
    return None, "failed"


# ---------------------------------------------------------------------------
# LLM 客户端
# ---------------------------------------------------------------------------
class FatalAPIError(RuntimeError):
    """不可重试的错误（400/401/403/404 等配置问题），应立即停止整个运行。"""


class ContentFilterError(RuntimeError):
    """服务端内容审核拦截了单条 prompt（如“检测到敏感词”）。只跳过该样本，不中断整个运行，也不重试。"""


CONTENT_FILTER_PATTERNS = ["敏感词", "敏感内容", "违规", "内容安全", "审核", "content_filter", "content filter",
                           "content_policy", "content management policy", "sensitive", "data_inspection_failed",
                           "inappropriate content", "safety system"]


def is_content_filter(text: str) -> bool:
    t = (text or "").lower()
    return any(p.lower() in t for p in CONTENT_FILTER_PATTERNS)


class LLMClient:
    def __init__(self, args, logger):
        self.args, self.logger = args, logger
        base, env = PROVIDERS[args.provider]
        raw = (args.base_url or os.environ.get(f"{args.provider.upper()}_BASE_URL") or base).strip().rstrip("/")
        # 兼容把完整接口地址填进 base_url 的情况：.../v1/chat/completions → .../v1
        fixed = re.sub(r"/chat/completions$", "", raw)
        if fixed != raw:
            logger.warning("base_url 末尾的 /chat/completions 已自动去掉：%s → %s", raw, fixed)
        self.base_url = fixed
        logger.info("API 地址: %s/chat/completions | model=%s", self.base_url, args.model)
        self.api_key = args.api_key or (os.environ.get(env) if env else None)
        self.json_mode = (args.provider != "local") if args.json_mode == "auto" else args.json_mode == "true"
        if args.provider not in ("mock", "local") and not self.api_key:
            raise SystemExit(f"未找到 API key：请设置环境变量 {env} 或 --api_key；无 key 时可用 --dry_run true")
        import requests  # noqa
        self.session = requests.Session()

    def chat(self, prompt: str, system: str = SYSTEM_PROMPT) -> Tuple[str, Dict[str, Any]]:
        if self.args.provider == "mock":
            return mock_llm(prompt), {"mock": True}
        import requests
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.args.model, "temperature": self.args.temperature, "max_tokens": self.args.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        last_err = None
        for attempt in range(self.args.max_retries):
            try:
                resp = self.session.post(url, headers=headers, json=body, timeout=self.args.timeout)
                if resp.status_code == 400 and self.json_mode and "response_format" in resp.text:
                    body.pop("response_format", None)  # 服务不支持 json mode 时自动降级
                    self.json_mode = False
                    continue
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code in (400, 403, 451) and is_content_filter(resp.text):
                    raise ContentFilterError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                if 400 <= resp.status_code < 500:
                    raise FatalAPIError(f"HTTP {resp.status_code} ({url}): {resp.text[:500]}")
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                return content, {"usage": data.get("usage"), "attempts": attempt + 1}
            except (FatalAPIError, ContentFilterError):
                raise
            except Exception as e:  # noqa
                last_err = e
                wait = min(60.0, (2 ** attempt) + random.random())
                self.logger.debug("请求失败（第 %d 次）：%s；%.1fs 后重试", attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"重试 {self.args.max_retries} 次后仍失败: {last_err}")


def mock_llm(prompt: str) -> str:
    """冒烟测试用启发式：RAG 时跟随记忆主导立场多数票，否则用关键词规则。"""
    target_rel = re.search(r"目标关系：(\S+)", prompt)
    doms = re.findall(r"主导 (favor|against|none)", prompt)
    ids = re.findall(r"\[((?:LP|SP|TR|CP)_[0-9a-f]+)\]", prompt)
    if target_rel and target_rel.group(1) in ("topic_shift", "reply_author_only"):
        lab = "none"
    elif doms:
        lab = Counter(doms[:3]).most_common(1)[0][0]
    else:
        cur = prompt.split("【当前评论】")[-1][:80]
        if any(k in cur for k in ["支持", "不错", "方便", "说得对", "确实"]):
            lab = "favor"
        elif any(k in cur for k in ["垃圾", "反对", "呵呵", "？", "?"]):
            lab = "against"
        else:
            lab = "none"
    return json.dumps({"predicted_label": lab, "confidence": 0.7, "reason_short": "mock",
                       "used_rag": bool(doms), "used_memory_ids": ids[:2],
                       "target_relation_warning": "none"}, ensure_ascii=False)


# ---------------------------------------------------------------------------

def predict_one(client: LLMClient, p: Dict[str, Any], args) -> Dict[str, Any]:
    t0 = time.time()
    raw, meta = client.chat(p["prompt"])
    pred, status = parse_response(raw)
    repair_raw = None
    if pred is None and args.repair_with_llm and args.provider != "mock":
        fix_prompt = ("下面是一段立场判断输出，请把它转换为合法 JSON，字段为 predicted_label(favor/against/none)、"
                      "confidence、reason_short、used_rag、used_memory_ids、target_relation_warning。只输出 JSON。\n\n" + raw)
        repair_raw, _ = client.chat(fix_prompt)
        pred, status = parse_response(repair_raw)
        status = f"llm_repair:{status}"
    if args.sleep:
        time.sleep(args.sleep)
    base = {"prompt_id": p["prompt_id"], "sample_uid": p["sample_uid"], "heldout_topic": p["heldout_topic"],
            "heldout_dirname": p.get("heldout_dirname"), "method": p["method"], "context_mode": p.get("context_mode"),
            "prompt_hash": p.get("prompt_hash"),
            "provider": args.provider, "model": args.model, "temperature": args.temperature,
            "raw_response": raw, "repair_response": repair_raw, "parse_status": status,
            "latency_sec": round(time.time() - t0, 3), "meta": meta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if pred is None:
        base["error"] = "parse_failed"
        return base
    if p["method"] in ("current_only", "context_only", "dynamic_semantic", "rag_empty"):
        pred["used_rag"] = False
    cand = set(p.get("candidate_memory_ids") or [])
    pred["used_memory_ids_valid"] = [i for i in pred["used_memory_ids"] if i in cand]
    base.update(pred)
    return base


def select_subset(prompts, args):
    if not args.sample_limit:
        return prompts
    if args.sample_order == "hash":
        import hashlib
        key = lambda p: hashlib.md5(f"{args.seed}::{p['sample_uid']}".encode("utf-8")).hexdigest()
        keep = {p["sample_uid"] for p in sorted(prompts, key=key)[: args.sample_limit]}
        return [p for p in prompts if p["sample_uid"] in keep]
    return prompts[: args.sample_limit]


def _is_stale(r, cur_hash):
    h = cur_hash.get(r.get("prompt_id"))
    return h is not None and r.get("prompt_hash") != h


def resume_done_ids(pf, prompts, out_dir, tname, m, logger, dry=False):
    """
    断点续跑：返回无需重新预测的 prompt_id。
    prompt 内容已变化（prompt_hash 不同或旧记录无 hash）的旧预测视为过期：
    移入 predictions_stale/ 备份并从预测文件中删除，随后重新预测，避免新旧 prompt 的结果混用。
    """
    cur_hash = {p["prompt_id"]: p.get("prompt_hash") for p in prompts}
    allrec = read_jsonl(pf, logger)
    stale = [r for r in allrec if _is_stale(r, cur_hash)]
    if stale:
        logger.warning("[%s/%s] %d 条旧预测对应的 prompt 已变化，%s", tname, m, len(stale),
                       "将在正式运行时重新预测" if dry else "已备份至 predictions_stale/ 并将重新预测")
        if not dry:
            write_jsonl(Path(out_dir) / "predictions_stale" / f"{pf.stem}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl", stale)
            write_jsonl(pf, [r for r in allrec if not _is_stale(r, cur_hash)])
    return {r["prompt_id"] for r in allrec
            if normalize_label(r.get("predicted_label")) and not _is_stale(r, cur_hash)}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    logger = setup_logger("06_run_cross_topic_llm_prediction", out_dir / "logs", args.debug)
    log_args(logger, argparse.Namespace(**{k: ("***" if k == "api_key" and v else v) for k, v in vars(args).items()}))
    if args.provider == "mock":
        logger.warning("provider=mock：仅用于流程测试，结果不可用于论文！")

    dirs = select_topics(list_topic_dirs(args.prompts_dir, "prompt_report.json") or
                         [p for p in Path(args.prompts_dir).iterdir() if p.is_dir() and list(p.glob("prompts_*.jsonl"))],
                         args.only_heldout_topic)
    jobs = []
    for d in dirs:
        for m in args.methods:
            f = d / f"prompts_{m}.jsonl"
            if f.exists():
                jobs.append((d.name, m, f))
            else:
                logger.warning("缺少 %s", f)
    if not jobs:
        raise SystemExit("没有可预测的 prompts 文件")

    report = {"provider": args.provider, "model": args.model, "dry_run": args.dry_run, "files": {}}
    if args.dry_run:
        pend_dir = out_dir / "pending"
        for tname, m, f in jobs:
            prompts = read_jsonl(f, logger)
            done = set()
            pf = out_dir / f"predictions_{tname}_{m}.jsonl"
            if pf.exists():
                done = resume_done_ids(pf, prompts, out_dir, tname, m, logger, dry=True)
            pending = [p for p in select_subset(prompts, args) if p["prompt_id"] not in done]
            write_jsonl(pend_dir / f"pending_{tname}_{m}.jsonl", pending)
            report["files"][f"{tname}/{m}"] = {"n_prompts": len(prompts), "n_done": len(done), "n_pending": len(pending)}
            logger.info("[dry_run] %s/%s 待预测 %d 条", tname, m, len(pending))
        write_json(report_path(out_dir / "prediction_run_report.json", args.overwrite), report)
        logger.info("dry_run 完成：待预测文件位于 %s", pend_dir)
        return

    client = LLMClient(args, logger)
    if args.provider != "mock" and args.preflight:
        try:
            txt, _ = client.chat('请只输出 JSON：{"predicted_label": "none"}')
            logger.info("连通性检查通过，模型返回：%s", txt[:100].replace("\n", " "))
        except Exception as e:
            raise SystemExit(f"连通性检查失败，未开始批量预测：{e}\n"
                             f"请检查 OPENAI_BASE_URL（应以 /v1 结尾）、API key 和 --model 名称")
    fail_file = out_dir / "prediction_failures.jsonl"
    lock = threading.Lock()
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    for tname, m, f in jobs:
        pf = out_dir / f"predictions_{tname}_{m}.jsonl"
        prompts = read_jsonl(f, logger)
        prompts = select_subset(prompts, args)
        done = set()
        if pf.exists():
            if args.resume:
                done = resume_done_ids(pf, read_jsonl(f, logger), out_dir, tname, m, logger)
            elif args.overwrite:
                pf.unlink()
            elif args.skip_existing:
                logger.info("%s 已存在，跳过", pf)
                report["files"][f"{tname}/{m}"] = {"skipped": True}
                continue
            else:
                raise FileExistsError(f"{pf} 已存在；使用 --resume true 续跑，或 --overwrite true 覆盖")
        todo = [p for p in prompts if p["prompt_id"] not in done]
        logger.info("[%s/%s] 共 %d 条，已完成 %d，待预测 %d", tname, m, len(prompts), len(done), len(todo))
        stats = Counter()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(predict_one, client, p, args): p for p in todo}
            it = as_completed(futs)
            if tqdm:
                it = tqdm(it, total=len(futs), desc=f"{tname}/{m}", ncols=90)
            for fu in it:
                p = futs[fu]
                try:
                    rec = fu.result()
                except ContentFilterError as e:
                    stats["content_filter"] += 1
                    with lock:
                        append_jsonl(fail_file, {"prompt_id": p["prompt_id"], "sample_uid": p["sample_uid"],
                                                 "heldout_topic": p["heldout_topic"], "method": m,
                                                 "error": f"content_filter: {e}",
                                                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                    logger.warning("[%s/%s] 样本 %s 被服务端内容审核拦截，已跳过（记录于 prediction_failures.jsonl）",
                                   tname, m, p["sample_uid"])
                    continue
                except FatalAPIError as e:
                    logger.error("不可重试的 API 错误，停止运行：%s", e)
                    for other in futs:
                        other.cancel()
                    write_json(report_path(out_dir / "prediction_run_report.json", args.overwrite),
                               {**report, "aborted": str(e)})
                    raise SystemExit(1)
                except Exception as e:
                    stats["api_failed"] += 1
                    with lock:
                        append_jsonl(fail_file, {"prompt_id": p["prompt_id"], "sample_uid": p["sample_uid"],
                                                 "heldout_topic": p["heldout_topic"], "method": m,
                                                 "error": f"api_error: {e}",
                                                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                    continue
                with lock:
                    if rec.get("error"):
                        stats["parse_failed"] += 1
                        append_jsonl(fail_file, rec)
                    else:
                        stats[rec["parse_status"]] += 1
                        append_jsonl(pf, rec)
        report["files"][f"{tname}/{m}"] = {"n_prompts": len(prompts), "previously_done": len(done),
                                           "attempted": len(todo), **dict(stats),
                                           "elapsed_sec": round(time.time() - t0, 1)}
        logger.info("[%s/%s] 完成 %s", tname, m, dict(stats))
    write_json(report_path(out_dir / "prediction_run_report.json", args.overwrite), report)
    logger.info("全部完成。失败记录（如有）见 %s；重新运行并保持 --resume true 可只重跑失败样本", fail_file)


if __name__ == "__main__":
    main()