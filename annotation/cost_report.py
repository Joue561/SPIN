# -*- coding: utf-8 -*-
"""A4: cost of SPIN's annotation step (tokens and USD), from the instance files.

Input tokens are counted with tiktoken (o200k_base, GPT-4o family) when installed, otherwise
estimated from character counts. Output tokens are counted from stored raw responses if the
instance file has them; otherwise the --out_tokens default is used and marked as an estimate.

python cost_report.py --annotator build_semantic_propagation_annotations_llm.py \
  --instances C=processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
              Z=zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
              M=MT-CSD/mtcsd_experiment/processed_mtcsd_semantic_llm_causal/mtcsd_semantic_instances_llm.jsonl
"""
import argparse, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--annotator", required=True, help="annotation script (to read the prompt template)")
ap.add_argument("--instances", nargs="+", required=True, help="NAME=path")
ap.add_argument("--price_in", type=float, default=0.15, help="USD per 1M input tokens (GPT-4o-mini list price)")
ap.add_argument("--price_out", type=float, default=0.60, help="USD per 1M output tokens")
ap.add_argument("--out_tokens", type=int, default=250, help="fallback output tokens per call")
a = ap.parse_args()

src = open(a.annotator, encoding="utf-8").read()
system = src.split('SYSTEM_PROMPT = """')[1].split('"""')[0]
user_tpl = src.split('USER_PROMPT_TEMPLATE = """')[1].split('"""')[0]
fixed_text = system + re.sub(r"\{[a-z_]+\}", "", user_tpl)

try:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    ntok = lambda s: len(enc.encode(s))
    how = "tiktoken o200k_base"
except Exception:
    ntok = lambda s: int(len(re.findall(r"[一-鿿]", s)) * 1.0 + len(re.sub(r"[一-鿿]", "", s)) / 3.5)
    how = "estimate (1 token per CJK char, 3.5 chars per token otherwise)"

fixed = ntok(fixed_text)
RAW_KEYS = ("llm_raw_response", "raw_response", "llm_response", "response_text")
print(f"tokenizer: {how}; fixed prompt part = {fixed} tokens per call")
print(f"{'data':6s} {'nodes':>7s} {'LLM calls':>9s} {'in tok/call':>11s} {'out tok/call':>12s} {'total in':>12s} {'total out':>11s} {'USD':>8s}")
for spec in a.instances:
    name, path = spec.split("=", 1)
    n = calls = tin = tout = nraw = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line); n += 1
        if str(r.get("llm_context_mode", "")).endswith("rule_filled") or r.get("llm_semantic_propagation") == "no_parent_context" and not r.get("parent_text"):
            continue                                  # root posts filled by rule, no call
        calls += 1
        ctx = str(r.get("causal_context_text") or "")[:600] + str(r.get("target_anchor") or r.get("main_target") or "")[:300] \
              + str(r.get("current_text") or "")
        tin += fixed + ntok(ctx)
        raw = next((r[k] for k in RAW_KEYS if r.get(k)), None)
        if raw:
            tout += ntok(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)); nraw += 1
    out_per = tout / nraw if nraw else a.out_tokens
    tot_out = out_per * calls
    usd = tin / 1e6 * a.price_in + tot_out / 1e6 * a.price_out
    flag = "" if nraw else " (output tokens estimated)"
    print(f"{name:6s} {n:7d} {calls:9d} {tin / max(calls, 1):11.0f} {out_per:12.0f} {tin:12,.0f} {tot_out:11,.0f} {usd:8.2f}{flag}")
print("Prices are list prices; a proxy may charge differently. Annotation is done once and shared by all base models.")