# -*- coding: utf-8 -*-
"""
check_annotation_context_mode.py
核对：语义传导标注（llm_* 字段）当时是用哪种上下文模式生成的。
做法：用 build_semantic_propagation_annotations_llm.py 里的 build_llm_prompt 按 causal / posthoc / both 三种模式
重建每条样本的 prompt，计算 md5，与样本中保存的 llm_prompt_hash 比对。

若 test/valid 样本匹配的是 both 或 posthoc，说明标注时 prompt 里包含了子回复（未来信息），
dynamic_semantic / full_rag 等方法名义上是因果设置，实际已用到子回复。

用法（在 make_rag_data 目录下运行）：
python spin/annotation/check_annotation_context_mode.py \
  --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \
  --annotator build_semantic_propagation_annotations_llm.py
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--annotator", default="build_semantic_propagation_annotations_llm.py")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("ann", a.annotator)
    ann = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ann)

    stats = defaultdict(Counter)
    child = defaultdict(Counter)
    seen = set()
    for line in open(a.instances, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        k = (r.get("topic"), str(r.get("node_id")))
        if k in seen:
            continue
        seen.add(k)
        sp = r.get("split", "?")
        h = r.get("llm_prompt_hash")
        if not h:
            stats[sp]["无 llm_prompt_hash"] += 1
            continue
        hit = [m for m in ("causal", "posthoc", "both") if ann.prompt_hash(ann.build_llm_prompt(r, m)) == h]
        stats[sp]["+".join(hit) if hit else "都不匹配"] += 1
        has_child = bool(r.get("downward_feedback_window_texts"))
        child[sp]["有子回复" if has_child else "无子回复"] += 1
        if has_child and r.get("llm_use_child_feedback"):
            child[sp]["有子回复且LLM声明使用了子回复"] += 1

    print("【prompt 哈希匹配的上下文模式】（posthoc 与 both 生成的 prompt 相同，会同时匹配）")
    for sp in sorted(stats):
        print(f"  {sp:6s} " + "  ".join(f"{k}: {v}" for k, v in stats[sp].most_common()))
    print("\n【子回复情况】")
    for sp in sorted(child):
        print(f"  {sp:6s} " + "  ".join(f"{k}: {v}" for k, v in child[sp].most_common()))
    print("\n解读：test/valid 若主要匹配 'posthoc+both'，说明标注时使用了子回复；匹配 'causal' 则为因果设置。"
          "\n'都不匹配' 可能是标注后又改过窗口字段或 prompt 模板，需另行确认。")


if __name__ == "__main__":
    main()