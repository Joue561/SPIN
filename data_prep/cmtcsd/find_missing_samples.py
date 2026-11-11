# -*- coding: utf-8 -*-
"""
find_missing_samples.py
定位：官方划分中的评论（不含原帖）在处理流程的哪一步丢失。

1. 以转换输出（processed_cmtscd/cmtscd_stance_instances.jsonl，含全部评论）为全集，
   找出不在实验实例文件中的评论；
2. 在 --scan 给出的所有 jsonl 文件（可用通配符）里，统计这些评论各出现在哪些文件，
   从而判断是哪一步丢的；同时汇总失败记录文件中的错误信息；
3. 输出丢失样本的话题、划分、深度分布和文本长度，并写出 id 文件，便于补跑。

用法（make_rag_data 目录下）：
python spin/data_prep/cmtcsd/find_missing_samples.py \
  --full processed_cmtscd/cmtscd_stance_instances.jsonl \
  --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl \
  --scan "processed_*/*.jsonl" "causal_fix/**/*.jsonl" \
  --out missing_samples
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    out = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except (OSError, UnicodeDecodeError):
        pass
    return out


def key(r):
    nid = str(r.get("node_id"))
    if "||" in nid:
        t, n = nid.split("||", 1)
        return (t, n)
    return (str(r.get("topic")), nid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="转换输出的全部评论")
    ap.add_argument("--instances", required=True, help="实验用的实例文件")
    ap.add_argument("--scan", nargs="*", default=[], help="要检查的中间文件（支持通配符，记得加引号）")
    ap.add_argument("--out", default="missing_samples")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    full = {key(r): r for r in load(a.full)}
    ins = {key(r) for r in load(a.instances)}
    miss = {k: r for k, r in full.items() if k not in ins}
    print(f"全部评论 {len(full)} | 实例文件 {len(ins)} | 丢失 {len(miss)}")
    print("  按划分：", dict(Counter(r.get("split") for r in miss.values())))
    print("  按话题：", dict(Counter(r.get("topic") for r in miss.values())))
    print("  按深度：", dict(sorted(Counter(r.get("depth") for r in miss.values()).items())))
    print("  按标签：", dict(Counter(r.get("stance_label") for r in miss.values())))
    lens = [len(str(r.get("current_text") or "")) for r in miss.values()]
    alll = [len(str(r.get("current_text") or "")) for r in full.values()]
    if lens:
        print(f"  当前评论平均长度：丢失样本 {sum(lens) / len(lens):.1f} 字 vs 全部 {sum(alll) / len(alll):.1f} 字；"
              f"空文本 {sum(1 for x in lens if x == 0)} 条")
    flags = Counter()
    for r in miss.values():
        for f in ("missing_root_text", "missing_parent_text", "ready_for_dynamic_window", "ready_for_semantic_propagation"):
            if f in r:
                flags[(f, bool(r[f]))] += 1
    if flags:
        print("  转换时的标记：" + "  ".join(f"{f}={v}: {c}" for (f, v), c in sorted(flags.items())))

    # ---- 在中间文件中查找
    files = sorted({p for pat in a.scan for p in glob.glob(pat, recursive=True)})
    print(f"\n【丢失样本在各中间文件中的出现情况】（共检查 {len(files)} 个文件）")
    errors = Counter()
    for p in files:
        recs = load(p)
        if not recs:
            continue
        ks = {key(r) for r in recs}
        hit = sum(1 for k in miss if k in ks)
        sp = Counter(miss[k].get("split") for k in miss if k in ks)
        tag = "  ← 失败记录" if "fail" in Path(p).name.lower() else ""
        print(f"  {hit:5d} / {len(miss)}  {p}（文件共 {len(recs)} 条）{dict(sp) if hit else ''}{tag}")
        if "fail" in Path(p).name.lower():
            for r in recs:
                if key(r) in miss:
                    errors[str(r.get("_error") or r.get("error") or "")[:100]] += 1
    if errors:
        print("\n【失败记录中的错误信息（丢失样本）】")
        for e, c in errors.most_common(8):
            print(f"  {c:5d}  {e}")

    with open(out / "missing_ids.jsonl", "w", encoding="utf-8") as f:
        for k, r in miss.items():
            f.write(json.dumps({"topic": k[0], "node_id": k[1], "split": r.get("split"), "depth": r.get("depth")},
                               ensure_ascii=False) + "\n")
    with open(out / "missing_full_records.jsonl", "w", encoding="utf-8") as f:
        for r in miss.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n丢失样本的 id 与完整记录已写到 {out}/")


if __name__ == "__main__":
    main()