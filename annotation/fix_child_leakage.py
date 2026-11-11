# -*- coding: utf-8 -*-
"""
fix_child_leakage.py
修复：语义传导标注当时用了 --context_mode both（prompt 含子回复）。
只有"有子回复"的样本受影响（无子回复的样本，both 与 causal 给模型的信息相同），因此只需重标注这部分样本。

三个子命令：
1) select    从实例文件中选出受影响样本（downward_feedback_window_texts 非空），写成单独文件
             --prefix_ids true：把 node_id 临时改成 "话题||node_id"（原标注脚本断点续跑只按 node_id 判断，
             跨话题重名会被误跳过；标注 prompt 不含 node_id，改名不影响 prompt）
2) merge_ann 把因果模式下的新标注合并回完整实例文件，生成新的实例文件
3) merge_pred 把受影响样本的新预测替换进原预测文件

完整流程见文件末尾的说明（运行 python fix_child_leakage.py -h）。
"""
import argparse
import json
from collections import Counter
from pathlib import Path

SEP = "||"


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def key(r):
    nid = str(r.get("node_id"))
    if SEP in nid:
        t, n = nid.split(SEP, 1)
        return (t, n)
    return (str(r.get("topic")), nid)


def dump(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_select(a):
    splits = set(a.splits.split(","))
    only = {key(r) for r in load(a.ids_file)} if a.ids_file else None
    seen, rows = set(), []
    for r in load(a.instances):
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        hit = (k in only) if only is not None else bool(r.get("downward_feedback_window_texts"))
        if r.get("split") in splits and hit:
            r = dict(r)
            if a.prefix_ids.lower() == "true":
                r["node_id"] = f"{k[0]}{SEP}{k[1]}"
            rows.append(r)
    dump(rows, a.out)
    print(f"受影响样本 {len(rows)} 条 → {a.out}")
    print("  按划分：", dict(Counter(r.get("split") for r in rows)))
    print("  按话题：", dict(Counter(r.get("topic") for r in rows)))


def cmd_merge_ann(a):
    new = {}
    for path in a.new:
        for r in load(path):
            new[key(r)] = r
    out, seen, n_rep, changed = [], set(), 0, Counter()
    for r in load(a.instances):
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        r = dict(r)
        if k in new:
            nr = new[k]
            for f in ("llm_semantic_propagation", "llm_reply_relation", "llm_target_relation",
                      "llm_stance_mapping_to_parent", "llm_inferred_stance_to_main_target"):
                if str(r.get(f)) != str(nr.get(f)):
                    changed[f] += 1
            for kk in [x for x in r if x.startswith("llm_")]:
                r.pop(kk)
            for kk, v in nr.items():
                if kk.startswith("llm_"):
                    r[kk] = v
            r["llm_context_mode"] = "causal_reannotated"
            n_rep += 1
        elif r.get("llm_context_mode") == "causal_reannotated":
            pass                                  # 之前已修复过，保留
        else:
            r["llm_context_mode"] = ("causal_equivalent_no_child" if not r.get("downward_feedback_window_texts")
                                    else "both_NOT_FIXED")
        out.append(r)
    n_add = 0
    for k, nr in new.items():                     # 原实例文件中没有的样本（如标注时丢失后补标的），直接追加
        if k in seen:
            continue
        r = dict(nr)
        r["topic"], r["node_id"] = k[0], k[1]
        r["llm_context_mode"] = "causal_reannotated"
        out.append(r); n_add += 1
    if n_add:
        print(f"  追加原实例文件中没有的样本 {n_add} 条")
    dump(out, a.out)
    modes = Counter(r["llm_context_mode"] for r in out)
    print(f"合并完成：共 {len(out)} 条，替换 {n_rep} 条 → {a.out}")
    print("  各划分状态：")
    for sp in sorted({r.get('split') for r in out}):
        print(f"    {sp}: " + str(dict(Counter(r['llm_context_mode'] for r in out if r.get('split') == sp))))
    print(f"  重新标注后字段发生变化的条数（共 {n_rep} 条中）：{dict(changed)}")
    if modes.get("both_NOT_FIXED"):
        print("  ⚠️ 仍有含子回复但未重新标注的样本（标注 both_NOT_FIXED），通常是 train 或标注失败的样本")


def cmd_merge_pred(a):
    new = {key(r): r for r in load(a.new)}
    out, n_rep, seen = [], 0, set()
    for r in load(a.old):
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        if k in new:
            out.append(new[k]); n_rep += 1
        else:
            out.append(r)
    for k, r in new.items():
        if k not in seen:
            out.append(r); n_rep += 1
    dump(out, a.out)
    print(f"预测合并：原 {len(seen)} 条，替换/新增 {n_rep} 条，输出 {len(out)} 条 → {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="""
流程（在 make_rag_data 目录下）：
  1. 选出 test/valid 中受影响的样本（ID 加话题前缀）
     python spin/annotation/fix_child_leakage.py select \\
       --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \\
       --splits test,valid --prefix_ids true --out causal_fix/affected_for_annotation.jsonl
  2. 因果模式重新标注（约 1,160 次调用）
     python -u build_semantic_propagation_annotations_llm.py \\
       --windowed_instances causal_fix/affected_for_annotation.jsonl \\
       --output_dir causal_fix/reannotated --context_mode causal --resume true
  3. 合并，生成新的实例文件
     python spin/annotation/fix_child_leakage.py merge_ann \\
       --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \\
       --new causal_fix/reannotated/cmtscd_semantic_instances_llm.jsonl \\
       --out processed_cmtscd_semantic_llm_causal/cmtscd_semantic_instances_llm.jsonl
  4. 选出受影响样本（保留原 ID），只对它们重跑 dynamic_semantic
     python spin/annotation/fix_child_leakage.py select \\
       --instances processed_cmtscd_semantic_llm_causal/cmtscd_semantic_instances_llm.jsonl \\
       --splits test,valid --prefix_ids false --out causal_fix/affected_for_prediction.jsonl
     python -u simple_rag_stance_predictor.py --instances causal_fix/affected_for_prediction.jsonl \\
       --knowledge processed_cmtscd_rag_memory/rag_knowledge_documents_llm.jsonl \\
       --output_dir causal_fix/pred --split test --method dynamic_semantic --resume true
  5. 合并预测
     python spin/annotation/fix_child_leakage.py merge_pred \\
       --old rag_prediction_outputs/predictions_dynamic_semantic_test.jsonl \\
       --new causal_fix/pred/predictions_dynamic_semantic_test.jsonl \\
       --out rag_prediction_outputs_causal/predictions_dynamic_semantic_test.jsonl
""")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--instances", required=True)
    s.add_argument("--splits", default="test,valid")
    s.add_argument("--prefix_ids", default="true")
    s.add_argument("--ids_file", default=None, help="只选这些样本（jsonl，含 topic 与 node_id），不再按是否有子回复筛选")
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge_ann")
    m.add_argument("--instances", required=True)
    m.add_argument("--new", required=True, nargs="+", help="一个或多个因果模式重新标注的输出文件")
    m.add_argument("--out", required=True)
    p = sub.add_parser("merge_pred")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "merge_ann": cmd_merge_ann, "merge_pred": cmd_merge_pred}[a.cmd](a)


if __name__ == "__main__":
    main()