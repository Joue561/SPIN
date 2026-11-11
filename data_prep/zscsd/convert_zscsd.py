# -*- coding: utf-8 -*-
"""
convert_zscsd.py
把 ZS-CSD（路径式实例）转成与 C-MTCSD 相同的实例格式，供现有的
语义传导标注（build_semantic_propagation_annotations_llm.py）、预测（simple_rag_stance_predictor.py）
和知识库推断脚本直接使用。

ZS-CSD 的每条实例 = 从第一层评论到当前句的完整路径 sentences，label 针对最后一句，目标为 target。
转换规则：
  - 节点 = (划分, target, 路径)；路径去掉最后一句即父节点（诊断显示父节点 100% 有标签）
  - node_id 按层级编号：<划分前缀><目标序号>-<子序号>-...，如 te12-3-1；全数据唯一，
    下游脚本按 "-" 还原父节点和所属会话树（root = 第一段）
  - 第一层评论 depth = 1，没有父节点（ZS-CSD 不含原帖）；其余 depth = 句数
  - topic 统一为 "ZS-CSD"：零样本设定下 test 目标在 train 中没有出现，知识库、混淆矩阵、
    knn 示例都必须跨目标使用；真实目标写在 main_target，prompt 用的是 main_target
  - 重复路径（同一目标下同一路径出现多次）合并，标签取多数（平票取先出现的）
  - 动态窗口与 C-MTCSD 一致：目标锚点（目标 + 第一层评论前 200 字）+ 最近 2 个祖先
    （深度 ≥ 3 或当前评论含义模糊时扩到 3 个）；只用祖先，不含子回复（因果设定）
  - 说话人信息保留在 speaker_id / ancestor_speakers / same_speaker_ancestor_ids，供后续使用

两个子命令：
  convert  生成
      <out_dir>/zscsd_windowed_instances.jsonl   深度 ≥ 2 的节点，送去 LLM 语义传导标注
      <out_dir>/zscsd_roots_rule_filled.jsonl    第一层评论，语义字段按规则填写（无父节点），不调 LLM
      <out_dir>/zscsd_convert_report.json
  merge    把标注输出与第一层评论合并成最终实例文件，并检查有没有漏标的节点

用法（make_rag_data 目录下）：
python spin/data_prep/zscsd/convert_zscsd.py convert --data_dir ZS-CSD/data --out_dir zscsd_exp/data
python spin/data_prep/zscsd/convert_zscsd.py merge \
  --annotated zscsd_exp/data/annotated/cmtscd_semantic_instances_llm.jsonl \
  --windowed zscsd_exp/data/zscsd_windowed_instances.jsonl \
  --roots zscsd_exp/data/zscsd_roots_rule_filled.jsonl \
  --out zscsd_exp/data/zscsd_semantic_instances_llm.jsonl
"""
import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

FILES = OrderedDict([("train", "train_data.json"), ("valid", "dev_data.json"), ("test", "test_data.json")])
PREFIX = {"train": "tr", "valid": "va", "test": "te"}
ZH = {"favor": "支持", "against": "反对", "none": "中立"}
TOPIC = "ZS-CSD"


def load_labels(d):
    p = d / "label.json"
    m = json.load(open(p, encoding="utf-8")) if p.exists() else {"favor": 0, "against": 1, "neutral": 2}
    return {v: ("none" if k == "neutral" else k) for k, v in m.items()}


def is_ambiguous(text):
    t = re.sub(r"\[[^\]]{1,8}\]", "", text)            # 去掉 [哈哈] 之类的表情
    t = re.sub(r"[\s\W_]+", "", t)
    return len(t) <= 8


def build_split(rows, sp, lab):
    """返回该划分的节点列表（按目标、按树的先序排列）。"""
    by_t = OrderedDict()
    for r in rows:
        by_t.setdefault(r["target"], []).append(r)
    out, stat = [], Counter()
    for ti, (target, rs) in enumerate(by_t.items()):
        # 合并重复路径
        labs, first = defaultdict(list), OrderedDict()
        for r in rs:
            path = tuple(str(s) for s in r["sentences"])
            labs[path].append(lab.get(r["label"], r["label"]))
            first.setdefault(path, r)
        stat["重复路径"] += sum(1 for v in labs.values() if len(v) > 1)
        stat["重复且标签不一致"] += sum(1 for v in labs.values() if len(set(v)) > 1)
        paths = sorted(first, key=lambda p: (len(p), first[p]["id"]))
        ids, kids = {}, defaultdict(list)
        n_root = 0
        for p in paths:
            par = p[:-1]
            if len(p) == 1 or par not in ids:
                if len(p) > 1:
                    stat["父节点缺失（作为新树根）"] += 1
                n_root += 1
                ids[p] = f"{PREFIX[sp]}{ti}" if n_root == 1 else f"{PREFIX[sp]}{ti}r{n_root}"
            else:
                kids[par].append(p)
                ids[p] = f"{ids[par]}-{len(kids[par])}"
        for p in paths:
            r = first[p]
            y = Counter(labs[p]).most_common(1)[0][0]
            nid = ids[p]
            parts = nid.split("-")
            chain = ["-".join(parts[:i + 1]) for i in range(len(parts))]
            depth = len(parts)                     # 第一层评论 = 1
            off = len(p) - depth                   # 父节点缺失时，路径前面的句子没有节点
            anc_texts = list(p[off:])              # 与 chain 对齐（含当前）
            spk = [int(s) for s in r.get("speakers", [])] or [None] * len(p)
            spk = spk[off:]
            par = p[:-1] if depth > 1 else None
            root_text = anc_texts[0]
            amb = is_ambiguous(p[-1])
            k = 3 if (depth >= 3 or amb) else 2
            up_ids = chain[1:-1][-k:]              # 根（第一层评论）在锚点里，不重复放
            up_texts = anc_texts[1:-1][-k:]
            anchor = f"目标对象：{target}。会话首条评论：{root_text[:200]}"
            ctx_lines = [f"目标锚点：\n{anchor}"]
            if up_texts:
                ctx_lines.append("上文窗口（由远到近）：\n" + "\n".join(
                    f"[{'父评论' if i == len(up_texts) - 1 else f'上文{i + 1}'}] {t}" for i, t in enumerate(up_texts)))
            elif depth == 2:
                ctx_lines.append("[父评论] 即会话首条评论")
            ctx_lines.append(f"[当前评论] {p[-1]}")
            ctx = "\n\n".join(ctx_lines)
            same_spk = [chain[i] for i in range(len(chain) - 1) if spk[i] is not None and spk[i] == spk[-1]]
            out.append({
                "dataset": "ZS-CSD", "topic": TOPIC, "main_target": target,
                "target_type": r.get("target_type"), "split": sp,
                "node_id": nid, "root_id": parts[0], "parent_id": chain[-2] if depth > 1 else None,
                "children_ids": [ids[c] for c in kids.get(p, [])], "depth": depth, "type": f"comment{depth}",
                "root_text": root_text, "parent_text": p[-2] if depth > 1 else "", "current_text": p[-1],
                "ancestor_chain": chain, "ancestor_texts": anc_texts,
                "context_text": "\n".join([f"会话首条评论：{root_text}"] +
                                          ([f"父评论：{p[-2]}"] if depth > 2 else []) + [f"当前评论：{p[-1]}"]),
                "stance_label": y, "stance_label_zh": ZH[y],
                "parent_stance_label": None, "parent_stance_label_zh": None,   # 下面补
                "has_parent": depth > 1, "has_children": bool(kids.get(p)), "children_count": len(kids.get(p, [])),
                "is_root": depth == 1, "is_comment": True,
                "missing_root_text": False, "missing_parent_text": False,
                "target_anchor": anchor,
                "upward_path_ids": chain[:-1], "current_node_id": nid, "downward_candidate_child_ids": [],
                "ready_for_dynamic_window": True, "ready_for_semantic_propagation": True,
                "reply_relation_candidate": None, "semantic_propagation_pattern": None,
                "stance_mapping_to_parent": None, "target_relation_candidate": None, "child_feedback_signal": None,
                "is_ambiguous_current": amb,
                "upward_window_ids": up_ids, "upward_window_texts": up_texts,
                "downward_feedback_window_ids": [], "downward_feedback_window_texts": [],
                "downward_feedback_quality": "none",
                "window_strategy": "expanded_upward" if k == 3 else "default_upward",
                "causal_context_text": ctx, "posthoc_context_text": ctx,   # 因果设定：不提供子回复
                "target_drift_candidate": None, "ready_for_semantic_propagation_windowed": True,
                "speaker_id": spk[-1], "ancestor_speakers": spk[:-1], "same_speaker_ancestor_ids": same_spk,
                "zscsd_source_id": r["id"], "zscsd_label_votes": labs[p],
            })
        stat["节点"] += len(paths)
        stat["树"] += n_root
    lab_of = {n["node_id"]: n["stance_label"] for n in out}
    for n in out:
        if n["parent_id"]:
            pl = lab_of.get(n["parent_id"])
            n["parent_stance_label"], n["parent_stance_label_zh"] = pl, ZH.get(pl)
    return out, stat


def rule_fill_root(n):
    n = dict(n)
    n.update({
        "llm_reply_relation": "no_parent", "llm_semantic_propagation": "no_parent_context",
        "llm_stance_mapping_to_parent": "none", "llm_target_relation": "direct",
        "llm_child_feedback_signal": "no_child", "llm_candidate_patterns": [],
        "llm_direct_target": "main_target", "llm_inferred_stance_to_main_target": None,
        "llm_confidence": None, "llm_reason_short": "", "llm_use_child_feedback": False,
        "llm_context_setting": "causal", "llm_annotation_source": "rule",
        "llm_context_mode": "root_comment_rule_filled",
    })
    return n


def dump(rows, p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def cmd_convert(a):
    d = Path(a.data_dir)
    lab = load_labels(d)
    allrows, report = [], {}
    for sp, f in FILES.items():
        rows = json.load(open(d / f, encoding="utf-8"))
        nodes, st = build_split(rows, sp, lab)
        allrows += nodes
        report[sp] = dict(st, 原始实例=len(rows))
        print(f"{sp:5s} 原始 {len(rows):6d} → 节点 {st['节点']:6d}，树 {st['树']:3d}，重复路径 {st['重复路径']}"
              f"（标签不一致 {st['重复且标签不一致']}），父节点缺失 {st['父节点缺失（作为新树根）']}")
    ids = Counter(n["node_id"] for n in allrows)
    assert max(ids.values()) == 1, "node_id 不唯一"
    win = [n for n in allrows if n["depth"] >= 2]
    roots = [rule_fill_root(n) for n in allrows if n["depth"] == 1]
    out = Path(a.out_dir)
    dump(win, out / "zscsd_windowed_instances.jsonl")
    dump(roots, out / "zscsd_roots_rule_filled.jsonl")
    report["送标注节点（深度≥2）"] = dict(Counter(n["split"] for n in win))
    report["第一层评论（规则填写）"] = dict(Counter(n["split"] for n in roots))
    report["深度分布"] = {sp: dict(sorted(Counter(min(n["depth"], 6) for n in allrows if n["split"] == sp).items()))
                      for sp in FILES}
    report["窗口策略"] = dict(Counter(n["window_strategy"] for n in win))
    report["含义模糊的当前评论"] = sum(n["is_ambiguous_current"] for n in allrows)
    json.dump(report, open(out / "zscsd_convert_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n送标注 {len(win)} 条 {report['送标注节点（深度≥2）']} → {out / 'zscsd_windowed_instances.jsonl'}")
    print(f"第一层评论 {len(roots)} 条（规则填写）→ {out / 'zscsd_roots_rule_filled.jsonl'}")
    ex = next(n for n in win if n["split"] == "test" and n["depth"] == 4)
    print("\n示例（test，深度 4）：")
    for k in ("node_id", "parent_id", "main_target", "stance_label", "parent_stance_label", "window_strategy",
              "same_speaker_ancestor_ids"):
        print(f"  {k}: {ex[k]}")
    print("  causal_context_text:\n    " + ex["causal_context_text"].replace("\n", "\n    "))


def cmd_merge(a):
    win = {r["node_id"]: r for r in load(a.windowed)}
    ann = {}
    for p in a.annotated:
        for r in load(p):
            if r.get("llm_semantic_propagation"):
                ann[str(r["node_id"])] = r
    roots = load(a.roots)
    miss = [nid for nid in win if nid not in ann]
    extra = [nid for nid in ann if nid not in win]
    rows = []
    for nid, w in win.items():
        r = dict(w)
        if nid in ann:
            r.update({k: v for k, v in ann[nid].items() if k.startswith("llm_")})
        elif a.keep_missing == "true":
            # 保留未标注节点，保证评测集完整、子节点仍能连到父节点；语义键记为 unknown，知识库检索时退回到深度 / 全局
            r.update({"llm_reply_relation": "unknown", "llm_semantic_propagation": "unknown",
                      "llm_target_relation": "unknown", "llm_stance_mapping_to_parent": "unknown",
                      "llm_candidate_patterns": [], "llm_annotation_source": "missing",
                      "llm_context_mode": "annotation_failed"})
        else:
            continue
        rows.append(r)
    rows += roots
    dump(rows, a.out)
    print(f"送标注 {len(win)}，已标注 {len(win) - len(miss)}，缺失 {len(miss)}（按划分 "
          f"{dict(Counter(win[n]['split'] for n in miss))}），多余 {len(extra)}")
    print(f"合并后 {len(rows)} 条（含第一层评论 {len(roots)}；未标注节点"
          f"{'保留，语义键记为 unknown' if a.keep_missing == 'true' else '已排除'}）→ {a.out}")
    print("各划分：", dict(Counter(r["split"] for r in rows)))
    print("传导类型分布（train）：", dict(Counter(r.get("llm_semantic_propagation") for r in rows
                                             if r["split"] == "train").most_common(10)))
    if miss:
        mp = Path(a.out).with_name("zscsd_missing_for_annotation.jsonl")
        dump([win[n] for n in miss], mp)
        print(f"⚠️ 缺失节点已写到 {mp}，可单独补标注后再 merge（--annotated 可给多个文件）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--data_dir", required=True)
    c.add_argument("--out_dir", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--annotated", required=True, nargs="+")
    m.add_argument("--windowed", required=True)
    m.add_argument("--roots", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--keep_missing", default="true", help="true：未标注节点保留（语义键 unknown）；false：排除")
    a = ap.parse_args()
    cmd_convert(a) if a.cmd == "convert" else cmd_merge(a)


if __name__ == "__main__":
    main()