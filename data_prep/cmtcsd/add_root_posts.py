# -*- coding: utf-8 -*-
"""
add_root_posts.py
把 C-MTCSD 官方划分中的原帖（node_id 不含 "-"）补进实例文件，使评测集与官方 test 完全一致，
并让深度 1 的评论在会话树推断中拥有父节点（原帖）的预测。

原帖记录的构造：
  - 文本、标签、划分取自官方数据（CSV 文本 + JSON 中的 stance 与划分）
  - depth = 0，无父节点；上下文只有目标锚点和原帖本身
  - 语义传导字段按规则填写（原帖没有父评论）：回复关系 no_parent，传导类型 no_parent_context，
    目标关系 direct，父子映射 none；不调用 LLM，会话树推断也不会修正原帖本身
  - 字段格式参照实例文件中已有的深度 1 评论（同一话题、同一原帖），保证预测脚本可直接使用

用法（make_rag_data 目录下）：
python spin/data_prep/cmtcsd/add_root_posts.py \
  --raw_dir C-MTCSD \
  --instances processed_cmtscd_semantic_llm_causal_v3/cmtscd_semantic_instances_llm.jsonl \
  --out processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl
"""
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

_s = importlib.util.spec_from_file_location("co", Path(__file__).resolve().parent / "check_official_split.py")
co = importlib.util.module_from_spec(_s)
_s.loader.exec_module(co)

TOPIC_TARGET = {"Apollo Go": "萝卜快跑", "Naked Resignation": "裸辞", "Non-marriage Doctrine": "不婚主义",
                "Pre-made Meals": "预制菜", "iPhone 15": "iPhone 15"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    topic_dirs, csvs = co.find_dirs(Path(a.raw_dir))
    rows = [json.loads(l) for l in open(a.instances, encoding="utf-8") if l.strip()]
    have = {(str(r.get("topic")), str(r.get("node_id"))) for r in rows}
    # 每个话题、每个原帖：找一条深度 1 的评论作为字段模板（取它的 root_text / target_anchor 等）
    child_of_root = {}
    for r in rows:
        nid = str(r.get("node_id"))
        if nid.count("-") == 1:
            child_of_root.setdefault((str(r.get("topic")), nid.split("-")[0]), r)
    template = rows[0]

    new = []
    stat = Counter()
    for t in co.TOPICS:
        d = topic_dirs.get(t)
        cm = co.load_csv(csvs[t]) if t in csvs else {}
        if not d:
            continue
        for sp in co.SPLITS:
            for rec in co.read_json_any(d / f"{sp}.json"):
                nid = co.rec_id(rec)
                if not nid or "-" in nid or (t, nid) in have:
                    continue
                lab, _ = co.rec_label(rec)
                text = str((cm.get(nid) or {}).get("text", "")).strip()
                if not lab or not text:
                    stat["跳过（无标签或无文本）"] += 1
                    continue
                target = TOPIC_TARGET.get(t, t)
                anchor = f"目标对象：{target}。原帖内容：{text[:200]}"
                ref = child_of_root.get((t, nid), template)
                r = {k: None for k in ref.keys()}           # 与已有记录同一套字段
                r.update({
                    "dataset": ref.get("dataset", "C-MTCSD"), "topic": t, "main_target": target, "split": sp,
                    "node_id": nid, "root_id": nid, "parent_id": None, "depth": 0, "type": "background",
                    "root_text": text, "parent_text": "", "current_text": text,
                    "ancestor_chain": [nid], "ancestor_texts": [text],
                    "context_text": f"原帖：{text}", "target_anchor": anchor,
                    "causal_context_text": f"{anchor}\n[当前评论（原帖）] {text}",
                    "posthoc_context_text": f"{anchor}\n[当前评论（原帖）] {text}",
                    "upward_window_texts": [], "downward_feedback_window_texts": [],
                    "stance_label": lab, "parent_stance_label": None,
                    "has_parent": False, "is_root": True, "is_comment": False,
                    "llm_reply_relation": "no_parent", "llm_semantic_propagation": "no_parent_context",
                    "llm_target_relation": "direct", "llm_stance_mapping_to_parent": "none",
                    "llm_candidate_patterns": [], "llm_inferred_stance_to_main_target": None,
                    "llm_context_mode": "root_post_rule_filled",
                })
                new.append(r)
                stat[sp] += 1
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows + new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"原实例 {len(rows)} 条，追加原帖 {len(new)} 条（{dict(stat)}）→ {out}，共 {len(rows) + len(new)} 条")
    print("各划分总数：", dict(Counter(r.get("split") for r in rows + new)))


if __name__ == "__main__":
    main()