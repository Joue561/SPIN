#!/usr/bin/env python3
"""
convert_mtcsd_to_stance_instances.py

将 MT-CSD（https://github.com/nfq729/MT-CSD）标准化为与
convert_cmtscd_to_stance_instances.py 完全同构的 stance instance schema，
以便后续 build_target_aware_windows_en.py / build_semantic_propagation_annotations_llm_en.py /
build_llm_pattern_memory_en.py / prepare_plm_stance_data.py / train_plm_stance_model.py
等下游脚本可以直接复用，无需改动接口。

与 C-MTCSD 转换脚本的关键数据差异（已在本脚本中处理）：
  1. text.csv 只有 id/text 列（外加 logical_relationship / dialogue_act 两个
     GLAN 论文自带的会话关系标注，本脚本保留为 gold_logical_relationship /
     gold_dialogue_act 附加字段，可用于和你 LLM 语义传导标注结果做交叉验证）。
     text.csv 没有 stance 列。
  2. stance 标签和 train/valid/test 划分都在 train.json / valid.json / test.json
     里，通过 index（从 root 到当前节点的完整路径）+ stance 给出，覆盖该 topic
     的每一个节点（包括 root 帖子本身）。
  3. 没有独立的 type 列，需要用 id 是否包含 "-" 推断 root / comment。
  4. 【重要修复】MT-CSD 每个 topic 的 id 都是从 "1" 重新计数的（Bitcoin 的 "1"
     和 Tesla 的 "1" 是两个不同节点）。如果像常见做法那样直接把所有 topic 的
     children_map / split_index 合并进同一个全局 dict，会发生 id 冲突、互相
     覆盖。本脚本统一在内部用 f"{topic}::{raw_id}" 作为跨 topic 唯一 node_id，
     所有下游字段（parent_id / root_id / children_ids / ancestor_chain）都基于
     这个带命名空间的 id，从根本上避免冲突。

用法：
python convert_mtcsd_to_stance_instances.py \
  --input_dir ./MT-CSD_repo \
  --output_dir ./processed_mtcsd
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

# ============ 常量 ============

DATASET_NAME = "MT-CSD"
TOPICS = ["Bitcoin", "Tesla", "SpaceX", "Biden", "Trump"]
SHORT_TEXT_WORD_THRESHOLD = 2  # 英文按词数判断"极短文本"


# ============ IO 工具 ============

def read_csv_robust(filepath: Path) -> list:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            rows = []
            with open(filepath, "r", encoding=enc, errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            return rows
        except Exception:
            continue
    print(f"  ⚠️  无法读取 CSV：{filepath}")
    return []


def read_json_list(filepath: Path) -> list:
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        results = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return results


def write_jsonl(records: list, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath.name}（{len(records)} 条）")


def write_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath.name}")


def safe_str(val) -> str:
    return str(val).strip() if val else ""


def is_short_text(text: str) -> bool:
    return len(safe_str(text).split()) <= SHORT_TEXT_WORD_THRESHOLD


# ============ 带命名空间的 id 工具 ============
# raw_id 形如 "1-2-1"（topic 内部路径），node_id 形如 "Bitcoin::1-2-1"（全局唯一）

def make_nid(topic: str, raw_id: str) -> str:
    return f"{topic}::{raw_id}"


def get_depth(raw_id: str) -> int:
    return raw_id.count("-")


def get_parent_raw(raw_id: str) -> Optional[str]:
    parts = raw_id.rsplit("-", 1)
    return parts[0] if len(parts) > 1 else None


def get_root_raw(raw_id: str) -> str:
    return raw_id.split("-")[0]


def get_ancestor_raw_chain(raw_id: str) -> list:
    parts = raw_id.split("-")
    return ["-".join(parts[:i]) for i in range(1, len(parts) + 1)]


# ============ 读取单个 topic ============

def load_topic_text(csv_path: Path) -> dict:
    """返回 {raw_id: {text, gold_logical_relationship, gold_dialogue_act}}"""
    rows = read_csv_robust(csv_path)
    node_map = {}
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if not rid:
            continue
        node_map[rid] = {
            "text": str(row.get("text", "")).strip(),
            "gold_logical_relationship": str(row.get("logical_relationship", "")).strip() or None,
            "gold_dialogue_act": str(row.get("dialogue_act", "")).strip() or None,
        }
    return node_map


def load_topic_stance_split(topic_dir: Path) -> tuple:
    """
    返回 (stance_map, split_map)，key 都是 raw_id。
    MT-CSD 的 train/valid/test.json 覆盖该 topic 下的每一个节点（含 root）。
    """
    stance_map = {}
    split_map = {}
    duplicates = []
    for split_name in ("train", "valid", "test"):
        records = read_json_list(topic_dir / f"{split_name}.json")
        for rec in records:
            index = rec.get("index")
            if not (isinstance(index, list) and index):
                continue
            rid = str(index[-1]).strip()
            stance = str(rec.get("stance", "")).strip().lower()
            if rid in split_map and split_map[rid] != split_name:
                duplicates.append(rid)
            split_map[rid] = split_name
            stance_map[rid] = stance
    return stance_map, split_map


# ============ 构建单条 stance instance ============

def build_stance_instance(
    topic: str,
    raw_id: str,
    text_map: dict,
    stance_map: dict,
    split_map: dict,
    main_target: str,
    warnings: list,
) -> dict:
    depth = get_depth(raw_id)
    root_raw = get_root_raw(raw_id)
    parent_raw = get_parent_raw(raw_id)
    ancestor_raw_chain = get_ancestor_raw_chain(raw_id)

    nid = make_nid(topic, raw_id)
    root_id = make_nid(topic, root_raw)
    parent_id = make_nid(topic, parent_raw) if parent_raw else None
    ancestor_chain = [make_nid(topic, a) for a in ancestor_raw_chain]

    split = split_map.get(raw_id, "unknown")

    # root_text
    root_node = text_map.get(root_raw)
    root_text = root_node["text"] if root_node else None
    missing_root_text = root_node is None
    if missing_root_text:
        warnings.append(f"missing_root: {nid} → root_id={root_id}")

    # parent_text / parent_stance
    parent_text = None
    parent_stance = None
    missing_parent_text = False
    if parent_raw:
        parent_node = text_map.get(parent_raw)
        if parent_node:
            parent_text = parent_node["text"]
            parent_stance = stance_map.get(parent_raw)
        else:
            missing_parent_text = True
            warnings.append(f"missing_parent: {nid} → parent_id={parent_id}")
    else:
        missing_parent_text = True  # root 节点无父

    ancestor_texts = []
    for araw in ancestor_raw_chain:
        anode = text_map.get(araw)
        ancestor_texts.append(anode["text"] if anode else None)

    current_node = text_map.get(raw_id, {})
    current_text = current_node.get("text", "")
    stance_label = stance_map.get(raw_id, "none")

    # context_text（英文模板）
    if depth == 0:
        context_text = f"Post: {current_text}"
    elif depth == 1:
        context_text = f"Post: {root_text or ''}\nCurrent comment: {current_text}"
    else:
        context_text = (
            f"Post: {root_text or ''}\n"
            f"Parent comment: {parent_text or ''}\n"
            f"Current comment: {current_text}"
        )

    root_summary = (root_text or "")[:200]
    target_anchor = f"Target: {main_target}. Post content: {root_summary}"

    upward_path_ids = ancestor_chain[:-1] if len(ancestor_chain) > 1 else []

    ready_for_dynamic_window = bool(root_text and current_text and ancestor_chain)
    ready_for_semantic_propagation = bool(parent_text and parent_stance)

    return {
        "dataset":        DATASET_NAME,
        "topic":          topic,
        "main_target":    main_target,
        "split":          split,

        "node_id":        nid,
        "root_id":        root_id,
        "parent_id":      parent_id,
        "children_ids":   [],   # 后面统一填充

        "depth":          depth,
        "type":           "root" if depth == 0 else "comment",

        "root_text":      root_text,
        "parent_text":    parent_text,
        "current_text":   current_text,

        "ancestor_chain": ancestor_chain,
        "ancestor_texts": ancestor_texts,
        "context_text":   context_text,

        "stance_label":    stance_label,
        "stance_label_zh": stance_label,  # MT-CSD 为英文数据集，此字段与 stance_label 相同，仅为保持下游脚本字段兼容

        "has_parent":     parent_id is not None,
        "has_children":   False,  # 后面统一填充
        "children_count": 0,      # 后面统一填充

        "is_root":        depth == 0,
        "is_comment":     depth >= 1,

        "missing_root_text":   missing_root_text,
        "missing_parent_text": missing_parent_text,

        # 附加：GLAN 论文自带的会话关系标注，可用于交叉验证你的 LLM 语义传导标注
        "gold_logical_relationship": current_node.get("gold_logical_relationship"),
        "gold_dialogue_act":         current_node.get("gold_dialogue_act"),

        # 动态窗口预留字段
        "target_anchor":                target_anchor,
        "upward_path_ids":              upward_path_ids,
        "current_node_id":              nid,
        "downward_candidate_child_ids": [],  # 后面统一填充
        "ready_for_dynamic_window":     ready_for_dynamic_window,

        # 语义传导预留字段
        "parent_stance_label":           parent_stance,
        "parent_stance_label_zh":        parent_stance,
        "reply_relation_candidate":      None,
        "semantic_propagation_pattern":  None,
        "stance_mapping_to_parent":      None,
        "target_relation_candidate":     None,
        "child_feedback_signal":         None,
        "ready_for_semantic_propagation": ready_for_semantic_propagation,
    }


# ============ 质量报告 ============

def build_quality_report(all_instances: list, warnings: list, duplicate_splits: list) -> dict:
    total = len(all_instances)
    roots = [r for r in all_instances if r["is_root"]]
    comments = [r for r in all_instances if r["is_comment"]]

    depth_dist = Counter(r["depth"] for r in all_instances)
    stance_dist = Counter(r["stance_label"] for r in comments)
    split_dist = Counter(r["split"] for r in all_instances)
    topic_dist = Counter(r["topic"] for r in all_instances)

    missing_parent_count = sum(1 for r in comments if r.get("missing_parent_text"))
    missing_root_count = sum(1 for r in comments if r.get("missing_root_text"))

    nodes_with_children = sum(1 for r in all_instances if r["has_children"])
    children_counts = [r["children_count"] for r in all_instances if r["has_children"]]
    avg_children = round(sum(children_counts) / len(children_counts), 2) if children_counts else 0
    max_children = max(children_counts) if children_counts else 0

    comment_with_children = sum(1 for r in comments if r["has_children"])
    comment_ratio = round(comment_with_children / len(comments), 4) if comments else 0

    short_texts = [r for r in comments if is_short_text(r["current_text"])]
    short_with_children = sum(1 for r in short_texts if r["has_children"])

    max_depth = max((r["depth"] for r in all_instances), default=0)

    return {
        "total_nodes":               total,
        "root_count":                len(roots),
        "comment_count":             len(comments),
        "stance_instances_count":    len(comments),
        "topic_count":               len(set(r["topic"] for r in all_instances)),
        "max_depth":                 max_depth,
        "depth_distribution":        dict(depth_dist),
        "stance_distribution":       dict(stance_dist),
        "topic_distribution":        dict(topic_dist),
        "split_distribution":        dict(split_dist),
        "missing_parent_count":      missing_parent_count,
        "missing_root_count":        missing_root_count,
        "duplicate_split_assignment_count": len(duplicate_splits),
        "nodes_with_children_count": nodes_with_children,
        "average_children_count":    avg_children,
        "max_children_count":        max_children,
        "comment_nodes_with_children_count": comment_with_children,
        "comment_nodes_with_children_ratio": comment_ratio,
        "short_text_count":          len(short_texts),
        "short_text_with_children_count": short_with_children,
        "warning_count":             len(warnings),
        "sample_warnings":           warnings[:20],
    }


def build_topic_summary(all_instances: list) -> list:
    topics = defaultdict(list)
    for r in all_instances:
        topics[r["topic"]].append(r)

    summaries = []
    for topic, records in sorted(topics.items()):
        roots = [r for r in records if r["is_root"]]
        comments = [r for r in records if r["is_comment"]]
        depth_dist = Counter(r["depth"] for r in records)
        stance_dist = Counter(r["stance_label"] for r in comments)
        split_dist = Counter(r["split"] for r in records)
        comment_with_children = sum(1 for r in comments if r["has_children"])
        short_texts = [r for r in comments if is_short_text(r["current_text"])]
        short_with_children = sum(1 for r in short_texts if r["has_children"])
        max_depth = max((r["depth"] for r in records), default=0)

        summaries.append({
            "topic":         topic,
            "main_target":   records[0]["main_target"] if records else topic,
            "total_nodes":   len(records),
            "root_count":    len(roots),
            "comment_count": len(comments),
            "max_depth":     max_depth,
            "depth_distribution":  dict(depth_dist),
            "stance_distribution": dict(stance_dist),
            "split_distribution":  dict(split_dist),
            "comment_nodes_with_children_count": comment_with_children,
            "short_text_count":              len(short_texts),
            "short_text_with_children_count": short_with_children,
        })
    return summaries


def build_root_context(root_rec: dict, children_map: dict, all_comments: list) -> dict:
    nid = root_rec["node_id"]
    children_ids = children_map.get(nid, [])

    own_split = root_rec["split"]
    if own_split == "unknown":
        child_splits = set(
            r["split"] for r in all_comments
            if r["root_id"] == nid and r["split"] != "unknown"
        )
        if len(child_splits) == 0:
            final_split = "unknown"
        elif len(child_splits) == 1:
            final_split = child_splits.pop()
        else:
            final_split = "mixed"
    else:
        final_split = own_split

    return {
        "dataset":        DATASET_NAME,
        "topic":          root_rec["topic"],
        "main_target":    root_rec["main_target"],
        "root_id":        nid,
        "root_text":      root_rec["current_text"],
        "children_ids":   children_ids,
        "children_count": len(children_ids),
        "split":          final_split,
    }


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="MT-CSD 数据集标准化转换")
    parser.add_argument("--input_dir", required=True, help="MT-CSD 仓库根目录（含 data/<Topic>/ 子目录）")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = input_dir / "data"
    if not data_root.exists():
        data_root = input_dir  # 兼容直接传入 data/ 目录的情况

    found_topics = [t for t in TOPICS if (data_root / t).exists()]
    if not found_topics:
        raise SystemExit(
            f"❌ 在 {data_root} 下没有找到任何 topic 目录（期望 {TOPICS} 中的一个或多个）。\n"
            f"   input_dir 参数当前是：{input_dir}\n"
            f"   请检查 --input_dir 是否指向正确的 MT-CSD 根目录（应包含 data/Bitcoin 等子目录），\n"
            f"   或者直接指向 data 目录本身（应包含 Bitcoin 等子目录）。"
        )
    print(f"📂 发现 topic 目录：{found_topics}")

    all_instances = []
    all_warnings = []
    all_duplicate_splits = []
    global_children_map = defaultdict(list)   # key: node_id（带命名空间）
    global_split_index = {}

    for topic in found_topics:
        print(f"\n🔄 处理 topic：{topic}")
        topic_dir = data_root / topic
        csv_path = topic_dir / "text.csv"

        text_map = load_topic_text(csv_path)
        print(f"  CSV 节点数：{len(text_map)}")

        stance_map, split_map = load_topic_stance_split(topic_dir)
        print(f"  stance/split 映射数：{len(split_map)}")

        # 检查 json 中的 id 是否都能在 csv 中找到文本
        missing_in_csv = [rid for rid in split_map if rid not in text_map]
        if missing_in_csv:
            all_warnings.append(f"{topic}: {len(missing_in_csv)} ids in json but missing text in csv")

        main_target = topic  # MT-CSD topic 名本身就是英文 target 名，无需额外映射表

        topic_warnings = []
        topic_instances = []
        for raw_id in text_map:
            try:
                inst = build_stance_instance(
                    topic, raw_id, text_map, stance_map, split_map,
                    main_target, topic_warnings,
                )
                topic_instances.append(inst)
                global_split_index[inst["node_id"]] = inst["split"]
            except Exception as e:
                topic_warnings.append(f"build_error: {topic}:{raw_id}：{e}")

        # ---- 填充 children_ids / has_children / children_count ----
        topic_children_map = defaultdict(list)
        for inst in topic_instances:
            if inst["parent_id"] is not None:
                topic_children_map[inst["parent_id"]].append(inst["node_id"])

        for inst in topic_instances:
            children = topic_children_map.get(inst["node_id"], [])
            inst["children_ids"] = children
            inst["has_children"] = len(children) > 0
            inst["children_count"] = len(children)
            inst["downward_candidate_child_ids"] = children

        global_children_map.update(topic_children_map)
        all_instances.extend(topic_instances)
        all_warnings.extend(topic_warnings[:50])
        print(f"  生成样本：{len(topic_instances)}")

    root_instances = [r for r in all_instances if r["is_root"]]
    comment_instances = [r for r in all_instances if r["is_comment"]]
    print(f"\n📊 总计：{len(all_instances)} 节点，root={len(root_instances)}，comment={len(comment_instances)}")

    root_contexts = []
    for r in root_instances:
        try:
            root_contexts.append(build_root_context(r, global_children_map, comment_instances))
        except Exception as e:
            all_warnings.append(f"root_context_error: {r['node_id']}：{e}")

    quality_report = build_quality_report(all_instances, all_warnings, all_duplicate_splits)
    topic_summary = build_topic_summary(all_instances)

    print(f"\n📝 写出文件到 {output_dir}/")
    write_jsonl(comment_instances, output_dir / "mtcsd_stance_instances.jsonl")
    write_jsonl(root_contexts, output_dir / "mtcsd_root_context.jsonl")
    write_json(dict(global_children_map), output_dir / "mtcsd_children_map.json")
    write_json(global_split_index, output_dir / "mtcsd_split_index.json")
    write_json(quality_report, output_dir / "mtcsd_tree_quality_report.json")
    write_json(topic_summary, output_dir / "mtcsd_topic_summary.json")

    print(f"""
📊 处理摘要：
  总节点：          {quality_report['total_nodes']}
  root 节点：       {quality_report['root_count']}
  评论节点：        {quality_report['comment_count']}
  最大深度：        {quality_report['max_depth']}
  缺失父节点文本：  {quality_report['missing_parent_count']}
  缺失根节点文本：  {quality_report['missing_root_count']}
  短文本节点（<=2词）：{quality_report['short_text_count']}
  立场分布：        {quality_report['stance_distribution']}
  split 分布：      {quality_report['split_distribution']}
  topic 分布：      {quality_report['topic_distribution']}
    """)


if __name__ == "__main__":
    main()
