#!/usr/bin/env python3
"""
convert_cmtscd_to_stance_instances.py

用法：
python convert_cmtscd_to_stance_instances.py \
  --input_dir ./C-MTCSD-main \
  --output_dir ./processed_cmtscd
"""

import json
import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# ============ 常量 ============

TOPIC_TARGET_MAP = {
    "Apollo Go":          "萝卜快跑",
    "Naked Resignation":  "裸辞",
    "Non-marriage Doctrine": "不婚主义",
    "Pre-made Meals":     "预制菜",
    "iPhone 15":          "iPhone 15",
}

STANCE_ZH_MAP = {
    "favor":   "支持",
    "against": "反对",
    "none":    "中立",
}

DATASET_NAME = "C-MTCSD"
SHORT_TEXT_THRESHOLD = 4  # 去空格后长度 <= 4


# ============ IO 工具 ============

def read_csv_robust(filepath: Path) -> list:
    """尝试多种编码读取 CSV"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
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


def read_json_robust(filepath: Path) -> list:
    """读取 JSON 或 JSONL，返回列表"""
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        # 尝试 JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass
        # 尝试 JSONL
        results = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return results
    except Exception as e:
        print(f"  ⚠️  读取 JSON 失败 {filepath}：{e}")
        return []


def write_jsonl(records: list, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✅ {filepath.name}（{len(records)} 条）")


def write_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {filepath.name}")


# ============ ID 路径工具 ============

def get_depth(node_id: str) -> int:
    return node_id.count("-")


def get_parent_id(node_id: str) -> Optional[str]:
    parts = node_id.rsplit("-", 1)
    return parts[0] if len(parts) > 1 else None


def get_root_id(node_id: str) -> str:
    return node_id.split("-")[0]


def get_ancestor_chain(node_id: str) -> list:
    parts = node_id.split("-")
    chain = []
    for i in range(1, len(parts) + 1):
        chain.append("-".join(parts[:i]))
    return chain


# ============ 读取 CSV ============

def load_topic_csv(csv_path: Path, topic: str) -> dict:
    """读取一个 topic 的 CSV，返回 {node_id: row}"""
    rows = read_csv_robust(csv_path)
    node_map = {}
    for row in rows:
        nid = str(row.get("id", "")).strip()
        if not nid:
            continue
        node_map[nid] = {
            "node_id": nid,
            "type":    str(row.get("type", "")).strip(),
            "text":    str(row.get("text", "")).strip(),
            "stance":  str(row.get("stance", "")).strip().lower(),
            "topic":   topic,
        }
    return node_map


# ============ 读取 split JSON ============

def load_split_index(topic_dir: Path) -> dict:
    """
    读取 train/valid/test JSON，返回 {node_id: split}
    JSON 中每条样本通常有 index 字段（路径列表），取最后一个 ID 作为 node_id
    """
    split_map = {}
    duplicates = []

    for split_name in ("train", "valid", "test"):
        json_path = topic_dir / f"{split_name}.json"
        records = read_json_robust(json_path)
        for rec in records:
            # 兼容多种字段名
            index = rec.get("index") or rec.get("path") or rec.get("ids") or []
            if isinstance(index, list) and index:
                nid = str(index[-1]).strip()
            elif isinstance(index, str):
                nid = index.strip()
            else:
                # 有些数据集直接用 id 字段
                nid = str(rec.get("id", "")).strip()
            if not nid:
                continue
            if nid in split_map and split_map[nid] != split_name:
                duplicates.append(nid)
            split_map[nid] = split_name

    return split_map, duplicates


# ============ 构建 children_map ============

def build_children_map(node_ids: list) -> dict:
    children_map = defaultdict(list)
    for nid in node_ids:
        pid = get_parent_id(nid)
        if pid is not None:
            children_map[pid].append(nid)
    return dict(children_map)


# ============ 构建 stance instance ============

def build_stance_instance(
    node: dict,
    node_map: dict,
    children_map: dict,
    split_map: dict,
    topic: str,
    main_target: str,
    warnings: list,
) -> dict:
    nid = node["node_id"]
    depth = get_depth(nid)
    root_id = get_root_id(nid)
    parent_id = get_parent_id(nid)
    ancestor_chain = get_ancestor_chain(nid)
    children_ids = children_map.get(nid, [])
    split = split_map.get(nid, "unknown")

    # root_text
    root_node = node_map.get(root_id)
    if root_node:
        root_text = root_node["text"]
        missing_root_text = False
    else:
        root_text = None
        missing_root_text = True
        warnings.append(f"missing_root: {nid} → root_id={root_id}")

    # parent_text + parent_stance
    parent_text = None
    parent_stance = None
    parent_stance_zh = None
    missing_parent_text = False

    if parent_id:
        parent_node = node_map.get(parent_id)
        if parent_node:
            parent_text = parent_node["text"]
            parent_stance = parent_node.get("stance")
            parent_stance_zh = STANCE_ZH_MAP.get(parent_stance)
        else:
            missing_parent_text = True
            warnings.append(f"missing_parent: {nid} → parent_id={parent_id}")
    else:
        missing_parent_text = True  # root 节点无父

    # ancestor_texts
    ancestor_texts = []
    for aid in ancestor_chain:
        anode = node_map.get(aid)
        ancestor_texts.append(anode["text"] if anode else None)

    current_text = node["text"]
    stance_label = node.get("stance", "none")
    stance_label_zh = STANCE_ZH_MAP.get(stance_label, "中立")

    # context_text
    if depth == 0:
        context_text = f"原帖：{current_text}"
    elif depth == 1:
        context_text = f"原帖：{root_text or ''}\n当前评论：{current_text}"
    else:
        context_text = (
            f"原帖：{root_text or ''}\n"
            f"父评论：{parent_text or ''}\n"
            f"当前评论：{current_text}"
        )

    # target_anchor
    root_summary = (root_text or "")[:200]
    target_anchor = f"目标对象：{main_target}。原帖内容：{root_summary}"

    # upward_path_ids（root 到 parent，不含 current）
    upward_path_ids = ancestor_chain[:-1] if len(ancestor_chain) > 1 else []

    # ready flags
    ready_for_dynamic_window = bool(root_text and current_text and ancestor_chain)
    ready_for_semantic_propagation = bool(parent_text and parent_stance)

    return {
        "dataset":         DATASET_NAME,
        "topic":           topic,
        "main_target":     main_target,
        "split":           split,

        "node_id":         nid,
        "root_id":         root_id,
        "parent_id":       parent_id,
        "children_ids":    children_ids,

        "depth":           depth,
        "type":            node.get("type", ""),

        "root_text":       root_text,
        "parent_text":     parent_text,
        "current_text":    current_text,

        "ancestor_chain":  ancestor_chain,
        "ancestor_texts":  ancestor_texts,
        "context_text":    context_text,

        "stance_label":    stance_label,
        "stance_label_zh": stance_label_zh,

        "has_parent":      parent_id is not None,
        "has_children":    len(children_ids) > 0,
        "children_count":  len(children_ids),

        "is_root":         depth == 0,
        "is_comment":      depth >= 1,

        "missing_root_text":   missing_root_text,
        "missing_parent_text": missing_parent_text,

        # 动态窗口预留字段
        "target_anchor":               target_anchor,
        "upward_path_ids":             upward_path_ids,
        "current_node_id":             nid,
        "downward_candidate_child_ids": children_ids,
        "ready_for_dynamic_window":    ready_for_dynamic_window,

        # 语义传导预留字段
        "parent_stance_label":         parent_stance,
        "parent_stance_label_zh":      parent_stance_zh,
        "reply_relation_candidate":    None,
        "semantic_propagation_pattern": None,
        "stance_mapping_to_parent":    None,
        "target_relation_candidate":   None,
        "child_feedback_signal":       None,
        "ready_for_semantic_propagation": ready_for_semantic_propagation,
    }


# ============ 质量报告 ============

def build_quality_report(
    all_instances: list,
    root_contexts: list,
    children_map: dict,
    split_index: dict,
    warnings: list,
    duplicate_splits: list,
    missing_node_in_csv: list,
    nodes_without_split: list,
) -> dict:
    total = len(all_instances)
    roots = [r for r in all_instances if r["is_root"]]
    comments = [r for r in all_instances if r["is_comment"]]

    depth_dist = Counter(r["depth"] for r in all_instances)
    type_dist  = Counter(r["type"]  for r in all_instances)
    stance_dist = Counter(r["stance_label"] for r in comments)

    stance_by_topic = defaultdict(Counter)
    for r in comments:
        stance_by_topic[r["topic"]][r["stance_label"]] += 1

    split_dist = Counter(r["split"] for r in all_instances)

    missing_parent_count = sum(1 for r in comments if r.get("missing_parent_text"))
    missing_root_count   = sum(1 for r in comments if r.get("missing_root_text"))

    nodes_with_children = sum(1 for r in all_instances if r["has_children"])
    children_counts = [r["children_count"] for r in all_instances if r["has_children"]]
    avg_children = round(sum(children_counts) / len(children_counts), 2) if children_counts else 0
    max_children = max(children_counts) if children_counts else 0

    comment_with_children = sum(1 for r in comments if r["has_children"])
    comment_ratio = round(comment_with_children / len(comments), 4) if comments else 0

    short_texts = [r for r in comments
                   if len((r["current_text"] or "").replace(" ", "")) <= SHORT_TEXT_THRESHOLD]
    short_with_children = sum(1 for r in short_texts if r["has_children"])

    max_depth = max((r["depth"] for r in all_instances), default=0)

    # root split 统计
    root_split_counter = Counter(r["split"] for r in roots)

    # root split overlap（同一 root 下子节点跨 split）
    root_child_splits = defaultdict(set)
    for r in comments:
        root_child_splits[r["root_id"]].add(r["split"])
    root_split_overlap = [rid for rid, splits in root_child_splits.items()
                          if len(splits - {"unknown"}) > 1]

    return {
        "total_nodes":                   total,
        "root_count":                    len(roots),
        "comment_count":                 len(comments),
        "stance_instances_count":        len(comments),
        "topic_count":                   len(set(r["topic"] for r in all_instances)),
        "max_depth":                     max_depth,
        "depth_distribution":            dict(depth_dist),
        "type_distribution":             dict(type_dist),
        "stance_distribution":           dict(stance_dist),
        "stance_distribution_by_topic":  {t: dict(v) for t, v in stance_by_topic.items()},
        "split_distribution":            dict(split_dist),
        "missing_parent_count":          missing_parent_count,
        "missing_root_count":            missing_root_count,
        "missing_parent_text_count":     missing_parent_count,
        "missing_root_text_count":       missing_root_count,
        "nodes_without_split_count":     len(nodes_without_split),
        "duplicate_split_assignment_count": len(duplicate_splits),
        "missing_node_in_csv_count":     len(missing_node_in_csv),
        "nodes_with_children_count":     nodes_with_children,
        "average_children_count":        avg_children,
        "max_children_count":            max_children,
        "train_root_count":              root_split_counter.get("train", 0),
        "valid_root_count":              root_split_counter.get("valid", 0),
        "test_root_count":               root_split_counter.get("test", 0),
        "root_split_overlap_warning":    len(root_split_overlap),
        "root_split_overlap_root_ids":   root_split_overlap[:20],
        "comment_nodes_with_children_count": comment_with_children,
        "comment_nodes_with_children_ratio": comment_ratio,
        "short_text_count":              len(short_texts),
        "short_text_with_children_count": short_with_children,
        "warning_count":                 len(warnings),
        "sample_warnings":               warnings[:20],
    }


# ============ 话题报告 ============

def build_topic_summary(all_instances: list) -> list:
    topics = defaultdict(list)
    for r in all_instances:
        topics[r["topic"]].append(r)

    summaries = []
    for topic, records in sorted(topics.items()):
        roots    = [r for r in records if r["is_root"]]
        comments = [r for r in records if r["is_comment"]]

        depth_dist  = Counter(r["depth"] for r in records)
        stance_dist = Counter(r["stance_label"] for r in comments)
        split_dist  = Counter(r["split"] for r in records)

        comment_with_children = sum(1 for r in comments if r["has_children"])
        short_texts = [r for r in comments
                       if len((r["current_text"] or "").replace(" ", "")) <= SHORT_TEXT_THRESHOLD]
        short_with_children = sum(1 for r in short_texts if r["has_children"])
        max_depth = max((r["depth"] for r in records), default=0)

        summaries.append({
            "topic":          topic,
            "main_target":    records[0]["main_target"] if records else topic,
            "total_nodes":    len(records),
            "root_count":     len(roots),
            "comment_count":  len(comments),
            "max_depth":      max_depth,
            "depth_distribution":  dict(depth_dist),
            "stance_distribution": dict(stance_dist),
            "split_distribution":  dict(split_dist),
            "comment_nodes_with_children_count": comment_with_children,
            "short_text_count":              len(short_texts),
            "short_text_with_children_count": short_with_children,
        })
    return summaries


# ============ root context 构建 ============

def build_root_context(
    root_node: dict,
    children_map: dict,
    all_comments: list,
) -> dict:
    nid = root_node["node_id"]
    children_ids = children_map.get(nid, [])

    # 判断 split：若 root 自身有 split 则用，否则看子节点
    own_split = root_node["split"]
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
        "topic":          root_node["topic"],
        "main_target":    root_node["main_target"],
        "root_id":        nid,
        "root_text":      root_node["current_text"],
        "children_ids":   children_ids,
        "children_count": len(children_ids),
        "split":          final_split,
    }


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="C-MTCSD 数据集标准化转换")
    parser.add_argument("--input_dir",  required=True, help="C-MTCSD 根目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 查找 CSV 文件 ----
    # 支持 data/data/*.csv 或 data/*.csv 或直接 *.csv
    csv_candidates = (
        list((input_dir / "data" / "data").glob("*.csv")) +
        list((input_dir / "data").glob("*.csv")) +
        list(input_dir.glob("*.csv"))
    )
    # 去重
    csv_paths = {p.stem: p for p in csv_candidates}
    print(f"📂 发现 CSV 文件：{list(csv_paths.keys())}")

    # ---- 查找 topic 目录（JSON split 文件） ----
    # 支持 data/data/<topic>/ 或 data/<topic>/ 或 <topic>/
    topic_dirs = {}
    for search_root in [input_dir / "data" / "data", input_dir / "data", input_dir]:
        if search_root.exists():
            for d in search_root.iterdir():
                if d.is_dir() and d.name in TOPIC_TARGET_MAP:
                    topic_dirs[d.name] = d
    print(f"📂 发现 topic 目录：{list(topic_dirs.keys())}")

    all_topics = set(csv_paths.keys()) | set(topic_dirs.keys())

    # ---- 全局数据收集 ----
    all_instances      = []
    all_warnings       = []
    all_duplicate_splits   = []
    all_missing_node_csv   = []
    all_nodes_without_split = []
    global_split_index     = {}
    global_children_map    = {}

    for topic in sorted(all_topics):
        print(f"\n🔄 处理 topic：{topic}")
        main_target = TOPIC_TARGET_MAP.get(topic, topic)

        # 读取 CSV
        node_map = {}
        if topic in csv_paths:
            node_map = load_topic_csv(csv_paths[topic], topic)
            print(f"  CSV 节点数：{len(node_map)}")
        else:
            print(f"  ⚠️  未找到 CSV：{topic}")

        # 读取 split
        split_map = {}
        dup_splits = []
        if topic in topic_dirs:
            split_map, dup_splits = load_split_index(topic_dirs[topic])
            print(f"  split 映射数：{len(split_map)}，重复 split：{len(dup_splits)}")
            all_duplicate_splits.extend(dup_splits)

            # 检查 JSON 中的 node_id 是否在 CSV 中
            for nid in split_map:
                if nid not in node_map:
                    all_missing_node_csv.append(f"{topic}:{nid}")

        # 构建 children_map（基于当前 topic）
        topic_children_map = build_children_map(list(node_map.keys()))
        global_children_map.update(topic_children_map)

        # 构建 stance instances
        topic_warnings = []
        for nid, node in node_map.items():
            try:
                instance = build_stance_instance(
                    node, node_map, topic_children_map,
                    split_map, topic, main_target, topic_warnings
                )
                all_instances.append(instance)
                global_split_index[nid] = instance["split"]

                if instance["split"] == "unknown":
                    all_nodes_without_split.append(f"{topic}:{nid}")

            except Exception as e:
                topic_warnings.append(f"build_error: {topic}:{nid}：{e}")

        all_warnings.extend(topic_warnings[:50])  # 每个 topic 最多记录 50 条警告
        print(f"  生成样本：{len([i for i in all_instances if i['topic'] == topic])}")

    # ---- 拆分 root / comment ----
    root_instances    = [r for r in all_instances if r["is_root"]]
    comment_instances = [r for r in all_instances if r["is_comment"]]
    print(f"\n📊 总计：{len(all_instances)} 节点，root={len(root_instances)}，comment={len(comment_instances)}")

    # ---- 构建 root_context ----
    root_contexts = []
    for r in root_instances:
        try:
            rc = build_root_context(r, global_children_map, comment_instances)
            root_contexts.append(rc)
        except Exception as e:
            all_warnings.append(f"root_context_error: {r['node_id']}：{e}")

    # ---- 生成报告 ----
    quality_report = build_quality_report(
        all_instances, root_contexts, global_children_map,
        global_split_index, all_warnings,
        all_duplicate_splits, all_missing_node_csv, all_nodes_without_split
    )
    topic_summary = build_topic_summary(all_instances)

    # ---- 写出文件 ----
    print(f"\n📝 写出文件到 {output_dir}/")
    write_jsonl(comment_instances, output_dir / "cmtscd_stance_instances.jsonl")
    write_jsonl(root_contexts,     output_dir / "cmtscd_root_context.jsonl")
    write_json(global_children_map, output_dir / "cmtscd_children_map.json")
    write_json(global_split_index,  output_dir / "cmtscd_split_index.json")
    write_json(quality_report,      output_dir / "cmtscd_tree_quality_report.json")
    write_json(topic_summary,       output_dir / "cmtscd_topic_summary.json")

    # ---- 控制台摘要 ----
    print(f"""
📊 处理摘要：
  总节点：          {quality_report['total_nodes']}
  root 节点：       {quality_report['root_count']}
  评论节点：        {quality_report['comment_count']}
  最大深度：        {quality_report['max_depth']}
  缺失父节点文本：  {quality_report['missing_parent_count']}
  缺失根节点文本：  {quality_report['missing_root_count']}
  无 split 节点：   {quality_report['nodes_without_split_count']}
  短文本节点：      {quality_report['short_text_count']}
  立场分布：        {quality_report['stance_distribution']}
  split 分布：      {quality_report['split_distribution']}
    """)


if __name__ == "__main__":
    main()