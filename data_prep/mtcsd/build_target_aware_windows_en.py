#!/usr/bin/env python3
"""
build_target_aware_windows.py

用法：
python build_target_aware_windows.py \
  --instances processed_mtcsd/mtcsd_stance_instances.jsonl \
  --children_map processed_mtcsd/mtcsd_children_map.json \
  --output_dir processed_mtcsd_windows \
  --up_k_default 2 \
  --up_k_expanded 3 \
  --down_k 5
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# ============ 词表常量（英文版，适配 MT-CSD / Reddit 语料） ============
# 说明：这些词表是从 C-MTCSD（中文微博）版本迁移过来的语言学等价物，
# 用于目标感知动态窗口构建中"模糊文本/反问/澄清/反驳"等判断。
# MT-CSD 是英文 Reddit 语料，按英文网络用语重新设计，而非直译中文词表。

AMBIGUOUS_PHRASES = [
    "true", "fair", "same", "this", "this is it", "exactly", "facts",
    "lol", "lmao", "haha", "yep", "yeah", "agreed", "+1", "word",
]

RHETORICAL_MARKERS = [
    "?", "really?", "seriously?", "you can't be serious", "who believes that",
    "no way", "as if", "come on",
]

CLARIFICATION_CUES = [
    "what do you mean", "are you saying", "are you for or against",
    "clearly sarcastic", "he means", "that's not what i meant",
    "you misunderstood", "you're missing the point",
]

REBUTTAL_CUES = [
    "no", "that's wrong", "nonsense", "don't twist my words",
    "your logic", "lol", "lmao", "that's not true",
]

EXPLANATION_CUES = [
    "what he means is", "the point is", "not saying that",
    "what i mean is", "to clarify",
]

STANCE_CUES = [
    "support", "oppose", "agree", "disagree", "in favor", "against",
]

REPLY_AUTHOR_CUES = [
    "you", "you guys", "op", "the guy above", "stop trolling", "troll",
    "reading comprehension", "your logic", "learn to read",
]

RELATED_ATTRIBUTE_WORDS = [
    "price", "cost", "efficiency", "experience", "management",
    "execution", "risk", "consequence", "safety", "service",
    "fee", "performance", "quality", "regulation", "environment",
]

# 英文停用词（用于 detect_target_drift_candidate 的核心词提取，
# 过滤掉没有实际指代意义的高频功能词）
ENGLISH_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "and", "or", "but",
    "if", "then", "so", "of", "in", "on", "at", "to", "for", "with",
    "as", "by", "from", "about", "into", "than", "not", "no", "just",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "i", "you", "he", "she", "they", "we", "his",
    "her", "their", "our", "your", "my", "me", "him", "them", "us",
}


# ============ IO 工具 ============

def load_jsonl(filepath: str) -> list:
    records = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第 {i+1} 行 JSON 解析失败，已跳过：{e}")
    return records


def load_json(filepath: str) -> dict:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


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


def contains_any(text: str, keywords: list) -> bool:
    """
    英文关键词匹配：短语（含空格）或含特殊符号（如 "?"）用子串匹配；
    单个英文词用 \\b 词边界匹配，避免 "you" 命中 "your"/"youtube" 这类误判。
    """
    import re
    t = (text or "").lower()
    for kw in keywords:
        kwl = kw.lower()
        if " " in kwl or not kwl.isalpha():
            if kwl in t:
                return True
        else:
            if re.search(r'\b' + re.escape(kwl) + r'\b', t):
                return True
    return False


# ============ 模糊文本判断 ============

SHORT_TEXT_WORD_THRESHOLD = 2  # 英文按词数判断"极短文本"，而非中文字符数


def is_short_text(text: str) -> bool:
    """英文短文本判断：按空白分词后的词数，而不是中文的字符数。"""
    words = safe_str(text).lower().split()
    return len(words) <= SHORT_TEXT_WORD_THRESHOLD


BACKCHANNEL_MAX_WORDS = 6  # 只在整条评论较短时才用 AMBIGUOUS_PHRASES/RHETORICAL_MARKERS 判断，
                            # 避免 "this"/"true"/"same" 这类高频英文词在长文本里被当成子串误命中


def matches_backchannel_lexicon(text: str) -> bool:
    """
    判断整条评论是否属于"简短附和/反问"类语用表达。
    只在评论本身较短（<= BACKCHANNEL_MAX_WORDS 词）时才做词表匹配，
    否则像 "this"/"true"/"same" 这类高频词几乎必然作为子串出现在任何长评论里，
    会导致误判。
    """
    words = safe_str(text).lower().split()
    if len(words) > BACKCHANNEL_MAX_WORDS:
        return False
    return contains_any(text, AMBIGUOUS_PHRASES) or contains_any(text, RHETORICAL_MARKERS)


def is_ambiguous_text(text: str) -> bool:
    t_lower = safe_str(text).lower()
    if is_short_text(t_lower):
        return True
    if matches_backchannel_lexicon(t_lower):
        return True
    # 判断是否只有标点/表情（简单规则，英文只需 \w，无需 CJK 区间）
    import re
    non_punct = re.sub(r'[^\w]', '', t_lower)
    if len(non_punct) <= 2:
        return True
    return False


# ============ 上游窗口选择 ============

def select_upward_window(
    rec: dict,
    up_k_default: int,
    up_k_expanded: int,
) -> tuple:
    """
    返回 (upward_window_ids, upward_window_texts, used_expanded, strategy_tag)
    """
    ancestor_chain = rec.get("ancestor_chain") or []
    ancestor_texts = rec.get("ancestor_texts") or []
    current_text   = safe_str(rec.get("current_text"))
    depth          = rec.get("depth", 0)

    # ancestor_chain 包含 current 自身，去掉最后一个
    parent_chain       = ancestor_chain[:-1] if len(ancestor_chain) > 1 else []
    parent_chain_texts = ancestor_texts[:-1]  if len(ancestor_texts) > 1 else []

    # 判断是否需要 expanded
    use_expanded = (
        is_short_text(current_text)
        or matches_backchannel_lexicon(current_text)
        or depth >= 3
    )

    k = up_k_expanded if use_expanded else up_k_default

    # 取最近 k 层祖先（不含 root，root 由 target_anchor 覆盖）
    # 如果 parent_chain 第一个是 root，跳过 root 节点以避免与 target_anchor 重复
    effective_chain = parent_chain
    effective_texts = parent_chain_texts

    if effective_chain and effective_chain[0] == rec.get("root_id"):
        effective_chain = effective_chain[1:]
        effective_texts = effective_texts[1:] if len(effective_texts) > 1 else []

    # 取最近 k 层
    window_ids   = effective_chain[-k:] if effective_chain else []
    window_texts = effective_texts[-k:] if effective_texts else []

    # 对齐长度
    min_len = min(len(window_ids), len(window_texts))
    window_ids   = window_ids[:min_len]
    window_texts = window_texts[:min_len]

    strategy_tag = "expanded_upward" if use_expanded else "default_upward"
    return window_ids, window_texts, use_expanded, strategy_tag


# ============ 下游反馈窗口选择 ============

def select_downward_feedback(
    rec: dict,
    children_map: dict,
    down_k: int,
) -> tuple:
    """
    返回 (feedback_ids, feedback_texts, quality)
    """
    node_id         = rec.get("node_id", "")
    is_ambiguous    = rec.get("_is_ambiguous", False)
    depth           = rec.get("depth", 0)
    children_ids    = children_map.get(node_id, [])

    # 只在模糊或深层时启用
    if not (is_ambiguous or depth >= 3):
        return [], [], "none"

    if not children_ids:
        return [], [], "none"

    # 简单文本索引（children 节点文本需从 all_node_map 获取）
    child_node_map: dict = rec.get("_child_node_map", {})

    high_value = []
    low_value  = []

    all_cues = CLARIFICATION_CUES + REBUTTAL_CUES + EXPLANATION_CUES + STANCE_CUES

    for cid in children_ids:
        child_text = safe_str(child_node_map.get(cid, ""))
        if contains_any(child_text, all_cues):
            high_value.append((cid, child_text))
        else:
            low_value.append((cid, child_text))

    if high_value:
        selected = high_value[:down_k]
        quality  = "high"
    else:
        selected = low_value[:down_k]
        quality  = "low"

    feedback_ids   = [x[0] for x in selected]
    feedback_texts = [x[1] for x in selected]
    return feedback_ids, feedback_texts, quality


# ============ 目标漂移候选检测 ============

def detect_target_drift_candidate(
    rec: dict,
) -> str:
    current_text = safe_str(rec.get("current_text"))
    main_target  = safe_str(rec.get("main_target"))
    root_text    = safe_str(rec.get("root_text") or "")

    # 提取 root_text 核心词（英文：正则取词长>=3的实词，过滤停用词，取前15个）
    import re
    root_words_all = re.findall(r"[A-Za-z']{3,}", root_text.lower())
    root_words = [w for w in root_words_all if w not in ENGLISH_STOPWORDS][:15]
    target_words_all = re.findall(r"[A-Za-z']+", main_target.lower())
    target_words = [w for w in target_words_all if w not in ENGLISH_STOPWORDS]

    all_anchor_words = list(set(target_words + root_words))

    # 1. 包含 main_target 或 root 核心词 → likely_same
    if contains_any(current_text, all_anchor_words) or main_target.lower() in current_text.lower():
        return "likely_same_or_related"

    # 2. 包含评论作者词且无 target 词 → reply_author_only
    if contains_any(current_text, REPLY_AUTHOR_CUES):
        return "reply_author_only"

    # 3. 极短且无足够上下文 → unclear
    if is_short_text(current_text):
        return "unclear"

    # 4. 包含相关属性词 → likely_same
    if contains_any(current_text, RELATED_ATTRIBUTE_WORDS):
        return "likely_same_or_related"

    # 5. 其他
    return "possible_topic_shift"


# ============ 构建 context_text ============

def build_context_texts(
    rec: dict,
    target_anchor: str,
    window_ids: list,
    window_texts: list,
    feedback_ids: list,
    feedback_texts: list,
) -> tuple:
    """
    返回 (causal_context_text, posthoc_context_text)
    注意：不写入任何 stance_label
    """
    node_id      = rec.get("node_id", "")
    current_text = safe_str(rec.get("current_text"))

    # ---- 上游部分 ----
    upstream_lines = []
    for wid, wtxt in zip(window_ids, window_texts):
        upstream_lines.append(f"[{wid}] {safe_str(wtxt)}")
    upstream_str = "\n".join(upstream_lines) if upstream_lines else "(no upstream comments)"

    # ---- causal_context_text ----
    causal = (
        f"Target anchor:\n{target_anchor}\n\n"
        f"Upstream conversation:\n{upstream_str}\n\n"
        f"Current comment:\n[{node_id}] {current_text}"
    )

    # ---- posthoc_context_text ----
    if feedback_ids:
        feedback_lines = []
        for fid, ftxt in zip(feedback_ids, feedback_texts):
            feedback_lines.append(f"[{fid}] {safe_str(ftxt)}")
        feedback_str = "\n".join(feedback_lines)
        posthoc = causal + f"\n\nDownstream feedback:\n{feedback_str}"
    else:
        posthoc = causal

    return causal, posthoc


# ============ 核心窗口构建 ============

def build_windows(
    instances: list,
    children_map: dict,
    up_k_default: int,
    up_k_expanded: int,
    down_k: int,
) -> tuple:
    """
    返回 (windowed_instances, warnings)
    """
    # 构建全局文本索引（用于下游反馈文本查找）
    node_text_map = {
        rec.get("node_id", ""): safe_str(rec.get("current_text"))
        for rec in instances
    }

    windowed = []
    warnings = []

    for rec in instances:
        try:
            nid          = rec.get("node_id", "")
            depth        = rec.get("depth", 0)
            current_text = safe_str(rec.get("current_text"))
            main_target  = safe_str(rec.get("main_target"))
            root_text    = safe_str(rec.get("root_text") or "")

            if not rec.get("ready_for_dynamic_window", False):
                warnings.append(f"skip_not_ready: {nid}")
                continue

            # ---- target_anchor ----
            target_anchor = safe_str(rec.get("target_anchor"))
            if not target_anchor:
                root_summary = root_text[:200]
                target_anchor = f"Target: {main_target}. Post content: {root_summary}"

            # ---- 上游窗口 ----
            window_ids, window_texts, use_expanded, strategy_tag = select_upward_window(
                rec, up_k_default, up_k_expanded
            )

            # ---- 模糊判断 ----
            is_ambiguous = is_ambiguous_text(current_text)
            rec["_is_ambiguous"]    = is_ambiguous
            rec["_child_node_map"]  = {
                cid: node_text_map.get(cid, "")
                for cid in children_map.get(nid, [])
            }

            # ---- 下游反馈 ----
            feedback_ids, feedback_texts, feedback_quality = select_downward_feedback(
                rec, children_map, down_k
            )

            # ---- 最终 strategy ----
            if feedback_ids:
                if feedback_quality == "high":
                    strategy = "expanded_upward_with_child_feedback" if use_expanded else "expanded_upward_with_child_feedback"
                else:
                    strategy = "child_feedback_only_low_quality"
            elif use_expanded:
                strategy = "expanded_upward"
            else:
                strategy = "default_upward"

            if not current_text and not window_texts:
                strategy = "no_valid_window"

            # ---- 构建 context_text ----
            causal_ctx, posthoc_ctx = build_context_texts(
                rec, target_anchor,
                window_ids, window_texts,
                feedback_ids, feedback_texts,
            )

            # ---- 目标漂移候选 ----
            target_drift = detect_target_drift_candidate(rec)

            # ---- ready flag ----
            ready_windowed = bool(
                causal_ctx
                and current_text
                and (window_texts or rec.get("parent_text"))
            )

            # ---- 组装输出 ----
            out = dict(rec)
            # 清理内部临时字段
            out.pop("_is_ambiguous",   None)
            out.pop("_child_node_map", None)

            out.update({
                "target_anchor":                  target_anchor,
                "is_ambiguous_current":           is_ambiguous,
                "upward_window_ids":              window_ids,
                "upward_window_texts":            window_texts,
                "downward_feedback_window_ids":   feedback_ids,
                "downward_feedback_window_texts": feedback_texts,
                "downward_feedback_quality":      feedback_quality,
                "window_strategy":                strategy,
                "causal_context_text":            causal_ctx,
                "posthoc_context_text":           posthoc_ctx,
                "target_drift_candidate":         target_drift,
                "ready_for_semantic_propagation_windowed": ready_windowed,
            })
            windowed.append(out)

        except Exception as e:
            warnings.append(f"build_error: {rec.get('node_id', '?')}：{e}")

    return windowed, warnings


# ============ 质量报告 ============

def generate_report(
    windowed: list,
    original_count: int,
    warnings: list,
) -> dict:
    total = len(windowed)

    ambiguous_count = sum(1 for r in windowed if r.get("is_ambiguous_current"))
    with_feedback   = sum(1 for r in windowed if r.get("downward_feedback_window_ids"))
    high_feedback   = sum(1 for r in windowed
                          if r.get("downward_feedback_quality") == "high")
    no_feedback     = sum(1 for r in windowed
                          if not r.get("downward_feedback_window_ids"))

    strategy_dist = Counter(r.get("window_strategy") for r in windowed)
    drift_dist    = Counter(r.get("target_drift_candidate") for r in windowed)
    split_dist    = Counter(r.get("split") for r in windowed)
    topic_dist    = Counter(r.get("topic") for r in windowed)
    depth_dist    = Counter(r.get("depth") for r in windowed)

    up_sizes   = [len(r.get("upward_window_ids", [])) for r in windowed]
    down_sizes = [len(r.get("downward_feedback_window_ids", [])) for r in windowed]

    avg_up   = round(sum(up_sizes)   / total, 3) if total else 0
    avg_down = round(sum(down_sizes) / total, 3) if total else 0
    max_up   = max(up_sizes,   default=0)
    max_down = max(down_sizes, default=0)

    short_text = [r for r in windowed
                  if is_short_text(safe_str(r.get("current_text")))]
    short_with_feedback = sum(1 for r in short_text
                              if r.get("downward_feedback_window_ids"))

    return {
        "total_original_instances":                   original_count,
        "windowed_instances_count":                   total,
        "skipped_count":                              original_count - total,
        "ambiguous_current_count":                    ambiguous_count,
        "instances_with_child_feedback_count":        with_feedback,
        "instances_with_high_quality_child_feedback_count": high_feedback,
        "instances_without_child_feedback_count":     no_feedback,
        "default_upward_count":                       strategy_dist.get("default_upward", 0),
        "expanded_upward_count":                      strategy_dist.get("expanded_upward", 0),
        "expanded_upward_with_child_feedback_count":  strategy_dist.get("expanded_upward_with_child_feedback", 0),
        "child_feedback_low_quality_count":           strategy_dist.get("child_feedback_only_low_quality", 0),
        "no_valid_window_count":                      strategy_dist.get("no_valid_window", 0),
        "target_drift_candidate_distribution":        dict(drift_dist),
        "average_upward_window_size":                 avg_up,
        "average_downward_feedback_size":             avg_down,
        "max_upward_window_size":                     max_up,
        "max_downward_feedback_size":                 max_down,
        "split_distribution":                         dict(split_dist),
        "topic_distribution":                         dict(topic_dist),
        "depth_distribution":                         dict(depth_dist),
        "short_text_count":                           len(short_text),
        "short_text_with_child_feedback_count":       short_with_feedback,
        "warning_count":                              len(warnings),
        "sample_warnings":                            warnings[:20],
    }


# ============ 示例文件生成 ============

def generate_examples(windowed: list) -> list:
    examples = []
    seen_ids = set()

    EXAMPLE_FIELDS = [
        "topic", "node_id", "main_target", "current_text",
        "upward_window_texts", "downward_feedback_window_texts",
        "causal_context_text", "posthoc_context_text",
        "window_strategy", "target_drift_candidate",
        "depth", "is_ambiguous_current", "stance_label",
    ]

    def add(rec):
        nid = rec.get("node_id", "")
        if nid in seen_ids:
            return
        seen_ids.add(nid)
        examples.append({k: rec.get(k) for k in EXAMPLE_FIELDS})

    # 1. 短文本 + 有子回复
    short_with_child = [
        r for r in windowed
        if is_short_text(safe_str(r.get("current_text")))
        and r.get("downward_feedback_window_ids")
    ]
    for r in short_with_child[:100]:
        add(r)

    # 2. depth >= 3
    deep = [r for r in windowed if r.get("depth", 0) >= 3]
    for r in deep[:100]:
        add(r)

    # 3. reply_author_only
    author_only = [r for r in windowed
                   if r.get("target_drift_candidate") == "reply_author_only"]
    for r in author_only[:100]:
        add(r)

    # 4. expanded_upward_with_child_feedback
    expanded_child = [r for r in windowed
                      if r.get("window_strategy") == "expanded_upward_with_child_feedback"]
    for r in expanded_child[:100]:
        add(r)

    return examples


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="构建目标感知动态会话窗口")
    parser.add_argument("--instances",     required=True)
    parser.add_argument("--children_map",  required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--up_k_default",  type=int, default=2)
    parser.add_argument("--up_k_expanded", type=int, default=3)
    parser.add_argument("--down_k",        type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print(f"📂 读取 stance instances：{args.instances}")
    instances = load_jsonl(args.instances)
    print(f"  共加载 {len(instances)} 条")

    print(f"📂 读取 children_map：{args.children_map}")
    children_map = load_json(args.children_map)
    print(f"  共加载 {len(children_map)} 个父节点")

    # 2. 构建窗口
    print(f"\n🪟 构建动态窗口（up_k_default={args.up_k_default}，"
          f"up_k_expanded={args.up_k_expanded}，down_k={args.down_k}）...")
    windowed, warnings = build_windows(
        instances, children_map,
        args.up_k_default, args.up_k_expanded, args.down_k
    )
    print(f"  完成：{len(windowed)} 条窗口样本")

    # 3. 质量报告
    report = generate_report(windowed, len(instances), warnings)

    # 4. 示例文件
    examples = generate_examples(windowed)

    # 5. 写出
    print(f"\n📝 写出文件到 {output_dir}/")
    write_jsonl(windowed,  output_dir / "mtcsd_windowed_instances.jsonl")
    write_jsonl(examples,  output_dir / "mtcsd_window_examples.jsonl")
    write_json(report,     output_dir / "mtcsd_window_quality_report.json")

    # 6. 控制台摘要
    print(f"""
📊 窗口构建摘要：
  总输入：          {len(instances)}
  成功窗口：        {len(windowed)}
  模糊评论：        {report['ambiguous_current_count']}
  有子反馈：        {report['instances_with_child_feedback_count']}
  高质量子反馈：    {report['instances_with_high_quality_child_feedback_count']}
  策略分布：
    default_upward：                        {report['default_upward_count']}
    expanded_upward：                       {report['expanded_upward_count']}
    expanded_upward_with_child_feedback：   {report['expanded_upward_with_child_feedback_count']}
    child_feedback_low_quality：            {report['child_feedback_low_quality_count']}
  目标漂移分布：    {report['target_drift_candidate_distribution']}
  短文本：          {report['short_text_count']}
  示例文件：        {len(examples)} 条
    """)

    if warnings:
        print(f"⚠️  警告（前5条）：")
        for w in warnings[:5]:
            print(f"  {w}")


if __name__ == "__main__":
    main()