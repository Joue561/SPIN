#!/usr/bin/env python3
"""
check_semantic_annotation_quality.py

诊断假设 A：LLM 语义传导标注（llm_reply_relation / llm_semantic_propagation）
在英文 Reddit 语料上是不是不靠谱，用 MT-CSD 论文自带的人工标注
gold_logical_relationship / gold_dialogue_act 做交叉验证（这两个字段是
convert_mtcsd_to_stance_instances.py 从原始 text.csv 里保留下来的，一路
带到 mtcsd_semantic_instances_llm.jsonl 里，之前没有用上，现在正好拿来
当验证集用）。

做两件事：
1. 打印完整交叉表（llm_reply_relation × gold_dialogue_act，
   llm_semantic_propagation × gold_logical_relationship），不做任何假设，
   方便你自己判断类别聚集得合不合理。
2. 额外算一个"粗略一致率"——只对能找到清晰对应关系的类别子集
   （比如 agree_with_parent ↔ Agreement）计算，其余类别不参与这个数字
   （因为两套标签体系设计目的不同，不是所有类别都能一一对应，强行算
   整体准确率会误导）。这个数字只是一个参考信号，不是严格意义上的
   "LLM 标注准确率"。

用法：
python check_semantic_annotation_quality.py \
  --semantic_instances mtcsd_experiment/processed_mtcsd_semantic_llm_all/mtcsd_semantic_instances_llm.jsonl \
  --split test
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# 只对有清晰对应关系的类别子集计算"粗略一致率"，其余类别在交叉表里
# 展示但不参与这个数字的计算。这张表是人工按语义相近程度划的，
# 不是标准答案，仅供参考。
REPLY_RELATION_TO_DIALOGUE_ACT = {
    "agree_with_parent":    {"Agreement"},
    "disagree_with_parent": {"Disagreement", "Refusal"},
    "mock_parent":          {"Disagreement"},
    "clarify_parent":       {"Clarification"},
    "question_parent":      {"Question"},
    "supplement_parent":    {"Suggestion", "Summarize"},
}

SEMANTIC_PROP_TO_LOGICAL_REL = {
    "stance_reversal":         {"Contrastive"},
    "sarcastic_reversal":      {"Contrastive"},
    "stance_amplification":    {"Causal", "Succession"},
    "direct_stance_expression": {"Summary"},
}

# 反讽/嘲讽相关类别 vs 非反讽类别的分组，用来验证"英文反讽缺少像中文
# '呵呵'/'典'这样固化的语言标记，导致这一类识别可靠性明显更差"这个假设
SARCASM_RELATED_REPLY_RELATIONS = {"mock_parent", "sarcastic_agreement"}
NON_SARCASM_REPLY_RELATIONS = {
    "agree_with_parent", "disagree_with_parent", "clarify_parent",
    "question_parent", "supplement_parent",
}

SARCASM_RELATED_SEMANTIC_PROPS = {"sarcastic_reversal"}
NON_SARCASM_SEMANTIC_PROPS = {
    "stance_reversal", "stance_amplification", "direct_stance_expression",
}


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_crosstab(records: list, field_a: str, field_b: str, title: str):
    pairs = [
        (r.get(field_a) or "(empty)", r.get(field_b) or "(empty)")
        for r in records
        if r.get(field_b)  # gold 字段为空的样本（比如 root 节点）跳过
    ]
    if not pairs:
        print(f"\n{title}：没有可用样本（{field_b} 全部为空，可能是 root 节点或该字段没有覆盖）")
        return

    a_vals = sorted(set(p[0] for p in pairs))
    b_vals = sorted(set(p[1] for p in pairs))
    table = defaultdict(Counter)
    for a, b in pairs:
        table[a][b] += 1

    print(f"\n{'='*90}")
    print(f"{title}（共 {len(pairs)} 条有 gold 标注的样本）")
    print(f"{'='*90}")
    col_width = max(10, max(len(b) for b in b_vals) + 2)
    header = f"{'':<28}" + "".join(f"{b:>{col_width}}" for b in b_vals) + f"{'row_total':>12}"
    print(header)
    for a in a_vals:
        row_total = sum(table[a].values())
        row = f"{a:<28}" + "".join(f"{table[a].get(b,0):>{col_width}}" for b in b_vals) + f"{row_total:>12}"
        print(row)


def compute_rough_agreement(records: list, llm_field: str, gold_field: str, mapping: dict) -> dict:
    covered = 0
    matched = 0
    excluded = 0
    for r in records:
        gold = r.get(gold_field)
        llm_val = r.get(llm_field)
        if not gold:
            continue
        if llm_val not in mapping:
            excluded += 1
            continue
        covered += 1
        if gold in mapping[llm_val]:
            matched += 1
    return {
        "covered_samples": covered,
        "excluded_samples_no_mapping": excluded,
        "matched": matched,
        "rough_agreement_rate": round(matched / covered, 4) if covered else None,
    }


def compute_grouped_agreement(records: list, llm_field: str, gold_field: str,
                                mapping: dict, sarcasm_vals: set, non_sarcasm_vals: set) -> dict:
    """把 compute_rough_agreement 的样本按 llm_field 取值是否属于'反讽相关'
    类别拆成两组，分别算一致率，用来对比反讽类和非反讽类的标注可靠性差多少。"""
    def agreement_for_subset(allowed_vals):
        covered = 0
        matched = 0
        for r in records:
            gold = r.get(gold_field)
            llm_val = r.get(llm_field)
            if not gold or llm_val not in mapping or llm_val not in allowed_vals:
                continue
            covered += 1
            if gold in mapping[llm_val]:
                matched += 1
        return covered, matched, (round(matched / covered, 4) if covered else None)

    sarcasm_covered, sarcasm_matched, sarcasm_rate = agreement_for_subset(sarcasm_vals)
    non_sarcasm_covered, non_sarcasm_matched, non_sarcasm_rate = agreement_for_subset(non_sarcasm_vals)

    return {
        "sarcasm": {"covered": sarcasm_covered, "matched": sarcasm_matched, "rate": sarcasm_rate},
        "non_sarcasm": {"covered": non_sarcasm_covered, "matched": non_sarcasm_matched, "rate": non_sarcasm_rate},
    }


def main():
    parser = argparse.ArgumentParser(description="交叉验证 LLM 语义传导标注 vs MT-CSD 自带 gold 会话关系标注")
    parser.add_argument("--semantic_instances", required=True,
                         help="build_semantic_propagation_annotations_llm_en.py 的输出文件"
                              "（mtcsd_semantic_instances_llm.jsonl）")
    parser.add_argument("--split", default=None,
                         help="只看某个 split，默认不过滤（全部 split 一起看）")
    args = parser.parse_args()

    records = load_jsonl(Path(args.semantic_instances))
    print(f"共加载 {len(records)} 条")

    if args.split:
        records = [r for r in records if r.get("split") == args.split]
        print(f"过滤 split={args.split} 后剩 {len(records)} 条")

    has_gold_dialogue = sum(1 for r in records if r.get("gold_dialogue_act"))
    has_gold_logical = sum(1 for r in records if r.get("gold_logical_relationship"))
    print(f"有 gold_dialogue_act 的样本：{has_gold_dialogue}/{len(records)}")
    print(f"有 gold_logical_relationship 的样本：{has_gold_logical}/{len(records)}")

    print_crosstab(records, "llm_reply_relation", "gold_dialogue_act",
                    "llm_reply_relation × gold_dialogue_act 交叉表")

    print_crosstab(records, "llm_semantic_propagation", "gold_logical_relationship",
                    "llm_semantic_propagation × gold_logical_relationship 交叉表")

    print(f"\n{'='*90}")
    print("粗略一致率（仅供参考，只统计能找到清晰映射关系的类别子集）")
    print(f"{'='*90}")

    r1 = compute_rough_agreement(records, "llm_reply_relation", "gold_dialogue_act",
                                  REPLY_RELATION_TO_DIALOGUE_ACT)
    print(f"\nllm_reply_relation vs gold_dialogue_act：")
    print(f"  覆盖样本数：{r1['covered_samples']}（另有 {r1['excluded_samples_no_mapping']} 条因为"
          f"llm 标签没有清晰映射，未参与计算）")
    print(f"  粗略一致率：{r1['rough_agreement_rate']}")

    r2 = compute_rough_agreement(records, "llm_semantic_propagation", "gold_logical_relationship",
                                  SEMANTIC_PROP_TO_LOGICAL_REL)
    print(f"\nllm_semantic_propagation vs gold_logical_relationship：")
    print(f"  覆盖样本数：{r2['covered_samples']}（另有 {r2['excluded_samples_no_mapping']} 条因为"
          f"llm 标签没有清晰映射，未参与计算）")
    print(f"  粗略一致率：{r2['rough_agreement_rate']}")

    print(f"\n{'='*90}")
    print("反讽/嘲讽相关类别 vs 非反讽类别 —— 分组对比")
    print(f"{'='*90}")
    print("验证假设：英文缺少像中文'呵呵'/'典'这样固化的反讽标记词，导致")
    print("反讽相关类别（mock_parent/sarcastic_reversal 等）的识别可靠性")
    print("明显低于非反讽类别（agree/disagree_with_parent 等）\n")

    g1 = compute_grouped_agreement(records, "llm_reply_relation", "gold_dialogue_act",
                                    REPLY_RELATION_TO_DIALOGUE_ACT,
                                    SARCASM_RELATED_REPLY_RELATIONS, NON_SARCASM_REPLY_RELATIONS)
    print("llm_reply_relation 分组对比：")
    print(f"  反讽相关类别（{sorted(SARCASM_RELATED_REPLY_RELATIONS)}）："
          f"覆盖 {g1['sarcasm']['covered']} 条，一致率 = {g1['sarcasm']['rate']}")
    print(f"  非反讽类别（{sorted(NON_SARCASM_REPLY_RELATIONS)}）："
          f"覆盖 {g1['non_sarcasm']['covered']} 条，一致率 = {g1['non_sarcasm']['rate']}")

    g2 = compute_grouped_agreement(records, "llm_semantic_propagation", "gold_logical_relationship",
                                    SEMANTIC_PROP_TO_LOGICAL_REL,
                                    SARCASM_RELATED_SEMANTIC_PROPS, NON_SARCASM_SEMANTIC_PROPS)
    print("\nllm_semantic_propagation 分组对比：")
    print(f"  反讽相关类别（{sorted(SARCASM_RELATED_SEMANTIC_PROPS)}）："
          f"覆盖 {g2['sarcasm']['covered']} 条，一致率 = {g2['sarcasm']['rate']}")
    print(f"  非反讽类别（{sorted(NON_SARCASM_SEMANTIC_PROPS)}）："
          f"覆盖 {g2['non_sarcasm']['covered']} 条，一致率 = {g2['non_sarcasm']['rate']}")

    print(f"\n（如果反讽相关类别的一致率明显低于非反讽类别，就实锤支持'英文缺少"
          f"固化反讽标记词、导致反讽识别机制难以跨语言迁移'这个假设；"
          f"如果两组差不多，说明整体标注质量问题是均匀分布的，不特别集中在反讽类）")

    print(f"""
{'='*90}
怎么解读
{'='*90}
- 一致率明显偏低（比如 < 0.3~0.4）：说明 LLM 语义传导标注和人工会话关系
  标注在"能对应上的类别"上都对不齐，标注质量本身有问题的可能性较大，
  建议抽查交叉表里最大的"错位格子"（比如 llm 标了 agree_with_parent 但
  gold 是 Disagreement 的那些样本），看 LLM 是不是系统性理解反了。
- 一致率还可以（比如 > 0.5），但整体实验结果仍然是"叠加语义信息越叠越差"：
  说明问题更可能出在 prompt 里"谨慎判断"这类指令导致的预测偏差
  （对应假设 B），而不是标注本身错得离谱，建议接着跑
  compare_confusion_matrices.py 确认是不是"过度预测 none"。
- 两个都不理想：两个原因可能同时存在，建议先着手改 prompt 措辞
  （去掉容易导致模型回避表态的"谨慎判断"类描述），效果有限的话再
  考虑重新设计语义传导标注的 prompt。
""")


if __name__ == "__main__":
    main()
