#!/usr/bin/env python3
"""
check_rag_feature_methods_diff.py

验证问题②：run_full_plm_pipeline.py 汇总表里 plm_rag_feature 和
plm_semantic_rag_feature 两行的 (use_dynamic_window, use_semantic,
use_rag_text, use_rag_feature) 完全一样，是不是意味着这两个方法实际训练
数据也一样（如果一样，这组对照实验就没有区分度，需要重跑）。

结论先说在前面（我已经读过 prepare_plm_stance_data.py 源码核实过）：
这大概率不是数据 bug，而是 run_full_plm_pipeline.py 汇总表里
SEMANTIC_METHODS 集合的分类口径问题——它把 plm_rag_feature 也归进了
"use_semantic=True"（可能是因为 RAG 检索过程本身依赖语义标注字段做
query），但这不代表 plm_rag_feature 的 input_text 里注入了 semantic_block
文本。实际两方法的区别只在 build_input_text() 里：
  - plm_rag_feature：input_text 只有 目标+会话上下文，RAG 信息全部走
    独立的 rag_features 数值特征
  - plm_semantic_rag_feature：input_text 额外拼了一段 semantic_block
    文本，RAG 信息同样走 rag_features 数值特征
本脚本用于用你自己实际生成的数据实锤验证这个判断，而不是只信我读代码的
结论。

用法：
python check_rag_feature_methods_diff.py \
  --dir_a mtcsd_experiment/plm_experiment_data/plm_rag_feature \
  --dir_b mtcsd_experiment/plm_experiment_data/plm_semantic_rag_feature \
  --split train
"""

import json
import argparse
from pathlib import Path


def load_jsonl_by_id(path: Path) -> dict:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["node_id"]] = rec
    return records


def diff_rag_features(fa: dict, fb: dict) -> bool:
    """判断两个 rag_features 数值特征字典是否（近似）相等"""
    keys = set(fa.keys()) | set(fb.keys())
    for k in keys:
        va, vb = fa.get(k), fb.get(k)
        if isinstance(va, float) or isinstance(vb, float):
            if abs((va or 0.0) - (vb or 0.0)) > 1e-6:
                return False
        else:
            if va != vb:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="对比 plm_rag_feature 与 plm_semantic_rag_feature 的实际生成数据")
    parser.add_argument("--dir_a", required=True, help="通常传 plm_rag_feature 目录")
    parser.add_argument("--dir_b", required=True, help="通常传 plm_semantic_rag_feature 目录")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--show_examples", type=int, default=2, help="打印几条具体差异样本")
    args = parser.parse_args()

    path_a = Path(args.dir_a) / f"{args.split}.jsonl"
    path_b = Path(args.dir_b) / f"{args.split}.jsonl"
    if not path_a.exists() or not path_b.exists():
        raise SystemExit(f"❌ 文件不存在：{path_a if not path_a.exists() else path_b}")

    recs_a = load_jsonl_by_id(path_a)
    recs_b = load_jsonl_by_id(path_b)

    common_ids = set(recs_a.keys()) & set(recs_b.keys())
    only_a = set(recs_a.keys()) - set(recs_b.keys())
    only_b = set(recs_b.keys()) - set(recs_a.keys())

    print(f"📂 {args.dir_a}：{len(recs_a)} 条")
    print(f"📂 {args.dir_b}：{len(recs_b)} 条")
    print(f"共同 node_id：{len(common_ids)} 条（只在 A：{len(only_a)}，只在 B：{len(only_b)}）\n")

    if not common_ids:
        raise SystemExit("❌ 没有共同的 node_id，两个目录的数据来源可能不一致，无法比较。")

    input_text_identical = 0
    input_text_different = 0
    rag_features_identical = 0
    rag_features_different = 0
    label_mismatch = 0

    diff_examples = []

    for nid in common_ids:
        ra, rb = recs_a[nid], recs_b[nid]

        if ra.get("label") != rb.get("label"):
            label_mismatch += 1

        if ra.get("input_text") == rb.get("input_text"):
            input_text_identical += 1
        else:
            input_text_different += 1
            if len(diff_examples) < args.show_examples:
                diff_examples.append((nid, ra.get("input_text", ""), rb.get("input_text", "")))

        fa = ra.get("rag_features", {})
        fb = rb.get("rag_features", {})
        if diff_rag_features(fa, fb):
            rag_features_identical += 1
        else:
            rag_features_different += 1

    n = len(common_ids)
    print("=" * 70)
    print("对比结果")
    print("=" * 70)
    print(f"label 不一致的样本数：           {label_mismatch}  "
          f"{'✅ 应为 0（同一份原始数据切出来的）' if label_mismatch == 0 else '⚠️ 异常，两目录数据来源可能不同！'}")
    print(f"input_text 完全相同的样本数：    {input_text_identical}/{n}"
          f"（{round(input_text_identical/n*100,1)}%）")
    print(f"input_text 不同的样本数：        {input_text_different}/{n}"
          f"（{round(input_text_different/n*100,1)}%）")
    print(f"rag_features 数值特征相同的样本数：{rag_features_identical}/{n}"
          f"（{round(rag_features_identical/n*100,1)}%）")
    print(f"rag_features 数值特征不同的样本数：{rag_features_different}/{n}"
          f"（{round(rag_features_different/n*100,1)}%）")

    print("\n" + "=" * 70)
    print("结论判定")
    print("=" * 70)
    if input_text_different / n > 0.9 and rag_features_identical / n > 0.9:
        print("✅ 符合预期：两个方法的 input_text 绝大多数不同（semantic_rag_feature 多了语义文本块），")
        print("   而 rag_features 数值特征绝大多数相同（两者用的是同一套 RAG 检索特征）。")
        print("   → 这是一组*真实有效*的对照实验，不是数据重复 bug。")
        print("   → run_full_plm_pipeline.py 汇总表里两行 use_semantic 都是 True 只是分类口径问题：")
        print("     它把'RAG检索依赖语义标注做query'也算作 use_semantic=True，")
        print("     但没有单独一列表示'semantic_block 文本是否被拼进 input_text'。")
        print("     建议：把汇总表里 plm_rag_feature 单独标注为 use_semantic=False（或新增一列")
        print("     use_semantic_text_injected），避免以后再产生这个误会。")
    elif input_text_identical / n > 0.9:
        print("⚠️ 异常：两个方法的 input_text 绝大多数相同，说明 build_input_text() 没有正确区分这两个 method，")
        print("   这组对照实验目前确实没有区分度，需要检查 prepare_plm_stance_data.py 里的 method 分支逻辑并重跑。")
    else:
        print("⚠️ 结果介于两者之间，建议人工抽查下面打印的具体样本，确认差异是否符合预期。")

    if diff_examples:
        print("\n" + "=" * 70)
        print(f"具体差异样本（前 {len(diff_examples)} 条）")
        print("=" * 70)
        for nid, ta, tb in diff_examples:
            print(f"\n--- node_id: {nid} ---")
            print(f"[plm_rag_feature 侧] input_text:\n{ta}\n")
            print(f"[plm_semantic_rag_feature 侧] input_text:\n{tb}\n")


if __name__ == "__main__":
    main()
