#!/usr/bin/env python3
"""
collect_plm_summary_results.py

独立汇总 8 种 PLM 方法的评估结果，生成 summary_results.json / summary_results.csv。

为什么需要这个脚本：run_full_plm_pipeline.py 自带的汇总逻辑要求 8 个方法
全部训练成功才会生成 summary_results.json（之前你遇到过因为网络问题
8 个方法全挂、"没有任何方法成功完成训练，跳过汇总表生成"的情况）。
这个脚本改为直接扫描每个方法目录下 train_plm_stance_model.py 产出的
evaluation_test.json（或 evaluation_valid.json），哪个方法训练成功了就把
哪个收进汇总表，不要求全部成功，方便你部分方法失败时依然能拿到已完成
部分的结果表。

用法：
python collect_plm_summary_results.py \
  --model_output_dir mtcsd_experiment/plm_model_outputs_v2 \
  --base_model roberta-base \
  --eval_split test \
  --output_dir mtcsd_experiment/plm_model_outputs_v2
"""

import json
import argparse
import csv
from pathlib import Path

# 方法 → (use_dynamic_window, use_semantic, use_rag_text, use_rag_feature)
# 与 run_full_plm_pipeline.py 里 WINDOW_METHODS / SEMANTIC_METHODS /
# RAG_TEXT_METHODS / FEATURE_METHODS 的口径保持一致
METHOD_FLAGS = {
    "plm_current":               (False, False, False, False),
    "plm_parent":                (False, False, False, False),
    "plm_window":                (True,  False, False, False),
    "plm_semantic_text":         (True,  True,  False, False),
    "plm_rag_text":               (True,  False, True,  False),
    "plm_semantic_rag_text":      (True,  True,  True,  False),
    "plm_rag_feature":            (True,  True,  False, True),
    "plm_semantic_rag_feature":   (True,  True,  False, True),
}

METHOD_ORDER = list(METHOD_FLAGS.keys())


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_method_dir(model_output_dir: Path, method: str, base_model: str) -> Path:
    """
    train_plm_stance_model.py 的 --output_dir 命名习惯是
    {method}_{base_model 的最后一段}，比如 plm_current_roberta-base。
    这里做一个宽松匹配：优先精确匹配，找不到就用前缀匹配兜底，
    避免因为 base_model 传参写法不同（比如带路径）导致匹配不上。
    """
    base_model_tag = base_model.rstrip("/").split("/")[-1]
    exact = model_output_dir / f"{method}_{base_model_tag}"
    if exact.exists():
        return exact

    candidates = [
        d for d in model_output_dir.iterdir()
        if d.is_dir() and d.name.startswith(f"{method}_")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # 有多个候选（比如同一个方法跑过不同 base_model），取最新修改的
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return candidates[0]
    return exact  # 都找不到，返回期望路径，后续会报 not found


def extract_row(method: str, eval_data: dict, base_model: str) -> dict:
    flags = METHOD_FLAGS[method]
    per_class = eval_data.get("per_class", {})

    def f1(label):
        return per_class.get(label, {}).get("f1")

    return {
        "method":              method,
        "base_model":          base_model,
        "use_dynamic_window":  flags[0],
        "use_semantic":        flags[1],
        "use_rag_text":        flags[2],
        "use_rag_feature":     flags[3],
        "accuracy":            eval_data.get("accuracy"),
        "macro_f1":            eval_data.get("macro_f1"),
        "weighted_f1":         eval_data.get("weighted_f1"),
        "favor_f1":            f1("favor"),
        "against_f1":          f1("against"),
        "none_f1":             f1("none"),
    }


def main():
    parser = argparse.ArgumentParser(description="独立汇总 PLM 8 种方法的评估结果")
    parser.add_argument("--model_output_dir", required=True,
                         help="train_plm_stance_model.py 的 --output_dir 所在的父目录")
    parser.add_argument("--base_model", default="roberta-base")
    parser.add_argument("--eval_split", default="test", choices=["test", "valid"],
                         help="用哪个 split 的 evaluation_*.json 汇总，默认 test")
    parser.add_argument("--methods", nargs="*", default=None,
                         help="只汇总指定的方法，默认汇总全部 8 种")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model_output_dir = Path(args.model_output_dir)
    methods = args.methods or METHOD_ORDER

    rows = []
    missing = []

    for method in methods:
        method_dir = find_method_dir(model_output_dir, method, args.base_model)
        eval_path = method_dir / f"evaluation_{args.eval_split}.json"
        eval_data = load_json(eval_path)
        if eval_data is None:
            missing.append((method, str(eval_path)))
            continue
        rows.append(extract_row(method, eval_data, args.base_model))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "summary_results.json"
    csv_path  = output_dir / "summary_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"✅ 成功汇总 {len(rows)}/{len(methods)} 个方法")
    print(f"   {json_path}")
    if rows:
        print(f"   {csv_path}")

    if missing:
        print(f"\n⚠️ 以下 {len(missing)} 个方法没有找到 evaluation_{args.eval_split}.json（可能训练失败或还没跑）：")
        for method, path in missing:
            print(f"   - {method}: 期望路径 {path}")

    if rows:
        print("\n结果预览（按 macro_f1 排序）：")
        for r in sorted(rows, key=lambda x: (x["macro_f1"] is None, -(x["macro_f1"] or 0))):
            print(f"   {r['method']:<28} acc={r['accuracy']}  macro_f1={r['macro_f1']}  weighted_f1={r['weighted_f1']}")


if __name__ == "__main__":
    main()
