#!/usr/bin/env python3
"""
collect_llm_summary_results.py

把 rag_evaluation_outputs/{method}_{split}/evaluation_report.json 这些
分散的单方法结果文件，汇总成一张 summary_results.json / .csv，
格式和字段命名跟 collect_plm_summary_results.py 输出的 PLM 汇总表对齐，
方便你把 LLM 组和 PLM 组的结果放在一起看、或者接到
compare_cmtscd_mtcsd_results.py 里做跨语言对照。

用法：
python collect_llm_summary_results.py \
  --eval_dir mtcsd_experiment/rag_evaluation_outputs \
  --split test \
  --output_dir mtcsd_experiment/rag_evaluation_outputs
"""

import json
import argparse
import csv
from pathlib import Path

# method → (use_dynamic_window, use_semantic, use_rag)
# 依据 simple_rag_stance_predictor_en.py 里 SEMANTIC_METHODS / RAG_METHODS /
# 使用 causal_ctx 的方法集合还原出来的口径
METHOD_FLAGS = {
    "current_only":      (False, False, False),
    "root_current":      (False, False, False),
    "parent_current":    (False, False, False),
    "full_path":         (False, False, False),
    "dynamic_window":    (True,  False, False),
    "dynamic_semantic":  (True,  True,  False),
    "dynamic_rag":       (True,  False, True),
    "full_rag":          (True,  True,  True),
    "full_rag_posthoc":  (True,  True,  True),  # 额外用到 child feedback，见下方备注
}

METHOD_ORDER = [
    "current_only", "root_current", "parent_current", "full_path",
    "dynamic_window", "dynamic_semantic", "dynamic_rag", "full_rag",
    "full_rag_posthoc",
]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_row(method: str, report: dict) -> dict:
    flags = METHOD_FLAGS.get(method, (None, None, None))
    per_class = report.get("per_class", {})

    def f1(label):
        return per_class.get(label, {}).get("f1")

    return {
        "method":               method,
        "split":                report.get("split"),
        "use_dynamic_window":   flags[0],
        "use_semantic":         flags[1],
        "use_rag":              flags[2],
        "uses_child_feedback":  method == "full_rag_posthoc",
        "total_samples":        report.get("total_samples"),
        "valid_samples":        report.get("valid_samples"),
        "invalid_count":        report.get("invalid_count"),
        "accuracy":             report.get("accuracy"),
        "macro_f1":             report.get("macro_f1"),
        "weighted_f1":          report.get("weighted_f1"),
        "favor_f1":             f1("favor"),
        "against_f1":           f1("against"),
        "none_f1":              f1("none"),
    }


def main():
    parser = argparse.ArgumentParser(description="汇总 LLM 零样本消融实验的各方法结果")
    parser.add_argument("--eval_dir", required=True,
                         help="evaluate_stance_predictions.py 的 --output_dir 所在的父目录"
                              "（即 run_mtcsd_llm_ablation_test.sh 里的 rag_evaluation_outputs）")
    parser.add_argument("--split", default="test",
                         help="只汇总这个 split 的结果，默认 test")
    parser.add_argument("--methods", nargs="*", default=None,
                         help="只汇总指定的方法，默认按 METHOD_ORDER 全部尝试")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    methods = args.methods or METHOD_ORDER

    rows = []
    missing = []

    for method in methods:
        report_path = eval_dir / f"{method}_{args.split}" / "evaluation_report.json"
        report = load_json(report_path)
        if report is None:
            missing.append((method, str(report_path)))
            continue
        rows.append(extract_row(method, report))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "llm_summary_results.json"
    csv_path  = output_dir / "llm_summary_results.csv"

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
        print(f"\n⚠️ 以下 {len(missing)} 个方法没有找到 evaluation_report.json（可能还没跑或者失败了）：")
        for method, path in missing:
            print(f"   - {method}: 期望路径 {path}")

    if rows:
        rows_sorted = sorted(rows, key=lambda x: (x["macro_f1"] is None, -(x["macro_f1"] or 0)))
        print("\n结果预览（按 macro_f1 排序）：")
        for r in rows_sorted:
            print(f"   {r['method']:<20} acc={r['accuracy']}  macro_f1={r['macro_f1']}  weighted_f1={r['weighted_f1']}")

        baseline = next((r for r in rows if r["method"] == "current_only"), None)
        best = rows_sorted[0]
        if baseline and best["method"] != "current_only" and baseline.get("macro_f1"):
            gain = (best["macro_f1"] - baseline["macro_f1"]) / baseline["macro_f1"] * 100
            print(f"\n📈 最佳方法 {best['method']} 相对 current_only 基线的 macro_f1 相对提升：{gain:+.2f}%")


if __name__ == "__main__":
    main()
