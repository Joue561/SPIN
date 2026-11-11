#!/usr/bin/env python3
"""
compare_confusion_matrices.py

诊断假设 B：叠加语义传导信息 / RAG 记忆后，模型是不是"过度谨慎"，
系统性地把更多样本预测成 none（回避表态），导致 favor/against 的召回率
下降、准确率跟着掉。

对比两个（或多个）方法的混淆矩阵（evaluate_stance_predictions.py 产出的
confusion_matrix.json），把行归一化成召回率，直接看每个方法在
favor/against/none 三个类别上的召回率变化趋势。

用法：
python compare_confusion_matrices.py \
  --eval_dir mtcsd_experiment/rag_evaluation_outputs \
  --split test \
  --methods current_only dynamic_window dynamic_semantic dynamic_rag full_rag
"""

import json
import argparse
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_per_class(cm: dict) -> dict:
    """cm['matrix'] 行=gold，列=predicted，labels 顺序固定"""
    labels = cm["labels"]
    matrix = cm["matrix"]
    result = {}
    for i, gold_label in enumerate(labels):
        row = matrix[i]
        row_total = sum(row)
        result[gold_label] = {
            "row_total": row_total,
            "recall": round(row[i] / row_total, 4) if row_total else None,
            # 这一类 gold 样本，被预测成每个类别的比例
            "predicted_as": {
                labels[j]: round(row[j] / row_total, 4) if row_total else None
                for j in range(len(labels))
            },
        }
    return result


def precision_per_class(cm: dict) -> dict:
    """
    精确率 = 预测成某类的样本里，真正是这一类的比例。
    用于诊断"是不是把大量 none 样本也拉去判成了 favor/against"——
    如果 favor/against 的 precision 明显走低，说明有大量误报。
    """
    labels = cm["labels"]
    matrix = cm["matrix"]
    n = len(labels)
    col_totals = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    result = {}
    for j, pred_label in enumerate(labels):
        col_total = col_totals[j]
        tp = matrix[j][j]
        result[pred_label] = {
            "col_total": col_total,
            "precision": round(tp / col_total, 4) if col_total else None,
            # 预测成这一类的样本，实际 gold 是什么
            "actual_gold": {
                labels[i]: round(matrix[i][j] / col_total, 4) if col_total else None
                for i in range(n)
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="对比多个方法的混淆矩阵，诊断是否存在过度预测 none 的偏差")
    parser.add_argument("--eval_dir", required=True,
                         help="evaluate_stance_predictions.py 的 --output_dir 所在的父目录")
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", nargs="+", required=True,
                         help="按你关心的顺序传入方法名，比如从简单到复杂："
                              "current_only dynamic_window dynamic_semantic dynamic_rag full_rag")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    all_recalls = {}
    all_precisions = {}
    for method in args.methods:
        cm_path = eval_dir / f"{method}_{args.split}" / "confusion_matrix.json"
        cm = load_json(cm_path)
        if cm is None or not cm:
            print(f"⚠️ {method}: 找不到或为空 {cm_path}，跳过")
            continue
        all_recalls[method] = recall_per_class(cm)
        all_precisions[method] = precision_per_class(cm)

    if not all_recalls:
        raise SystemExit("❌ 没有任何方法能加载到混淆矩阵，检查 --eval_dir / --split / --methods 是否正确。")

    labels = ["favor", "against", "none"]

    print(f"\n{'='*100}")
    print(f"各方法在每个 gold 类别上的召回率（recall），split={args.split}")
    print(f"{'='*100}")
    header = f"{'method':<20}" + "".join(f"{l+'_recall':>16}" for l in labels)
    print(header)
    for method in args.methods:
        if method not in all_recalls:
            continue
        r = all_recalls[method]
        row = f"{method:<20}"
        for l in labels:
            v = r.get(l, {}).get("recall")
            row += f"{v if v is not None else '-':>16}"
        print(row)

    print(f"\n{'='*100}")
    print(f"各方法在每个预测类别上的精确率（precision），split={args.split}")
    print(f"（精确率低 = 预测成这一类的样本里，很多其实不是这一类，说明存在大量误报）")
    print(f"{'='*100}")
    header = f"{'method':<20}" + "".join(f"{l+'_precision':>16}" for l in labels)
    print(header)
    for method in args.methods:
        if method not in all_precisions:
            continue
        p = all_precisions[method]
        row = f"{method:<20}"
        for l in labels:
            v = p.get(l, {}).get("precision")
            row += f"{v if v is not None else '-':>16}"
        print(row)

    print(f"\n{'='*100}")
    print("gold=favor 的样本，被预测成什么（看 favor 是不是被'吃'成了 none）")
    print(f"{'='*100}")
    header = f"{'method':<20}" + "".join(f"{'pred='+l:>16}" for l in labels)
    print(header)
    for method in args.methods:
        if method not in all_recalls:
            continue
        dist = all_recalls[method]["favor"]["predicted_as"]
        row = f"{method:<20}"
        for l in labels:
            row += f"{dist.get(l, '-'):>16}"
        print(row)

    print(f"\n{'='*100}")
    print("gold=against 的样本，被预测成什么")
    print(f"{'='*100}")
    header = f"{'method':<20}" + "".join(f"{'pred='+l:>16}" for l in labels)
    print(header)
    for method in args.methods:
        if method not in all_recalls:
            continue
        dist = all_recalls[method]["against"]["predicted_as"]
        row = f"{method:<20}"
        for l in labels:
            row += f"{dist.get(l, '-'):>16}"
        print(row)

    print(f"\n{'='*100}")
    print("gold=none 的样本，被预测成什么（看 none 是不是被大量误判成了 favor/against）")
    print(f"{'='*100}")
    header = f"{'method':<20}" + "".join(f"{'pred='+l:>16}" for l in labels)
    print(header)
    for method in args.methods:
        if method not in all_recalls:
            continue
        dist = all_recalls[method]["none"]["predicted_as"]
        row = f"{method:<20}"
        for l in labels:
            row += f"{dist.get(l, '-'):>16}"
        print(row)

    print(f"\n{'='*100}")
    print("预测成 favor / against 的样本，实际 gold 是什么（precision 的分解）")
    print(f"{'='*100}")
    for pred_l in ["favor", "against"]:
        print(f"\n预测成 {pred_l}：")
        header = f"  {'method':<18}" + "".join(f"{'gold='+l:>16}" for l in labels)
        print(header)
        for method in args.methods:
            if method not in all_precisions:
                continue
            dist = all_precisions[method][pred_l]["actual_gold"]
            row = f"  {method:<18}"
            for l in labels:
                row += f"{dist.get(l, '-'):>16}"
            print(row)

    # ---- 自动判断趋势 ----
    methods_with_data = [m for m in args.methods if m in all_recalls]
    if len(methods_with_data) >= 2:
        first, last = methods_with_data[0], methods_with_data[-1]

        favor_recall_first = all_recalls[first]["favor"]["recall"] or 0
        favor_recall_last = all_recalls[last]["favor"]["recall"] or 0
        against_recall_first = all_recalls[first]["against"]["recall"] or 0
        against_recall_last = all_recalls[last]["against"]["recall"] or 0
        none_recall_first = all_recalls[first]["none"]["recall"] or 0
        none_recall_last = all_recalls[last]["none"]["recall"] or 0

        favor_precision_first = all_precisions[first]["favor"]["precision"] or 0
        favor_precision_last = all_precisions[last]["favor"]["precision"] or 0
        against_precision_first = all_precisions[first]["against"]["precision"] or 0
        against_precision_last = all_precisions[last]["against"]["precision"] or 0

        none_to_favor_or_against_first = (
            all_recalls[first]["none"]["predicted_as"].get("favor", 0)
            + all_recalls[first]["none"]["predicted_as"].get("against", 0)
        )
        none_to_favor_or_against_last = (
            all_recalls[last]["none"]["predicted_as"].get("favor", 0)
            + all_recalls[last]["none"]["predicted_as"].get("against", 0)
        )

        print(f"\n{'='*100}")
        print(f"自动诊断：{first} → {last}")
        print(f"{'='*100}")
        print(f"favor 召回率：{favor_recall_first} → {favor_recall_last}"
              f"（{'上升' if favor_recall_last > favor_recall_first else '下降'}）")
        print(f"against 召回率：{against_recall_first} → {against_recall_last}"
              f"（{'上升' if against_recall_last > against_recall_first else '下降'}）")
        print(f"none 召回率：{none_recall_first} → {none_recall_last}"
              f"（{'上升' if none_recall_last > none_recall_first else '下降'}）")
        print(f"favor 精确率：{favor_precision_first} → {favor_precision_last}"
              f"（{'上升' if favor_precision_last > favor_precision_first else '下降'}）")
        print(f"against 精确率：{against_precision_first} → {against_precision_last}"
              f"（{'上升' if against_precision_last > against_precision_first else '下降'}）")
        print(f"gold=none 被误判为 favor/against 的比例：{round(none_to_favor_or_against_first,4)}"
              f" → {round(none_to_favor_or_against_last,4)}"
              f"（{'上升' if none_to_favor_or_against_last > none_to_favor_or_against_first else '没有明显上升'}）")

        none_predict_rate_first = (
            all_recalls[first]["favor"]["predicted_as"].get("none", 0)
            + all_recalls[first]["against"]["predicted_as"].get("none", 0)
        )
        none_predict_rate_last = (
            all_recalls[last]["favor"]["predicted_as"].get("none", 0)
            + all_recalls[last]["against"]["predicted_as"].get("none", 0)
        )

        over_cautious = (
            favor_recall_last < favor_recall_first and against_recall_last < against_recall_first
            and none_predict_rate_last > none_predict_rate_first
        )
        over_eager = (
            (favor_recall_last > favor_recall_first or against_recall_last > against_recall_first)
            and none_recall_last < none_recall_first
            and (favor_precision_last < favor_precision_first or against_precision_last < against_precision_first)
        )

        print()
        if over_cautious:
            print("✅ 符合'过度谨慎、过度预测 none'假设：favor/against 召回率同时下降，且都更多地被误判成 none。"
                  "\n   建议去掉/弱化 prompt 里'reply_author_only/topic_shift 时应谨慎判断'这类容易"
                  "\n   诱导模型回避表态的措辞。")
        elif over_eager:
            print("✅ 符合'过度敢下判断、把 none 错判成 favor/against'假设：favor/against 召回率上升但"
                  "\n   精确率下降，同时 none 召回率明显走低——说明模型看到语义传导/RAG 信息后，"
                  "\n   变得更容易把中立/无关的评论也强行关联到 main_target 并给出明确立场。"
                  "\n   建议检查 prompt 里是否有类似'不要把无关/目标不清简单当作 none'这类指令"
                  "\n   矫枉过正，导致模型对本该是 none 的样本也强行找理由归到 favor/against。")
        else:
            print("⚠️ 误判模式比较混合，没有呈现单一清晰趋势，建议直接抽查"
                  "\n   error_cases.jsonl 里 gold=none 但 predicted 是 favor/against 的具体样本，"
                  "\n   看 reason_short 里 LLM 自己给出的理由，人工判断问题出在哪一步。")


if __name__ == "__main__":
    main()
