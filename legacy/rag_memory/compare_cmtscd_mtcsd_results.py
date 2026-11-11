#!/usr/bin/env python3
"""
compare_cmtscd_mtcsd_results.py

汇总 MT-CSD（英文 Reddit）上的 LLM 消融实验 + PLM 消融实验结果，
并且如果提供了 C-MTCSD（中文微博）的历史结果文件，会生成一份
"中文 vs 英文" 跨语言、跨数据集的对照表 —— 这是本次 MT-CSD 实验
最核心的产出：验证你的方法（目标感知动态窗口 + 语义传导分析 +
结构化语言模式 RAG）是否具有跨语言 / 跨数据集的普适性。

用法：
python compare_cmtscd_mtcsd_results.py \
  --mtcsd_llm_eval_dir mtcsd_experiment/rag_evaluation_outputs \
  --mtcsd_plm_summary  mtcsd_experiment/plm_model_outputs/summary_results.json \
  --cmtscd_llm_phase2_summary  make_rag_data/experiment_logs/phase2_summary.json \
  --cmtscd_plm_summary         make_rag_data/plm_model_outputs/summary_results.json \
  --output mtcsd_experiment/cmtscd_vs_mtcsd_comparison.md
"""

import json
import argparse
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_llm_eval_dir(eval_dir: str) -> dict:
    """
    扫描 rag_evaluation_outputs/{method}_{split}/evaluation_report.json，
    返回 {method: report}（只取 split=valid 的记录，多个 split 时优先 valid）
    """
    results = {}
    root = Path(eval_dir)
    if not root.exists():
        return results
    for sub in sorted(root.iterdir()):
        report_path = sub / "evaluation_report.json"
        report = load_json(report_path)
        if not report:
            continue
        method = report.get("method") or sub.name
        split = report.get("split", "")
        # 同一 method 有多个 split 目录时，优先保留 valid
        # 优先取 test split（和 collect_plm_summary_results.py 默认汇总
        # test split 的口径保持一致，避免 LLM/PLM 两组对比时用了不同 split）
        if method not in results or split == "test":
            results[method] = report
    return results


def fmt(v, digits=4):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def build_llm_table(mtcsd_llm: dict, cmtscd_llm_summary: dict) -> str:
    methods = [
        "current_only", "root_current", "parent_current", "full_path",
        "dynamic_window", "dynamic_semantic", "dynamic_rag", "full_rag",
    ]
    lines = [
        "| Method | C-MTCSD (zh) Acc | C-MTCSD (zh) Macro-F1 | MT-CSD (en) Acc | MT-CSD (en) Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for m in methods:
        zh = (cmtscd_llm_summary or {}).get(m, {})
        en = mtcsd_llm.get(m, {})
        lines.append(
            f"| `{m}` | {fmt(zh.get('accuracy'))} | {fmt(zh.get('macro_f1'))} "
            f"| {fmt(en.get('accuracy'))} | {fmt(en.get('macro_f1'))} |"
        )
    return "\n".join(lines)


def build_plm_table(mtcsd_plm_rows: list, cmtscd_plm_rows: list) -> str:
    methods = [
        "plm_current", "plm_parent", "plm_window",
        "plm_semantic_text", "plm_rag_text", "plm_semantic_rag_text",
        "plm_rag_feature", "plm_semantic_rag_feature",
    ]
    zh_by_method = {r["method"]: r for r in (cmtscd_plm_rows or [])}
    en_by_method = {r["method"]: r for r in (mtcsd_plm_rows or [])}
    lines = [
        "| Method | C-MTCSD (Chinese-MacBERT) Acc | C-MTCSD Macro-F1 | MT-CSD (English PLM) Acc | MT-CSD Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for m in methods:
        zh = zh_by_method.get(m, {})
        en = en_by_method.get(m, {})
        lines.append(
            f"| `{m}` | {fmt(zh.get('accuracy'))} | {fmt(zh.get('macro_f1'))} "
            f"| {fmt(en.get('accuracy'))} | {fmt(en.get('macro_f1'))} |"
        )
    return "\n".join(lines)


def relative_gain(baseline: float, full: float) -> str:
    if baseline is None or full is None or baseline == 0:
        return "-"
    return f"{(full - baseline) / baseline * 100:+.1f}%"


def main():
    parser = argparse.ArgumentParser(description="C-MTCSD vs MT-CSD 实验结果对照汇总")
    parser.add_argument("--mtcsd_llm_eval_dir", required=True,
                         help="MT-CSD LLM 实验的 evaluation 输出目录（含各 method 子目录）")
    parser.add_argument("--mtcsd_plm_summary", required=True,
                         help="MT-CSD PLM 实验的 summary_results.json")
    parser.add_argument("--cmtscd_llm_phase2_summary", default=None,
                         help="（可选）C-MTCSD 已有的 LLM phase2_summary.json，用于跨语言对照")
    parser.add_argument("--cmtscd_plm_summary", default=None,
                         help="（可选）C-MTCSD 已有的 PLM summary_results.json，用于跨语言对照")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mtcsd_llm = collect_llm_eval_dir(args.mtcsd_llm_eval_dir)
    mtcsd_plm = load_json(args.mtcsd_plm_summary) or []

    # 从实际收集到的报告里读出用的是哪个 split，标题动态显示，
    # 不再写死"valid split"（之前切到 test split 后忘了同步更新文案）
    actual_splits = {r.get("split") for r in mtcsd_llm.values() if r.get("split")}
    if len(actual_splits) == 1:
        split_label = actual_splits.pop()
    elif len(actual_splits) > 1:
        split_label = "/".join(sorted(actual_splits)) + "（混用了多个 split，建议统一）"
    else:
        split_label = "未知"

    cmtscd_llm = load_json(args.cmtscd_llm_phase2_summary) if args.cmtscd_llm_phase2_summary else None
    cmtscd_plm = load_json(args.cmtscd_plm_summary) if args.cmtscd_plm_summary else None

    llm_table = build_llm_table(mtcsd_llm, cmtscd_llm)
    plm_table = build_plm_table(mtcsd_plm, cmtscd_plm)

    # 相对提升（best method vs baseline），MT-CSD 侧
    mtcsd_llm_baseline = mtcsd_llm.get("current_only", {}).get("macro_f1")
    mtcsd_llm_full = mtcsd_llm.get("full_rag", {}).get("macro_f1")
    mtcsd_llm_gain = relative_gain(mtcsd_llm_baseline, mtcsd_llm_full)

    mtcsd_plm_by_method = {r["method"]: r for r in mtcsd_plm}
    mtcsd_plm_baseline = mtcsd_plm_by_method.get("plm_current", {}).get("macro_f1")
    mtcsd_plm_full = mtcsd_plm_by_method.get("plm_semantic_rag_feature", {}).get("macro_f1")
    mtcsd_plm_gain = relative_gain(mtcsd_plm_baseline, mtcsd_plm_full)

    report = f"""# C-MTCSD（中文） vs MT-CSD（英文）跨语言实验结果对照

本报告自动汇总两套实验结果：
- **C-MTCSD**（中文 Weibo，此前实验）：LLM 零样本提示 + Chinese-MacBERT 微调
- **MT-CSD**（英文 Reddit，本次新增实验）：LLM 零样本提示 + 英文 PLM 微调

用于验证"目标感知动态会话窗口 + 语义传导分析 + 结构化语言模式 RAG"
这套方法是否具有跨语言、跨数据集的普适性。

---

## 一、LLM 零样本提示消融结果对照（{split_label} split）

{llm_table}

**MT-CSD 上 full_rag 相对 current_only 的 Macro-F1 相对提升：{mtcsd_llm_gain}**
（对照：C-MTCSD 上此前报告的相对提升为 +37.0%）

---

## 二、PLM 微调消融结果对照

{plm_table}

**MT-CSD 上 plm_semantic_rag_feature 相对 plm_current 的 Macro-F1 相对提升：{mtcsd_plm_gain}**
（对照：C-MTCSD 上此前报告的相对提升为 +5.76%）

---

## 三、解读提示（需要人工核对，本脚本仅做数字汇总）

1. 如果 MT-CSD 上的相对提升幅度与 C-MTCSD 相近（同方向、同数量级），
   可以作为方法**跨语言普适性**的直接证据。
2. 如果两者出现明显差异，请重点检查：
   - MT-CSD 的英文语言模式词表（`build_target_aware_windows_en.py` 中的
     `AMBIGUOUS_PHRASES` / `RHETORICAL_MARKERS` 等）是否需要针对 Reddit
     语域进一步调整；
   - 英文 PLM 基座模型的选择（roberta-base 是通用领域模型，Reddit 社交
     文本可以尝试 `vinai/bertweet-base` 等社交媒体预训练模型）；
   - MT-CSD 五个 target（Bitcoin/Tesla/SpaceX/Biden/Trump）之间的立场分布
     差异较大（比如 Trump 的 against 占比远高于 favor），可能需要按
     target 拆分做子集分析，而不是只看整体 valid 集指标。
3. MT-CSD 自带的 `gold_logical_relationship` / `gold_dialogue_act` 字段
   （已保留在 `mtcsd_stance_instances.jsonl` 里）可以和你 LLM 标注出的
   `llm_reply_relation` 做交叉验证，这是 C-MTCSD 没有的额外优势，建议
   作为一个独立的分析补充到论文里。
"""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ 对照报告已写出：{args.output}")


if __name__ == "__main__":
    main()
