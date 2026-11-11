#!/usr/bin/env python3
"""
run_all_plm_experiments.py

用法：
python run_all_plm_experiments.py \
  --instances processed_cmtscd_semantic_llm/cmtscd_semantic_instances_llm.jsonl \
  --knowledge processed_cmtscd_rag_memory/rag_knowledge_documents_llm.jsonl \
  --base_model hfl/chinese-macbert-base \
  --epochs 5 \
  --batch_size 16
"""

import json
import csv
import subprocess
import argparse
import sys
from pathlib import Path
from datetime import datetime


DEFAULT_METHODS = [
    "plm_current",
    "plm_parent",
    "plm_window",
    "plm_semantic_text",
    "plm_rag_text",
    "plm_semantic_rag_text",
    "plm_rag_feature",
    "plm_semantic_rag_feature",
]

FEATURE_METHODS = {"plm_rag_feature", "plm_semantic_rag_feature"}
RAG_METHODS      = {"plm_rag_text", "plm_semantic_rag_text"} | FEATURE_METHODS
SEMANTIC_METHODS = {
    "plm_semantic_text", "plm_semantic_rag_text",
    "plm_rag_feature", "plm_semantic_rag_feature",
}
WINDOW_METHODS = {
    "plm_window", "plm_semantic_text", "plm_rag_text", "plm_semantic_rag_text",
    "plm_rag_feature", "plm_semantic_rag_feature",
}


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "RUN": "🚀"}.get(level, "  ")
    print(f"[{ts}] {prefix} {msg}")


def run_cmd(cmd: list, log_path: Path) -> bool:
    log(f"执行：{' '.join(cmd)}", "RUN")
    with open(log_path, "w", encoding="utf-8") as lf:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8")
        lf.write(result.stdout or "")
        for line in (result.stdout or "").strip().splitlines()[-8:]:
            print(f"    {line}")
    return result.returncode == 0


def load_json(filepath: Path) -> dict:
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="批量运行 PLM 立场检测消融实验")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--knowledge", required=True)
    parser.add_argument("--data_dir",   default="plm_experiment_data")
    parser.add_argument("--model_output_dir", default="plm_model_outputs")
    parser.add_argument("--base_model", default="hfl/chinese-macbert-base")
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--fp16",       default="true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--methods",    default=None,
                        help="逗号分隔的方法列表，默认运行全部8种")
    parser.add_argument("--skip_data_prep", default="false",
                        help="如果数据已准备好，可跳过 prepare 步骤")
    args = parser.parse_args()

    methods = args.methods.split(",") if args.methods else DEFAULT_METHODS
    skip_prep = args.skip_data_prep.lower() == "true"

    model_tag = args.base_model.split("/")[-1]
    model_output_root = Path(args.model_output_dir)
    log_dir = model_output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for method in methods:
        log(f"=== 方法：{method} ===", "INFO")

        # ---- Step 1: 数据准备 ----
        if not skip_prep:
            cmd = [
                sys.executable, "prepare_plm_stance_data.py",
                "--instances", args.instances,
                "--knowledge", args.knowledge,
                "--output_dir", args.data_dir,
                "--method", method,
                "--top_k", str(args.top_k),
            ]
            ok = run_cmd(cmd, log_dir / f"prepare_{method}.log")
            if not ok:
                log(f"{method} 数据准备失败，跳过训练", "ERR")
                continue
        else:
            log(f"  跳过数据准备（使用已有数据）", "INFO")

        # ---- Step 2: 训练 ----
        use_rag_feat = method in FEATURE_METHODS
        out_dir = model_output_root / f"{method}_{model_tag}"

        cmd = [
            sys.executable, "train_plm_stance_model.py",
            "--data_dir",   str(Path(args.data_dir) / method),
            "--output_dir", str(out_dir),
            "--base_model", args.base_model,
            "--use_rag_features", "true" if use_rag_feat else "false",
            "--max_length", str(args.max_length),
            "--epochs",     str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--seed",       str(args.seed),
            "--fp16",       args.fp16,
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        ]
        ok = run_cmd(cmd, log_dir / f"train_{method}.log")
        if not ok:
            log(f"{method} 训练失败", "ERR")
            continue

        log(f"{method} 完成", "OK")

        # ---- Step 3: 汇总结果 ----
        eval_test = load_json(out_dir / "evaluation_test.json")
        if not eval_test:
            log(f"{method} 未找到评估结果，跳过汇总", "WARN")
            continue

        pc = eval_test.get("per_class", {})
        summary_rows.append({
            "method":              method,
            "base_model":          args.base_model,
            "use_dynamic_window":  method in WINDOW_METHODS,
            "use_semantic":        method in SEMANTIC_METHODS,
            "use_rag_text":        method in {"plm_rag_text", "plm_semantic_rag_text"},
            "use_rag_feature":     method in FEATURE_METHODS,
            "accuracy":            eval_test.get("accuracy"),
            "macro_f1":            eval_test.get("macro_f1"),
            "weighted_f1":         eval_test.get("weighted_f1"),
            "favor_f1":            pc.get("favor", {}).get("f1"),
            "against_f1":          pc.get("against", {}).get("f1"),
            "none_f1":             pc.get("none", {}).get("f1"),
        })

    # ---- 写出汇总表 ----
    model_output_root.mkdir(parents=True, exist_ok=True)
    csv_path  = model_output_root / "summary_results.csv"
    json_path = model_output_root / "summary_results.json"

    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        log(f"汇总表已保存：{csv_path}", "OK")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        log(f"汇总表已保存：{json_path}", "OK")

        # ---- 打印对比表 ----
        print("\n" + "=" * 100)
        print(f"{'方法':<26} {'窗口':>5} {'语义':>5} {'RAG文本':>8} {'RAG特征':>8} "
              f"{'Acc':>7} {'Macro-F1':>9} {'Favor-F1':>9} {'Against-F1':>11} {'None-F1':>8}")
        print("-" * 100)
        for r in sorted(summary_rows, key=lambda x: -(x.get("macro_f1") or 0)):
            print(f"{r['method']:<26} "
                  f"{'✓' if r['use_dynamic_window'] else '-':>5} "
                  f"{'✓' if r['use_semantic'] else '-':>5} "
                  f"{'✓' if r['use_rag_text'] else '-':>8} "
                  f"{'✓' if r['use_rag_feature'] else '-':>8} "
                  f"{str(r['accuracy']):>7} {str(r['macro_f1']):>9} "
                  f"{str(r['favor_f1']):>9} {str(r['against_f1']):>11} {str(r['none_f1']):>8}")
        print("=" * 100)

        # ---- 重点对比提示 ----
        by_method = {r["method"]: r for r in summary_rows}
        print("\n📌 重点对比：")
        pairs = [
            ("plm_current", "plm_window", "动态窗口是否增强专门模型"),
            ("plm_window", "plm_semantic_text", "语义传导字段是否有帮助"),
            ("plm_window", "plm_rag_feature", "RAG 特征是否有帮助"),
            ("plm_semantic_text", "plm_semantic_rag_feature", "在已有语义信息基础上 RAG 是否仍有增益"),
            ("plm_rag_text", "plm_rag_feature", "结构化 RAG 特征是否优于长文本拼接"),
        ]
        for a, b, desc in pairs:
            if a in by_method and b in by_method:
                fa = by_method[a].get("macro_f1") or 0
                fb = by_method[b].get("macro_f1") or 0
                delta = fb - fa
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                print(f"  {desc}：{a}({fa:.4f}) → {b}({fb:.4f})  {arrow} {delta:+.4f}")
            else:
                print(f"  {desc}：缺少 {a} 或 {b} 的结果，跳过")
    else:
        log("没有任何方法成功完成，未生成汇总表", "WARN")

    log(f"全部实验完成，共 {len(summary_rows)}/{len(methods)} 个方法成功", "OK")


if __name__ == "__main__":
    main()
