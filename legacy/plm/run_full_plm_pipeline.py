#!/usr/bin/env python3
"""
run_full_plm_pipeline.py

一键顺序运行 PLM 立场检测消融实验完整流程：
  Step 1: prepare_plm_stance_data.py    （为 8 种 method 分别准备数据）
  Step 2: train_plm_stance_model.py     （为 8 种 method 分别训练模型）
  Step 3: run_all_plm_experiments.py 的汇总逻辑（自动调用，生成 summary）
  Step 4: analyze_plm_errors.py         （对完整方法和关键 baseline 做错误分析）

所有终端输出会实时打印，同时完整保存到 pipeline_logs/ 下的日志文件。

用法（默认跑全部8种方法）：
python run_full_plm_pipeline.py \
  --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \
  --knowledge processed_cmtscd_rag_memory/rag_knowledge_documents_llm.jsonl \
  --base_model hfl/chinese-macbert-base \
  --epochs 5 --batch_size 16

只跑部分方法：
python run_full_plm_pipeline.py \
  --instances ... --knowledge ... \
  --methods plm_current,plm_window,plm_semantic_rag_feature

跳过数据准备（数据已生成过）：
python run_full_plm_pipeline.py --instances ... --knowledge ... --skip_data_prep true
"""

import subprocess
import argparse
import sys
import json
import csv
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

FEATURE_METHODS  = {"plm_rag_feature", "plm_semantic_rag_feature"}
RAG_TEXT_METHODS = {"plm_rag_text", "plm_semantic_rag_text"}
SEMANTIC_METHODS = {
    "plm_semantic_text", "plm_semantic_rag_text",
    "plm_rag_feature", "plm_semantic_rag_feature",
}
WINDOW_METHODS = {
    "plm_window", "plm_semantic_text", "plm_rag_text", "plm_semantic_rag_text",
    "plm_rag_feature", "plm_semantic_rag_feature",
}

REQUIRED_SCRIPTS = [
    "prepare_plm_stance_data.py",
    "train_plm_stance_model.py",
    "analyze_plm_errors.py",
]


# ============ 日志：同时输出到终端和文件 ============

class TeeLogger:
    """把 print 内容同时写入终端和日志文件"""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def write(self, msg: str):
        sys.__stdout__.write(msg)
        self._fh.write(msg)
        self._fh.flush()

    def flush(self):
        sys.__stdout__.flush()
        self._fh.flush()

    def close(self):
        self._fh.close()


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "RUN": "🚀", "STEP": "📍"}.get(level, "  ")
    print(f"[{ts}] {prefix} {msg}")


def run_cmd(cmd: list, step_log_path: Path) -> bool:
    """运行子进程，子进程的完整 stdout/stderr 实时打印并写入独立日志文件，
    同时（通过外层 TeeLogger）一并汇入总日志。"""
    log(f"执行：{' '.join(cmd)}", "RUN")
    with open(step_log_path, "w", encoding="utf-8") as lf:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1,
        )
        for line in process.stdout:
            print(f"    {line.rstrip()}")
            lf.write(line)
        process.wait()
    return process.returncode == 0


def check_required_files(instances: str, knowledge: str) -> bool:
    ok = True
    for f in REQUIRED_SCRIPTS:
        if not Path(f).exists():
            log(f"缺少必要脚本：{f}（需与本脚本放在同一目录）", "ERR")
            ok = False
    for f in (instances, knowledge):
        if not Path(f).exists():
            log(f"缺少输入文件：{f}", "ERR")
            ok = False
    return ok


def load_json(filepath: Path) -> dict:
    if not filepath.exists():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def print_comparison_table(summary_rows: list):
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


def print_key_comparisons(summary_rows: list):
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


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="PLM 立场检测消融实验一键完整流程")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--knowledge", required=True)
    parser.add_argument("--data_dir",         default="plm_experiment_data")
    parser.add_argument("--model_output_dir", default="plm_model_outputs")
    parser.add_argument("--error_output_dir", default="plm_error_analysis")
    parser.add_argument("--log_dir",          default="pipeline_logs")
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
                        help="逗号分隔，默认跑全部8种方法")
    parser.add_argument("--skip_data_prep", default="false")
    parser.add_argument("--skip_error_analysis", default="false")
    args = parser.parse_args()

    methods = args.methods.split(",") if args.methods else DEFAULT_METHODS
    skip_prep  = args.skip_data_prep.lower() == "true"
    skip_error = args.skip_error_analysis.lower() == "true"
    model_tag  = args.base_model.split("/")[-1]

    # ---- 日志初始化：主日志文件名带时间戳，避免覆盖历史记录 ----
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = log_dir / f"full_pipeline_{run_ts}.log"
    step_log_dir  = log_dir / f"steps_{run_ts}"
    step_log_dir.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(main_log_path)
    sys.stdout = tee

    try:
        start_time = datetime.now()
        log(f"完整流程启动：{start_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
        log(f"主日志文件：{main_log_path}", "INFO")
        log(f"方法列表：{methods}", "INFO")
        log(f"base_model={args.base_model}  epochs={args.epochs}  batch_size={args.batch_size}", "INFO")

        if not check_required_files(args.instances, args.knowledge):
            log("缺少必要文件，流程终止", "ERR")
            sys.exit(1)

        model_output_root = Path(args.model_output_dir)
        error_output_root  = Path(args.error_output_dir)
        model_output_root.mkdir(parents=True, exist_ok=True)
        error_output_root.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        method_status = {}

        # ================= Step 1 + 2：逐方法 数据准备 + 训练 =================
        for i, method in enumerate(methods, 1):
            log(f"=== [{i}/{len(methods)}] 方法：{method} ===", "STEP")

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
                ok = run_cmd(cmd, step_log_dir / f"01_prepare_{method}.log")
                if not ok:
                    log(f"{method} 数据准备失败，跳过该方法", "ERR")
                    method_status[method] = "prepare_failed"
                    continue
                log(f"{method} 数据准备完成", "OK")
            else:
                log("跳过数据准备（使用已有数据）", "INFO")

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
            ok = run_cmd(cmd, step_log_dir / f"02_train_{method}.log")
            if not ok:
                log(f"{method} 训练失败", "ERR")
                method_status[method] = "train_failed"
                continue

            log(f"{method} 训练完成", "OK")
            method_status[method] = "success"

            # ---- 收集汇总结果 ----
            eval_test = load_json(out_dir / "evaluation_test.json")
            if eval_test:
                pc = eval_test.get("per_class", {})
                summary_rows.append({
                    "method":             method,
                    "base_model":         args.base_model,
                    "use_dynamic_window": method in WINDOW_METHODS,
                    "use_semantic":       method in SEMANTIC_METHODS,
                    "use_rag_text":       method in RAG_TEXT_METHODS,
                    "use_rag_feature":    method in FEATURE_METHODS,
                    "accuracy":           eval_test.get("accuracy"),
                    "macro_f1":           eval_test.get("macro_f1"),
                    "weighted_f1":        eval_test.get("weighted_f1"),
                    "favor_f1":           pc.get("favor", {}).get("f1"),
                    "against_f1":         pc.get("against", {}).get("f1"),
                    "none_f1":            pc.get("none", {}).get("f1"),
                })
            else:
                log(f"{method} 未找到 evaluation_test.json，无法纳入汇总", "WARN")

        # ================= Step 3：写出汇总表 =================
        log("=== 汇总全部方法结果 ===", "STEP")
        csv_path  = model_output_root / "summary_results.csv"
        json_path = model_output_root / "summary_results.json"

        if summary_rows:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary_rows, f, ensure_ascii=False, indent=2)
            log(f"汇总表已保存：{csv_path}", "OK")
            log(f"汇总表已保存：{json_path}", "OK")

            print_comparison_table(summary_rows)
            print_key_comparisons(summary_rows)
        else:
            log("没有任何方法成功完成训练，跳过汇总表生成", "WARN")

        # ================= Step 4：错误分析 =================
        if not skip_error and summary_rows:
            log("=== 对各方法预测结果进行错误分析 ===", "STEP")

            # plm_window 作为错误分析中的统一 baseline 参照
            window_pred_path = model_output_root / f"plm_window_{model_tag}" / "predictions_test.jsonl"
            baseline_arg = []
            if window_pred_path.exists():
                baseline_arg = ["--baseline_predictions", str(window_pred_path)]
                log(f"使用 plm_window 作为错误分析的 baseline 参照", "INFO")
            else:
                log("未找到 plm_window 预测结果，错误分析将不带 baseline 对比", "WARN")

            for method in methods:
                if method_status.get(method) != "success":
                    continue
                pred_path = model_output_root / f"{method}_{model_tag}" / "predictions_test.jsonl"
                if not pred_path.exists():
                    log(f"{method} 缺少 predictions_test.jsonl，跳过错误分析", "WARN")
                    continue

                err_out_dir = error_output_root / method
                cmd = [
                    sys.executable, "analyze_plm_errors.py",
                    "--predictions", str(pred_path),
                    "--output_dir",  str(err_out_dir),
                ] + (baseline_arg if method != "plm_window" else [])

                ok = run_cmd(cmd, step_log_dir / f"04_analyze_{method}.log")
                if ok:
                    log(f"{method} 错误分析完成 → {err_out_dir}", "OK")
                else:
                    log(f"{method} 错误分析失败", "ERR")
        elif skip_error:
            log("已跳过错误分析（--skip_error_analysis true）", "INFO")

        # ================= 最终汇总 =================
        elapsed = (datetime.now() - start_time).seconds
        log("=== 完整流程结束 ===", "STEP")
        print(f"""
🎯 运行状态汇总：
""")
        for m in methods:
            status = method_status.get(m, "not_run")
            icon = {"success": "✅", "prepare_failed": "❌(数据准备失败)",
                    "train_failed": "❌(训练失败)", "not_run": "⏭️ (未运行)"}.get(status, status)
            print(f"  {m:<28} {icon}")

        print(f"""
输出目录：
  数据：       {args.data_dir}/
  模型与预测： {args.model_output_dir}/
  错误分析：   {args.error_output_dir}/
  完整日志：   {main_log_path}
  各步骤日志： {step_log_dir}/

总耗时：{elapsed // 3600} 时 {(elapsed % 3600) // 60} 分 {elapsed % 60} 秒
        """)
        log(f"完整流程完成，主日志已保存至：{main_log_path}", "OK")

    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == "__main__":
    main()