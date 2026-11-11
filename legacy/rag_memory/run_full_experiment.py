#!/usr/bin/env python3
"""
run_full_experiment.py

一键运行完整实验流程：
python run_full_experiment.py \
  --provider openai \
  --model gpt-4o-mini \
  --phase all

分阶段运行：
python run_full_experiment.py --phase 1   # 小样本验证
python run_full_experiment.py --phase 2   # valid 全量对比
python run_full_experiment.py --phase 3   # test 最终评估
python run_full_experiment.py --phase 4   # 专项子集分析
"""

import subprocess
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


# ============ 全局配置 ============

INSTANCES_FILE = "processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl"
KNOWLEDGE_FILE  = "processed_cmtscd_rag_memory/rag_knowledge_documents_llm.jsonl"
PRED_OUTPUT_DIR = "rag_prediction_outputs"
EVAL_OUTPUT_DIR = "rag_evaluation_outputs"
LOG_DIR         = "experiment_logs"

ALL_METHODS = [
    "current_only",
    "root_current",
    "parent_current",
    "full_path",
    "dynamic_window",
    "dynamic_semantic",
    "dynamic_rag",
    "full_rag",
]

SUBSET_CONFIGS = [
    {
        "name":   "short_text",
        "method": "full_rag",
        "flags":  ["--only_ambiguous", "true"],
        "desc":   "短文本/模糊评论子集",
    },
    {
        "name":   "deep_reply",
        "method": "full_rag",
        "flags":  ["--only_depth_ge", "3"],
        "desc":   "depth>=3 深层回复子集",
    },
    {
        "name":   "child_feedback",
        "method": "full_rag_posthoc",
        "flags":  ["--only_child_feedback", "true"],
        "desc":   "子回复消歧子集（posthoc）",
    },
    {
        "name":   "short_text_baseline",
        "method": "current_only",
        "flags":  ["--only_ambiguous", "true"],
        "desc":   "短文本 baseline 对比",
    },
    {
        "name":   "deep_reply_baseline",
        "method": "current_only",
        "flags":  ["--only_depth_ge", "3"],
        "desc":   "深层回复 baseline 对比",
    },
]


# ============ 工具函数 ============

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "RUN": "🚀"}.get(level, "  ")
    print(f"[{ts}] {prefix} {msg}")


def ensure_dirs():
    for d in (PRED_OUTPUT_DIR, EVAL_OUTPUT_DIR, LOG_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def check_files():
    ok = True
    for f in (INSTANCES_FILE, KNOWLEDGE_FILE,
              "simple_rag_stance_predictor.py",
              "evaluate_stance_predictions.py"):
        if not Path(f).exists():
            log(f"缺少必要文件：{f}", "ERR")
            ok = False
    return ok


def run_cmd(cmd: list, step_name: str, log_file: Path) -> bool:
    log(f"执行：{' '.join(cmd)}", "RUN")
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
            lf.write(result.stdout or "")
            # 实时打印最后几行
            lines = (result.stdout or "").strip().splitlines()
            for line in lines[-6:]:
                print(f"    {line}")

        if result.returncode == 0:
            log(f"{step_name} 完成", "OK")
            return True
        else:
            log(f"{step_name} 失败（returncode={result.returncode}），详见 {log_file}", "ERR")
            return False
    except Exception as e:
        log(f"{step_name} 异常：{e}", "ERR")
        return False


def pred_path(method: str, split: str, suffix: str = "") -> Path:
    name = f"predictions_{method}_{split}{suffix}.jsonl"
    return Path(PRED_OUTPUT_DIR) / name


def eval_dir(method: str, split: str, suffix: str = "") -> Path:
    d = Path(EVAL_OUTPUT_DIR) / f"{method}_{split}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_predictor_cmd(
    method: str,
    split: str,
    provider: str,
    model: str,
    api_base: Optional[str],
    sample_limit: Optional[int],
    top_k: int,
    extra_flags: list = [],
) -> list:
    cmd = [
        sys.executable, "simple_rag_stance_predictor.py",
        "--instances",  INSTANCES_FILE,
        "--knowledge",  KNOWLEDGE_FILE,
        "--output_dir", PRED_OUTPUT_DIR,
        "--split",      split,
        "--method",     method,
        "--provider",   provider,
        "--model",      model,
        "--top_k",      str(top_k),
        "--resume",     "true",
    ]
    if api_base:
        cmd += ["--api_base", api_base]
    if sample_limit:
        cmd += ["--sample_limit", str(sample_limit)]
    cmd += extra_flags
    return cmd


def build_evaluator_cmd(pred_file: Path, out_dir: Path) -> list:
    return [
        sys.executable, "evaluate_stance_predictions.py",
        "--predictions", str(pred_file),
        "--output_dir",  str(out_dir),
        "--group_by",    "true",
    ]


def load_eval_summary(eval_dir: Path) -> dict:
    report_path = eval_dir / "evaluation_report.json"
    if not report_path.exists():
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def print_comparison_table(results: dict):
    """打印方法对比表格"""
    print("\n" + "=" * 70)
    print(f"{'方法':<28} {'Acc':>7} {'Macro-F1':>10} {'favor-F1':>10} {'against-F1':>12} {'none-F1':>9}")
    print("-" * 70)
    for method, r in sorted(results.items(), key=lambda x: -x[1].get("macro_f1", 0)):
        acc    = r.get("accuracy", "-")
        mf1    = r.get("macro_f1", "-")
        pc     = r.get("per_class", {})
        ff1    = pc.get("favor",   {}).get("f1", "-")
        af1    = pc.get("against", {}).get("f1", "-")
        nf1    = pc.get("none",    {}).get("f1", "-")
        print(f"{method:<28} {str(acc):>7} {str(mf1):>10} {str(ff1):>10} {str(af1):>12} {str(nf1):>9}")
    print("=" * 70 + "\n")


# ============ 阶段函数 ============

def phase1_small_sample(args):
    """第一阶段：小样本快速验证（current_only vs full_rag，各300条 valid）"""
    log("=== 第一阶段：小样本快速验证 ===", "INFO")
    results = {}

    for method in ("current_only", "full_rag"):
        step = f"预测 {method} valid(300)"
        pf   = pred_path(method, "valid", "_sample300")
        ed   = eval_dir(method, "valid", "_sample300")
        lf   = Path(LOG_DIR) / f"phase1_{method}.log"

        # 预测
        cmd = build_predictor_cmd(
            method, "valid", args.provider, args.model,
            args.api_base, 300, args.top_k
        )
        # 覆盖输出文件名以区分 sample300
        cmd[cmd.index("--output_dir") + 1] = PRED_OUTPUT_DIR
        # 直接用默认路径，sample_limit 保证数量
        ok = run_cmd(cmd, step, lf)
        if not ok:
            continue

        # 评估
        actual_pred = pred_path(method, "valid")
        if actual_pred.exists():
            run_cmd(build_evaluator_cmd(actual_pred, ed),
                    f"评估 {method} valid", Path(LOG_DIR) / f"phase1_eval_{method}.log")
            results[method] = load_eval_summary(ed)

    log("第一阶段结果对比：", "INFO")
    print_comparison_table(results)
    save_phase_summary(results, "phase1_summary.json")
    return results


def phase2_valid_all_methods(args):
    """第二阶段：valid 全量对比所有方法"""
    log("=== 第二阶段：valid 全量对比（所有方法）===", "INFO")
    results = {}

    for method in ALL_METHODS:
        step = f"预测 {method} valid"
        pf   = pred_path(method, "valid")
        ed   = eval_dir(method, "valid")
        lf   = Path(LOG_DIR) / f"phase2_pred_{method}.log"
        le   = Path(LOG_DIR) / f"phase2_eval_{method}.log"

        # 预测
        cmd = build_predictor_cmd(
            method, "valid", args.provider, args.model,
            args.api_base, args.sample_limit, args.top_k
        )
        ok = run_cmd(cmd, step, lf)
        if not ok:
            log(f"  跳过 {method} 的评估", "WARN")
            continue

        # 评估
        if pf.exists():
            run_cmd(build_evaluator_cmd(pf, ed), f"评估 {method} valid", le)
            results[method] = load_eval_summary(ed)
        else:
            log(f"  预测文件不存在：{pf}", "WARN")

    log("第二阶段结果对比（valid）：", "INFO")
    print_comparison_table(results)
    save_phase_summary(results, "phase2_summary.json")

    # 找最优方法
    if results:
        best = max(results, key=lambda m: results[m].get("macro_f1", 0))
        log(f"valid 最优方法：{best}（Macro-F1={results[best].get('macro_f1')}）", "OK")
        return results, best
    return results, "full_rag"


def phase3_test_best(args, best_method: str = "full_rag"):
    """第三阶段：最优方法在 test 上最终评估"""
    log(f"=== 第三阶段：test 最终评估（method={best_method}）===", "INFO")

    pf = pred_path(best_method, "test")
    ed = eval_dir(best_method, "test")
    lf = Path(LOG_DIR) / f"phase3_pred_{best_method}.log"
    le = Path(LOG_DIR) / f"phase3_eval_{best_method}.log"

    cmd = build_predictor_cmd(
        best_method, "test", args.provider, args.model,
        args.api_base, None, args.top_k
    )
    ok = run_cmd(cmd, f"预测 {best_method} test", lf)
    if not ok:
        log("test 预测失败，跳过评估", "ERR")
        return {}

    if pf.exists():
        run_cmd(build_evaluator_cmd(pf, ed), f"评估 {best_method} test", le)
        result = load_eval_summary(ed)
        log(f"test 最终结果：Accuracy={result.get('accuracy')}  Macro-F1={result.get('macro_f1')}", "OK")
        save_phase_summary({best_method: result}, "phase3_summary.json")
        return result
    return {}


def phase4_subset_analysis(args):
    """第四阶段：专项子集分析"""
    log("=== 第四阶段：专项子集分析 ===", "INFO")
    results = {}

    for cfg in SUBSET_CONFIGS:
        name   = cfg["name"]
        method = cfg["method"]
        flags  = cfg["flags"]
        desc   = cfg["desc"]

        step = f"子集预测 {name}({method})"
        pf   = pred_path(method, "valid", f"_{name}")
        ed   = eval_dir(method, "valid", f"_{name}")
        lf   = Path(LOG_DIR) / f"phase4_pred_{name}.log"
        le   = Path(LOG_DIR) / f"phase4_eval_{name}.log"

        log(f"  {desc}", "INFO")

        cmd = build_predictor_cmd(
            method, "valid", args.provider, args.model,
            args.api_base, args.sample_limit, args.top_k,
            extra_flags=flags
        )
        # 修改输出文件后缀
        # 由于 predictor 固定输出 predictions_{method}_{split}.jsonl
        # 先预测再复制
        ok = run_cmd(cmd, step, lf)

        # 实际预测文件（不带 suffix）
        actual_pred = pred_path(method, "valid")
        if ok and actual_pred.exists():
            # 复制为带 suffix 的版本便于区分
            import shutil
            shutil.copy(str(actual_pred), str(pf))

            run_cmd(build_evaluator_cmd(pf, ed), f"评估子集 {name}", le)
            r = load_eval_summary(ed)
            r["subset"] = name
            r["desc"]   = desc
            results[name] = r

    log("第四阶段子集分析结果：", "INFO")
    print("\n" + "=" * 70)
    print(f"{'子集':<28} {'Acc':>7} {'Macro-F1':>10} {'样本数':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{r.get('desc', name):<28} "
              f"{str(r.get('accuracy', '-')):>7} "
              f"{str(r.get('macro_f1', '-')):>10} "
              f"{str(r.get('valid_samples', '-')):>8}")
    print("=" * 70 + "\n")

    save_phase_summary(results, "phase4_summary.json")
    return results


def save_phase_summary(data: dict, filename: str):
    path = Path(LOG_DIR) / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log(f"阶段汇总已保存：{path}", "OK")


def print_final_summary(phase_results: dict):
    print("\n" + "🎯 " * 20)
    print("完整实验汇总")
    print("🎯 " * 20)

    if "phase2" in phase_results:
        log("valid 各方法对比：", "INFO")
        print_comparison_table(phase_results["phase2"])

    if "phase3" in phase_results and "best_method" in phase_results:
        best = phase_results["best_method"]
        r    = phase_results["phase3"]
        log(f"test 最终结果（{best}）：", "INFO")
        print(f"  Accuracy：   {r.get('accuracy')}")
        print(f"  Macro-F1：   {r.get('macro_f1')}")
        print(f"  Weighted-F1：{r.get('weighted_f1')}")
        pc = r.get("per_class", {})
        for lbl in ("favor", "against", "none"):
            info = pc.get(lbl, {})
            print(f"  {lbl:<10} P={info.get('precision')}  R={info.get('recall')}  F1={info.get('f1')}")

    print("\n输出目录：")
    print(f"  预测结果：{PRED_OUTPUT_DIR}/")
    print(f"  评估报告：{EVAL_OUTPUT_DIR}/")
    print(f"  运行日志：{LOG_DIR}/")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="C-MTCSD RAG 立场检测完整实验流程")
    parser.add_argument("--phase",         default="all",
                        help="运行阶段：1/2/3/4/all（默认 all）")
    parser.add_argument("--provider",      default="openai",
                        choices=["openai", "deepseek", "qwen", "local"])
    parser.add_argument("--model",         default="gpt-4o-mini")
    parser.add_argument("--api_base",      default=None,
                        help="API base URL，国内代理填这里")
    parser.add_argument("--top_k",         type=int, default=5)
    parser.add_argument("--sample_limit",  type=int, default=None,
                        help="每个方法最多处理多少条（None=全量）")
    parser.add_argument("--best_method",   default=None,
                        help="指定第三阶段使用的方法（默认从第二阶段自动选取）")
    args = parser.parse_args()

    start_time = datetime.now()
    log(f"实验开始：{start_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    log(f"provider={args.provider}  model={args.model}  top_k={args.top_k}", "INFO")

    ensure_dirs()
    if not check_files():
        sys.exit(1)

    phase_results = {}
    phases = args.phase.strip()

    # ---- 阶段 1 ----
    if phases in ("1", "all"):
        r1 = phase1_small_sample(args)
        phase_results["phase1"] = r1

    # ---- 阶段 2 ----
    if phases in ("2", "all"):
        r2, best = phase2_valid_all_methods(args)
        phase_results["phase2"]     = r2
        phase_results["best_method"] = args.best_method or best
    elif args.best_method:
        phase_results["best_method"] = args.best_method
    else:
        phase_results["best_method"] = "full_rag"

    # ---- 阶段 3 ----
    if phases in ("3", "all"):
        r3 = phase3_test_best(args, phase_results["best_method"])
        phase_results["phase3"] = r3

    # ---- 阶段 4 ----
    if phases in ("4", "all"):
        r4 = phase4_subset_analysis(args)
        phase_results["phase4"] = r4

    # ---- 最终汇总 ----
    print_final_summary(phase_results)
    save_phase_summary(phase_results, "full_experiment_summary.json")

    elapsed = (datetime.now() - start_time).seconds
    log(f"实验完成，总耗时 {elapsed//60} 分 {elapsed%60} 秒", "OK")


# 兼容 Python 3.8 的 Optional
from typing import Optional

if __name__ == "__main__":
    main()