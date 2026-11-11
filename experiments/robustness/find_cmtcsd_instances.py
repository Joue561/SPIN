# -*- coding: utf-8 -*-
"""
find_cmtcsd_instances.py —— 找出论文 C-MTCSD 主实验所用的实例文件。
对每个候选实例文件，用 GPT-4o-mini kNN 的验证/测试预测跑一遍与 analyze_combined_prior.py 相同的 SPIN（λ 网格 0–6），
打印匹配到的节点数和测试集合并 Macro-F1。论文数字：合并 Macro-F1 72.51（base 68.25）。

  python -u spin/experiments/robustness/find_cmtcsd_instances.py --tools_dir spin/spin_core
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spin_robust_common as C  # noqa: E402

CANDS = ["causal_fix/reannotated/cmtscd_semantic_instances_llm.jsonl",
         "processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl",
         "processed_cmtscd_semantic_llm_causal/cmtscd_semantic_instances_llm.jsonl",
         "processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl",
         "processed_cmtscd_semantic_llm_causal_v3/cmtscd_semantic_instances_llm.jsonl",
         "processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools_dir", default="spin/spin_core")
    ap.add_argument("--valid_pred", default="baseline_outputs/predictions_knn_fewshot_valid.jsonl")
    ap.add_argument("--test_pred", default="baseline_outputs/predictions_knn_fewshot_test.jsonl")
    a = ap.parse_args()
    acp = C.load_acp(a.tools_dir)
    extra = [p for p in glob.glob("**/*cmtscd*semantic_instances*merged*.jsonl", recursive=True)
             + glob.glob("causal_fix/**/*.jsonl", recursive=True) if "MT-CSD" not in p]
    for path in dict.fromkeys(CANDS + sorted(extra)):
        if not Path(path).exists():
            continue
        try:
            nodes, _ = acp.dd.load_instances(path, "parent_stance_label")
            base, _ = acp.kc.load_pred_conf(a.test_pred, "predicted_stance", "confidence")
            n_test = sum(1 for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"])
            unk = sum(1 for n in nodes.values() if n["sp"] == "unknown")
            print(f"\n{path}\n  节点 {len(nodes)}  可评测测试节点 {n_test}（论文 3,498）  sp=unknown {unk}", flush=True)
            if n_test == 0:
                continue
            run = C.Run(acp, path, a.valid_pred, a.test_pred, name=path)
            b = run.evaluate({k: run.base[k] for k in run.keys})["macro_f1"]
            s = run.evaluate(run.spin)["macro_f1"]
            print(f"  base {100*b:.2f}  SPIN {100*s:.2f}  λ=({run.lr:g},{run.la:g})   ← 论文 base 68.25 / SPIN 72.51")
        except Exception as e:  # noqa
            print(f"  出错：{e}")


if __name__ == "__main__":
    main()