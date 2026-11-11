# -*- coding: utf-8 -*-
"""
analyze_kb_fewshot.py
离线检验（不调用 API）：跨话题语义传导知识库 + 少样本基准率校准。

背景：各话题的立场基准率（尤其 none 比例 0.04 ~ 0.47）由标注尺度决定，无法从 LLM 的无标注预测中估计；
用真实基准率校准后，跨话题知识库显著有效。问题：实际中需要多少目标话题的标注才能把基准率估准？

做法：从目标话题的 train（不碰 test）随机抽 k 条带标签样本，只用它们的标签分布：
  整体水平 = k 条样本的标签分布；按深度的形状借用源话题（其他话题 train）
  π_t(深度) ∝ π_s(深度) × p_k / π_s(整体)
k = 0（不校准）, 10, 20, 50, 100, 200, all（目标话题全部 train 的按深度分布），每个 k 重复多个随机种子。

协议：严格零样本 —— LLM 混淆矩阵只在其他话题上估计，λ 只在其他话题上选（与 analyze_kb_calibration.py 相同）。
对比：
  跨话题 KB:传导类型+深度 + 少样本校准         ← 迁移方法
  仅校准（无知识库） + 少样本校准              ← 对照：收益是否来自知识库
  参照：话题内 KB（用目标话题全部 train 构建，同一协议）、真实 test 基准率校准（上界）

用法：
python -u spin/experiments/negative/analyze_kb_fewshot.py \
  --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \
  --pred dynamic_semantic=rag_prediction_outputs/predictions_dynamic_semantic_test.jsonl \
  --output_dir kb_fewshot_analysis
"""
import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("cal", (Path(__file__).resolve().parent / "../../spin_core/analyze_kb_calibration.py").resolve())
cal = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cal)
kbm, dd = cal.kbm, cal.dd
LABELS = dd.LABELS


def fewshot_pi(src_depth, src_overall, labels):
    c = Counter(labels)
    p = np.array([c[l] + 1 for l in LABELS], float)
    p /= p.sum()
    ratio = p / src_overall
    out = {}
    for b, v in src_depth.items():
        x = v * ratio
        out[b] = x / x.sum()
    return out


def evaluate(ev, base, models, lams, topics, emis, mode="pred"):
    grid = {l: cal.smooth_e(ev, base, models, l, emis, mode) for l in lams}
    lab, desc = cal.select(ev, grid, lams, topics, "loto")
    return lab, desc


def metrics(ev, lab, base):
    r = dd.summarize(ev, lab, base, {})
    keep = ["话题平均MacroF1", "话题平均F_avg", "合并MacroF1", "改对", "改错"] + \
        [k for k in r if k.startswith("depth") and k.endswith("_F_avg")]
    return {k: r[k] for k in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--min_depth_n", type=int, default=80)
    ap.add_argument("--shots", default="10,20,50,100,200,all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output_dir", default="kb_fewshot_analysis")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    preds = {s.split("=", 1)[0]: dd.load_pred(s.split("=", 1)[1], a.pred_field) for s in a.pred}
    base = preds[a.base or next(iter(preds))]
    ev = [n for k, n in nodes.items() if k in base and n["split"] == a.eval_split and n["y"]]
    print(f"train {len(train)} | {a.eval_split} 可评估 {len(ev)}")

    def confusion(sub):
        cm = np.ones((3, 3))
        for n in sub:
            cm[dd.LID[n["y"]], dd.LID[base[(n["topic"], n["node_id"])]]] += 1
        return cm / cm.sum(1, keepdims=True)
    C_loto = {t: confusion([n for n in ev if n["topic"] != t]) for t in topics}
    emis = lambda n: C_loto[n["topic"]]

    src_depth, src_all, tgt_train, pi_true = {}, {}, {}, {}
    for t in topics:
        src_depth[t], src_all[t] = cal.dist_by_depth([n for n in train if n["topic"] != t], lambda n: n["y"], a.min_depth_n)
        tgt_train[t] = [n for n in train if n["topic"] == t]
        pi_true[t], _ = cal.dist_by_depth([n for n in ev if n["topic"] == t], lambda n: n["y"], a.min_depth_n)
    cross_kb = {t: kbm.PatternKB([n for n in train if n["topic"] != t], kbm.k_sp_depth, a.min_count) for t in topics}

    rows = []

    def add(method, shots, seed, models):
        lab, desc = evaluate(ev, base, models, lams, topics, emis)
        m = metrics(ev, lab, base)
        rows.append({"method": method, "shots": shots, "seed": seed, "lambda": desc, **m})
        safe = f"{method}_{shots}_{seed}".replace(":", "_").replace("+", "_").replace("(", "_").replace(")", "")
        with open(out / f"pred_{safe}.jsonl", "w", encoding="utf-8") as f:
            for n in ev:
                key = (n["topic"], n["node_id"])
                f.write(json.dumps({"topic": n["topic"], "node_id": n["node_id"], "depth": n["depth"],
                                    "stance_label": n["y"], "predicted_stance": lab[key],
                                    "base_prediction": base[key]}, ensure_ascii=False) + "\n")

    # 参照
    rows.append({"method": "dynamic_semantic(原始)", "shots": "-", "seed": "-", "lambda": "-", **metrics(ev, base, base)})
    add("跨话题KB(不校准)", "0", 0, cross_kb)
    in_kb = {t: kbm.PatternKB(tgt_train[t], kbm.k_sp_depth, a.min_count) for t in topics}
    add("话题内KB(参照)", "全部train", 0, in_kb)
    add("跨话题KB+真实test基准率(上界)", "上界", 0, {t: cal.Calibrated(cross_kb[t], pi_true[t], src_depth[t]) for t in topics})
    add("仅校准+真实test基准率(上界)", "上界", 0, {t: cal.PriorOnly(pi_true[t]) for t in topics})
    print("  参照组完成")

    # 少样本
    for shots in a.shots.split(","):
        seeds = [0] if shots == "all" else range(a.seeds)
        for seed in seeds:
            pi = {}
            for t in topics:
                if shots == "all":
                    pi[t], _ = cal.dist_by_depth(tgt_train[t], lambda n: n["y"], a.min_depth_n)
                else:
                    rng = random.Random(f"{seed}|{t}")
                    sample = rng.sample(tgt_train[t], min(int(shots), len(tgt_train[t])))
                    pi[t] = fewshot_pi(src_depth[t], src_all[t], [n["y"] for n in sample])
            add("跨话题KB+少样本校准", shots, seed, {t: cal.Calibrated(cross_kb[t], pi[t], src_depth[t]) for t in topics})
            add("仅校准+少样本校准", shots, seed, {t: cal.PriorOnly(pi[t]) for t in topics})
        print(f"  k={shots} 完成")
    dd.write_csv(out / "kb_fewshot_all_runs.csv", rows)

    # 汇总（多种子取均值 ± 标准差）
    print(f"\n{'方法':30s} {'k':>8s} {'种子数':>4s} {'话题平均MacroF1':>18s} {'话题平均F_avg':>18s} {'合并MacroF1':>18s} {'改对':>6s} {'改错':>6s} | depth4+ F_avg")
    summ = []
    order = []
    for r in rows:
        k = (r["method"], r["shots"])
        if k not in order:
            order.append(k)
    for meth, shots in order:
        g = [r for r in rows if r["method"] == meth and r["shots"] == shots]
        def ms(key):
            v = np.array([r[key] for r in g], float)
            return v.mean(), v.std()
        s = {"method": meth, "shots": shots, "n_seeds": len(g)}
        for key in ("话题平均MacroF1", "话题平均F_avg", "合并MacroF1", "改对", "改错", "depth4+_F_avg"):
            if key in g[0]:
                s[key + "_mean"], s[key + "_std"] = ms(key)
        summ.append(s)
        print(f"{meth[:30]:30s} {shots:>8s} {len(g):4d} "
              f"{s['话题平均MacroF1_mean']:10.4f}±{s['话题平均MacroF1_std']:.4f} "
              f"{s['话题平均F_avg_mean']:10.4f}±{s['话题平均F_avg_std']:.4f} "
              f"{s['合并MacroF1_mean']:10.4f}±{s['合并MacroF1_std']:.4f} "
              f"{s['改对_mean']:6.0f} {s['改错_mean']:6.0f} | {s.get('depth4+_F_avg_mean', float('nan')):.4f}")
    dd.write_csv(out / "kb_fewshot_summary.csv", summ)
    print(f"\n结果已保存到 {out.resolve()}")
    print("显著性检验示例（k=50 种子 0，跨话题KB vs 原始）：python spin/experiments/main/paired_significance.py "
          f"--a {out}/pred_跨话题KB_少样本校准_50_0.jsonl --b {out}/pred_跨话题KB_少样本校准_50_0.jsonl "
          "--pred_field predicted_stance --pred_field_b base_prediction --id_field topic node_id --gold_field stance_label --group_field topic")


if __name__ == "__main__":
    main()