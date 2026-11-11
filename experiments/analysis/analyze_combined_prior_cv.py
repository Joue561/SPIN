# -*- coding: utf-8 -*-
"""
analyze_combined_prior_cv.py
与 analyze_combined_prior.py 相同的推断（相对 + 绝对知识），只改 λ 的选择方式：
在 valid 内部按会话树做 K 折交叉验证。

  原协议：用全部 valid 估计混淆矩阵 E，再在同一份 valid 上选 λ。
          E 和 λ 来自同一批样本，无法反映“换一批目标 / 会话后 LLM 错误模式改变”，
          在错误模式随目标变化较大的基础模型（如微调模型）上会选出过大的 λ。
  CV 协议：把 valid 的会话树分成 K 折；每折用其余 K−1 折估计 E，在本折上推断；
          K 折预测合并后计算话题平均 MacroF1，选使其最高的 λ（同分取 λ 之和更小的）。
          最终仍用全部 valid 估计 E，用选出的 λ 推断 test。test 只用于报告。
          ZS-CSD 中一棵树对应一个目标，按树分折即按目标分折（模拟零样本下的目标迁移）。

同时输出原协议的选择，便于对比；三种用法（只用相对 / 只用绝对 / 组合）都用 CV 选择。
输出 pred_combined_cv.jsonl，字段与 analyze_combined_prior.py 的 pred_combined.jsonl 相同。

用法（make_rag_data 目录下）：
python -u spin/experiments/analysis/analyze_combined_prior_cv.py \
  --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --pred zscsd_exp/outputs/sitpcl/seed1234/predictions_sitpcl_test.jsonl \
  --valid_pred zscsd_exp/outputs/sitpcl/seed1234/predictions_sitpcl_valid.jsonl \
  --output_dir zscsd_exp/outputs/sitpcl/seed1234/combined_prior_cv
"""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

_TOOLS = (Path(__file__).resolve().parent / "../../spin_core").resolve()
_s = importlib.util.spec_from_file_location("cp", _TOOLS / "analyze_combined_prior.py")
cp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cp)
cd, kc, kbm, dd = cp.cd, cp.kc, cp.kbm, cp.dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--kb", default="full", choices=list(cd.KB_KEYS))
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2,3,4,6")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="会话树分折的随机种子")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--output_dir", default="combined_prior_cv")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, _ = kc.load_pred_conf(a.pred, a.pred_field, a.conf_field)
    vbase, vconf = kc.load_pred_conf(a.valid_pred, a.pred_field, a.conf_field)
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    ms = {t: kbm.PatternKB([n for n in train if n["topic"] == t], cd.KB_KEYS[a.kb], a.min_count) for t in topics}
    one = kc.make_binner(None)

    def est_E(nds):
        return {t: kc.confusion([n for n in nds if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}

    # 按会话树分折（树 = (话题, 根节点)）
    roots = sorted({(n["topic"], n["root"]) for n in vev})
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(roots))
    fold_of = {roots[i]: j % a.folds for j, i in enumerate(perm)}
    folds = [[n for n in vev if fold_of[(n["topic"], n["root"])] == f] for f in range(a.folds)]
    Ef = [est_E([n for n in vev if fold_of[(n["topic"], n["root"])] != f]) for f in range(a.folds)]
    E_all = est_E(vev)
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)}（{len(roots)} 棵树，{a.folds} 折，"
          f"每折 {', '.join(str(len(f)) for f in folds)} 个节点）| KB:{a.kb}")

    grid_cv, grid_in = {}, {}
    for c in itertools.product(lams, lams):
        lab = {}
        for f in range(a.folds):
            E = Ef[f]
            lab.update(cp.forward(folds[f], vbase, ms, lambda n, E=E: E[n["topic"]], *c))
        grid_cv[c] = cp.topic_avg(vev, lab, topics)
        grid_in[c] = cp.topic_avg(vev, cp.forward(vev, vbase, ms, lambda n: E_all[n["topic"]], *c), topics)

    def pick(gr, cands):
        return max(cands, key=lambda c: (gr[c], -(c[0] + c[1])))

    variants = {"只用相对知识": lambda c: c[1] == 0, "只用绝对知识": lambda c: c[0] == 0, "相对 + 绝对（组合）": lambda c: True}
    g = [n["y"] for n in ev]
    keys = [(n["topic"], n["node_id"]) for n in ev]
    emis = lambda n: E_all[n["topic"]]
    mf, fa, _ = cd.scores_of(g, [base[k] for k in keys])
    print(f"\n【test 结果】CV 协议选 λ（括号内为原协议的选择及其 test 结果，便于对比）")
    print(f"  {'基础预测':20s} 合并MacroF1 {mf:.4f}  合并F_avg {fa:.4f}  话题平均MacroF1 {cp.topic_avg(ev, {k: base[k] for k in keys}, topics):.4f}")
    preds = {}
    for name, ok in variants.items():
        cands = [c for c in grid_cv if ok(c)]
        c_cv, c_in = pick(grid_cv, cands), pick(grid_in, cands)
        lab = cp.forward(ev, base, ms, emis, *c_cv)
        lab_in = cp.forward(ev, base, ms, emis, *c_in)
        m1, f1, _ = cd.scores_of(g, [lab[k] for k in keys])
        m0 = cd.scores_of(g, [lab_in[k] for k in keys])[0]
        fx = sum(lab[k] == n["y"] != base[k] for k, n in zip(keys, ev))
        br = sum(base[k] == n["y"] != lab[k] for k, n in zip(keys, ev))
        print(f"  {name:20s} 合并MacroF1 {m1:.4f}  合并F_avg {f1:.4f}  话题平均MacroF1 {cp.topic_avg(ev, lab, topics):.4f}  "
              f"改对 {fx} 改错 {br}  λ={c_cv}（CV {grid_cv[c_cv]:.4f}）  [原协议 λ={c_in} → test {m0:.4f}]")
        preds[name] = lab

    print("\n【valid 上 CV 分数 vs 原协议分数】（同一 λ；两者差距大说明原协议在 valid 上过拟合）")
    for c in sorted({pick(grid_cv, list(grid_cv)), pick(grid_in, list(grid_in)), (0.0, 0.0)}):
        print(f"  λ={c}: CV {grid_cv[c]:.4f}   原协议 {grid_in[c]:.4f}")

    comb = preds["相对 + 绝对（组合）"]
    with open(out / "pred_combined_cv.jsonl", "w", encoding="utf-8") as f:
        for k, n in zip(keys, ev):
            f.write(json.dumps({"topic": k[0], "node_id": k[1], "depth": n["depth"], "stance_label": n["y"],
                                "predicted_stance": comb[k], "base_prediction": base[k],
                                "relative_only_prediction": preds["只用相对知识"][k],
                                "absolute_only_prediction": preds["只用绝对知识"][k]}, ensure_ascii=False) + "\n")
    print(f"\n已保存 {out / 'pred_combined_cv.jsonl'}")


if __name__ == "__main__":
    main()