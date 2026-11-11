# -*- coding: utf-8 -*-
"""
diagnose_transfer.py
诊断（不调用 API，结果不作为论文主结果）：知识库推断在某个基础模型上没有提升时，
区分两种原因——
  A. valid → test 迁移失败：从 valid 估计的混淆矩阵 E 和 λ 不适用于 test
  B. 信息重复：基础模型已经包含了知识库的信息，即使 E、λ 都“作弊”地在 test 上估计也涨不了

输出：
  1. E_valid 与 E_test（行归一化的 P(ŷ|y)）及两者每行的 JS 散度
  2. 三种设定下 test 上的最好结果（组合 λ 网格）：
       协议      ：E 来自 valid，λ 在 valid 上选（= 正式结果）
       λ 上界    ：E 来自 valid，λ 在 test 上选（只把 λ 换成作弊）
       完全上界  ：E 来自 test， λ 在 test 上选（E 和 λ 都作弊）
     若“完全上界”也几乎没有提升 → 原因 B；若上界明显提升而协议不提升 → 原因 A。

用法（make_rag_data 目录下）：
python spin/experiments/analysis/diagnose_transfer.py \
  --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --pred zscsd_exp/outputs/sitpcl/seed1234/predictions_sitpcl_test.jsonl \
  --valid_pred zscsd_exp/outputs/sitpcl/seed1234/predictions_sitpcl_valid.jsonl --name SITPCL
"""
import argparse
import importlib.util
import itertools
from pathlib import Path

import numpy as np

_TOOLS = (Path(__file__).resolve().parent / "../../spin_core").resolve()
_s = importlib.util.spec_from_file_location("cp", _TOOLS / "analyze_combined_prior.py")
cp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cp)
cd, kc, kbm, dd = cp.cd, cp.kc, cp.kbm, cp.dd
LABELS = dd.LABELS


def js(p, q):
    p, q = np.asarray(p) + 1e-12, np.asarray(q) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2,3,4,6")
    ap.add_argument("--min_count", type=int, default=20)
    a = ap.parse_args()
    lams = [float(x) for x in a.lambdas.split(",")]

    nodes, _ = dd.load_instances(a.instances, "parent_stance_label")
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, conf = kc.load_pred_conf(a.pred, "predicted_stance", "confidence")
    vbase, vconf = kc.load_pred_conf(a.valid_pred, "predicted_stance", "confidence")
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    ms = {t: kbm.PatternKB([n for n in train if n["topic"] == t], cd.KB_KEYS["full"], a.min_count) for t in topics}
    one = kc.make_binner(None)
    Ev = {t: kc.confusion([n for n in vev if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}
    Et = {t: kc.confusion([n for n in ev if n["topic"] == t], base, conf, one, 1)[0] for t in topics}

    print(f"===== {a.name}   test {len(ev)} | valid {len(vev)}")
    for t in topics:
        print(f"【混淆矩阵 P(预测 | 真实)】话题 {t}；行 = 真实 (favor / against / none)，列 = 预测")
        for i, y in enumerate(LABELS):
            rv = Ev[t][i] / Ev[t][i].sum(); rt = Et[t][i] / Et[t][i].sum()
            print(f"  {y:8s} valid [" + " ".join(f"{x:.2f}" for x in rv) + "]   test [" +
                  " ".join(f"{x:.2f}" for x in rt) + f"]   JS {js(rv, rt):.3f}")

    keys = [(n["topic"], n["node_id"]) for n in ev]
    g = [n["y"] for n in ev]
    # 与下面三种设定同一口径（话题平均 MacroF1）；单话题数据集（ZS-CSD）时与合并口径相同
    base_f1 = cp.topic_avg(ev, {k: base[k] for k in keys}, topics)

    def grid(ev_, base_, E):
        emis = lambda n: E[n["topic"]]
        return {c: cp.topic_avg(ev_, cp.forward(ev_, base_, ms, emis, *c), topics) for c in itertools.product(lams, lams)}

    gv = grid(vev, vbase, Ev)                 # valid 上（E 来自 valid）
    gt_ev = grid(ev, base, Ev)                # test 上（E 来自 valid）
    gt_et = grid(ev, base, Et)                # test 上（E 来自 test）
    pick = lambda gr: max(gr, key=lambda c: (gr[c], -sum(c)))
    c1, c2, c3 = pick(gv), pick(gt_ev), pick(gt_et)
    print(f"\n【test 话题平均 MacroF1】基础 {100 * base_f1:.2f}")
    print(f"  协议（E←valid，λ←valid） {100 * gt_ev[c1]:.2f}  Δ {100 * (gt_ev[c1] - base_f1):+.2f}  λ={c1}")
    print(f"  λ 上界（E←valid，λ←test） {100 * gt_ev[c2]:.2f}  Δ {100 * (gt_ev[c2] - base_f1):+.2f}  λ={c2}")
    print(f"  完全上界（E←test，λ←test）{100 * gt_et[c3]:.2f}  Δ {100 * (gt_et[c3] - base_f1):+.2f}  λ={c3}")
    print(f"  valid 上：基础 {100 * gv[(0.0, 0.0)]:.2f} → 选出的 λ {100 * gv[c1]:.2f}  Δ {100 * (gv[c1] - gv[(0.0, 0.0)]):+.2f}")


if __name__ == "__main__":
    main()