# -*- coding: utf-8 -*-
"""
channel_shift_diagnosis.py —— 问题一 / 实验 A：验证集与测试集之间的三类分布变化，以及它们对 SPIN 的实际影响。

分开度量三种变化（不能把“预测比例变了”直接当成“E 失效”）：
  1. Label Distribution Shift         π_val(y) vs π_test(y)                               —— 需要测试标签（离线诊断）
  2. Prediction Marginal Shift        P_val(ŷ) vs P_test(ŷ)                               —— 不需要标签
     其中“标签变化可以解释的部分”：  π_test · Ê_val（若只有标签变化、E 不变，测试预测分布应为此值）
     残差 TV(P_test(ŷ), π_test·Ê_val) 才可能来自 E 的变化
  3. Class-Conditional Error Shift    Σ_y π_test(y) · JS(Ê_val[y,:], Ê_test[y,:])        —— 需要测试标签（离线诊断）
每个统计量都给出“按对话树随机重划分验证/测试”的置换检验 p 值：即在同样样本量下，单纯抽样噪声会产生多大的差异。
SPIN 的 E 按话题估计，所以统计量先在每个话题内计算，再按测试节点数加权平均；置换也在话题内进行。

影响（全部为离线诊断，不作为方法性能）：
  official        E_val + λ(val)                       —— 论文中的 SPIN
  oracle_E        E_test + λ(val)                      —— 只把 E 换成测试集上的“真”信道
  oracle_lambda   E_val + λ(test)                      —— 只把 λ 换成测试集上最优
  oracle_both     E_test + λ(test)                     —— 两者都换（上界）
official 与 oracle_E 的差距 = “E 从验证集迁移到测试集”造成的损失。

用法：
  python -u channel_shift_diagnosis.py --tools_dir cross_topic_exp/tools --manifest runs.json \
      --output_dir robustness/channel --n_perm 1000
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import spin_robust_common as C


def smoothed_rows(cnt, a=0.5):
    return C.rownorm(cnt + a)


def topic_stats(gold_v, pred_v, gold_t, pred_t, labels):
    K = len(labels)
    cv = C.confusion_counts(gold_v, pred_v, labels)
    ct = C.confusion_counts(gold_t, pred_t, labels)
    pi_v, pi_t = cv.sum(1), ct.sum(1)
    pv, pt = cv.sum(0), ct.sum(0)
    Ev, Et = smoothed_rows(cv), smoothed_rows(ct)
    pi_t_n = pi_t / max(pi_t.sum(), 1)
    expected_pt = pi_t_n @ Ev
    row_js = np.array([C.js(Ev[y], Et[y]) for y in range(K)])
    return {
        "label_tv": C.tv(pi_v, pi_t),
        "pred_tv": C.tv(pv, pt),
        "pred_tv_given_label_shift": C.tv(pt, expected_pt),
        "ccs_js": float((pi_t_n * row_js).sum()),
        "row_js": row_js.tolist(),
        "pi_val": (pi_v / max(pi_v.sum(), 1)).tolist(), "pi_test": pi_t_n.tolist(),
        "pred_val": (pv / max(pv.sum(), 1)).tolist(), "pred_test": (pt / max(pt.sum(), 1)).tolist(),
        "E_val_counts": cv.tolist(), "E_test_counts": ct.tolist(), "n_val": int(cv.sum()), "n_test": int(ct.sum()),
    }


STATS = ("label_tv", "pred_tv", "pred_tv_given_label_shift", "ccs_js")


def aggregate(per_topic):
    if not per_topic:
        return {k: float("nan") for k in STATS}
    w = np.array([s["n_test"] for s in per_topic.values()], float)
    w = w / w.sum()
    return {k: float(sum(wi * s[k] for wi, s in zip(w, per_topic.values()))) for k in STATS}


def all_stats(run, assign_v, assign_t):
    """assign_*: {topic: list of nodes}，节点上的预测从 vbase/base 合并字典读取。"""
    per = {}
    for t in run.topics:
        nv, nt = assign_v.get(t, []), assign_t.get(t, [])
        if not nv or not nt:
            continue
        pv = [run._pred(n) for n in nv]
        pt = [run._pred(n) for n in nt]
        per[t] = topic_stats([n["y"] for n in nv], pv, [n["y"] for n in nt], pt, run.LABELS)
    return per


def permutation_null(run, n_perm, seed):
    """话题内按对话树把验证+测试节点随机重新划分，保持各自节点数近似不变。"""
    rng = np.random.default_rng(seed)
    pools = {}
    for t in run.topics:
        nv = [n for n in run.vev if n["topic"] == t]
        nt = [n for n in run.ev if n["topic"] == t]
        if nv and nt:
            pools[t] = (list(C.tree_groups(nv + nt).values()), len(nv))
    null = {k: [] for k in STATS}
    for _ in range(n_perm):
        av, at = {}, {}
        for t, (trees, nval) in pools.items():
            order = rng.permutation(len(trees))
            v, tt, cnt = [], [], 0
            for i in order:
                if cnt < nval:
                    v.extend(trees[i]); cnt += len(trees[i])
                else:
                    tt.extend(trees[i])
            av[t], at[t] = v, tt
        agg = aggregate(all_stats(run, av, at))
        for k in STATS:
            null[k].append(agg[k])
    return {k: np.array(v) for k, v in null.items()}


def confidence_summary(run):
    out = {}
    for split, nodes, base, conf in (("valid", run.vev, run.vbase, run.vconf), ("test", run.ev, run.base, run.conf)):
        for l in run.LABELS:
            vals = []
            for n in nodes:
                k = (n["topic"], n["node_id"])
                if base.get(k) == l:
                    c = conf.get(k) if isinstance(conf, dict) else None
                    try:
                        vals.append(float(c))
                    except (TypeError, ValueError):
                        pass
            out[f"{split}_{l}"] = {"n": len(vals), "mean": float(np.mean(vals)) if vals else None,
                                   "q25": float(np.percentile(vals, 25)) if vals else None,
                                   "q75": float(np.percentile(vals, 75)) if vals else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", default="robustness/channel")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_oracle", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    acp = C.load_acp(a.tools_dir)
    rows = []
    for spec in C.load_manifest(a.manifest):
        print(f"\n===== {spec['name']} ({spec['dataset']})", flush=True)
        run = C.make_run(acp, spec)
        merged = {**run.vbase, **run.base}
        run._pred = lambda n, m=merged: m.get((n["topic"], n["node_id"]))
        print(f"  valid {len(run.vev)} | test {len(run.ev)} | λ=({run.lr:g},{run.la:g})")

        per = all_stats(run, {t: [n for n in run.vev if n["topic"] == t] for t in run.topics},
                        {t: [n for n in run.ev if n["topic"] == t] for t in run.topics})
        both = [t for t in run.topics if any(n["topic"] == t for n in run.vev) and any(n["topic"] == t for n in run.ev)]
        if not per:
            print("  !! 没有任何话题同时含有验证和测试节点，无法比较同一话题内的 E；跳过分布诊断（E 按话题估计时，"
                  "测试话题的 E 只能来自平滑先验）")
        elif len(both) < len([t for t in run.topics if any(n["topic"] == t for n in run.ev)]):
            print(f"  !! 只有 {len(both)} 个话题同时含验证和测试节点，诊断只覆盖这些话题")
        obs = aggregate(per)
        null = permutation_null(run, a.n_perm, a.seed)
        pvals = {k: float((1 + (null[k] >= obs[k] - 1e-15).sum()) / (1 + a.n_perm)) for k in STATS}
        null_mean = {k: float(null[k].mean()) for k in STATS}
        null_q95 = {k: float(np.percentile(null[k], 95)) for k in STATS}

        base_lab = {k: run.base[k] for k in run.keys}
        res = {"base": run.evaluate(base_lab), "official": run.evaluate(run.spin, ref=base_lab)}
        if not a.skip_oracle:
            E_test = {t: run.confusion(run.ev, t, valid=False) for t in run.topics}
            res["oracle_E"] = run.evaluate(run.forward(run.ev, run.base, E_test, run.kb), ref=base_lab)
            lr_t, la_t, _ = run.select_lambda(run.E_val, run.kb, nodes=run.ev, base=run.base)
            res["oracle_lambda"] = run.evaluate(run.forward(run.ev, run.base, run.E_val, run.kb, lr_t, la_t), ref=base_lab)
            lr_b, la_b, _ = run.select_lambda(E_test, run.kb, nodes=run.ev, base=run.base)
            res["oracle_both"] = run.evaluate(run.forward(run.ev, run.base, E_test, run.kb, lr_b, la_b), ref=base_lab)
            res["oracle_lambda"]["lambda"] = [lr_t, la_t]
            res["oracle_both"]["lambda"] = [lr_b, la_b]
            # 正式 E 与测试集“真”信道的距离（按 SPIN 实际使用的、经 kc.confusion 平滑后的 E）
            w = {t: sum(1 for n in run.ev if n["topic"] == t) for t in run.topics}
            tot = sum(w.values())
            res["official_E_vs_test_E_js"] = float(sum(
                w[t] / tot * np.mean([C.js(run.E_val[t][y], E_test[t][y]) for y in range(len(run.LABELS))])
                for t in run.topics if w[t]))

        report = {"name": spec["name"], "dataset": spec["dataset"], "lambda": [run.lr, run.la],
                  "observed": obs, "perm_p": pvals, "null_mean": null_mean, "null_q95": null_q95,
                  "per_topic": per, "confidence": confidence_summary(run), "impact": res, "labels": run.LABELS}
        C.dump_json(out / f"channel_shift_{spec['name']}.json", report)

        b = res["base"]["macro_f1"]
        row = {"name": spec["name"], "dataset": spec["dataset"], "n_valid": len(run.vev), "n_test": len(run.ev),
               **{f"{k}": round(obs[k], 5) for k in STATS}, **{f"p_{k}": round(pvals[k], 4) for k in STATS},
               **{f"null95_{k}": round(null_q95[k], 5) for k in STATS},
               "base_mf1": round(100 * b, 2), "spin_mf1": round(100 * res["official"]["macro_f1"], 2),
               "gain": round(100 * (res["official"]["macro_f1"] - b), 2)}
        if not a.skip_oracle:
            for k in ("oracle_E", "oracle_lambda", "oracle_both"):
                row[f"{k}_gain"] = round(100 * (res[k]["macro_f1"] - b), 2)
            og = res["oracle_E"]["macro_f1"] - b
            row["frac_of_oracleE_gain"] = round((res["official"]["macro_f1"] - b) / og, 3) if og > 1e-9 else None
            row["E_val_vs_test_js"] = round(res["official_E_vs_test_E_js"], 5)
        rows.append(row)
        print("  观测值 / 置换 95% 分位 / p：")
        for k in STATS:
            print(f"    {k:28s} {obs[k]:.4f}  / {null_q95[k]:.4f}  / p={pvals[k]:.3f}")
        print(f"  MacroF1 base {row['base_mf1']}  SPIN {row['spin_mf1']}  (Δ {row['gain']:+.2f})", end="")
        if not a.skip_oracle:
            print(f"   oracle_E Δ {row['oracle_E_gain']:+.2f}  oracle_λ Δ {row['oracle_lambda_gain']:+.2f}  "
                  f"oracle_both Δ {row['oracle_both_gain']:+.2f}")
        else:
            print()

    with open(out / "channel_shift_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\n已保存 {out/'channel_shift_summary.csv'} 及每个 run 的 JSON。")
    print("判读：只有 p_ccs_js 小（class-conditional 误差变化超过抽样噪声）且 official 明显低于 oracle_E，"
          "才能说 E 的验证-测试漂移损害了 SPIN；pred_tv 显著但 pred_tv_given_label_shift 不显著，说明预测分布变化主要由标签分布变化解释。")


if __name__ == "__main__":
    main()