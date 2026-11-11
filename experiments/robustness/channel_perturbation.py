# -*- coding: utf-8 -*-
"""
channel_perturbation.py —— 问题一 / 实验 B（+ 可选实验 C）：观测信道 E 的扰动敏感性。

只检验“E 估计不准”时 SPIN 的敏感性，不等同于真实的验证-测试漂移（真实漂移见 channel_shift_diagnosis.py）。
所有扰动都作用在 SPIN 实际使用的 E_val 上（kc.confusion 的输出，逐话题），保持每行归一化；λ 默认固定为官方值，
以隔离 E 本身的影响（--reselect 时在验证集上用扰动后的 E 重新选 λ，模拟完整协议）。

扰动族：
  uniform    E_α = (1-α)E + αU                               α ∈ {0, .1, .2, .3, .5}   —— 信道变“更不可信”
  offdiag    非对角元素整体乘 s，对角元素补足到行和为 1      s ∈ {0.5, .75, 1, 1.5, 2}   —— 保留错误方向，只改错误强度
  toward_none 每行把 δ 比例的质量从其余列移到 NONE 列（δ<0 为移出）   δ ∈ {-.2,-.1,0,.1,.2}  —— 针对 LLM 的主要偏差方向
  bootstrap  按对话树对验证集有放回重采样后重新估计 E（B 次）                          —— 现实的估计噪声
可选实验 C（--em_label_shift，探索性，不属于 SPIN）：
  在无标签测试预测上用 EM（Saerens et al., 2002）估计测试类别先验 π̂_test（E 固定为 E_val），
  报告 π̂_test 与真实 π_test 的距离；并给出把绝对知识按 π̂_test/π_train 重加权后的结果。
  注意：完整 E 的无监督 EM 不可辨识（每个话题 9 个参数，只有 3 个预测边际），因此只做标签先验适应。

用法：
  python -u channel_perturbation.py --tools_dir cross_topic_exp/tools --manifest runs.json \
      --output_dir robustness/perturb --n_boot 200 [--reselect] [--em_label_shift]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import spin_robust_common as C


def perturb_uniform(E, a):
    K = E.shape[0]
    return C.rownorm((1 - a) * E + a * np.ones((K, K)) / K)


def perturb_offdiag(E, s):
    K = E.shape[0]
    off = E * (1 - np.eye(K)) * s
    rs = off.sum(1, keepdims=True)
    off = np.where(rs > 0.98, off * 0.98 / np.maximum(rs, 1e-12), off)
    return C.rownorm(off + np.diag(1 - off.sum(1)))


def perturb_toward_none(E, d, none_idx):
    E = E.copy()
    K = E.shape[0]
    for y in range(K):
        others = [j for j in range(K) if j != none_idx]
        if d >= 0:
            move = d * E[y, others].sum()
            E[y, others] *= (1 - d)
            E[y, none_idx] += move
        else:
            move = -d * E[y, none_idx]
            E[y, none_idx] -= move
            w = E[y, others] / max(E[y, others].sum(), 1e-12)
            E[y, others] += move * w
    return C.rownorm(np.maximum(E, 1e-9))


class AbsReweight:
    """把 kb.direct_proba 乘以类别权重 w 后归一化；其余方法原样转发。统计调用次数以确认 forward 确实用到了它。"""

    def __init__(self, kb, w):
        self._kb, self._w, self.calls = kb, np.asarray(w, float), 0

    def direct_proba(self, n):
        self.calls += 1
        p = np.asarray(self._kb.direct_proba(n), float) * self._w
        return p / p.sum()

    def __getattr__(self, k):
        return getattr(self._kb, k)


def em_label_prior(preds, E, pi0, iters=200, tol=1e-8):
    """Saerens et al. (2002) 形式的 EM：只用无标签预测 ŷ 与固定的 E[y,ŷ] 估计测试先验 π。"""
    pi = np.asarray(pi0, float)
    cols = np.array(preds)
    for _ in range(iters):
        lik = E[:, cols].T * pi          # n × K
        r = lik / lik.sum(1, keepdims=True)
        new = r.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", default="robustness/perturb")
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5")
    ap.add_argument("--scales", default="0.5,0.75,1,1.5,2")
    ap.add_argument("--none_shifts", default="-0.2,-0.1,0,0.1,0.2")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--reselect", action="store_true", help="每个扰动后在验证集上重新选 λ（较慢）")
    ap.add_argument("--em_label_shift", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    acp = C.load_acp(a.tools_dir)
    rows, boot_rows = [], []
    for spec in C.load_manifest(a.manifest):
        print(f"\n===== {spec['name']} ({spec['dataset']})", flush=True)
        run = C.make_run(acp, spec)
        base_lab = {k: run.base[k] for k in run.keys}
        b = run.evaluate(base_lab)
        off = run.evaluate(run.spin, ref=base_lab)
        print(f"  base {100*b['macro_f1']:.2f}  SPIN {100*off['macro_f1']:.2f}  λ=({run.lr:g},{run.la:g})")

        def score(Ep, family, level):
            if a.reselect:
                lr, la, vs = run.select_lambda(Ep, run.kb)
            else:
                lr, la, vs = run.lr, run.la, None
            m = run.evaluate(run.forward(run.ev, run.base, Ep, run.kb, lr, la), ref=base_lab)
            r = {"name": spec["name"], "dataset": spec["dataset"], "family": family, "level": level,
                 "lambda_r": lr, "lambda_a": la, "macro_f1": round(100 * m["macro_f1"], 3),
                 "accuracy": round(100 * m["accuracy"], 3), "f_avg": round(100 * m.get("f_avg", np.nan), 3),
                 "gain_vs_base": round(100 * (m["macro_f1"] - b["macro_f1"]), 3),
                 "delta_vs_official": round(100 * (m["macro_f1"] - off["macro_f1"]), 3),
                 "fixed": m["fixed"], "broken": m["broken"],
                 "mean_row_js_to_official": round(float(np.mean([np.mean([C.js(Ep[t][y], run.E_val[t][y])
                                                     for y in range(len(run.LABELS))]) for t in run.topics])), 5)}
            rows.append(r)
            return r

        for al in [float(x) for x in a.alphas.split(",")]:
            r = score({t: perturb_uniform(E, al) for t, E in run.E_val.items()}, "uniform", al)
            print(f"  uniform α={al:<4g} MF1 {r['macro_f1']:.2f}  Δbase {r['gain_vs_base']:+.2f}")
        for s in [float(x) for x in a.scales.split(",")]:
            r = score({t: perturb_offdiag(E, s) for t, E in run.E_val.items()}, "offdiag", s)
            print(f"  offdiag s={s:<4g} MF1 {r['macro_f1']:.2f}  Δbase {r['gain_vs_base']:+.2f}")
        for d in [float(x) for x in a.none_shifts.split(",")]:
            r = score({t: perturb_toward_none(E, d, run.none_idx) for t, E in run.E_val.items()}, "toward_none", d)
            print(f"  toward_none δ={d:<5g} MF1 {r['macro_f1']:.2f}  Δbase {r['gain_vs_base']:+.2f}")

        # bootstrap：按树重采样验证集重新估计 E
        rng = np.random.default_rng(a.seed)
        groups = {t: list(C.tree_groups([n for n in run.vev if n["topic"] == t]).values()) for t in run.topics}
        bs = []
        for i in range(a.n_boot):
            Eb = {}
            for t, trees in groups.items():
                pick = rng.integers(0, len(trees), len(trees))
                Eb[t] = run.confusion([n for j in pick for n in trees[j]], t, valid=True)
            r = score(Eb, "bootstrap", i)
            bs.append(r["macro_f1"])
        bs = np.array(bs)
        boot_rows.append({"name": spec["name"], "dataset": spec["dataset"], "official": round(100 * off["macro_f1"], 3),
                          "base": round(100 * b["macro_f1"], 3), "boot_mean": round(float(bs.mean()), 3),
                          "boot_sd": round(float(bs.std(ddof=1)), 3), "boot_p2.5": round(float(np.percentile(bs, 2.5)), 3),
                          "boot_p97.5": round(float(np.percentile(bs, 97.5)), 3),
                          "frac_boot_above_base": round(float((bs > 100 * b["macro_f1"]).mean()), 3)})
        print(f"  bootstrap E（{a.n_boot} 次）MF1 {bs.mean():.2f}±{bs.std(ddof=1):.2f}；"
              f"高于 base 的比例 {boot_rows[-1]['frac_boot_above_base']:.2f}")

        if a.em_label_shift:
            # π_train 来自训练标签；π_test 由无标签测试预测 + E_val 的 EM 估计
            run._em = {}
            for t in run.topics:
                tr = [n["y"] for n in run.train if n["topic"] == t]
                te = [n for n in run.ev if n["topic"] == t]
                if not tr or not te:          # 没有训练节点（无 π_train）或没有测试节点：该话题不做重加权
                    continue
                pi_tr = np.array([tr.count(l) for l in run.LABELS], float); pi_tr /= pi_tr.sum()
                preds = [run.LID[run.base[(n["topic"], n["node_id"])]] for n in te]
                pi_em = em_label_prior(preds, run.E_val[t], pi_tr)
                g = [n["y"] for n in te]
                pi_true = np.array([g.count(l) for l in run.LABELS], float); pi_true /= pi_true.sum()
                run._em[t] = (pi_tr, pi_em, pi_true)
                print(f"  [EM] {t}: π_train {np.round(pi_tr,3)}  π̂_test {np.round(pi_em,3)}  π_test(真,仅诊断) {np.round(pi_true,3)}"
                      f"  TV(π̂,π)={C.tv(pi_em, pi_true):.3f}  TV(π_train,π)={C.tv(pi_tr, pi_true):.3f}")
            skipped = [t for t in run.topics if t not in run._em and any(n["topic"] == t for n in run.ev)]
            if skipped:
                print(f"  [EM] {len(skipped)} 个有测试节点的话题没有训练节点，未做重加权：{skipped[:5]}")
            kbw = {t: AbsReweight(run.kb[t], (run._em[t][1] / np.maximum(run._em[t][0], 1e-9)) if t in run._em
                                  else np.ones(len(run.LABELS))) for t in run.topics}
            lab = run.forward(run.ev, run.base, run.E_val, kbw)
            if sum(k.calls for k in kbw.values()) == 0:
                print("  [EM] 警告：forward 未调用 kb.direct_proba，无法测试绝对知识重加权；已跳过。")
            else:
                m = run.evaluate(lab, ref=base_lab)
                rows.append({"name": spec["name"], "dataset": spec["dataset"], "family": "em_label_shift", "level": 0,
                             "lambda_r": run.lr, "lambda_a": run.la, "macro_f1": round(100 * m["macro_f1"], 3),
                             "accuracy": round(100 * m["accuracy"], 3), "f_avg": round(100 * m.get("f_avg", np.nan), 3),
                             "gain_vs_base": round(100 * (m["macro_f1"] - b["macro_f1"]), 3),
                             "delta_vs_official": round(100 * (m["macro_f1"] - off["macro_f1"]), 3),
                             "fixed": m["fixed"], "broken": m["broken"], "mean_row_js_to_official": 0.0})
                print(f"  [EM] SPIN + 绝对知识按 π̂_test 重加权（探索性）MF1 {100*m['macro_f1']:.2f}  "
                      f"Δofficial {rows[-1]['delta_vs_official']:+.2f}")

    with open(out / "perturbation_results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(out / "bootstrap_E_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys())); wr.writeheader(); wr.writerows(boot_rows)
    print(f"\n已保存 {out/'perturbation_results.csv'} 与 {out/'bootstrap_E_summary.csv'}")


if __name__ == "__main__":
    main()