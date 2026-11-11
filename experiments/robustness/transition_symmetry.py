# -*- coding: utf-8 -*-
"""
transition_symmetry.py —— 问题二：相对转移矩阵的极性对称假设检验（对称 vs 非对称）。

矩阵约定（与论文一致）：行 = 父节点立场，列 = 子节点立场，顺序按 acp.LABELS；
子节点的相对先验 = q_parent^T · M_k（再归一化）。检索键与 back-off 链直接调用原实现 cd.KB_KEYS["full"]（k_full）：
(c: rr,sp,tr,db) → (rs: rr,sp,db) → (spd: sp,db) → (sp) → (d: db) → (g: 全局)，n_min 与原实现相同。
与原 PatternKB 一致：父立场取 pa_gold；R（继承/反转/中立化）与 N（表态/不表态）两张表各自回退；平滑 (c+0.5)/(tot+0.5·类数)。

变体（每个“参数组”独立计数、独立平滑；样本不足时按同一 back-off 链退回上一级，而不是用不可靠的局部估计）：
  sym           原 SPIN：极性行共享 (ι,ρ,ν)，NONE 行 [ε/2, ε/2, 1-ε]
  full          完整 3×3：F 行、A 行、N 行各自独立估计
  polar_asym    F 行、A 行独立；NONE 行保持 ε/2 对称            （可选，--variants 中加入）
  neutral_asym  极性行共享；NONE 行三项独立                       （可选）

一致性闸门（必须通过才产出结果）：本脚本重写的 sym 必须与原 kb.prior 在全部验证/测试节点、多种父后验下数值一致；
脚本会在若干平滑/back-off 约定中自动找出与原实现一致的那一个，并把同一约定用于所有变体。
只替换 kb.prior（相对知识）；绝对知识 kb.direct_proba、观测信道 E、λ 网格与选择准则全部沿用原实现。
测试标签只用于最终评测与诊断（留出 NLL、分桶分析）。

输出：
  symmetry_summary.csv           每个 run × 变体：MacroF1 / Acc / F_avg / 话题平均 / 每类 P,R,F1 / 对 base 与对 sym 的配对检验
  symmetry_nll.csv               以真实父立场为条件的子立场负对数似然：train（样本内）/ valid / test（留出）
  symmetry_datatests.csv         训练集上直接检验两条对称约束（G 检验 / 二项检验，BH 校正）
  symmetry_breakdown.csv         测试集按键支持度、深度分桶的比较
  symmetry_backoff.csv           各变体各参数组实际使用的 back-off 层级分布
  preds_<run>_<variant>.jsonl    供 paired_significance.py 复核

用法：
  python -u transition_symmetry.py --tools_dir cross_topic_exp/tools --manifest runs.json \
      --output_dir robustness/symmetry --variants sym,full
"""
import argparse
import csv
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import spin_robust_common as C

DEFAULT_LEVELS = [("rr", "sp", "tr", "db"), ("rr", "sp", "db"), ("sp", "db"), ("sp",), ("db",), ()]
VARIANTS = {"sym": ("tied", "tied"), "full": ("free", "free"),
            "polar_asym": ("free", "tied"), "neutral_asym": ("tied", "free")}
SUPPORT_BINS = [(0, 0), (1, 19), (20, 49), (50, 199), (200, 10 ** 9)]


def field(n, f):
    return min(int(n["depth"]), 4) if f == "db" else n.get(f)


class AltKB:
    """在原 KB 之外重新计数的相对知识；可按变体决定哪些行共享参数。"""

    def __init__(self, train_nodes, all_nodes, labels, none_idx, min_count, keyfn, a, mode, eps_rule, parent_src="pa_gold"):
        self.L, self.K, self.N = labels, len(labels), none_idx
        self.P = [i for i in range(self.K) if i != none_idx]          # 两个极性类（按 LABELS 顺序）
        self.idx = {l: i for i, l in enumerate(labels)}
        # keyfn = 原实现的检索键函数 cd.KB_KEYS["full"]（返回从细到粗的键列表，最后一个是全局键 ("g",)）
        self.keyfn, self.a, self.mode, self.eps_rule, self.n_min = keyfn, a, mode, eps_rule, min_count
        # 某话题在训练集中没有节点时（原 PatternKB 此时只剩全局键、计数为 0），用任一节点确定键链结构
        probe = keyfn(train_nodes[0] if train_nodes else next(iter(all_nodes.values())))
        self.L = len(probe)
        self.level_names = [k[0] for k in probe]
        self.cnt = [defaultdict(lambda: np.zeros((self.K, self.K))) for _ in range(self.L)]
        self.n_pairs = 0
        for c in train_nodes:
            if parent_src == "pa_gold":          # 实例文件中的父节点真实立场字段（dd.load_instances 的 pa_gold）
                py = c.get("pa_gold")
            else:                                # 在节点表中查父节点的标签
                pk = c.get("parent_key")
                py = all_nodes[pk].get("y") if pk and pk in all_nodes else None
            if not py or py not in self.idx:
                continue
            yp, yc = self.idx[py], self.idx[c["y"]]
            for li, k in enumerate(keyfn(c)):
                self.cnt[li][k][yp, yc] += 1
            self.n_pairs += 1

    def counts(self, n, li):
        return self.cnt[li].get(self.keyfn(n)[li])

    def support(self, n):
        c = self.counts(n, 0)
        return 0 if c is None else int(c.sum())

    def _pick(self, n, need):
        """返回第一个满足 need(counts) >= n_min 的层级（全局层总是可用）。"""
        last = self.L - 1
        for li in range(self.L):
            c = self.counts(n, li)
            if c is not None and (li == last or need(c) >= self.n_min):
                return li, c
        return last, np.zeros((self.K, self.K))

    def matrix(self, n, variant, return_levels=False):
        polar, neutral = VARIANTS[variant]
        F, A, N, a = self.P[0], self.P[1], self.N, self.a
        M = np.zeros((self.K, self.K))
        lv = {}
        if self.mode == "joint":
            lj, cj = self._pick(n, lambda c: c.sum())
        sel = (lambda need: (lj, cj)) if self.mode == "joint" else (lambda need: self._pick(n, need))
        if polar == "tied":
            li, c = sel(lambda c: c[[F, A]].sum())
            tot = c[[F, A]].sum()
            inh, rev, neu = c[F, F] + c[A, A], c[F, A] + c[A, F], c[F, N] + c[A, N]
            i_, r_, v_ = [(x + a) / (tot + 3 * a) for x in (inh, rev, neu)]
            M[F, F], M[F, A], M[F, N] = i_, r_, v_
            M[A, A], M[A, F], M[A, N] = i_, r_, v_
            lv["polar"] = li
        else:
            for p, nm in ((F, "F_row"), (A, "A_row")):
                li, c = sel(lambda c, p=p: c[p].sum())
                M[p] = (c[p] + a) / (c[p].sum() + 3 * a)
                lv[nm] = li
        if neutral == "tied":
            li, c = sel(lambda c: c[N].sum())
            tot = c[N].sum()
            if self.eps_rule == "binary":
                eps = (c[N, F] + c[N, A] + a) / (tot + 2 * a)
            else:   # ternary：先三类各加 a，再把两极性合并
                eps = (c[N, F] + c[N, A] + 2 * a) / (tot + 3 * a)
            M[N, F] = M[N, A] = eps / 2
            M[N, N] = 1 - eps
            lv["none_row"] = li
        else:
            li, c = sel(lambda c: c[N].sum())
            M[N] = (c[N] + a) / (c[N].sum() + 3 * a)
            lv["none_row"] = li
        M = C.rownorm(M)
        return (M, lv) if return_levels else M

    def prior(self, n, qp, variant):
        p = np.asarray(qp, float) @ self.matrix(n, variant)
        return p / p.sum()


class KBView:
    """替换 prior（相对知识），其余（direct_proba 等）原样转发给原 KB。"""

    def __init__(self, kb, alt, variant):
        self._kb, self._alt, self._v, self.calls = kb, alt, variant, 0

    def prior(self, n, qp):
        self.calls += 1
        return self._alt.prior(n, qp, self._v)

    def __getattr__(self, k):
        return getattr(self._kb, k)


def parity(run, alt_by_topic, n_random=3, seed=0):
    rng = np.random.default_rng(seed)
    K = len(run.LABELS)
    qs = [np.eye(K)[i] for i in range(K)] + [rng.dirichlet(np.ones(K)) for _ in range(n_random)]
    worst, ex, n_checked = 0.0, None, 0
    for n in run.vev + run.ev:
        if not n.get("parent_key"):
            continue
        t = n["topic"]
        for q in qs:
            o = np.asarray(run.kb[t].prior(n, q), float)
            o = o / o.sum()
            m = alt_by_topic[t].prior(n, q, "sym")
            d = float(np.abs(o - m).max())
            n_checked += 1
            if d > worst:
                worst, ex = d, (n["topic"], n["node_id"], alt_by_topic[t].keyfn(n)[0],
                                np.round(q, 3).tolist(), np.round(o, 6).tolist(), np.round(m, 6).tolist())
    return worst, ex, n_checked


def build_alt(run, keyfn, a, mode, eps_rule, parent_src="pa_gold"):
    return {t: AltKB([n for n in run.train if n["topic"] == t], run.nodes, run.LABELS, run.none_idx,
                     run.min_count, keyfn, a, mode, eps_rule, parent_src) for t in run.topics}


def nll(run, alt, nodes, variant):
    vals = []
    for n in nodes:
        pk = n.get("parent_key")
        py = n.get("pa_gold") or (run.nodes[pk].get("y") if pk and pk in run.nodes else None)
        if not py or py not in run.LID:
            continue
        M = alt[n["topic"]].matrix(n, variant)
        vals.append(-np.log(max(M[run.LID[py], run.LID[n["y"]]], 1e-12)))
    return float(np.mean(vals)) if vals else None, len(vals)


def data_tests(alt, labels, n_min):
    """在训练计数上直接检验：(1) F 行与“镜像后的”A 行是否同分布；(2) NONE 父节点下 F 与 A 是否等概率。"""
    try:
        from scipy.stats import binomtest, chi2_contingency
    except ImportError:
        return {"error": "需要 scipy"}
    F, A, N = alt.P[0], alt.P[1], alt.N

    def bh(ps):
        ps = np.asarray(ps)
        if not len(ps):
            return np.array([])
        o = np.argsort(ps)
        q = ps[o] * len(ps) / (np.arange(len(ps)) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty_like(q); out[o] = np.minimum(q, 1)
        return out

    res = {}
    for li in range(alt.L):
        name = alt.level_names[li]
        p1, tv1, w1, p2, r2, w2 = [], [], [], [], [], []
        for key, c in alt.cnt[li].items():
            rf = c[F][[F, A, N]]          # F 父：继承 / 反转 / 中立
            ra = c[A][[A, F, N]]          # A 父镜像后：继承 / 反转 / 中立
            if rf.sum() >= n_min and ra.sum() >= n_min:
                tab = np.vstack([rf, ra]) + 0.5
                p1.append(chi2_contingency(tab, correction=False, lambda_="log-likelihood")[1])
                tv1.append(C.tv(rf, ra)); w1.append(rf.sum() + ra.sum())
            nf, na = int(c[N, F]), int(c[N, A])
            if c[N].sum() >= n_min and nf + na > 0:
                p2.append(binomtest(nf, nf + na, 0.5).pvalue)
                r2.append(nf / (nf + na)); w2.append(nf + na)
        q1, q2 = bh(p1), bh(p2)
        res[name] = {
            "polar_symmetry_keys_tested": len(p1), "polar_symmetry_rejected_q05": int((q1 < 0.05).sum()),
            "polar_mirror_tv_weighted": float(np.average(tv1, weights=w1)) if tv1 else None,
            "polar_mirror_tv_median": float(np.median(tv1)) if tv1 else None,
            "neutral_equal_keys_tested": len(p2), "neutral_equal_rejected_q05": int((q2 < 0.05).sum()),
            "neutral_share_favor_weighted": float(np.average(r2, weights=w2)) if r2 else None,
            "neutral_share_favor_range": [float(min(r2)), float(max(r2))] if r2 else None,
            "polar_symmetry_min_p": float(min(p1)) if p1 else None, "neutral_equal_min_p": float(min(p2)) if p2 else None,
        }
    g = alt.cnt[alt.L - 1].get(("g",), np.zeros((alt.K, alt.K)))
    res["global_counts_parent_x_child"] = g.tolist()
    return res


def bucket_of(s):
    for lo, hi in SUPPORT_BINS:
        if lo <= s <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 8 else f">={lo}"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", default="robustness/symmetry")
    ap.add_argument("--variants", default="sym,full")
    ap.add_argument("--smooth_grid", default="0.5,1.0")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--allow_mismatch", action="store_true", help="仅用于调试：一致性闸门不通过也继续")
    ap.add_argument("--n_boot", type=int, default=5000)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    acp = C.load_acp(a.tools_dir)
    variants = [v.strip() for v in a.variants.split(",")]
    assert variants[0] == "sym" and all(v in VARIANTS for v in variants), "variants 必须以 sym 开头，且取自 " + str(list(VARIANTS))
    S, NL, DT, BD, BO = [], [], [], [], []
    for spec in C.load_manifest(a.manifest):
        print(f"\n===== {spec['name']} ({spec['dataset']})", flush=True)
        run = C.make_run(acp, spec)
        print(f"  LABELS={run.LABELS}  NONE={run.LABELS[run.none_idx]}  cd.KB_KEYS['full']={acp.cd.KB_KEYS['full']}")

        # ---------- 一致性闸门
        best = None
        for ps, sm, mode, er in itertools.product(("pa_gold", "lookup"), [float(x) for x in a.smooth_grid.split(",")],
                                                  ("group", "joint"), ("binary", "ternary")):
            alt = build_alt(run, acp.cd.KB_KEYS["full"], sm, mode, er, ps)
            worst, ex, nchk = parity(run, alt)
            print(f"  约定 父标签={ps:7s} a={sm:g} backoff={mode:5s} ε={er:7s}  最大差 {worst:.2e}")
            if best is None or worst < best[0]:
                best = (worst, ex, (sm, mode, er, ps), alt)
            if worst < a.tol:
                break
        worst, ex, conv, alt = best
        if worst >= a.tol:
            print(f"\n  !! 一致性闸门未通过：重写的对称矩阵与原 kb.prior 最大差 {worst:.3e}。示例（话题, 节点, 键, q_parent, 原, 新）：\n  {ex}")
            print("  请运行 inspect_spin_code.py 并把 PatternKB 的源码发给我，以便对齐计数/平滑/back-off 细节。")
            if not a.allow_mismatch:
                continue
        else:
            print(f"  ✓ 一致性闸门通过（约定 a={conv[0]:g}, backoff={conv[1]}, ε={conv[2]}, 父标签={conv[3]}，{nchk} 次比较，最大差 {worst:.1e}）")

        # ---------- 确认 forward 通过 kb.prior 使用相对知识
        probe = {t: KBView(run.kb[t], alt[t], "sym") for t in run.topics}
        run.forward(run.vev, run.vbase, run.E_val, probe)
        if sum(p.calls for p in probe.values()) == 0:
            print("  !! acp.forward 没有调用 kb.prior；无法在不改主代码的前提下替换矩阵。请把 analyze_combined_prior.forward 发给我。")
            continue

        base_lab = {k: run.base[k] for k in run.keys}
        mb = run.evaluate(base_lab)
        preds = {}
        for v in variants:
            view = {t: KBView(run.kb[t], alt[t], v) for t in run.topics}
            lr, la, vs = run.select_lambda(run.E_val, view)
            lab = run.forward(run.ev, run.base, run.E_val, view, lr, la)
            preds[v] = lab
            m = run.evaluate(lab, ref=base_lab)
            row = {"name": spec["name"], "dataset": spec["dataset"], "variant": v, "lambda_r": lr, "lambda_a": la,
                   "valid_topic_avg": round(100 * vs, 3),
                   **{k: (round(100 * x, 3) if isinstance(x, float) else x) for k, x in m.items()
                      if k not in ("warning",)},
                   "gain_vs_base": round(100 * (m["macro_f1"] - mb["macro_f1"]), 3)}
            tb = C.paired_tests(run.gold, [lab[k] for k in run.keys], [base_lab[k] for k in run.keys], run.LABELS,
                                n_boot=a.n_boot, n_perm=a.n_boot)
            row.update({"vs_base_ci": f"[{100*tb['ci_low']:+.2f}, {100*tb['ci_high']:+.2f}]", "vs_base_p_ar": tb["p_ar"]})
            if v != "sym":
                ts = C.paired_tests(run.gold, [lab[k] for k in run.keys], [preds["sym"][k] for k in run.keys], run.LABELS,
                                    n_boot=a.n_boot, n_perm=a.n_boot)
                row.update({"vs_sym_delta": round(100 * ts["delta"], 3),
                            "vs_sym_ci": f"[{100*ts['ci_low']:+.2f}, {100*ts['ci_high']:+.2f}]", "vs_sym_p_ar": ts["p_ar"],
                            "changed_vs_sym": int(sum(lab[k] != preds["sym"][k] for k in run.keys))})
            S.append(row)
            print(f"  {v:13s} λ=({lr:g},{la:g})  MF1 {row['macro_f1']:.2f}  Acc {row['accuracy']:.2f}  "
                  f"Δbase {row['gain_vs_base']:+.2f}" + (f"  Δsym {row['vs_sym_delta']:+.2f} {row['vs_sym_ci']} p={row['vs_sym_p_ar']:.3f}"
                                                        if v != "sym" else ""))
            if v == "sym":
                o = run.evaluate(run.spin)["macro_f1"]
                if abs(o - m["macro_f1"]) > 1e-9:
                    print(f"  !! 重写的 sym 预测与官方 SPIN 不同（{100*o:.3f} vs {100*m['macro_f1']:.3f}），请检查")
            C.write_jsonl(out / f"preds_{spec['name']}_{v}.jsonl",
                          [{"topic": k[0], "node_id": k[1], "stance_label": n["y"], "predicted_stance": lab[k],
                            "sym_prediction": preds["sym"][k], "base_prediction": base_lab[k]}
                           for k, n in zip(run.keys, run.ev)])

            # 留出似然（以真实父立场为条件），train 为样本内
            for split, nodes in (("train", run.train), ("valid", run.vev), ("test", run.ev)):
                val, cnt = nll(run, alt, nodes, v)
                NL.append({"name": spec["name"], "dataset": spec["dataset"], "variant": v, "split": split,
                           "nll": round(val, 5) if val is not None else None, "n_pairs": cnt})

            # back-off 层级分布（不需要标签）
            lvc = defaultdict(Counter)
            for n in run.ev:
                if n.get("parent_key"):
                    _, lv = alt[n["topic"]].matrix(n, v, return_levels=True)
                    for g, li in lv.items():
                        lvc[g][li] += 1
            for g, cnt in lvc.items():
                tot = sum(cnt.values())
                BO.append({"name": spec["name"], "dataset": spec["dataset"], "variant": v, "group": g,
                           **{f"L{li}": round(cnt.get(li, 0) / tot, 4) for li in range(alt[run.topics[0]].L)}})

        # 分桶：键支持度（训练中该完整键的父子对数）与深度
        for bname, fn in (("support", lambda n: bucket_of(alt[n["topic"]].support(n))),
                          ("depth", lambda n: str(min(int(n["depth"]), 4)))):
            groups = defaultdict(list)
            for k, n in zip(run.keys, run.ev):
                groups[fn(n)].append(k)
            for g, ks in sorted(groups.items()):
                gold = [run.nodes[k]["y"] for k in ks]
                r = {"name": spec["name"], "dataset": spec["dataset"], "by": bname, "bucket": g, "n": len(ks),
                     "base_acc": round(100 * np.mean([run.base[k] == y for k, y in zip(ks, gold)]), 2),
                     "base_mf1": round(100 * C.macro_f1(gold, [run.base[k] for k in ks], run.LABELS), 2)}
                for v in variants:
                    r[f"{v}_acc"] = round(100 * np.mean([preds[v][k] == y for k, y in zip(ks, gold)]), 2)
                    r[f"{v}_mf1"] = round(100 * C.macro_f1(gold, [preds[v][k] for k in ks], run.LABELS), 2)
                BD.append(r)

        # 训练集上直接检验对称约束（每个话题的 KB 分别检验）
        for t in run.topics:
            d = data_tests(alt[t], run.LABELS, run.min_count)
            if "error" in d:
                print("  data tests:", d["error"]); break
            for lvl, dv in d.items():
                if lvl != "global_counts_parent_x_child":
                    DT.append({"name": spec["name"], "dataset": spec["dataset"], "topic": t, "level": lvl, **dv})
            for lvl, dv in d.items():
                if lvl == "global_counts_parent_x_child":
                    continue
                print(f"  [训练集检验] {t} {lvl:14s} 极性对称被拒 {dv['polar_symmetry_rejected_q05']}/{dv['polar_symmetry_keys_tested']}"
                      f"（加权镜像TV {dv['polar_mirror_tv_weighted'] if dv['polar_mirror_tv_weighted'] is None else round(dv['polar_mirror_tv_weighted'],3)}）"
                      f"  NONE等概率被拒 {dv['neutral_equal_rejected_q05']}/{dv['neutral_equal_keys_tested']}"
                      f"（NONE父下F占极性 {dv['neutral_share_favor_weighted'] if dv['neutral_share_favor_weighted'] is None else round(dv['neutral_share_favor_weighted'],3)}）")

    def save(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for nm, rows in (("symmetry_summary.csv", S), ("symmetry_nll.csv", NL), ("symmetry_datatests.csv", DT),
                     ("symmetry_breakdown.csv", BD), ("symmetry_backoff.csv", BO)):
        save(nm, rows)
    print(f"\n结果已写入 {out}/")


if __name__ == "__main__":
    main()