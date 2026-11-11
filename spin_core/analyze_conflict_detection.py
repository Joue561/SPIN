# -*- coding: utf-8 -*-
"""
analyze_conflict_detection.py
离线验证（不调用 API）：“冲突驱动的知识引导复核”是否值得做？

做法：复现 基础预测 + KB 会话树推断（valid_shared 协议：观测模型按话题在 valid 上估计，λ 在全部 valid 上统一选择），
记录每个 test 节点的
  o_i     = LLM 观测给出的分布（混淆矩阵的列，归一化）
  p~_i    = 传导先验（知识库 + 父节点后验）
  q_par   = 父节点后验
  q_i     = 最终后验
然后比较几种“挑选待复核节点”的打分方式，在预算 B（挑前 B% 的节点）下：
  - 挑中的节点里，最终预测（基础+KB）的错误率是多少（越高越好；整体错误率约 30%）
  - 覆盖了全部错误的百分之几（召回）
  - 上界：如果复核总能把挑中的节点改对，Macro-F1 / F_avg 能到多少

打分方式：
  conflict        JS(o_i, p~_i) × max q_par          ← 方案中的冲突分数
  conflict_raw    JS(o_i, p~_i)
  post_entropy    最终后验 q_i 的熵
  post_margin     1 − (q_i 最大值 − 第二大值)
  llm_lowconf     LLM 自报置信度越低越优先
  kb_changed      KB 推断改动过的节点优先，其次按 conflict
  random          随机（多次平均）
没有父节点后验的节点（深度 1 或父节点不在评估集）没有传导先验，conflict 类分数记为 0。

用法（make_rag_data 目录下）：
python -u spin/spin_core/analyze_conflict_detection.py \
  --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl \
  --pred baseline_outputs/predictions_knn_fewshot_test.jsonl \
  --valid_pred baseline_outputs/predictions_knn_fewshot_valid.jsonl \
  --output_dir conflict_detection_knn
"""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("kc", Path(__file__).resolve().parent / "analyze_kb_confidence.py")
kc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(kc)
kbm, dd = kc.kbm, kc.dd
LABELS, LID = dd.LABELS, dd.LID

KB_KEYS = {"full": kbm.k_full, "sp_depth": kbm.k_sp_depth, "sp": kbm.k_sp, "depth": kbm.k_depth}


def js(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def forward(ev, base, ms, lam, emis):
    """与 analyze_kb_calibration.smooth_e(mode='pred') 相同的前向推断，同时记录中间量。"""
    q, rec = {}, {}
    for n in sorted(ev, key=lambda x: x["depth"]):
        k = (n["topic"], n["node_id"])
        e = emis(n)[:, LID[base[k]]]
        o = e / e.sum()
        qp = q.get(n["parent_key"]) if n["parent_key"] else None
        if qp is None:
            post, lab, prior = o, base[k], None
        else:
            prior = ms[n["topic"]].prior(n, qp)
            post = e * prior ** lam
            post = post / post.sum()
            lab = LABELS[int(post.argmax())] if lam > 0 else base[k]
        q[k] = post
        rec[k] = {"o": o, "prior": prior, "qp": qp, "post": post, "label": lab}
    return rec


def scores_of(g, p):
    s = dd.scores(g, p)
    return s["macro_f1"], s["f_avg"], s["acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--kb", default="full", choices=list(KB_KEYS))
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2,2.5,3")
    ap.add_argument("--lambda_fixed", type=float, default=None, help="不在 valid 上选，直接指定 λ")
    ap.add_argument("--budgets", default="5,10,20,30")
    ap.add_argument("--n_random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="conflict_detection")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lams = [float(x) for x in a.lambdas.split(",")]
    budgets = [float(x) for x in a.budgets.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, conf = kc.load_pred_conf(a.pred, a.pred_field, a.conf_field)
    vbase, vconf = kc.load_pred_conf(a.valid_pred, a.pred_field, a.conf_field)
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)} | 知识库 KB:{a.kb}")

    ms = {t: kbm.PatternKB([n for n in train if n["topic"] == t], KB_KEYS[a.kb], a.min_count) for t in topics}
    one = kc.make_binner(None)
    E = {t: kc.confusion([n for n in vev if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}
    emis = lambda n: E[n["topic"]]

    # ---- λ：valid_shared
    if a.lambda_fixed is not None:
        lam = a.lambda_fixed
    else:
        def vscore(l):
            r = forward(vev, vbase, ms, l, emis)
            return np.mean([scores_of([n["y"] for n in vev if n["topic"] == t],
                                      [r[(n["topic"], n["node_id"])]["label"] for n in vev if n["topic"] == t])[0]
                            for t in topics])
        grid = {l: vscore(l) for l in lams}
        lam = max(lams, key=grid.get)
        print("  valid 上各 λ 的话题平均 MacroF1：" + "  ".join(f"{l:g}:{v:.4f}" for l, v in grid.items()))
    print(f"  选定 λ = {lam:g}")

    rec = forward(ev, base, ms, lam, emis)
    keys = [(n["topic"], n["node_id"]) for n in ev]
    gold = {(n["topic"], n["node_id"]): n["y"] for n in ev}
    depth = {(n["topic"], n["node_id"]): n["depth"] for n in ev}
    final = {k: rec[k]["label"] for k in keys}
    g = [gold[k] for k in keys]
    m0 = scores_of(g, [base[k] for k in keys]); m1 = scores_of(g, [final[k] for k in keys])
    err_final = {k: final[k] != gold[k] for k in keys}
    n_err = sum(err_final.values())
    has_prior = {k: rec[k]["prior"] is not None for k in keys}
    print(f"\n基础预测    MacroF1 {m0[0]:.4f}  F_avg {m0[1]:.4f}  Acc {m0[2]:.4f}")
    print(f"基础 + KB   MacroF1 {m1[0]:.4f}  F_avg {m1[1]:.4f}  Acc {m1[2]:.4f}")
    print(f"最终预测整体错误率 {n_err / len(keys):.3f}（{n_err}/{len(keys)}）；有传导先验的节点 {sum(has_prior.values())} 个，"
          f"其中错误率 {sum(err_final[k] for k in keys if has_prior[k]) / max(sum(has_prior.values()), 1):.3f}")

    # ---- 打分
    def ent(p):
        p = np.asarray(p) + 1e-12
        return float(-(p * np.log(p)).sum())

    S = {}
    S["conflict"] = {k: (js(rec[k]["o"], rec[k]["prior"]) * float(np.max(rec[k]["qp"])) if has_prior[k] else 0.0) for k in keys}
    S["conflict_raw"] = {k: (js(rec[k]["o"], rec[k]["prior"]) if has_prior[k] else 0.0) for k in keys}
    S["post_entropy"] = {k: ent(rec[k]["post"]) for k in keys}
    S["post_margin"] = {k: 1 - float(np.sort(rec[k]["post"])[-1] - np.sort(rec[k]["post"])[-2]) for k in keys}
    S["llm_lowconf"] = {k: -(conf.get(k) if conf.get(k) == conf.get(k) else 0.5) for k in keys}
    S["kb_changed"] = {k: (1.0 if final[k] != base[k] else 0.0) + S["conflict"][k] * 1e-3 for k in keys}

    def evaluate(sel):
        sel = set(sel)
        e = sum(err_final[k] for k in sel)
        orc = [gold[k] if k in sel else final[k] for k in keys]
        mf, fa, _ = scores_of(g, orc)
        return e / max(len(sel), 1), e / max(n_err, 1), mf, fa

    rows = []
    print(f"\n【挑选待复核节点】错误率=挑中节点里最终预测错的比例（整体 {n_err / len(keys):.3f}）；召回=覆盖全部错误的比例；"
          f"上界=复核总能改对时的 MacroF1 / F_avg")
    print(f"  {'打分方式':14s} {'预算':>5s} {'n':>5s} {'错误率':>7s} {'召回':>6s} {'上界MacroF1':>11s} {'上界F_avg':>9s}")
    for B in budgets:
        kn = int(round(len(keys) * B / 100))
        for name, sc in S.items():
            order = sorted(keys, key=lambda k: -sc[k])
            r = evaluate(order[:kn])
            rows.append({"score": name, "budget%": B, "n": kn, "err_rate": round(r[0], 4), "recall": round(r[1], 4),
                         "oracle_macro_f1": round(r[2], 4), "oracle_f_avg": round(r[3], 4)})
            print(f"  {name:14s} {B:4g}% {kn:5d} {r[0]:7.3f} {r[1]:6.3f} {r[2]:11.4f} {r[3]:9.4f}")
        rr = np.array([evaluate([keys[i] for i in rng.choice(len(keys), kn, replace=False)])
                       for _ in range(a.n_random)])
        rows.append({"score": "random", "budget%": B, "n": kn, "err_rate": round(rr[:, 0].mean(), 4),
                     "recall": round(rr[:, 1].mean(), 4), "oracle_macro_f1": round(rr[:, 2].mean(), 4),
                     "oracle_f_avg": round(rr[:, 3].mean(), 4)})
        print(f"  {'random(均值)':14s} {B:4g}% {kn:5d} {rr[:, 0].mean():7.3f} {rr[:, 1].mean():6.3f} "
              f"{rr[:, 2].mean():11.4f} {rr[:, 3].mean():9.4f}")
        print()
    dd.write_csv(out / "conflict_detection_budgets.csv", rows)

    # ---- conflict 前 20%：按话题、按深度、按错误类型
    kn = int(round(len(keys) * 0.2))
    top = sorted(keys, key=lambda k: -S["conflict"][k])[:kn]
    print("【conflict 前 20% 的构成】")
    by_t = defaultdict(lambda: [0, 0, 0])
    for k in keys:
        by_t[k[0]][2] += 1
    for k in top:
        by_t[k[0]][0] += 1; by_t[k[0]][1] += err_final[k]
    for t in topics:
        c, e, n = by_t[t]
        tot_e = sum(err_final[k] for k in keys if k[0] == t)
        print(f"  {t:24s} 挑中 {c:4d}/{n:4d}  其中错 {e:4d}（错误率 {e / max(c, 1):.3f}；该话题错误覆盖 {e / max(tot_e, 1):.3f}）")
    by_d = defaultdict(lambda: [0, 0])
    for k in top:
        db = dd.depth_bucket(depth[k]); by_d[db][0] += 1; by_d[db][1] += err_final[k]
    print("  按深度：" + "  ".join(f"深度{d}: {by_d[d][0]} 个，错误率 {by_d[d][1] / max(by_d[d][0], 1):.3f}" for d in sorted(by_d)))
    trans = defaultdict(int)
    for k in top:
        if err_final[k]:
            trans[(gold[k], final[k])] += 1
    print("  挑中节点里的错误类型（gold→预测）：" + "  ".join(f"{a}→{b}: {v}" for (a, b), v in sorted(trans.items(), key=lambda x: -x[1])))
    agree_prior = sum(int(np.argmax(rec[k]["prior"])) == LID[gold[k]] for k in top if err_final[k] and rec[k]["prior"] is not None)
    print(f"  挑中且预测错的节点里，传导先验最大类恰好是 gold 的比例：{agree_prior / max(sum(err_final[k] for k in top), 1):.3f}"
          f"（越高说明“相信知识库”就能改对；越低说明需要复核提供新信息）")

    with open(out / "conflict_per_node.jsonl", "w", encoding="utf-8") as f:
        for k in keys:
            r = rec[k]
            f.write(json.dumps({"topic": k[0], "node_id": k[1], "depth": depth[k], "stance_label": gold[k],
                                "base_prediction": base[k], "kb_prediction": final[k],
                                "conflict": round(S["conflict"][k], 5), "post_entropy": round(S["post_entropy"][k], 5),
                                "obs": [round(x, 4) for x in r["o"]],
                                "prior": None if r["prior"] is None else [round(float(x), 4) for x in r["prior"]],
                                "parent_post": None if r["qp"] is None else [round(float(x), 4) for x in r["qp"]],
                                "post": [round(float(x), 4) for x in r["post"]]}, ensure_ascii=False) + "\n")
    print(f"\n结果已保存到 {out.resolve()}")


if __name__ == "__main__":
    main()