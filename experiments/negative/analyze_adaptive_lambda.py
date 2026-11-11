# -*- coding: utf-8 -*-
"""
analyze_adaptive_lambda.py
离线（不调用 API）：冲突感知的自适应传导强度。
在会话树推断中，每个节点的 λ 不再全局固定，而是按“LLM 观测与传导先验的冲突程度”取两档：
    冲突分数 s_i <  τ ：λ_i = λ_low
    冲突分数 s_i >= τ ：λ_i = λ_high
冲突分数在前向推断过程中逐节点计算（父节点后验已被更新，因此会沿树传递）。
三个参数 (λ_low, λ_high, 分位数 q → τ) 全部在 valid 上按话题平均 MacroF1 选择（与 valid_shared 一致，各话题共用），
test 只用于报告。τ 取 valid 上（以 λ_low 推断时）冲突分数的 (1-q) 分位数。

打分方式（--scores 可多选）：
  conflict      JS(o_i, p~_i) × max q_par
  conflict_raw  JS(o_i, p~_i)
  llm_conf      对照：按 LLM 自报置信度（低于阈值的节点用 λ_high），检验“冲突”是否优于“LLM 不自信”

用法（make_rag_data 目录下）：
python -u spin/experiments/negative/analyze_adaptive_lambda.py \
  --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl \
  --pred baseline_outputs/predictions_knn_fewshot_test.jsonl \
  --valid_pred baseline_outputs/predictions_knn_fewshot_valid.jsonl \
  --output_dir adaptive_lambda_knn
"""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("cd", (Path(__file__).resolve().parent / "../../spin_core/analyze_conflict_detection.py").resolve())
cd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cd)
kc, kbm, dd = cd.kc, cd.kbm, cd.dd
LABELS, LID = dd.LABELS, dd.LID


def node_score(kind, o, prior, qp, conf):
    if kind == "conflict":
        return cd.js(o, prior) * float(np.max(qp))
    if kind == "conflict_raw":
        return cd.js(o, prior)
    if kind == "llm_conf":
        return -(conf if conf == conf else 0.5)
    raise ValueError(kind)


def forward_adaptive(ev, base, conf, ms, emis, lam_low, lam_high, tau, kind):
    q, lab, n_high = {}, {}, 0
    for n in sorted(ev, key=lambda x: x["depth"]):
        k = (n["topic"], n["node_id"])
        e = emis(n)[:, LID[base[k]]]
        qp = q.get(n["parent_key"]) if n["parent_key"] else None
        if qp is None:
            q[k], lab[k] = e / e.sum(), base[k]
            continue
        prior = ms[n["topic"]].prior(n, qp)
        s = node_score(kind, e / e.sum(), prior, qp, conf.get(k, float("nan")))
        lam = lam_high if (tau is not None and s >= tau) else lam_low
        n_high += lam == lam_high and tau is not None
        post = e * prior ** lam
        post = post / post.sum()
        q[k] = post
        lab[k] = LABELS[int(post.argmax())] if lam > 0 else base[k]
    return lab, n_high


def collect_scores(ev, base, conf, ms, emis, lam, kind):
    """以固定 λ 推断时，各有先验节点的打分（用来确定 τ 的分位数）。"""
    q, out = {}, []
    for n in sorted(ev, key=lambda x: x["depth"]):
        k = (n["topic"], n["node_id"])
        e = emis(n)[:, LID[base[k]]]
        qp = q.get(n["parent_key"]) if n["parent_key"] else None
        if qp is None:
            q[k] = e / e.sum(); continue
        prior = ms[n["topic"]].prior(n, qp)
        out.append(node_score(kind, e / e.sum(), prior, qp, conf.get(k, float("nan"))))
        post = e * prior ** lam
        q[k] = post / post.sum()
    return np.array(out)


def topic_avg(ev, lab, topics):
    return float(np.mean([cd.scores_of([n["y"] for n in ev if n["topic"] == t],
                                       [lab[(n["topic"], n["node_id"])] for n in ev if n["topic"] == t])[0] for t in topics]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--kb", default="full", choices=list(cd.KB_KEYS))
    ap.add_argument("--scores", default="conflict,conflict_raw,llm_conf")
    ap.add_argument("--lam_low", default="0.25,0.5,0.75,1")
    ap.add_argument("--lam_high", default="1,1.5,2,3,5")
    ap.add_argument("--quantiles", default="5,10,20,30,50", help="用 λ_high 的节点占有先验节点的百分比（在 valid 上换算成阈值）")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--output_dir", default="adaptive_lambda")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, conf = kc.load_pred_conf(a.pred, a.pred_field, a.conf_field)
    vbase, vconf = kc.load_pred_conf(a.valid_pred, a.pred_field, a.conf_field)
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    ms = {t: kbm.PatternKB([n for n in train if n["topic"] == t], cd.KB_KEYS[a.kb], a.min_count) for t in topics}
    one = kc.make_binner(None)
    E = {t: kc.confusion([n for n in vev if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}
    emis = lambda n: E[n["topic"]]
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)} | 知识库 KB:{a.kb}")

    lows = [float(x) for x in a.lam_low.split(",")]
    highs = [float(x) for x in a.lam_high.split(",")]
    qs = [float(x) for x in a.quantiles.split(",")]
    g = [n["y"] for n in ev]

    def report(name, lab, extra=""):
        mf, fa, acc = cd.scores_of(g, [lab[(n["topic"], n["node_id"])] for n in ev])
        ta = topic_avg(ev, lab, topics)
        ch = sum(lab[(n["topic"], n["node_id"])] != base[(n["topic"], n["node_id"])] for n in ev)
        fx = sum(lab[(n["topic"], n["node_id"])] == n["y"] != base[(n["topic"], n["node_id"])] for n in ev)
        br = sum(base[(n["topic"], n["node_id"])] == n["y"] != lab[(n["topic"], n["node_id"])] for n in ev)
        print(f"  {name:34s} 合并MacroF1 {mf:.4f}  合并F_avg {fa:.4f}  话题平均MacroF1 {ta:.4f}  改对 {fx:4d} 改错 {br:4d} {extra}")
        return {"method": name, "pooled_macro_f1": round(mf, 4), "pooled_f_avg": round(fa, 4),
                "topic_macro_f1": round(ta, 4), "fixed": fx, "broken": br, "note": extra}

    rows = []
    print("\n【test 结果】（所有参数均在 valid 上选择）")
    base_lab = {(n["topic"], n["node_id"]): base[(n["topic"], n["node_id"])] for n in ev}
    rows.append(report("基础预测", base_lab))

    # 固定 λ（valid_shared）
    allg = sorted(set(lows + highs))
    vfix = {l: topic_avg(vev, forward_adaptive(vev, vbase, vconf, ms, emis, l, l, None, "conflict")[0], topics) for l in allg}
    lfix = max(allg, key=vfix.get)
    lab_fix, _ = forward_adaptive(ev, base, conf, ms, emis, lfix, lfix, None, "conflict")
    rows.append(report("基础 + KB（固定 λ，valid_shared）", lab_fix, f"λ={lfix:g}，valid {vfix[lfix]:.4f}"))
    save = {"fixed": lab_fix}

    for kind in a.scores.split(","):
        best = None
        for lo in lows:
            vs = collect_scores(vev, vbase, vconf, ms, emis, lo, kind)
            for hi, qq in itertools.product(highs, qs):
                if hi <= lo:
                    continue
                tau = float(np.quantile(vs, 1 - qq / 100)) if len(vs) else None
                vl, _ = forward_adaptive(vev, vbase, vconf, ms, emis, lo, hi, tau, kind)
                v = topic_avg(vev, vl, topics)
                if best is None or v > best[0]:
                    best = (v, lo, hi, qq, tau)
        v, lo, hi, qq, tau = best
        lab, nh = forward_adaptive(ev, base, conf, ms, emis, lo, hi, tau, kind)
        rows.append(report(f"自适应 λ（{kind}）", lab,
                           f"λ_low={lo:g} λ_high={hi:g} 前{qq:g}%（test 上用 λ_high 的节点 {nh} 个），valid {v:.4f}"))
        save[kind] = lab
    dd.write_csv(out / "adaptive_lambda_results.csv", rows)

    for name, lab in save.items():
        with open(out / f"pred_{name}.jsonl", "w", encoding="utf-8") as f:
            for n in ev:
                k = (n["topic"], n["node_id"])
                f.write(json.dumps({"topic": k[0], "node_id": k[1], "depth": n["depth"], "stance_label": n["y"],
                                    "predicted_stance": lab[k], "base_prediction": base[k],
                                    "fixed_lambda_prediction": lab_fix[k]}, ensure_ascii=False) + "\n")
    print(f"\n结果已保存到 {out.resolve()}")
    print("显著性：--a 用 pred_conflict.jsonl，--b 用同一文件并加 --pred_field_b fixed_lambda_prediction（对比固定 λ）")


if __name__ == "__main__":
    main()