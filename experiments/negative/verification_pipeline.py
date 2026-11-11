# -*- coding: utf-8 -*-
"""
verification_pipeline.py
冲突感知闭环（第二阶段）：对“KB 会话树推断 + 自适应 λ”之后仍不确定的节点做定向复核，再把复核结果融合回推断。

两个子命令：
  select  在 valid 与 test 上分别运行 自适应 λ 推断，按最终后验的不确定度（1 − 最大值与次大值之差）
          在“有传导先验的节点”中挑出前 B%（相对该划分全部节点数），写出 id 文件，供预测脚本 --id_file 使用。
  fuse    读取复核预测（simple_rag_stance_predictor.py --method relation_verify 的输出），
          在 valid 上估计复核的可信度（混淆矩阵），在 valid 上选择融合方式，并报告 test 结果。

融合方式（复核过的节点）：
  replace    直接用复核给出的立场
  stance     把复核立场当作第二个立场观测：q ∝ E[y,ŷ]·E2[y,ŷ2]·p~(y)^λ
  relation   把复核给出的关系当作关系观测：q ∝ E[y,ŷ]·p~(y)^λ·Σ_yp q_par(yp)·R[r(yp,y), r̂]
  both       stance 与 relation 两个因子都乘上
E2、R 在 valid 的复核节点上估计（各话题合并，加一平滑）；融合方式按 valid 话题平均 MacroF1 选择。
未复核的节点与原推断完全相同。

用法（make_rag_data 目录下）：
  python spin/experiments/negative/verification_pipeline.py select \
    --instances processed_cmtscd_semantic_llm_causal_v2/cmtscd_semantic_instances_llm.jsonl \
    --pred baseline_outputs/predictions_knn_fewshot_test.jsonl \
    --valid_pred baseline_outputs/predictions_knn_fewshot_valid.jsonl \
    --budget 10 --output_dir verification_knn
  （再用预测脚本分别对 valid / test 跑 relation_verify，--id_file 指向本步输出）
  python spin/experiments/negative/verification_pipeline.py fuse \
    --instances ... --pred ... --valid_pred ... \
    --verify_valid verification_knn/pred/predictions_relation_verify_valid.jsonl \
    --verify_test  verification_knn/pred/predictions_relation_verify_test.jsonl \
    --output_dir verification_knn
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("aa", Path(__file__).resolve().parent / "analyze_adaptive_lambda.py")
aa = importlib.util.module_from_spec(_s)
_s.loader.exec_module(aa)
cd, kc, kbm, dd = aa.cd, aa.kc, aa.kbm, aa.dd
LABELS, LID = dd.LABELS, dd.LID
RELS4 = ["inherit", "reverse", "neutralize", "independent"]
RID = {r: i for i, r in enumerate(RELS4)}


def true_rel(yp, y):
    """父立场 yp、当前立场 y 对应的关系（与复核 prompt 的四个选项一致）。"""
    if yp in ("favor", "against"):
        return "neutralize" if y == "none" else ("inherit" if y == yp else "reverse")
    return "independent" if y != "none" else "neutralize"


def norm_rel(x):
    x = str(x or "").strip().lower()
    return x if x in RID else None


def forward(ev, base, conf, ms, emis, lo, hi, tau, kind, verify=None, mode="none", E2=None, R=None):
    """自适应 λ 前向推断；verify={key: {"stance":..., "rel":...}} 时按 mode 融合复核结果。返回 (标签, 后验, 是否有先验)。"""
    q, lab, has = {}, {}, {}
    for n in sorted(ev, key=lambda x: x["depth"]):
        k = (n["topic"], n["node_id"])
        e = emis(n)[:, LID[base[k]]]
        qp = q.get(n["parent_key"]) if n["parent_key"] else None
        v = verify.get(k) if verify else None
        if qp is None:
            post, has[k] = e / e.sum(), False
            if v and mode != "none" and v.get("stance") in LID:
                post = post * E2[:, LID[v["stance"]]] if mode in ("stance", "both") else post
                post = np.eye(3)[LID[v["stance"]]] * 0.9 + 0.05 if mode == "replace" else post
                post = post / post.sum()
            q[k] = post
            lab[k] = LABELS[int(post.argmax())] if (v and mode != "none") else base[k]
            continue
        has[k] = True
        prior = ms[n["topic"]].prior(n, qp)
        s = aa.node_score(kind, e / e.sum(), prior, qp, conf.get(k, float("nan")))
        lam = hi if (tau is not None and s >= tau) else lo
        post = e * prior ** lam
        if v and mode in ("stance", "both") and v.get("stance") in LID:
            post = post * E2[:, LID[v["stance"]]]
        if v and mode in ("relation", "both") and v.get("rel") in RID:
            r_obs = RID[v["rel"]]
            post = post * np.array([sum(qp[LID[yp]] * R[RID[true_rel(yp, y)], r_obs] for yp in LABELS) for y in LABELS])
        if v and mode == "replace" and v.get("stance") in LID:
            post = np.eye(3)[LID[v["stance"]]] * 0.9 + 0.05
        post = post / post.sum()
        q[k] = post
        lab[k] = LABELS[int(post.argmax())] if (lam > 0 or (v and mode != "none")) else base[k]
    return lab, q, has


def setup(a):
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
    vs = aa.collect_scores(vev, vbase, vconf, ms, emis, a.lam_low, a.score)
    tau = float(np.quantile(vs, 1 - a.q / 100))
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)} | KB:{a.kb} | 自适应 λ：{a.score}，"
          f"λ_low={a.lam_low:g} λ_high={a.lam_high:g} 前{a.q:g}%（τ={tau:.4f}，在 valid 上确定）")
    return nodes, topics, base, conf, vbase, vconf, ev, vev, ms, emis, tau


def cmd_select(a):
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    nodes, topics, base, conf, vbase, vconf, ev, vev, ms, emis, tau = setup(a)
    for split, E_, B_, C_ in (("valid", vev, vbase, vconf), ("test", ev, base, conf)):
        lab, q, has = forward(E_, B_, C_, ms, emis, a.lam_low, a.lam_high, tau, a.score)
        mf, fa, _ = cd.scores_of([n["y"] for n in E_], [lab[(n["topic"], n["node_id"])] for n in E_])
        cand = [(n["topic"], n["node_id"]) for n in E_ if has[(n["topic"], n["node_id"])]]
        margin = {k: 1 - float(np.sort(q[k])[-1] - np.sort(q[k])[-2]) for k in cand}
        kn = int(round(len(E_) * a.budget / 100))
        sel = sorted(cand, key=lambda k: -margin[k])[:kn]
        gold = {(n["topic"], n["node_id"]): n["y"] for n in E_}
        err = sum(lab[k] != gold[k] for k in sel) / max(len(sel), 1)
        path = out / f"verify_ids_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i, k in enumerate(sel):
                f.write(json.dumps({"topic": k[0], "node_id": k[1], "rank": i, "uncertainty": round(margin[k], 5),
                                    "current_prediction": lab[k]}, ensure_ascii=False) + "\n")
        print(f"  {split}: 自适应 λ 推断 MacroF1 {mf:.4f} F_avg {fa:.4f}；挑出 {len(sel)} 个节点（{a.budget:g}%），"
              f"其中当前预测错误率 {err:.3f} → {path}")


def load_verify(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[(str(r.get("topic")), str(r.get("node_id")))] = {
            "stance": dd.norm(r.get("predicted_stance")), "rel": norm_rel(r.get("relation_to_parent_pred")),
            "parent": dd.norm(r.get("parent_stance_pred"))}
    return out


def cmd_fuse(a):
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    nodes, topics, base, conf, vbase, vconf, ev, vev, ms, emis, tau = setup(a)
    Vv, Vt = load_verify(a.verify_valid), load_verify(a.verify_test)
    Vv = {k: v for k, v in Vv.items() if k in {(n["topic"], n["node_id"]) for n in vev}}
    Vt = {k: v for k, v in Vt.items() if k in {(n["topic"], n["node_id"]) for n in ev}}
    print(f"复核结果：valid {len(Vv)} 条，test {len(Vt)} 条")

    # 复核可信度（valid 上估计）
    lab_v0, q_v0, _ = forward(vev, vbase, vconf, ms, emis, a.lam_low, a.lam_high, tau, a.score)
    gold_v = {(n["topic"], n["node_id"]): n for n in vev}
    E2 = np.ones((3, 3)); R = np.ones((4, 4))
    for k, v in Vv.items():
        n = gold_v[k]
        if v["stance"] in LID:
            E2[LID[n["y"]], LID[v["stance"]]] += 1
        if v["rel"] in RID and n["pa_gold"] in LID:
            R[RID[true_rel(n["pa_gold"], n["y"])], RID[v["rel"]]] += 1
    E2 = E2 / E2.sum(1, keepdims=True); R = R / R.sum(1, keepdims=True)

    def acc_on(V, gold, lab0):
        ks = list(V)
        a_v = np.mean([V[k]["stance"] == gold[k] for k in ks]) if ks else float("nan")
        a_0 = np.mean([lab0[k] == gold[k] for k in ks]) if ks else float("nan")
        return a_v, a_0

    gv = {k: n["y"] for k, n in gold_v.items()}
    a_v, a_0 = acc_on(Vv, gv, lab_v0)
    print(f"\n【valid 复核节点】复核立场准确率 {a_v:.3f}  vs  当前推断准确率 {a_0:.3f}（复核需要明显高于后者才有收益）")
    rel_ok = [v["rel"] == true_rel(gold_v[k]["pa_gold"], gold_v[k]["y"]) for k, v in Vv.items()
              if v["rel"] in RID and gold_v[k]["pa_gold"] in LID]
    par_ok = [v["parent"] == gold_v[k]["pa_gold"] for k, v in Vv.items() if v["parent"] and gold_v[k]["pa_gold"] in LID]
    print(f"  复核关系准确率 {np.mean(rel_ok):.3f}（n={len(rel_ok)}）  复核父立场准确率 {np.mean(par_ok):.3f}（n={len(par_ok)}）")
    print("  复核立场混淆矩阵 E2（行=真实 favor/against/none）：\n" + np.array2string(E2, precision=3))

    modes = ["none", "replace", "stance", "relation", "both"]
    vscore = {}
    for m in modes:
        lab, _, _ = forward(vev, vbase, vconf, ms, emis, a.lam_low, a.lam_high, tau, a.score, Vv, m, E2, R)
        vscore[m] = aa.topic_avg(vev, lab, topics)
    best = max(modes, key=vscore.get)
    print("\n【valid 上各融合方式的话题平均 MacroF1】" + "  ".join(f"{m}:{vscore[m]:.4f}" for m in modes) + f"  → 选 {best}")

    gold_t = {(n["topic"], n["node_id"]): n["y"] for n in ev}
    lab_t0, _, _ = forward(ev, base, conf, ms, emis, a.lam_low, a.lam_high, tau, a.score)
    a_t, a_t0 = acc_on(Vt, gold_t, lab_t0)
    print(f"\n【test 复核节点】复核立场准确率 {a_t:.3f}  vs  当前推断准确率 {a_t0:.3f}")
    g = [n["y"] for n in ev]
    res = {}
    print("【test 结果】")
    for m in modes:
        lab, _, _ = forward(ev, base, conf, ms, emis, a.lam_low, a.lam_high, tau, a.score, Vt, m, E2, R)
        mf, fa, _ = cd.scores_of(g, [lab[(n["topic"], n["node_id"])] for n in ev])
        ta = aa.topic_avg(ev, lab, topics)
        fx = sum(lab[k] == gold_t[k] != lab_t0[k] for k in gold_t)
        br = sum(lab_t0[k] == gold_t[k] != lab[k] for k in gold_t)
        tag = "  ← valid 选定" if m == best else ""
        print(f"  {m:9s} 合并MacroF1 {mf:.4f}  合并F_avg {fa:.4f}  话题平均MacroF1 {ta:.4f}  相对自适应λ 改对 {fx:3d} 改错 {br:3d}{tag}")
        res[m] = lab
    with open(out / "pred_verification_fused.jsonl", "w", encoding="utf-8") as f:
        for n in ev:
            k = (n["topic"], n["node_id"])
            f.write(json.dumps({"topic": k[0], "node_id": k[1], "depth": n["depth"], "stance_label": n["y"],
                                "predicted_stance": res[best][k], "adaptive_prediction": lab_t0[k],
                                "base_prediction": base[k], "verified": k in Vt,
                                **{f"pred_{m}": res[m][k] for m in modes}}, ensure_ascii=False) + "\n")
    print(f"\n已保存 {out / 'pred_verification_fused.jsonl'}；显著性用 --pred_field_b adaptive_prediction 对比自适应 λ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "fuse"):
        p = sub.add_parser(name)
        p.add_argument("--instances", required=True)
        p.add_argument("--pred", required=True)
        p.add_argument("--valid_pred", required=True)
        p.add_argument("--kb", default="full", choices=list(cd.KB_KEYS))
        p.add_argument("--score", default="conflict_raw", choices=["conflict", "conflict_raw", "llm_conf"])
        p.add_argument("--lam_low", type=float, default=0.5, help="自适应 λ 参数，默认取 analyze_adaptive_lambda.py 在 valid 上选出的值")
        p.add_argument("--lam_high", type=float, default=5)
        p.add_argument("--q", type=float, default=5, help="用 λ_high 的节点比例（%）")
        p.add_argument("--pred_field", default="predicted_stance")
        p.add_argument("--conf_field", default="confidence")
        p.add_argument("--parent_field", default="parent_stance_label")
        p.add_argument("--min_count", type=int, default=20)
        p.add_argument("--output_dir", default="verification")
        if name == "select":
            p.add_argument("--budget", type=float, default=10, help="复核节点占该划分全部节点的百分比")
        else:
            p.add_argument("--verify_valid", required=True)
            p.add_argument("--verify_test", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "fuse": cmd_fuse}[a.cmd](a)


if __name__ == "__main__":
    main()