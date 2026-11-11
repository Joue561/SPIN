# -*- coding: utf-8 -*-
"""
analyze_speaker_prior.py
离线（不调用 API）：在会话树推断中加入第三种知识——同说话人一致性。

    q_i(y) ∝ E[y, ŷ_i] · p_rel_i(y)^λr · p_abs_i(y)^λa · p_spk_i(y)^λs

  p_rel：相对传导知识 + 父节点后验（与 analyze_combined_prior.py 相同）
  p_abs：该知识单元的立场分布（与 analyze_combined_prior.py 相同）
  p_spk：若当前说话人在祖先中出现过，取最近一次发言 j 的后验 q_j，
         p_spk(y) = Σ_y' q_j(y') · T_g[y', y]
         T_g = P(当前立场 | 同一说话人上一次的立场)，在 train 上用人工标签统计，
         按两次发言的深度差 g 分桶（1 = 回复自己，2 = A-B-A，3–4，≥5），样本少于 min_count 时退回全局
  没有同说话人祖先的节点不使用 p_spk；仍然只用祖先信息（因果设定）。

λr、λa、λs 在 valid 上联合选择（话题平均 MacroF1，各话题共用；同分取 λ 之和更小的），test 只用于报告。
同时报告 λs=0（即现有的相对 + 绝对组合）和只用说话人知识两种特例，各自也在 valid 上选择。

依赖实例文件中的 same_speaker_ancestor_ids 字段（convert_zscsd.py 生成）。

用法（make_rag_data 目录下）：
python -u spin/experiments/negative/analyze_speaker_prior.py \
  --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --pred zscsd_exp/outputs/gpt4omini/predictions_knn_fewshot_test.jsonl \
  --valid_pred zscsd_exp/outputs/gpt4omini/predictions_knn_fewshot_valid.jsonl \
  --output_dir zscsd_exp/outputs/gpt4omini/speaker_prior
"""
import argparse
import importlib.util
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_TOOLS = (Path(__file__).resolve().parent / "../../spin_core").resolve()
_s = importlib.util.spec_from_file_location("cd", _TOOLS / "analyze_conflict_detection.py")
cd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cd)
kc, kbm, dd = cd.kc, cd.kbm, cd.dd
LABELS, LID = dd.LABELS, dd.LID


def gap_bucket(g):
    return "1" if g <= 1 else "2" if g == 2 else "3-4" if g <= 4 else "5+"


class SpeakerKB:
    """P(y_i | y_j)：同一说话人前一次发言 j 的立场 → 本次立场，按深度差分桶。"""

    def __init__(self, train, spk_of, min_count):
        self.C = defaultdict(lambda: np.zeros((3, 3)))
        for n in train:
            j = spk_of.get((n["topic"], n["node_id"]))
            if not j:
                continue
            jn, g = j
            if jn["y"] not in LID or n["y"] not in LID:
                continue
            for b in (gap_bucket(g), "all"):
                self.C[b][LID[jn["y"]], LID[n["y"]]] += 1
        self.m = min_count

    def T(self, g):
        c = self.C.get(gap_bucket(g))
        if c is None or c.sum() < self.m:
            c = self.C["all"]
        c = c + 0.5
        return c / c.sum(axis=1, keepdims=True)

    def describe(self):
        print("【train 中的同说话人立场转移 P(本次 | 上一次)】行 = 上一次，列 = 本次 (favor / against / none)")
        for b in ("1", "2", "3-4", "5+", "all"):
            c = self.C.get(b)
            if c is None or c.sum() == 0:
                continue
            same = np.trace(c) / c.sum()
            p = (c + 0.5) / (c + 0.5).sum(axis=1, keepdims=True)
            print(f"  深度差 {b:4s} n={int(c.sum()):5d}  与上一次相同 {same:.1%}  " +
                  "  ".join(f"{LABELS[i]}→[" + " ".join(f"{x:.2f}" for x in p[i]) + "]" for i in range(3)))


def rel_matrix(m, n):
    pr, pe = m.rel_proba(n), m.express_proba(n)
    return np.array([[pr["inherit"], pr["reverse"], pe * 0.5],
                     [pr["reverse"], pr["inherit"], pe * 0.5],
                     [pr["neutralize"], pr["neutralize"], 1 - pe]])


def prepare(ev, base, ms, E, skb, spk_of):
    """把每个节点用到的量预先算好，网格搜索时只做向量乘法。"""
    order = sorted(ev, key=lambda x: x["depth"])
    items = []
    for n in order:
        k = (n["topic"], n["node_id"])
        m = ms[n["topic"]]
        j = spk_of.get(k)
        items.append({
            "k": k, "pk": n["parent_key"], "e": E[n["topic"]][:, LID[base[k]]],
            "M": rel_matrix(m, n), "abs": m.direct_proba(n),
            "jk": (j[0]["topic"], j[0]["node_id"]) if j else None, "T": skb.T(j[1]) if j else None,
        })
    return items


def forward(items, base, lr, la, ls):
    q, lab = {}, {}
    for it in items:
        post = it["e"].copy()
        used = False
        if la > 0:
            post = post * it["abs"] ** la; used = True
        qp = q.get(it["pk"]) if it["pk"] else None
        if qp is not None and lr > 0:
            pr = it["M"] @ qp
            post = post * (pr / pr.sum()) ** lr; used = True
        qj = q.get(it["jk"]) if it["jk"] else None
        if qj is not None and ls > 0:
            ps = qj @ it["T"]
            post = post * (ps / ps.sum()) ** ls; used = True
        post = post / post.sum()
        q[it["k"]] = post
        lab[it["k"]] = LABELS[int(post.argmax())] if used else base[it["k"]]
    return lab


def topic_avg(ev, lab, topics):
    return float(np.mean([cd.scores_of([n["y"] for n in ev if n["topic"] == t],
                                       [lab[(n["topic"], n["node_id"])] for n in ev if n["topic"] == t])[0]
                          for t in topics]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--kb", default="full", choices=list(cd.KB_KEYS))
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2,3,4,6")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--output_dir", default="speaker_prior")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    raw = {}
    for line in open(a.instances, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            raw[(str(r.get("topic")), str(r.get("node_id")))] = r
    spk_of = {}
    for k, n in nodes.items():
        ids = raw.get(k, {}).get("same_speaker_ancestor_ids") or []
        jk = (k[0], str(ids[-1])) if ids else None
        if jk and jk in nodes:
            spk_of[k] = (nodes[jk], n["depth"] - nodes[jk]["depth"])
    if not spk_of:
        print("实例文件中没有 same_speaker_ancestor_ids（或全为空），无法使用说话人知识。")
        return

    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    base, _ = kc.load_pred_conf(a.pred, a.pred_field, a.conf_field)
    vbase, vconf = kc.load_pred_conf(a.valid_pred, a.pred_field, a.conf_field)
    ev = [n for k, n in nodes.items() if k in base and n["split"] == "test" and n["y"]]
    vev = [n for k, n in nodes.items() if k in vbase and n["split"] == "valid" and n["y"]]
    ms = {t: kbm.PatternKB([n for n in train if n["topic"] == t], cd.KB_KEYS[a.kb], a.min_count) for t in topics}
    one = kc.make_binner(None)
    E = {t: kc.confusion([n for n in vev if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}
    skb = SpeakerKB(train, spk_of, a.min_count)
    print(f"train {len(train)} | test {len(ev)} | valid {len(vev)} | KB:{a.kb}")
    print(f"有同说话人祖先的节点：train {sum(1 for n in train if (n['topic'], n['node_id']) in spk_of)}，"
          f"valid {sum(1 for n in vev if (n['topic'], n['node_id']) in spk_of)}，"
          f"test {sum(1 for n in ev if (n['topic'], n['node_id']) in spk_of)}")
    # 对照：父子立场相同的比例
    par_same = [n["y"] == n["pa_gold"] for n in train if n["pa_gold"] in LID]
    print(f"对照：train 中当前立场与父节点相同 {np.mean(par_same):.1%}（n={len(par_same)}）")
    skb.describe()

    vitems, titems = prepare(vev, vbase, ms, E, skb, spk_of), prepare(ev, base, ms, E, skb, spk_of)
    t0 = time.time()
    grid = {c: topic_avg(vev, forward(vitems, vbase, *c), topics) for c in itertools.product(lams, lams, lams)}
    print(f"\nvalid 网格搜索 {len(grid)} 组，用时 {time.time() - t0:.0f} 秒")

    def pick(cands):
        return max(cands, key=lambda c: (grid[c], -sum(c)))

    choices = {
        "只用说话人知识": pick([c for c in grid if c[0] == 0 and c[1] == 0]),
        "相对 + 绝对（λs=0）": pick([c for c in grid if c[2] == 0]),
        "相对 + 绝对 + 说话人": pick(list(grid)),
    }
    g = [n["y"] for n in ev]
    keys = [(n["topic"], n["node_id"]) for n in ev]
    has = [k in spk_of for k in keys]
    b_lab = {k: base[k] for k in keys}

    def line(name, lab, extra=""):
        mf, fa, _ = cd.scores_of(g, [lab[k] for k in keys])
        sub = []
        for flag, nm in ((True, "同说话人"), (False, "新说话人")):
            gg = [y for y, h in zip(g, has) if h == flag]
            pp = [lab[k] for k, h in zip(keys, has) if h == flag]
            sub.append(f"{nm} {cd.scores_of(gg, pp)[0]:.4f}")
        print(f"  {name:22s} MacroF1 {mf:.4f}  F_avg {fa:.4f}  话题平均MacroF1 {topic_avg(ev, lab, topics):.4f}  "
              f"[{' / '.join(sub)}]{extra}")

    print("\n【test 结果】（λ 均在 valid 上选择；方括号内为同说话人 / 新说话人节点上的 MacroF1）")
    line("基础预测", b_lab)
    preds = {}
    for name, c in choices.items():
        lab = forward(titems, base, *c)
        fx = sum(lab[k] == n["y"] != base[k] for k, n in zip(keys, ev))
        br = sum(base[k] == n["y"] != lab[k] for k, n in zip(keys, ev))
        line(name, lab, f"  改对 {fx} 改错 {br}  λr={c[0]:g} λa={c[1]:g} λs={c[2]:g}（valid {grid[c]:.4f}）")
        preds[name] = lab

    best = choices["相对 + 绝对 + 说话人"]
    print(f"\n【valid：在选出的 λr={best[0]:g}、λa={best[1]:g} 下，λs 的影响】")
    print("  " + "  ".join(f"λs={ls:g}:{grid[(best[0], best[1], ls)]:.4f}" for ls in lams))

    full, comb = preds["相对 + 绝对 + 说话人"], preds["相对 + 绝对（λs=0）"]
    with open(out / "pred_speaker.jsonl", "w", encoding="utf-8") as f:
        for k, n, h in zip(keys, ev, has):
            f.write(json.dumps({"topic": k[0], "node_id": k[1], "depth": n["depth"], "stance_label": n["y"],
                                "has_same_speaker_ancestor": h,
                                "predicted_stance": full[k], "combined_prediction": comb[k],
                                "speaker_only_prediction": preds["只用说话人知识"][k],
                                "base_prediction": base[k]}, ensure_ascii=False) + "\n")
    print(f"\n已保存 {out / 'pred_speaker.jsonl'}")
    print("显著性：--pred_field_b 可用 combined_prediction / base_prediction / speaker_only_prediction；"
          "--group_field has_same_speaker_ancestor 可按是否同说话人分组")


if __name__ == "__main__":
    main()