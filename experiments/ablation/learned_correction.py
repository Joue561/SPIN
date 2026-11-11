# -*- coding: utf-8 -*-
"""
learned_correction.py  (A2: learned-correction baselines vs. SPIN; offline, no API)

Same inputs as SPIN — base prediction, channel E, retrieved relative prior p_rel (from the parent's
posterior), retrieved absolute distribution P_abs, key fields — but the combination is LEARNED on
the validation split instead of SPIN's two exponents:

  LR-log : multinomial logistic regression on [log E[:,ŷ], log p_rel, log P_abs, has_parent].
           SPIN is the special case with weights (1, λr, λa) shared across classes and no bias.
  GBDT   : gradient-boosted trees on the above plus base one-hot, parent posterior, and the
           one-hot key fields (reply relation, propagation type, target relation, depth bucket).

Parent posteriors: like SPIN, nodes are processed by depth and each node uses the model's own
posterior for its parent. Training features use out-of-fold posteriors (5 folds grouped by tree)
so that the model is trained on parent posteriors of the same quality it sees at test time.
Hyperparameters are fixed in advance (no tuning on test).

Put this file in cross_topic_exp/tools/ (it reuses analyze_combined_prior.py). Usage:
python -u spin/experiments/ablation/learned_correction.py \
  --instances zscsd_exp/data/zscsd_semantic_instances_llm.jsonl \
  --pred zscsd_exp/outputs/gpt4omini/predictions_knn_fewshot_test.jsonl \
  --valid_pred zscsd_exp/outputs/gpt4omini/predictions_knn_fewshot_valid.jsonl \
  --output_dir zscsd_exp/outputs/gpt4omini/learned_correction
Output: pred_learned.jsonl with predicted_stance (=LR-log), gbdt_prediction, spin_prediction, base_prediction,
for paired_significance.py / collect_paper_numbers.py.
"""
import argparse
import importlib.util
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

_s = importlib.util.spec_from_file_location("acp", (Path(__file__).resolve().parent / "../../spin_core/analyze_combined_prior.py").resolve())
acp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(acp)
cd, kc, kbm, dd = acp.cd, acp.kc, acp.kbm, acp.dd
LABELS, LID = acp.LABELS, acp.LID
EPS = 1e-6


def build_vocab(nodes):
    v = {}
    for f in ("rr", "sp", "tr"):
        for x in sorted({n[f] for n in nodes}):
            v[(f, x)] = len(v)
    for b in range(5):
        v[("db", b)] = len(v)
    return v


def features(n, qp, yhat, m, E, vocab, rich):
    e = E[n["topic"]][:, LID[yhat]]
    e = e / e.sum()
    pa = m[n["topic"]].direct_proba(n)
    hp = qp is not None
    pr = m[n["topic"]].prior(n, qp) if hp else np.ones(3) / 3
    x = [*np.log(e + EPS), *np.log(pr + EPS), *np.log(pa + EPS), float(hp)]
    if rich:
        oh = np.zeros(len(vocab))
        for f in ("rr", "sp", "tr"):
            if (f, n[f]) in vocab:
                oh[vocab[(f, n[f])]] = 1
        oh[vocab[("db", min(int(n["depth"]), 4))]] = 1
        base_oh = np.eye(3)[LID[yhat]]
        x += [*base_oh, *(qp if hp else np.zeros(3)), *oh]
    return np.array(x, dtype=float)


def forward_learned(ev, base, model, m, E, vocab, rich, init_q=None):
    """Depth-ordered pass; each node uses the model's own posterior for its parent."""
    q = dict(init_q or {})
    by_depth = defaultdict(list)
    for n in ev:
        by_depth[n["depth"]].append(n)
    for d in sorted(by_depth):
        layer = by_depth[d]
        X = np.stack([features(n, q.get(n["parent_key"]) if n["parent_key"] else None,
                               base[(n["topic"], n["node_id"])], m, E, vocab, rich) for n in layer])
        P = model.predict_proba(X)
        cls = [int(c) for c in model.classes_]          # classes are label indices (LID)
        for n, p in zip(layer, P):
            full = np.array([p[cls.index(i)] if i in cls else 0.0 for i in range(len(LABELS))]) + EPS
            q[(n["topic"], n["node_id"])] = full / full.sum()
    return q


def make_model(rich, seed):
    if rich:
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                              l2_regularization=1.0, random_state=seed)
    return LogisticRegression(C=1.0, max_iter=3000)


def proxy_posterior(n, base, E):
    e = E[n["topic"]][:, LID[base[(n["topic"], n["node_id"])]]]
    return e / e.sum()


def fit_with_oof(vev, vbase, m, E, vocab, rich, folds=5, seed=0):
    """Round 1: parent posterior = E-proxy of the parent's base prediction.
    Round 2: parent posterior = out-of-fold posterior of the round-1 model (grouped by tree)."""
    vkey = {(n["topic"], n["node_id"]): n for n in vev}
    y = np.array([LID[n["y"]] for n in vev])

    def feats(qmap):
        return np.stack([features(n, qmap.get(n["parent_key"]) if n["parent_key"] in vkey else None,
                                  vbase[(n["topic"], n["node_id"])], m, E, vocab, rich) for n in vev])

    proxy = {k: proxy_posterior(n, vbase, E) for k, n in vkey.items()}
    m1 = make_model(rich, seed).fit(feats(proxy), y)
    roots = sorted({(n["topic"], n["root"]) for n in vev})
    random.Random(seed).shuffle(roots)
    fold_of = {r: i % folds for i, r in enumerate(roots)}
    oof = {}
    for f in range(folds):
        tr = [i for i, n in enumerate(vev) if fold_of[(n["topic"], n["root"])] != f]
        te = [n for n in vev if fold_of[(n["topic"], n["root"])] == f]
        mf = make_model(rich, seed).fit(feats(proxy)[tr], y[tr])
        oof.update(forward_learned(te, vbase, mf, m, E, vocab, rich))
    return make_model(rich, seed).fit(feats(oof), y), m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--valid_pred", required=True)
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2,3,4,6")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--conf_field", default="confidence")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="learned_correction")
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
    m = {t: kbm.PatternKB([n for n in train if n["topic"] == t], cd.KB_KEYS["full"], a.min_count) for t in topics}
    one = kc.make_binner(None)
    E = {t: kc.confusion([n for n in vev if n["topic"] == t], vbase, vconf, one, 1)[0] for t in topics}
    vocab = build_vocab(train + vev + ev)
    print(f"train {len(train)} | valid {len(vev)} | test {len(ev)} | topics {len(topics)}")

    # SPIN (same protocol as analyze_combined_prior.py)
    emis = lambda n: E[n["topic"]]
    grid = {c: acp.topic_avg(vev, acp.forward(vev, vbase, m, emis, *c), topics) for c in itertools.product(lams, lams)}
    lr_, la_ = max(grid, key=lambda c: (grid[c], -(c[0] + c[1])))
    spin = acp.forward(ev, base, m, emis, lr_, la_)

    keys = [(n["topic"], n["node_id"]) for n in ev]
    g = [n["y"] for n in ev]
    res = {"基础预测": {k: base[k] for k in keys}, f"SPIN（λr={lr_:g}, λa={la_:g}）": spin}
    for name, rich in [("LR-log（学习式组合）", False), ("GBDT（学习式，含键特征）", True)]:
        model, _ = fit_with_oof(vev, vbase, m, E, vocab, rich, seed=a.seed)
        vq = forward_learned(vev, vbase, model, m, E, vocab, rich)
        vlab = {k: LABELS[int(np.argmax(p))] for k, p in vq.items()}
        vscore = acp.topic_avg(vev, vlab, topics)
        q = forward_learned(ev, base, model, m, E, vocab, rich)
        res[name] = {k: LABELS[int(np.argmax(q[k]))] for k in keys}
        print(f"  {name}: valid 话题平均MacroF1 {vscore:.4f}（训练集内，偏乐观）")
        if not rich:
            coef = model.coef_
            print("    LR 系数（行=favor/against/none 的类别权重；列=logE×3, log p_rel×3, log P_abs×3, has_parent）")
            for c, row in zip(model.classes_, coef):
                print(f"    {LABELS[int(c)]:8s} " + " ".join(f"{v:+.2f}" for v in row))
    print(f"\n  SPIN valid 话题平均MacroF1 {grid[(lr_, la_)]:.4f}")

    print("\n【test】")
    for name, lab in res.items():
        mf, fa, _ = cd.scores_of(g, [lab[k] for k in keys])
        ta = acp.topic_avg(ev, lab, topics)
        fx = sum(lab[k] == n["y"] != base[k] for k, n in zip(keys, ev))
        br = sum(base[k] == n["y"] != lab[k] for k, n in zip(keys, ev))
        print(f"  {name:28s} 合并MacroF1 {mf:.4f}  合并F_avg {fa:.4f}  话题平均MacroF1 {ta:.4f}  改对 {fx:4d} 改错 {br:4d}")

    names = list(res)
    with open(out / "pred_learned.jsonl", "w", encoding="utf-8") as f:
        for k, n in zip(keys, ev):
            f.write(json.dumps({"topic": k[0], "node_id": k[1], "stance_label": n["y"],
                                "predicted_stance": res[names[2]][k], "gbdt_prediction": res[names[3]][k],
                                "spin_prediction": res[names[1]][k], "base_prediction": base[k]},
                               ensure_ascii=False) + "\n")
    print(f"\n已保存 {out / 'pred_learned.jsonl'}（predicted_stance = LR-log）")
    print("显著性：paired_significance.py --pred_field predicted_stance|gbdt_prediction --pred_field_b spin_prediction|base_prediction")


if __name__ == "__main__":
    main()