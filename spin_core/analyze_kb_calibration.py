# -*- coding: utf-8 -*-
"""
analyze_kb_calibration.py
离线检验（不调用 API）：跨话题时，语义传导知识库"排序对、概率水平不对"的问题，
能否用目标话题的【无标注】数据做基准率校准来解决？

问题：跨话题知识库判断父子关系比多数类高约 10 个点，但会话树推断时没有收益。
假设：不同话题的立场基准率差异很大（如 none 占比 3.7% ~ 40%），源话题知识库的概率水平与目标话题不匹配。

校准（标签分布偏移校正，Saerens et al. 2002 的 EM 方法）：
  1. 用目标话题 test 节点上 LLM 的预测（不看 gold），结合 LLM 的混淆矩阵，按深度用 EM 估计目标话题的立场基准率 π_t
  2. 源话题基准率 π_s：知识库训练数据（其他话题 train）按深度统计
  3. 知识库给出的先验乘以 π_t / π_s 再归一化

对比（均为跨话题知识库，即目标话题的 train 完全不用）：
  不校准 / EM 校准（无监督）/ 真实基准率校准（上界：用目标话题 gold 分布，说明校准能带来多少）
  另设对照"仅校准（无知识库）"：只用 π_t 做先验，检验收益是否来自知识库本身
  并列出话题内知识库（in_topic）作为参照

两种协议：
  严格零样本：LLM 混淆矩阵与 λ 都只在其他话题上估计/选择，目标话题的 gold 完全不参与
  交叉拟合：与之前脚本一致，λ 在目标话题另一半会话树上选（会用到目标话题的部分 gold，仅作参照）

用法：
python -u spin/spin_core/analyze_kb_calibration.py \
  --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \
  --pred dynamic_semantic=rag_prediction_outputs/predictions_dynamic_semantic_test.jsonl \
  --output_dir kb_calibration_analysis
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("kbm", _here / "analyze_propagation_kb.py")
kbm = importlib.util.module_from_spec(_s)
_s.loader.exec_module(kbm)
dd = kbm.dd
LABELS, LID = dd.LABELS, dd.LID


# --------------------------------------------------------------------------- 基准率
def dist_by_depth(nodes_list, label_fn, min_n):
    """按深度统计立场分布；样本不足时回退到整体分布。"""
    allc = Counter(label_fn(n) for n in nodes_list)
    overall = np.array([allc[l] + 1 for l in LABELS], float)
    overall /= overall.sum()
    by = defaultdict(Counter)
    for n in nodes_list:
        by[dd.depth_bucket(n["depth"])][label_fn(n)] += 1
    out = {}
    for b in dd.DEPTH_BUCKETS:
        c = by.get(b, Counter())
        tot = sum(c.values())
        out[b] = (np.array([c[l] + 1 for l in LABELS], float) / (tot + 3)) if tot >= min_n else overall
    return out, overall


def em_prior(preds, C, iters=200):
    """Saerens EM：preds 为预测标签列表，C[y, p] = P(预测 p | 真实 y)。返回估计的真实标签分布。"""
    if not preds:
        return np.ones(3) / 3
    lik = np.array([C[:, LID[p]] for p in preds])          # n × 3
    pi = np.ones(3) / 3
    for _ in range(iters):
        post = lik * pi
        post /= post.sum(1, keepdims=True)
        new = post.mean(0)
        if np.abs(new - pi).max() < 1e-7:
            break
        pi = new
    return pi


def em_by_depth(nodes_list, base, C, min_n):
    overall = em_prior([base[(n["topic"], n["node_id"])] for n in nodes_list], C)
    by = defaultdict(list)
    for n in nodes_list:
        by[dd.depth_bucket(n["depth"])].append(base[(n["topic"], n["node_id"])])
    return {b: (em_prior(by[b], C) if len(by.get(b, [])) >= min_n else overall) for b in dd.DEPTH_BUCKETS}, overall


# --------------------------------------------------------------------------- 模型包装
class Calibrated:
    """把知识库给出的先验乘以 π_t / π_s（按深度）。"""

    def __init__(self, kb, pi_t, pi_s, clip=5.0):
        self.kb, self.r = kb, {b: np.clip(pi_t[b] / pi_s[b], 1 / clip, clip) for b in pi_t}
        self.r_major = getattr(kb, "r_major", "inherit")

    def _adj(self, n, p):
        p = p * self.r[dd.depth_bucket(n["depth"])]
        return p / p.sum()

    def prior(self, n, qp):
        return self._adj(n, self.kb.prior(n, qp))

    def direct_proba(self, n):
        return self._adj(n, self.kb.direct_proba(n))

    def rel_proba(self, n, **kw):
        return self.kb.rel_proba(n)

    def precompute(self, nodes):
        pass


class PriorOnly:
    """对照：没有知识库，只用估计的目标基准率作为先验（经典的标签偏移校正）。"""

    def __init__(self, pi_t):
        self.pi = pi_t
        self.r_major = "inherit"

    def prior(self, n, qp):
        return self.pi[dd.depth_bucket(n["depth"])]

    def direct_proba(self, n):
        return self.pi[dd.depth_bucket(n["depth"])]

    def precompute(self, nodes):
        pass


# --------------------------------------------------------------------------- 推断
def smooth_e(ev, base, models, lam, emis, mode):
    """与 analyze_depth_dynamics.smooth 相同，但观测似然由 emis(n) 给出（允许按目标话题使用不同的混淆矩阵）。"""
    q, out = {}, {}
    for n in sorted(ev, key=lambda x: x["depth"]):
        key = (n["topic"], n["node_id"])
        pred = base[key]
        e = emis(n)[:, LID[pred]]
        qp = None
        if mode == "gold":
            if n["pa_gold"] in LID:
                qp = np.eye(3)[LID[n["pa_gold"]]]
        elif n["parent_key"] in q:
            qp = q[n["parent_key"]]
        if qp is None:
            post, lab = e / e.sum(), pred
        else:
            m = models[n["topic"]]
            prior = m.direct_proba(n) if mode == "noparent" else m.prior(n, qp)
            post = e * prior ** lam
            post /= post.sum()
            lab = LABELS[int(post.argmax())] if lam > 0 else pred
        q[key], out[key] = post, lab
    return out


def per_topic_macro(ev, lab, topics):
    return {t: dd.scores([n["y"] for n in ev if n["topic"] == t],
                         [lab[(n["topic"], n["node_id"])] for n in ev if n["topic"] == t])["macro_f1"] for t in topics}


def select(ev, grid, lams, topics, protocol):
    """protocol = 'loto'：每个目标话题的 λ 只在其他话题上选；'crossfit'：在目标话题另一半会话树上选。"""
    lab, chosen = {}, {}
    if protocol == "loto":
        pt = {l: per_topic_macro(ev, grid[l], topics) for l in lams}
        for t in topics:
            best = max(lams, key=lambda l: np.mean([pt[l][o] for o in topics if o != t]))
            chosen[t] = best
            for n in ev:
                if n["topic"] == t:
                    lab[(n["topic"], n["node_id"])] = grid[best][(n["topic"], n["node_id"])]
        desc = "/".join(f"{chosen[t]:g}" for t in topics)
    else:
        for k in (0, 1):
            other = [n for n in ev if dd.fold_of(n) != k]
            best = max(lams, key=lambda l: dd.topic_macro(
                [(n["topic"], n["y"], grid[l][(n["topic"], n["node_id"])]) for n in other])["macro_f1"])
            chosen[k] = best
            for n in ev:
                if dd.fold_of(n) == k:
                    lab[(n["topic"], n["node_id"])] = grid[best][(n["topic"], n["node_id"])]
        desc = f"{chosen[0]:g}/{chosen[1]:g}"
    return lab, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2")
    ap.add_argument("--min_count", type=int, default=20, help="知识单元最少样本数")
    ap.add_argument("--min_depth_n", type=int, default=80, help="按深度估计基准率的最少样本数，不足则用话题整体")
    ap.add_argument("--output_dir", default="kb_calibration_analysis")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]

    nodes, _ = dd.load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    preds = {s.split("=", 1)[0]: dd.load_pred(s.split("=", 1)[1], a.pred_field) for s in a.pred}
    base_name = a.base or next(iter(preds))
    base = preds[base_name]
    ev = [n for k, n in nodes.items() if k in base and n["split"] == a.eval_split and n["y"]]
    print(f"train {len(train)} | {a.eval_split} 可评估 {len(ev)} | 基础方法 {base_name}")

    # ---- 混淆矩阵：严格协议下，目标话题 t 只用其他话题的 (gold, 预测) 估计
    def confusion(sub):
        cm = np.ones((3, 3))
        for n in sub:
            cm[LID[n["y"]], LID[base[(n["topic"], n["node_id"])]]] += 1
        return cm / cm.sum(1, keepdims=True)
    C_loto = {t: confusion([n for n in ev if n["topic"] != t]) for t in topics}
    E_fold = dd.emission_by_fold(ev, base)
    emis_loto = lambda n: C_loto[n["topic"]]
    emis_fold = lambda n: E_fold[dd.fold_of(n)]

    # ---- 基准率：源（其他话题 train）、EM 估计（目标话题无标注预测）、真实（目标话题 gold，仅作上界）
    print("\n【基准率估计】none 比例（源 = 其他话题 train；EM = 目标话题无标注估计；真实 = 目标话题 test gold）")
    pi_s, pi_em, pi_true, prows = {}, {}, {}, []
    for t in topics:
        src = [n for n in train if n["topic"] != t]
        tgt = [n for n in ev if n["topic"] == t]
        pi_s[t], s_all = dist_by_depth(src, lambda n: n["y"], a.min_depth_n)
        pi_em[t], em_all = em_by_depth(tgt, base, C_loto[t], a.min_depth_n)
        pi_true[t], tr_all = dist_by_depth(tgt, lambda n: n["y"], a.min_depth_n)
        raw = Counter(base[(n["topic"], n["node_id"])] for n in tgt)
        raw_none = raw["none"] / max(len(tgt), 1)
        prows.append({"topic": t, "源none": round(s_all[2], 3), "LLM原始预测none": round(raw_none, 3),
                      "EM估计none": round(em_all[2], 3), "真实none": round(tr_all[2], 3),
                      "EM估计(f/a/n)": "/".join(f"{x:.2f}" for x in em_all),
                      "真实(f/a/n)": "/".join(f"{x:.2f}" for x in tr_all)})
        print(f"  {t[:22]:22s} 源 {s_all[2]:.3f} | LLM原始预测 {raw_none:.3f} | EM {em_all[2]:.3f} | 真实 {tr_all[2]:.3f}"
              f"   （f/a/n：EM {prows[-1]['EM估计(f/a/n)']}，真实 {prows[-1]['真实(f/a/n)']}）")
    err_em = np.mean([np.abs(np.array([float(x) for x in r["EM估计(f/a/n)"].split("/")]) -
                             np.array([float(x) for x in r["真实(f/a/n)"].split("/")])).sum() / 2 for r in prows])
    err_src = np.mean([np.abs(pi_s[t][dd.DEPTH_BUCKETS[1]] - pi_true[t][dd.DEPTH_BUCKETS[1]]).sum() / 2 for t in topics])
    print(f"  EM 估计与真实分布的平均总变差距离 {err_em:.3f}（深度 1 处源分布与真实的距离 {err_src:.3f}，供参照）")
    dd.write_csv(out / "base_rates.csv", prows)

    # ---- 构建模型
    keyfns = {"KB:完整组合": kbm.k_full, "KB:传导类型+深度": kbm.k_sp_depth, "KB:仅深度": kbm.k_depth}
    configs = []   # (知识库, 来源, 校准, models, emis)
    for kname, kf in keyfns.items():
        cross = {t: kbm.PatternKB([n for n in train if n["topic"] != t], kf, a.min_count) for t in topics}
        configs.append((kname, "cross_topic", "不校准", cross))
        configs.append((kname, "cross_topic", "EM校准(无监督)", {t: Calibrated(cross[t], pi_em[t], pi_s[t]) for t in topics}))
        configs.append((kname, "cross_topic", "真实基准率校准(上界)", {t: Calibrated(cross[t], pi_true[t], pi_s[t]) for t in topics}))
    configs.append(("仅校准(无知识库)", "-", "EM校准(无监督)", {t: PriorOnly(pi_em[t]) for t in topics}))
    configs.append(("仅校准(无知识库)", "-", "真实基准率校准(上界)", {t: PriorOnly(pi_true[t]) for t in topics}))
    configs.append(("KB:完整组合", "in_topic(参照)", "不校准",
                    {t: kbm.PatternKB([n for n in train if n["topic"] == t], kbm.k_full, a.min_count) for t in topics}))

    # ---- 运行
    print(f"\n【会话树推断】修正 {base_name}；每个配置给出两种协议")
    rows = [dd.summarize(ev, base, base, {"knowledge": f"{base_name}(原始)", "source": "-", "calibration": "-",
                                          "usage": "-", "protocol": "-", "lambda": "-"})]
    usages = {"pred": "相对知识+预测父立场", "noparent": "绝对知识(不看父立场)"}
    for kname, src, cal, models in configs:
        for mode, mname in usages.items():
            if kname.startswith("仅校准") and mode == "noparent":
                continue
            for proto, emis in (("严格零样本", emis_loto), ("交叉拟合", emis_fold)):
                if src.startswith("in_topic") and proto == "严格零样本":
                    continue
                grid = {l: smooth_e(ev, base, models, l, emis, mode) for l in lams}
                lab, desc = select(ev, grid, lams, topics, "loto" if proto == "严格零样本" else "crossfit")
                r = dd.summarize(ev, lab, base, {"knowledge": kname, "source": src, "calibration": cal,
                                                 "usage": mname, "protocol": proto, "lambda": desc})
                rows.append(r)
                safe = f"{kname}_{src}_{cal}_{mode}_{proto}".replace(":", "_").replace("+", "_").replace("(", "_").replace(")", "")
                with open(out / f"pred_{safe}.jsonl", "w", encoding="utf-8") as f:
                    for n in ev:
                        key = (n["topic"], n["node_id"])
                        f.write(json.dumps({"topic": n["topic"], "node_id": n["node_id"], "depth": n["depth"],
                                            "stance_label": n["y"], "predicted_stance": lab[key],
                                            "base_prediction": base[key]}, ensure_ascii=False) + "\n")
        print(f"  完成 {kname} / {src} / {cal}")
    dd.write_csv(out / "kb_calibration_results.csv", rows)

    dcols = [c for c in rows[0] if c.startswith("depth") and c.endswith("_F_avg")]
    print(f"\n{'知识库':16s} {'来源':14s} {'校准':16s} {'用法':18s} {'协议':8s} {'λ':>20s} {'话题平均MacroF1':>14s} "
          f"{'话题平均F_avg':>12s} {'合并MacroF1':>10s} {'改对':>5s} {'改错':>5s} | " + " ".join(f"{c.replace('_F_avg', ''):>8s}" for c in dcols))
    for r in rows:
        print(f"{r['knowledge'][:16]:16s} {r['source'][:14]:14s} {r['calibration'][:16]:16s} {r['usage'][:18]:18s} {r['protocol']:8s} "
              f"{str(r['lambda'])[:20]:>20s} {r['话题平均MacroF1']:14.4f} {r['话题平均F_avg']:12.4f} {r['合并MacroF1']:10.4f} "
              f"{r['改对']:5d} {r['改错']:5d} | " + " ".join(f"{r.get(c, float('nan')):8.4f}" for c in dcols))
    print("  （严格零样本协议的 λ 按话题列出，顺序同话题列表；按深度的列为各深度合并口径 F_avg）")
    print(f"\n结果已保存到 {out.resolve()}")


if __name__ == "__main__":
    main()