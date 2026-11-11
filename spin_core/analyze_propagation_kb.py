# -*- coding: utf-8 -*-
"""
analyze_propagation_kb.py
离线检验（不调用 API）：把"语义传导模式"保存为可检索的知识库，
检索结果作为会话树推断的转移先验，能否修正 LLM 的立场预测？哪种检索键有效？

知识库（只用 train 的 gold 构建）：每个"模式单元" = 一个检索键 →
    相对变换分布  P(继承 / 反转 / 中立化 | 键)      —— 父评论有立场时
    表态分布      P(表态 / 不表态 | 键)             —— 父评论无立场时
    绝对立场分布  P(favor / against / none | 键)    —— 原设计存的内容，用作对照

检索键（比较哪种键能检索到有用的经验；均按从细到粗回退，样本数不足 --min_count 时退到下一级）：
    KB:传导类型          LLM 标注的 llm_semantic_propagation                  ← 原设计
    KB:传导类型+深度      传导类型 × 深度
    KB:完整组合          回复关系 × 传导类型 × 目标关系 × 深度
    KB:仅深度            只按深度（不含语义传导信息，检验传导类型是否有额外贡献）
    KB:随机键            用一个随机 train 样本的键检索（检验"检索到相关经验"是否重要）
    LR:表层特征          analyze_depth_dynamics.py 里的逻辑回归（参照）

每种知识库 × 两种来源（in_topic 同话题 train / cross_topic 留一话题）× 三种用法：
    相对知识 + 预测的父立场（主方法）
    绝对知识（原设计：不看父立场，直接用 P(立场 | 键) 作先验）
    相对知识 + gold 父立场（上界）
λ 与混淆矩阵按会话树 2 折交叉拟合，与 analyze_depth_dynamics.py 完全一致。

用法：
python -u spin/spin_core/analyze_propagation_kb.py \
  --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \
  --pred dynamic_semantic=rag_prediction_outputs/predictions_dynamic_semantic_test.jsonl \
  --output_dir propagation_kb_analysis
"""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "dd", Path(__file__).resolve().parent / "analyze_depth_dynamics.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

LABELS, RELS, LID = dd.LABELS, dd.RELS, dd.LID


# --------------------------------------------------------------------------- 检索键
def k_sp(n):
    return [("sp", n["sp"]), ("g",)]


def k_sp_depth(n):
    db = dd.depth_bucket(n["depth"])
    return [("spd", n["sp"], db), ("sp", n["sp"]), ("d", db), ("g",)]


def k_full(n):
    db = dd.depth_bucket(n["depth"])
    return [("c", n["rr"], n["sp"], n["tr"], db), ("rs", n["rr"], n["sp"], db),
            ("spd", n["sp"], db), ("sp", n["sp"]), ("d", db), ("g",)]


def k_depth(n):
    return [("d", dd.depth_bucket(n["depth"])), ("g",)]


KEYS = {"KB:传导类型": k_sp, "KB:传导类型+深度": k_sp_depth, "KB:完整组合": k_full, "KB:仅深度": k_depth}
LEVEL_NAMES = {"c": "完整组合", "rs": "回复关系+传导+深度", "spd": "传导类型+深度", "sp": "传导类型", "d": "仅深度", "g": "全局"}


class PatternKB:
    """语义传导模式知识库。接口与 analyze_depth_dynamics.Transition 相同，可直接接入会话树推断。"""

    def __init__(self, train, keyfn, min_count, eval_keyfn=None):
        self.keyfn, self.eval_keyfn, self.m = keyfn, eval_keyfn or keyfn, min_count
        self.R, self.N, self.A = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        for n in train:
            for k in keyfn(n):
                self.A[k][n["y"]] += 1
                if n["pa_gold"] in ("favor", "against"):
                    self.R[k][dd.relation(n["pa_gold"], n["y"])] += 1
                elif n["pa_gold"] == "none":
                    self.N[k]["express" if n["y"] != "none" else "stay_none"] += 1
        self.r_major = max(RELS, key=lambda r: self.R[("g",)].get(r, 0))
        self.hits = Counter()

    def _lookup(self, table, n, keys, count_hit=False):
        for k in self.eval_keyfn(n):
            c = table.get(k)
            if k == ("g",) or (c and sum(c.values()) >= self.m):
                if count_hit:
                    self.hits[k[0]] += 1
                c = c or Counter()
                tot = sum(c.values())
                return {x: (c.get(x, 0) + 0.5) / (tot + 0.5 * len(keys)) for x in keys}
        return {x: 1 / len(keys) for x in keys}

    def rel_proba(self, n, count_hit=False):
        return self._lookup(self.R, n, RELS, count_hit)

    def express_proba(self, n):
        return self._lookup(self.N, n, ["express", "stay_none"])["express"]

    def direct_proba(self, n):
        d = self._lookup(self.A, n, LABELS)
        return np.array([d[l] for l in LABELS])

    def prior(self, n, qp):
        pr, pe = self.rel_proba(n), self.express_proba(n)
        f, a, z = qp
        out = np.array([f * pr["inherit"] + a * pr["reverse"] + z * pe * 0.5,
                        a * pr["inherit"] + f * pr["reverse"] + z * pe * 0.5,
                        (f + a) * pr["neutralize"] + z * (1 - pe)])
        return out / out.sum()

    def precompute(self, nodes):
        pass


def random_keyfn(train, base_keyfn, seed_tag):
    """随机键：每个待检索节点固定映射到一个随机 train 样本，用它的键去检索。"""
    pool = [base_keyfn(n) for n in train]

    def f(n):
        h = int(hashlib.md5(f"{seed_tag}|{n['topic']}|{n['node_id']}".encode()).hexdigest(), 16)
        return pool[h % len(pool)]
    return f


# --------------------------------------------------------------------------- 主流程
MODES = {"pred": "相对知识+预测父立场", "noparent": "绝对知识(原设计,不看父立场)", "gold": "相对知识+gold父立场(上界)"}


def run(ev, nodes, base, ms, lams, E, out, vname, src, rows, save):
    for mode in MODES:
        grid = {lam: dd.smooth(ev, nodes, base, ms, lam, E, mode)[0] for lam in lams}
        chosen, lab_cf = {}, {}
        for k in (0, 1):
            other = [n for n in ev if dd.fold_of(n) != k]
            best = max(lams, key=lambda l: dd.topic_macro(
                [(n["topic"], n["y"], grid[l][(n["topic"], n["node_id"])]) for n in other])["macro_f1"])
            chosen[k] = best
            for n in ev:
                if dd.fold_of(n) == k:
                    key = (n["topic"], n["node_id"])
                    lab_cf[key] = grid[best][key]
        r = dd.summarize(ev, lab_cf, base, {"knowledge": vname, "memory_source": src, "usage": MODES[mode],
                                             "lambda": f"{chosen[0]}/{chosen[1]}"})
        r["λ网格(话题平均MacroF1)"] = " ".join(
            f"{l}:{dd.topic_macro([(n['topic'], n['y'], grid[l][(n['topic'], n['node_id'])]) for n in ev])['macro_f1']:.4f}"
            for l in lams)
        rows.append(r)
        if save and mode in ("pred", "noparent"):
            safe = vname.replace(":", "_").replace("+", "_")
            with open(out / f"pred_{safe}_{src}_{mode}_crossfit.jsonl", "w", encoding="utf-8") as f:
                for n in ev:
                    key = (n["topic"], n["node_id"])
                    f.write(json.dumps({"topic": n["topic"], "node_id": n["node_id"], "depth": n["depth"],
                                        "stance_label": n["y"], "predicted_stance": lab_cf[key],
                                        "base_prediction": base[key]}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="方法名=预测文件；第一个（或 --base）作为树修正的基础")
    ap.add_argument("--base", default=None)
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2")
    ap.add_argument("--min_count", type=int, default=20, help="知识单元最少样本数，不足则回退到更粗的键")
    ap.add_argument("--output_dir", default="propagation_kb_analysis")
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

    # ---- 知识库概况
    sp_train = Counter(n["sp"] for n in train)
    print(f"\n【知识库概况】train 中传导类型 {len(sp_train)} 种：" +
          ", ".join(f"{k}({v})" for k, v in sp_train.most_common(12)))
    unk = sum(1 for n in ev if n["sp"] == "unknown")
    if unk / max(len(ev), 1) > 0.2:
        print(f"  ⚠️ {unk}/{len(ev)} 个 test 节点缺少 llm_semantic_propagation 字段，请检查实例文件")
    kb_rows = []
    for sp, cnt in sp_train.most_common(15):
        g = [n for n in train if n["sp"] == sp]
        pol = [n for n in g if n["pa_gold"] in ("favor", "against")]
        rel = Counter(dd.relation(n["pa_gold"], n["y"]) for n in pol)
        ys = Counter(n["y"] for n in g)
        kb_rows.append({"传导类型": sp, "n": cnt, "父有立场n": len(pol),
                        **{f"P({r})": round(rel[r] / max(len(pol), 1), 3) for r in RELS},
                        **{f"P({l})": round(ys[l] / cnt, 3) for l in LABELS}})
    dd.write_csv(out / "kb_units_by_propagation.csv", kb_rows)
    print(f"  {'传导类型':28s} {'n':>6s} {'继承':>6s} {'反转':>6s} {'中立化':>6s} | {'favor':>6s} {'against':>7s} {'none':>6s}")
    for r in kb_rows:
        print(f"  {r['传导类型'][:28]:28s} {r['n']:6d} {r['P(inherit)']:6.3f} {r['P(reverse)']:6.3f} {r['P(neutralize)']:6.3f}"
              f" | {r['P(favor)']:6.3f} {r['P(against)']:7.3f} {r['P(none)']:6.3f}")

    # ---- 构建各知识库
    variants = {}
    for vname, kf in KEYS.items():
        variants[vname] = {
            "in_topic": {t: PatternKB([n for n in train if n["topic"] == t], kf, a.min_count) for t in topics},
            "cross_topic": {t: PatternKB([n for n in train if n["topic"] != t], kf, a.min_count) for t in topics}}
    variants["KB:随机键"] = {
        "in_topic": {t: PatternKB([n for n in train if n["topic"] == t], k_sp_depth, a.min_count,
                                  random_keyfn([n for n in train if n["topic"] == t], k_sp_depth, "in" + t)) for t in topics},
        "cross_topic": {t: PatternKB([n for n in train if n["topic"] != t], k_sp_depth, a.min_count,
                                     random_keyfn([n for n in train if n["topic"] != t], k_sp_depth, "x" + t)) for t in topics}}
    variants["LR:表层特征"] = {
        "in_topic": {t: dd.Transition([n for n in train if n["topic"] == t]) for t in topics},
        "cross_topic": {t: dd.Transition([n for n in train if n["topic"] != t]) for t in topics}}
    for v in variants.values():
        for ms in v.values():
            for t in topics:
                ms[t].precompute([n for n in ev if n["topic"] == t])

    # ---- 关系检索准确率（gold 父立场）+ 命中层级
    print("\n【检索质量】在 test 上用 gold 父立场评估：检索到的知识单元预测的关系是否正确（LLM 自行判断约 0.40）")
    qrows = []
    for vname, v in variants.items():
        for src, ms in v.items():
            if isinstance(next(iter(ms.values())), PatternKB):
                for t in topics:
                    ms[t].hits = Counter()
                for n in ev:
                    if n["pa_gold"] in ("favor", "against"):
                        ms[n["topic"]].rel_proba(n, count_hit=True)
            r = dd.eval_transition({src: ms}, ev, topics)[0]
            r = {"knowledge": vname, **r}
            if isinstance(next(iter(ms.values())), PatternKB):
                h = sum((ms[t].hits for t in topics), Counter())
                tot = sum(h.values()) or 1
                r["命中层级"] = ", ".join(f"{LEVEL_NAMES.get(k, k)} {v / tot:.0%}" for k, v in h.most_common())
            qrows.append(r)
            print(f"  {vname:16s} {src:12s} 关系准确率 {r['关系准确率']:.4f}（多数类 {r['多数类关系准确率']:.4f}）"
                  f" 关系macroF1 {r['关系macroF1']:.4f}  {r.get('命中层级', '')}")
    dd.write_csv(out / "kb_retrieval_quality.csv", qrows)

    # ---- 会话树推断
    print(f"\n【会话树推断】修正 {base_name}（交叉拟合 λ）")
    E = dd.emission_by_fold(ev, base)
    rows = [dd.summarize(ev, base, base, {"knowledge": f"{base_name}(原始)", "memory_source": "-", "usage": "-", "lambda": "-"})]
    for vname, v in variants.items():
        for src, ms in v.items():
            run(ev, nodes, base, ms, lams, E, out, vname, src, rows, save=True)
            print(f"  完成 {vname} / {src}")
    dd.write_csv(out / "kb_tree_inference.csv", rows)

    dcols = [c for c in rows[0] if c.startswith("depth") and c.endswith("_F_avg")]
    print(f"\n{'知识库':16s} {'来源':12s} {'用法':26s} {'λ':>9s} {'话题平均MacroF1':>14s} {'话题平均F_avg':>12s} "
          f"{'合并MacroF1':>10s} {'改对':>5s} {'改错':>5s} | " + " ".join(f"{c.replace('_F_avg', ''):>8s}" for c in dcols))
    for r in rows:
        print(f"{r['knowledge'][:16]:16s} {r['memory_source']:12s} {r['usage'][:26]:26s} {str(r['lambda']):>9s} "
              f"{r['话题平均MacroF1']:14.4f} {r['话题平均F_avg']:12.4f} {r['合并MacroF1']:10.4f} {r['改对']:5d} {r['改错']:5d} | "
              + " ".join(f"{r.get(c, float('nan')):8.4f}" for c in dcols))
    print("  （按深度的列为各深度合并口径 F_avg）")

    print("\n  λ 网格（相对知识 + 预测父立场，话题平均 MacroF1）")
    for r in rows[1:]:
        if r["usage"] == MODES["pred"]:
            print(f"  {r['knowledge'][:16]:16s} {r['memory_source']:12s} {r['λ网格(话题平均MacroF1)']}")

    print(f"\n结果已保存到 {out.resolve()}")
    print("显著性检验示例（某知识库 vs 原始预测）：python spin/experiments/main/paired_significance.py "
          f"--a {out}/pred_KB_传导类型_深度_in_topic_pred_crossfit.jsonl --b {out}/pred_KB_传导类型_深度_in_topic_pred_crossfit.jsonl "
          "--pred_field predicted_stance --pred_field_b base_prediction --id_field topic node_id --gold_field stance_label --group_field topic")


if __name__ == "__main__":
    main()