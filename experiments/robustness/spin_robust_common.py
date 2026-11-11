# -*- coding: utf-8 -*-
"""
spin_robust_common.py —— 观测信道鲁棒性 / 转移矩阵对称性实验的公共函数。

只复用现有 SPIN 代码中已经被 learned_correction.py 用过的接口，不修改 SPIN 本身：
  acp.forward(ev, base, kb, emis, λr, λa) -> {key: label}
  acp.topic_avg(ev, labels, topics)       -> 话题平均 Macro-F1（λ 选择准则）
  kb.prior(n, q_parent) / kb.direct_proba(n)
  kc.confusion(nodes, base, conf, binner, 1)[0] -> E（行=真实立场 y，列=预测 ŷ）
  kc.load_pred_conf / dd.load_instances / cd.KB_KEYS / cd.scores_of

数据协议：E 与 λ 只用验证集；测试集标签只用于最终评测与标注为“诊断/oracle”的离线分析。
"""
import importlib.util
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------- loading
def load_acp(tools_dir):
    p = Path(tools_dir) / "analyze_combined_prior.py"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}；请用 --tools_dir 指向 analyze_combined_prior.py 所在目录")
    s = importlib.util.spec_from_file_location("acp", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class Run:
    """一个 (数据集, 基础模型) 组合：节点、验证/测试预测、按话题的 KB 与 E、官方 λ。"""

    def __init__(self, acp, instances, valid_pred, test_pred, pred_field="predicted_stance",
                 conf_field="confidence", parent_field="parent_stance_label", min_count=20,
                 lambdas="0,0.25,0.5,0.75,1,1.5,2,3,4,6", name="run"):
        self.acp, self.name = acp, name
        cd, kc, kbm, dd = acp.cd, acp.kc, acp.kbm, acp.dd
        self.LABELS, self.LID = list(acp.LABELS), acp.LID
        self.lams = [float(x) for x in str(lambdas).split(",")]
        self.min_count = min_count
        self.nodes, _ = dd.load_instances(instances, parent_field)
        self.topics = sorted({n["topic"] for n in self.nodes.values()})
        self.train = [n for n in self.nodes.values() if n["split"] == "train" and n["y"]]
        self.base, self.conf = kc.load_pred_conf(test_pred, pred_field, conf_field)
        self.vbase, self.vconf = kc.load_pred_conf(valid_pred, pred_field, conf_field)
        self.ev = [n for k, n in self.nodes.items() if k in self.base and n["split"] == "test" and n["y"]]
        self.vev = [n for k, n in self.nodes.items() if k in self.vbase and n["split"] == "valid" and n["y"]]
        self.kb = {t: kbm.PatternKB([n for n in self.train if n["topic"] == t], cd.KB_KEYS["full"], min_count)
                   for t in self.topics}
        self.binner = kc.make_binner(None)
        self.E_val = {t: self.confusion(self.vev, t, valid=True) for t in self.topics}
        self.keys = [(n["topic"], n["node_id"]) for n in self.ev]
        self.gold = [n["y"] for n in self.ev]
        self.none_idx = find_none(self.LABELS)
        self.lr, self.la, self.val_score = self.select_lambda(self.E_val, self.kb)
        self.spin = self.forward(self.ev, self.base, self.E_val, self.kb)

    # E 估计：完全沿用 kc.confusion（含其平滑）
    def confusion(self, nodes, topic, valid):
        base, conf = (self.vbase, self.vconf) if valid else (self.base, self.conf)
        sub = [n for n in nodes if n["topic"] == topic]
        return np.asarray(self.acp.kc.confusion(sub, base, conf, self.binner, 1)[0], dtype=float)

    def forward(self, nodes, base, E, kb, lr=None, la=None):
        lr = self.lr if lr is None else lr
        la = self.la if la is None else la
        return self.acp.forward(nodes, base, kb, lambda n: E[n["topic"]], lr, la)

    def select_lambda(self, E, kb, nodes=None, base=None):
        """与 analyze_combined_prior.py 相同的协议：验证集话题平均 Macro-F1 最大，平局取 λr+λa 更小者。"""
        nodes = self.vev if nodes is None else nodes
        base = self.vbase if base is None else base
        grid = {}
        for c in itertools.product(self.lams, self.lams):
            lab = self.acp.forward(nodes, base, kb, lambda n: E[n["topic"]], *c)
            grid[c] = self.acp.topic_avg(nodes, lab, self.topics)
        lr, la = max(grid, key=lambda c: (grid[c], -(c[0] + c[1])))
        return lr, la, grid[(lr, la)]

    def evaluate(self, labels, ref=None):
        pred = [labels[k] for k in self.keys]
        return metrics(self.gold, pred, self.LABELS, ref_pred=None if ref is None else [ref[k] for k in self.keys],
                       acp=self.acp, nodes=self.ev, labels_dict=labels, topics=self.topics)


def make_run(acp, spec):
    """按 manifest 项构造 Run；可选字段 lambdas、pred_field、conf_field、reference_pred（论文主实验的 pred_combined.jsonl）。"""
    run = Run(acp, spec["instances"], spec["valid_pred"], spec["test_pred"],
              spec.get("pred_field", "predicted_stance"), spec.get("conf_field", "confidence"),
              lambdas=spec.get("lambdas", "0,0.25,0.5,0.75,1,1.5,2,3,4,6"), name=spec["name"])
    ref = spec.get("reference_pred")
    if ref:
        import json as _j
        refd = {}
        for line in open(ref, encoding="utf-8"):
            if line.strip():
                r = _j.loads(line)
                refd[(str(r["topic"]), str(r["node_id"]))] = r["predicted_stance"]
        common = [k for k in run.keys if k in refd]
        diff = sum(run.spin[k] != refd[k] for k in common)
        print(f"  [复现检查] 与 {ref} 比较 {len(common)} 个节点，预测不同 {diff} 个"
              + ("  ✓" if diff == 0 else "  !! 与论文主实验不一致，请检查 λ 网格（lambdas）或输入文件"))
    ntr = {t: sum(1 for n in run.train if n["topic"] == t) for t in run.topics}
    empty = [t for t in run.topics if ntr[t] == 0]
    if empty:
        nev = sum(1 for n in run.ev if n["topic"] in empty)
        print(f"  !! {len(empty)}/{len(run.topics)} 个话题在训练集中没有节点（KB 只剩全局键且计数为 0）："
              f"{empty[:5]}{'…' if len(empty) > 5 else ''}；涉及 {nev}/{len(run.ev)} 个测试节点")
    print(f"  [复现] SPIN λ=({run.lr:g},{run.la:g})  valid 话题平均 {100*run.val_score:.2f}  "
          f"test 合并MacroF1 {100*run.evaluate(run.spin)['macro_f1']:.2f}")
    return run


def find_none(labels):
    for i, l in enumerate(labels):
        if str(l).strip().lower() in ("none", "neutral", "无", "无立场", "中立", "其他"):
            return i
    raise ValueError(f"无法在 LABELS={labels} 中识别 NONE 类，请检查")


def load_manifest(path):
    """manifest：JSON 列表，每项 {name, dataset, instances, valid_pred, test_pred, [pred_field, conf_field]}"""
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)
    ok = []
    for r in runs:
        for k in ("name", "dataset", "instances", "valid_pred", "test_pred"):
            if k not in r:
                raise ValueError(f"manifest 项缺少字段 {k}: {r}")
        missing = [r[k] for k in ("instances", "valid_pred", "test_pred") if not Path(r[k]).exists()]
        if missing:
            print(f"  !! 跳过 {r['name']}：找不到文件 {missing}")
            continue
        ok.append(r)
    if not ok:
        raise SystemExit(f"{path} 中没有可运行的 run，请检查路径（当前目录 {Path.cwd()}）")
    print(f"manifest: {len(ok)}/{len(runs)} 个 run 可运行")
    return ok


# ----------------------------------------------------------------------------- metrics
def confusion_counts(gold, pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)))
    for g, p in zip(gold, pred):
        if p in idx:
            C[idx[g], idx[p]] += 1
    return C


def prf_from_counts(C):
    tp = np.diag(C)
    p = tp / np.maximum(C.sum(0), EPS)
    r = tp / np.maximum(C.sum(1), EPS)
    f = 2 * p * r / np.maximum(p + r, EPS)
    return p, r, f


def macro_f1(gold, pred, labels):
    return float(prf_from_counts(confusion_counts(gold, pred, labels))[2].mean())


def metrics(gold, pred, labels, ref_pred=None, acp=None, nodes=None, labels_dict=None, topics=None):
    C = confusion_counts(gold, pred, labels)
    p, r, f = prf_from_counts(C)
    out = {"macro_f1": float(f.mean()), "accuracy": float(np.trace(C) / max(C.sum(), 1)), "n": int(C.sum())}
    if acp is not None:
        mf, fa, _ = acp.cd.scores_of(gold, pred)
        out["macro_f1_cd"], out["f_avg"] = float(mf), float(fa)
        if abs(out["macro_f1_cd"] - out["macro_f1"]) > 1e-6:
            # 定义不一致时以现有 cd.scores_of 为准，并提示
            out["warning"] = f"内部 MacroF1 {out['macro_f1']:.6f} 与 cd.scores_of {mf:.6f} 不一致，以后者为准"
            out["macro_f1"] = float(mf)
        if nodes is not None and topics is not None:
            out["topic_avg_macro_f1"] = float(acp.topic_avg(nodes, labels_dict, topics))
    for i, l in enumerate(labels):
        out[f"P_{l}"], out[f"R_{l}"], out[f"F1_{l}"] = float(p[i]), float(r[i]), float(f[i])
    if ref_pred is not None:
        out["fixed"] = int(sum(a == g != b for a, b, g in zip(pred, ref_pred, gold)))
        out["broken"] = int(sum(b == g != a for a, b, g in zip(pred, ref_pred, gold)))
    return out


def paired_tests(gold, pred_a, pred_b, labels, n_boot=5000, n_perm=5000, seed=0):
    """与论文一致：配对 bootstrap（ΔMacro-F1 的 95% CI）与近似随机化检验（双侧）。"""
    rng = np.random.default_rng(seed)
    idx = {l: i for i, l in enumerate(labels)}
    g = np.array([idx[x] for x in gold])
    a = np.array([idx.get(x, -1) for x in pred_a])
    b = np.array([idx.get(x, -1) for x in pred_b])
    K = len(labels)

    def mf(gg, pp):
        C = np.zeros((K, K))
        ok = pp >= 0
        np.add.at(C, (gg[ok], pp[ok]), 1)
        return prf_from_counts(C)[2].mean()

    obs = mf(g, a) - mf(g, b)
    n = len(g)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        boots[i] = mf(g[s], a[s]) - mf(g[s], b[s])
    cnt = 0
    for _ in range(n_perm):
        sw = rng.random(n) < 0.5
        aa, bb = np.where(sw, b, a), np.where(sw, a, b)
        if abs(mf(g, aa) - mf(g, bb)) >= abs(obs) - 1e-12:
            cnt += 1
    return {"delta": float(obs), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "p_ar": float((cnt + 1) / (n_perm + 1))}


# ----------------------------------------------------------------------------- divergences
def js(p, q):
    p = np.asarray(p, float) + EPS
    q = np.asarray(q, float) + EPS
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def tv(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    return float(0.5 * np.abs(p / max(p.sum(), EPS) - q / max(q.sum(), EPS)).sum())


def rownorm(M):
    M = np.asarray(M, float)
    return M / np.maximum(M.sum(1, keepdims=True), EPS)


def tree_groups(nodes):
    g = defaultdict(list)
    for n in nodes:
        g[(n["topic"], n["root"])].append(n)
    return g


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")