# -*- coding: utf-8 -*-
"""
analyze_depth_dynamics.py
离线检验（不调用 API）：深层会话中是否存在跨话题共性的"立场传递规律"，
以及把这种规律当作"转移记忆"，沿会话树修正 LLM 预测能否提分。

【第 0 部分】数据检查：test 节点的父节点是否也有预测（决定能否沿树传递）
【第 1 部分】描述统计（只用 train 的 gold）：
    按深度统计 立场分布 / 父评论有立场比例 / 相对父评论的变换（继承 / 反转 / 中立化）/
    到最近"有明确立场的祖先"的距离 / 文本长度 / 疑问句比例 / 是否提及主目标；
    并计算每个深度下，绝对立场分布 与 相对变换分布 在 5 个话题间的 JSD（越小越有跨话题共性）
【第 2 部分】转移记忆：只用表层特征（深度、长度、疑问、附和/反驳/嘲讽标记、指人词、是否提主目标、
    父评论的长度与疑问），不使用任何 LLM 生成的字段（避免"按模型自己的理解检索"的自我确认）。
      R 模型：父评论有立场时 → P(继承 / 反转 / 中立化)
      N 模型：父评论无立场时 → P(当前评论表态)
    三种记忆来源：all_train（全部 train）/ in_topic（同话题 train）/ cross_topic（留一话题：只用其他话题 train）
    先在 test 上用 gold 父立场评估转移模型本身（关系准确率可与 LLM 的约 0.40 对比）
【第 3 部分】沿会话树从浅到深修正（因果：只用祖先，不用子回复）：
      观测：基础方法（默认 dynamic_semantic）的预测，用混淆矩阵转成似然 P(预测 | 真实)
      转移：父节点的后验分布 × 转移记忆
      后验 ∝ P(预测 | y) × P(y | 父后验, 表层特征)^λ
    混淆矩阵与 λ 用"按会话树 2 折交叉拟合"估计（在一半会话上估计，用到另一半），不在同一批样本上调参；
    另给出 λ 网格结果（看稳健性）和"已知 gold 父立场"的上界。
    结果按深度分组输出。

用法：
python spin/spin_core/analyze_depth_dynamics.py \
  --instances processed_cmtscd_semantic_llm_all/cmtscd_semantic_instances_llm.jsonl \
  --pred dynamic_semantic=rag_prediction_outputs/predictions_dynamic_semantic_test.jsonl \
         topic_memory=rag_prediction_outputs/predictions_topicmem_rag_test.jsonl \
         mixed_memory=rag_prediction_outputs/predictions_full_rag_test.jsonl \
         crosstopic_memory=rag_prediction_outputs/predictions_crosstopicmem_rag_test.jsonl \
  --base dynamic_semantic \
  --output_dir depth_dynamics_analysis
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
except ImportError:
    sys.exit("需要 scikit-learn：pip install scikit-learn")

LABELS = ["favor", "against", "none"]
LID = {l: i for i, l in enumerate(LABELS)}
RELS = ["inherit", "reverse", "neutralize"]
OPP = {"favor": "against", "against": "favor"}
ALIAS = {"favor": "favor", "favour": "favor", "support": "favor", "支持": "favor",
         "against": "against", "oppose": "against", "反对": "against",
         "none": "none", "neutral": "none", "中立": "none", "无立场": "none"}
TARGET_ALIASES = {
    "apollo": ["萝卜快跑", "萝卜", "无人驾驶", "自动驾驶", "无人车", "robotaxi", "apollo"],
    "naked": ["裸辞"],
    "marriage": ["不婚", "结婚", "婚姻"],
    "meal": ["预制菜", "预制"],
    "iphone": ["iphone", "苹果", "15"],
}
DEPTH_BUCKETS = ["0", "1", "2", "3", "4+"]

RE_Q = re.compile(r"[?？]|吗|呢|难道|怎么|为什么|凭什么|为啥|啥")
RE_AGREE = re.compile(r"确实|同意|赞同|说得对|说的对|有道理|没错|对的|是的|就是|\+1|顶|支持你|太对了|真相")
RE_DISAGREE = re.compile(r"不是|不对|错了|胡说|瞎说|扯|并不|反驳|别洗|你懂|无知|双标|洗地|搞笑")
RE_MOCK = re.compile(r"呵呵|笑死|哈哈|典|666|绷|乐了|离谱|服了|好家伙|🤣|😅|doge|狗头|孝")
RE_YOU = re.compile(r"你|您|楼上|博主|up主")


def norm(x):
    return None if x is None else ALIAS.get(str(x).strip().lower())


def depth_bucket(d):
    return str(d) if d < 4 else "4+"


def len_bucket(n):
    return "<=5" if n <= 5 else "<=15" if n <= 15 else "<=40" if n <= 40 else ">40"


def target_aliases(topic, main_target):
    t = str(topic or "").lower()
    out = [str(main_target).lower()] if main_target else []
    for k, v in TARGET_ALIASES.items():
        if k in t:
            out += v
    return [x for x in out if x]


def jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist(counter, keys, alpha=0.5):
    v = np.array([counter.get(k, 0) for k in keys], float) + alpha
    return v / v.sum()


def scores(g, p):
    if not g:
        return {"n": 0, "macro_f1": float("nan"), "f_avg": float("nan"), "acc": float("nan")}
    gi = np.array([LID[x] for x in g]); pi = np.array([LID[x] for x in p])
    cm = np.bincount(gi * 3 + pi, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(3), where=den > 0)
    return {"n": len(g), "macro_f1": float(f1.mean()), "f_avg": float((f1[0] + f1[1]) / 2),
            "acc": float(tp.sum() / len(g))}


def topic_macro(items):
    """items: [(topic, gold, pred)] → 话题平均的 macro_f1 / f_avg / acc"""
    by = defaultdict(lambda: ([], []))
    for t, g, p in items:
        by[t][0].append(g); by[t][1].append(p)
    s = [scores(*by[t]) for t in sorted(by)]
    return {k: float(np.mean([x[k] for x in s])) for k in ("macro_f1", "f_avg", "acc")} if s else {}


def relation(parent, child):
    if parent not in ("favor", "against"):
        return None
    if child == "none":
        return "neutralize"
    return "inherit" if child == parent else "reverse"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- 读数据
def load_instances(path, parent_field):
    nodes, dup = {}, 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        nid = str(r.get("node_id") or r.get("id") or "")
        topic = str(r.get("topic"))
        if not nid:
            continue
        key = (topic, nid)
        if key in nodes:
            dup += 1
        parts = nid.split("-")
        pid = r.get("parent_id")
        pid = str(pid) if pid not in (None, "", "None") else ("-".join(parts[:-1]) if len(parts) > 1 else None)
        d = r.get("depth")
        try:
            d = int(d)
        except (TypeError, ValueError):
            d = len(parts) - 1
        text = str(r.get("current_text") or r.get("text") or "")
        ptext = str(r.get("parent_text") or "")
        nodes[key] = {
            "topic": topic, "node_id": nid, "root": parts[0], "parent_key": (topic, pid) if pid else None,
            "depth": d, "split": str(r.get("split", "")).lower(), "y": norm(r.get("stance_label")),
            "pa_gold": norm(r.get(parent_field)), "text": text, "ptext": ptext,
            "aliases": target_aliases(topic, r.get("main_target")),
            # LLM 语义传导字段（供语义传导模式知识库检索使用）
            "sp": nv(r.get("llm_semantic_propagation")), "rr": nv(r.get("llm_reply_relation")),
            "tr": nv(r.get("llm_target_relation")),
        }
    return nodes, dup


def nv(x):
    """规范化 LLM 生成的类别字段：小写、空格/连字符转下划线；缺失记为 unknown。"""
    if x is None:
        return "unknown"
    if isinstance(x, (list, tuple)):
        x = "|".join(map(str, x))
    s = re.sub(r"[\s\-]+", "_", str(x).strip().lower())
    return s or "unknown"


def load_pred(path, pred_field):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = norm(r.get(pred_field))
        if p is None:
            continue
        out[(str(r.get("topic")), str(r.get("node_id")))] = p
    return out


def features(n):
    t, pt = n["text"], n["ptext"]
    low = t.lower()
    db, lb = depth_bucket(n["depth"]), len_bucket(len(t))
    f = {f"depth={db}": 1, f"len={lb}": 1, f"depth={db}|len={lb}": 1,
         f"p_len={len_bucket(len(pt))}": 1}
    for name, rx, s in (("q", RE_Q, t), ("agree", RE_AGREE, t), ("disagree", RE_DISAGREE, t),
                        ("mock", RE_MOCK, t), ("you", RE_YOU, t), ("p_q", RE_Q, pt)):
        if rx.search(s):
            f[name] = 1
            f[f"{name}|depth={db}"] = 1
    if any(a in low for a in n["aliases"]):
        f["tgt"] = 1
    if any(a in pt.lower() for a in n["aliases"]):
        f["p_tgt"] = 1
    return f


# --------------------------------------------------------------------------- 第 1 部分
def anchor_distance(n, nodes):
    """到最近一个 gold 立场为 favor/against 的祖先的距离（1 = 父评论）。"""
    k, d = n["parent_key"], 1
    if n["pa_gold"] in ("favor", "against"):
        return 1
    k = nodes[k]["parent_key"] if k in nodes else None
    d = 2
    while k is not None and k in nodes:
        if nodes[k]["y"] in ("favor", "against"):
            return d
        k, d = nodes[k]["parent_key"], d + 1
    return None


def part1(train, nodes, topics, min_topic_count):
    rows, jrows = [], []
    by = defaultdict(list)
    for n in train:
        by[depth_bucket(n["depth"])].append(n)
    for b in DEPTH_BUCKETS:
        g = by.get(b, [])
        if not g:
            continue
        ys = Counter(n["y"] for n in g)
        polar = [n for n in g if n["pa_gold"] in ("favor", "against")]
        nonep = [n for n in g if n["pa_gold"] == "none"]
        rel = Counter(relation(n["pa_gold"], n["y"]) for n in polar)
        anc = Counter()
        for n in g:
            a = anchor_distance(n, nodes)
            anc["无" if a is None else (str(a) if a < 3 else "3+")] += 1
        N = len(g)
        rows.append({
            "depth": b, "n": N,
            **{f"P({l})": round(ys[l] / N, 3) for l in LABELS},
            "父有立场比例": round(len(polar) / N, 3),
            **{f"P({r}|父有立场)": round(rel[r] / max(len(polar), 1), 3) for r in RELS},
            "P(表态|父无立场)": round(sum(n["y"] != "none" for n in nonep) / max(len(nonep), 1), 3),
            **{f"锚点距离={k}": round(anc[k] / N, 3) for k in ("1", "2", "3+", "无")},
            "长度中位数": int(np.median([len(n["text"]) for n in g])),
            "疑问句比例": round(np.mean([bool(RE_Q.search(n["text"])) for n in g]), 3),
            "提及主目标比例": round(np.mean([any(a in n["text"].lower() for a in n["aliases"]) for n in g]), 3),
        })
        # 跨话题 JSD
        abs_c, rel_c = defaultdict(Counter), defaultdict(Counter)
        for n in polar:
            abs_c[n["topic"]][n["y"]] += 1
            rel_c[n["topic"]][relation(n["pa_gold"], n["y"])] += 1
        cov = [t for t in topics if sum(abs_c[t].values()) >= min_topic_count]
        if len(cov) >= 2:
            pa = dist(sum((abs_c[t] for t in cov), Counter()), LABELS)
            pr = dist(sum((rel_c[t] for t in cov), Counter()), RELS)
            w = np.array([sum(abs_c[t].values()) for t in cov], float)
            ja = float(np.average([jsd(dist(abs_c[t], LABELS), pa) for t in cov], weights=w))
            jr = float(np.average([jsd(dist(rel_c[t], RELS), pr) for t in cov], weights=w))
            jrows.append({"depth": b, "n_topics": len(cov), "JSD_绝对立场": round(ja, 4),
                          "JSD_相对变换": round(jr, 4), "相对/绝对": round(jr / ja, 2) if ja > 0 else None,
                          **{f"{t}:P(inherit,reverse,neutralize)": "/".join(f"{x:.2f}" for x in dist(rel_c[t], RELS, 0))
                             for t in cov}})
    return rows, jrows


# --------------------------------------------------------------------------- 第 2 部分
class Transition:
    def __init__(self, train):
        polar = [n for n in train if n["pa_gold"] in ("favor", "against") and n["y"]]
        nonep = [n for n in train if n["pa_gold"] == "none" and n["y"]]
        alln = [n for n in train if n["y"]]
        self.vec = DictVectorizer()
        self.vec.fit([features(n) for n in alln])
        self.R = self._fit(polar, lambda n: relation(n["pa_gold"], n["y"]))
        self.N = self._fit(nonep, lambda n: "express" if n["y"] != "none" else "stay_none")
        # 对照模型 D：不看父评论立场，只用 深度 + 表层特征 直接预测立场（= 话题/深度先验）
        self.D = self._fit(alln, lambda n: n["y"])
        self.r_major = Counter(relation(n["pa_gold"], n["y"]) for n in polar).most_common(1)[0][0] if polar else "inherit"

    def _fit(self, data, lab):
        ys = [lab(n) for n in data]
        if len(set(ys)) < 2:
            return None
        m = LogisticRegression(max_iter=3000, C=1.0)
        m.fit(self.vec.transform([features(n) for n in data]), ys)
        return m

    def precompute(self, nodes):
        """批量计算并缓存（逐条调用 sklearn 很慢）。"""
        self.cache = getattr(self, "cache", {})
        todo = [n for n in nodes if (n["topic"], n["node_id"]) not in self.cache]
        if not todo:
            return
        X = self.vec.transform([features(n) for n in todo])
        if self.R is not None:
            PR = self.R.predict_proba(X)
            ri = {c: i for i, c in enumerate(self.R.classes_)}
        if self.N is not None:
            PN = self.N.predict_proba(X)
            ei = list(self.N.classes_).index("express") if "express" in self.N.classes_ else None
        if self.D is not None:
            PD = self.D.predict_proba(X)
            di = {c: i for i, c in enumerate(self.D.classes_)}
        for i, n in enumerate(todo):
            rel = {r: (float(PR[i, ri[r]]) if r in ri else 0.0) for r in RELS} if self.R is not None else {r: 1 / 3 for r in RELS}
            ex = float(PN[i, ei]) if self.N is not None and ei is not None else 0.5
            dp = np.array([float(PD[i, di[l]]) if l in di else 1e-6 for l in LABELS]) if self.D is not None else np.ones(3) / 3
            self.cache[(n["topic"], n["node_id"])] = (rel, ex, dp / dp.sum())

    def rel_proba(self, n):
        self.precompute([n])
        return self.cache[(n["topic"], n["node_id"])][0]

    def express_proba(self, n):
        self.precompute([n])
        return self.cache[(n["topic"], n["node_id"])][1]

    def direct_proba(self, n):
        self.precompute([n])
        return self.cache[(n["topic"], n["node_id"])][2]

    def prior(self, n, qp):
        """qp: 父节点在 (favor, against, none) 上的分布 → 当前节点的先验分布"""
        pr, pe = self.rel_proba(n), self.express_proba(n)
        f, a, z = qp
        out = np.array([
            f * pr["inherit"] + a * pr["reverse"] + z * pe * 0.5,
            a * pr["inherit"] + f * pr["reverse"] + z * pe * 0.5,
            (f + a) * pr["neutralize"] + z * (1 - pe),
        ])
        return out / out.sum()


def build_models(train, topics):
    models = {"all_train": {t: None for t in topics}, "in_topic": {}, "cross_topic": {}}
    shared = Transition(train)
    for t in topics:
        models["all_train"][t] = shared
        models["in_topic"][t] = Transition([n for n in train if n["topic"] == t])
        models["cross_topic"][t] = Transition([n for n in train if n["topic"] != t])
    return models


def eval_transition(models, test_nodes, topics):
    rows = []
    polar = [n for n in test_nodes if n["pa_gold"] in ("favor", "against") and n["y"]]
    for src, ms in models.items():
        gold_rel, pred_rel, maj_rel, items_mem, items_maj = [], [], [], [], []
        for n in polar:
            m = ms[n["topic"]]
            pr = m.rel_proba(n)
            r = max(RELS, key=pr.get)
            gold_rel.append(relation(n["pa_gold"], n["y"])); pred_rel.append(r); maj_rel.append(m.r_major)
            to = {"inherit": n["pa_gold"], "reverse": OPP[n["pa_gold"]], "neutralize": "none"}
            items_mem.append((n["topic"], n["y"], to[r])); items_maj.append((n["topic"], n["y"], to[m.r_major]))
        acc = float(np.mean([a == b for a, b in zip(gold_rel, pred_rel)])) if polar else float("nan")
        acc_maj = float(np.mean([a == b for a, b in zip(gold_rel, maj_rel)])) if polar else float("nan")
        rel_f1 = np.mean([
            (lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)(
                sum(g == r and p == r for g, p in zip(gold_rel, pred_rel)),
                sum(g != r and p == r for g, p in zip(gold_rel, pred_rel)),
                sum(g == r and p != r for g, p in zip(gold_rel, pred_rel)))
            for r in RELS]) if polar else float("nan")
        tm, tj = topic_macro(items_mem), topic_macro(items_maj)
        rows.append({"memory_source": src, "n_父有立场": len(polar), "关系准确率": round(acc, 4),
                     "关系macroF1": round(float(rel_f1), 4), "多数类关系准确率": round(acc_maj, 4),
                     "仅转移记忆预测立场_MacroF1(话题平均,已知gold父立场)": round(tm.get("macro_f1", float("nan")), 4),
                     "多数类关系_MacroF1(话题平均)": round(tj.get("macro_f1", float("nan")), 4)})
    return rows


# --------------------------------------------------------------------------- 第 3 部分
def fold_of(n):
    return int(hashlib.md5(f"{n['topic']}|{n['root']}".encode()).hexdigest(), 16) % 2


def emission_by_fold(nodes_eval, base):
    E = {}
    for k in (0, 1):
        cm = np.ones((3, 3))
        for n in nodes_eval:
            if fold_of(n) != k:     # 用另一折估计
                cm[LID[n["y"]], LID[base[(n["topic"], n["node_id"])]]] += 1
        E[k] = cm / cm.sum(1, keepdims=True)    # E[k][y, pred] = P(pred | y)
    return E


def smooth(nodes_eval, all_nodes, base, ms, lam, E, parent_mode):
    """从浅到深前向传递。返回 {key: label}，以及每个节点是否有父信息。"""
    order = sorted(nodes_eval, key=lambda n: n["depth"])
    q, out, has_parent = {}, {}, {}
    for n in order:
        key = (n["topic"], n["node_id"])
        pred = base[key]
        e = E[fold_of(n)][:, LID[pred]]
        qp = None
        if parent_mode == "gold":
            if n["pa_gold"] in LID:
                qp = np.eye(3)[LID[n["pa_gold"]]]
        elif n["parent_key"] in q:
            qp = q[n["parent_key"]]
        if qp is None:
            post = e / e.sum()
            lab = pred
        else:
            # noparent 对照：作用于完全相同的节点，但先验不看父节点（只用深度+表层特征）
            prior = ms[n["topic"]].direct_proba(n) if parent_mode == "noparent" else ms[n["topic"]].prior(n, qp)
            post = e * prior ** lam
            post = post / post.sum()
            lab = LABELS[int(post.argmax())] if lam > 0 else pred
        q[key], out[key], has_parent[key] = post, lab, qp is not None
    return out, has_parent


def summarize(nodes_eval, labels, base, tag):
    items = [(n["topic"], n["y"], labels[(n["topic"], n["node_id"])]) for n in nodes_eval]
    pooled = scores([g for _, g, _ in items], [p for _, _, p in items])
    tm = topic_macro(items)
    row = {**tag, "合并MacroF1": round(pooled["macro_f1"], 4), "合并F_avg": round(pooled["f_avg"], 4),
           "合并Acc": round(pooled["acc"], 4), "话题平均MacroF1": round(tm["macro_f1"], 4),
           "话题平均F_avg": round(tm["f_avg"], 4)}
    fix = brk = 0
    for n in nodes_eval:
        k = (n["topic"], n["node_id"])
        b, s = base[k], labels[k]
        if b != s:
            fix += s == n["y"]; brk += b == n["y"]
    row["改对"], row["改错"] = fix, brk
    by = defaultdict(list)
    for n in nodes_eval:
        by[depth_bucket(n["depth"])].append(n)
    for b in DEPTH_BUCKETS:
        if b in by:
            s = scores([n["y"] for n in by[b]], [labels[(n["topic"], n["node_id"])] for n in by[b]])
            row[f"depth{b}_F_avg"] = round(s["f_avg"], 4)
            row[f"depth{b}_MacroF1"] = round(s["macro_f1"], 4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="方法名=预测文件，可多个")
    ap.add_argument("--base", default=None, help="用于树修正的基础方法（默认第一个 --pred）")
    ap.add_argument("--pred_field", default="predicted_stance")
    ap.add_argument("--parent_field", default="parent_stance_label")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1,1.5,2")
    ap.add_argument("--min_topic_count", type=int, default=30)
    ap.add_argument("--output_dir", default="depth_dynamics_analysis")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    lams = [float(x) for x in a.lambdas.split(",")]

    nodes, dup = load_instances(a.instances, a.parent_field)
    topics = sorted({n["topic"] for n in nodes.values()})
    train = [n for n in nodes.values() if n["split"] == "train" and n["y"]]
    print(f"实例 {len(nodes)} 条（重复 ID {dup}）| train {len(train)} | 话题 {topics}")

    preds = {}
    for spec in a.pred:
        name, path = spec.split("=", 1)
        preds[name] = load_pred(path, a.pred_field)
    base_name = a.base or next(iter(preds))
    base = preds[base_name]

    # ---------------- 第 0 部分
    ev = [n for k, n in nodes.items() if k in base and n["split"] == a.eval_split and n["y"]]
    miss = [k for k in base if k not in nodes]
    print(f"\n【第 0 部分】数据检查")
    print(f"  {base_name}: {len(base)} 条预测，能和实例对上的 {a.eval_split} 样本 {len(ev)} 条，对不上 {len(miss)} 条")
    pk_in_pred = sum(1 for n in ev if n["parent_key"] in base)
    pk_in_inst = Counter(nodes[n["parent_key"]]["split"] if n["parent_key"] in nodes else "不在数据中" for n in ev)
    print(f"  父节点也有预测的比例：{pk_in_pred}/{len(ev)} = {pk_in_pred / max(len(ev), 1):.3f}")
    print(f"  父节点所在划分：{dict(pk_in_inst)}")
    print(f"  {a.eval_split} 深度分布：{dict(sorted(Counter(depth_bucket(n['depth']) for n in ev).items()))}")
    if pk_in_pred / max(len(ev), 1) < 0.5:
        print("  ⚠️ 大部分父节点没有预测：树修正只能作用于少数样本，第 3 部分的结果参考意义有限。")

    # ---------------- 第 1 部分
    rows, jrows = part1(train, nodes, topics, a.min_topic_count)
    write_csv(out / "part1_depth_stats_train.csv", rows)
    write_csv(out / "part1_depth_cross_topic_jsd.csv", jrows)
    print("\n【第 1 部分】train 按深度统计（gold）")
    cols = ["depth", "n", "P(favor)", "P(against)", "P(none)", "父有立场比例", "P(inherit|父有立场)",
            "P(reverse|父有立场)", "P(neutralize|父有立场)", "P(表态|父无立场)", "锚点距离=1", "锚点距离=2",
            "锚点距离=3+", "锚点距离=无", "长度中位数", "疑问句比例", "提及主目标比例"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))
    print("\n  跨话题差异（JSD，越小越有共性；仅父评论有立场的样本）")
    for r in jrows:
        print(f"  depth {r['depth']:>3s}: 绝对立场 {r['JSD_绝对立场']:.4f} | 相对变换 {r['JSD_相对变换']:.4f} | 相对/绝对 {r['相对/绝对']}")

    # ---------------- 第 2 部分
    print("\n【第 2 部分】训练转移记忆（只用表层特征）……")
    models = build_models(train, topics)
    for ms in models.values():
        for t in topics:
            ms[t].precompute([n for n in ev if n["topic"] == t])
    trows = eval_transition(models, ev, topics)
    write_csv(out / "part2_transition_eval.csv", trows)
    print(f"  在 {a.eval_split} 上用 gold 父立场评估（LLM 判断关系的准确率约 0.40，可对比）")
    for r in trows:
        print(f"  {r['memory_source']:12s} 关系准确率 {r['关系准确率']:.4f}（多数类 {r['多数类关系准确率']:.4f}）"
              f" 关系macroF1 {r['关系macroF1']:.4f} | 仅靠转移记忆预测立场 MacroF1 {r['仅转移记忆预测立场_MacroF1(话题平均,已知gold父立场)']:.4f}"
              f"（多数类 {r['多数类关系_MacroF1(话题平均)']:.4f}）")

    # ---------------- 第 3 部分
    print(f"\n【第 3 部分】沿会话树修正 {base_name}（因果，只用祖先）")
    E = emission_by_fold(ev, base)
    res = [summarize(ev, base, base, {"variant": f"{base_name}(原始)", "memory_source": "-", "lambda": "-"})]
    for name, p in preds.items():
        if name == base_name:
            continue
        sub = [n for n in ev if (n["topic"], n["node_id"]) in p]
        if len(sub) == len(ev):
            res.append(summarize(ev, p, base, {"variant": f"{name}(原始)", "memory_source": "-", "lambda": "-"}))
        else:
            print(f"  （{name} 只覆盖 {len(sub)}/{len(ev)} 条，按深度对比见 part3 的单独表）")
    grid = {}
    for src, ms in models.items():
        for mode in ("pred", "noparent", "gold"):
            for lam in lams:
                lab, hp = smooth(ev, nodes, base, ms, lam, E, mode)
                grid[(src, mode, lam)] = lab
                tag = {"variant": {"pred": "树修正", "gold": "树修正(上界:gold父立场)", "noparent": "对照:不看父立场(深度+表层先验)"}[mode],
                       "memory_source": src, "lambda": lam}
                res.append(summarize(ev, lab, base, tag))
            # 交叉拟合选 λ：在另一折上选最好的 λ
            chosen, lab_cf = {}, {}
            for k in (0, 1):
                other = [n for n in ev if fold_of(n) != k]
                best = max(lams, key=lambda l: topic_macro(
                    [(n["topic"], n["y"], grid[(src, mode, l)][(n["topic"], n["node_id"])]) for n in other])["macro_f1"])
                chosen[k] = best
                for n in ev:
                    if fold_of(n) == k:
                        key = (n["topic"], n["node_id"])
                        lab_cf[key] = grid[(src, mode, best)][key]
            tag = {"variant": {"pred": "树修正", "gold": "树修正(上界:gold父立场)", "noparent": "对照:不看父立场(深度+表层先验)"}[mode] + "[交叉拟合λ]",
                   "memory_source": src, "lambda": f"{chosen[0]}/{chosen[1]}"}
            res.append(summarize(ev, lab_cf, base, tag))
            if mode in ("pred", "noparent"):
                fname = f"pred_tree_{src}_crossfit.jsonl" if mode == "pred" else f"pred_noparent_{src}_crossfit.jsonl"
                with open(out / fname, "w", encoding="utf-8") as f:
                    for n in ev:
                        key = (n["topic"], n["node_id"])
                        f.write(json.dumps({"topic": n["topic"], "node_id": n["node_id"], "depth": n["depth"],
                                            "stance_label": n["y"], "predicted_stance": lab_cf[key],
                                            "base_prediction": base[key]}, ensure_ascii=False) + "\n")
    write_csv(out / "part3_tree_smoothing.csv", res)

    show = [r for r in res if "(原始)" in r["variant"] or "交叉拟合" in r["variant"]]
    dcols = [c for c in res[0] if c.endswith("_F_avg") and c.startswith("depth")]
    print(f"{'方法':40s} {'来源':12s} {'λ':>8s} {'话题平均MacroF1':>14s} {'话题平均F_avg':>12s} {'合并MacroF1':>10s} {'改对':>5s} {'改错':>5s} | "
          + " ".join(f"{c.replace('_F_avg', ''):>8s}" for c in dcols))
    for r in show:
        print(f"{r['variant'][:40]:40s} {r['memory_source']:12s} {str(r['lambda']):>8s} {r['话题平均MacroF1']:14.4f} "
              f"{r['话题平均F_avg']:12.4f} {r['合并MacroF1']:10.4f} {r['改对']:5d} {r['改错']:5d} | "
              + " ".join(f"{r.get(c, float('nan')):8.4f}" for c in dcols))
    print("  （按深度的列为各深度合并口径 F_avg）")

    # λ 稳健性
    print("\n  λ 网格（话题平均 MacroF1；树修正，父立场来自模型预测）")
    print(f"  {'来源':12s} " + " ".join(f"{l:>7}" for l in lams))
    for v_name, v_lab in (("树修正", "树修正"), ("对照:不看父立场(深度+表层先验)", "对照")):
        for src in models:
            vals = [next(r for r in res if r["variant"] == v_name and r["memory_source"] == src and r["lambda"] == l)["话题平均MacroF1"] for l in lams]
            print(f"  {v_lab:4s} {src:12s} " + " ".join(f"{v:7.4f}" for v in vals))

    # 其他方法按深度
    drows = []
    for name, p in preds.items():
        sub = [n for n in ev if (n["topic"], n["node_id"]) in p]
        by = defaultdict(list)
        for n in sub:
            by[depth_bucket(n["depth"])].append(n)
        for b in DEPTH_BUCKETS:
            if b in by:
                s = scores([n["y"] for n in by[b]], [p[(n["topic"], n["node_id"])] for n in by[b]])
                drows.append({"method": name, "depth": b, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}})
    write_csv(out / "part3_methods_by_depth.csv", drows)
    print("\n  各方法按深度（F_avg / MacroF1，合并口径）")
    for name in preds:
        cells = [f"d{r['depth']}: {r['f_avg']:.3f}/{r['macro_f1']:.3f} (n={r['n']})" for r in drows if r["method"] == name]
        print(f"  {name:20s} " + " | ".join(cells))

    print(f"\n结果已保存到 {out.resolve()}")
    print("显著性检验示例：python spin/experiments/main/paired_significance.py "
          f"--a {out}/pred_tree_cross_topic_crossfit.jsonl --b <{base_name} 预测文件> "
          "--id_field topic node_id --pred_field predicted_stance --gold_field stance_label --group_field topic")


if __name__ == "__main__":
    main()