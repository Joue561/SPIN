# -*- coding: utf-8 -*-
"""
zscsd_stats.py
ZS-CSD 数据诊断（不调用 API）：判断会话树推断在该数据集上有没有修正空间。

ZS-CSD 每条实例是从第一层评论到当前句的完整路径（sentences），label 只针对最后一句，
没有显式 node_id / parent_id。这里用 (target, 路径) 作为节点，路径去掉最后一句即为父节点。

输出：
  1. 各划分的样本数、目标数、目标类型、标签分布、深度分布
  2. 节点重复（同一目标下同一路径出现多次）及其标签是否一致
  3. 父节点覆盖率：父路径是否也作为带标签实例出现（会话树推断需要父节点的预测）
  4. 父子立场关系分布：继承 / 反转 / 中立化 / 表态 / 保持中立（按深度）
     —— 对比 MT-CSD：反转和中立化越多，可修正空间越大
  5. 同一会话是否标注了多个目标；会话（第一句）是否跨划分出现
  6. speakers 与说话人重复（当前说话人是否在祖先中出现过）

用法（make_rag_data 目录下）：
python spin/data_prep/zscsd/zscsd_stats.py --data_dir ZS-CSD/data
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

FILES = {"train": "train_data.json", "valid": "dev_data.json", "test": "test_data.json"}


def load_labels(d):
    p = d / "label.json"
    m = json.load(open(p, encoding="utf-8")) if p.exists() else {"favor": 0, "against": 1, "neutral": 2}
    inv = {v: k for k, v in m.items()}
    return {k: ("none" if v == "neutral" else v) for k, v in inv.items()}


def rel(p, c):
    if p in ("favor", "against"):
        if c == p:
            return "继承"
        if c == "none":
            return "中立化"
        return "反转"
    return "表态" if c != "none" else "保持中立"


def pct(c, n):
    return f"{c / max(n, 1):.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    a = ap.parse_args()
    d = Path(a.data_dir)
    lab = load_labels(d)
    data = {sp: json.load(open(d / f, encoding="utf-8")) for sp, f in FILES.items()}

    print("【1. 基本统计】")
    for sp, rows in data.items():
        tg = Counter(r["target"] for r in rows)
        tt = Counter(r.get("target_type") for r in rows)
        lb = Counter(lab.get(r["label"], r["label"]) for r in rows)
        dp = Counter(min(len(r["sentences"]), 6) for r in rows)
        print(f"  {sp:5s} {len(rows):6d} 条 | 目标 {len(tg):3d} 个 | 类型 {dict(tt)} | "
              f"标签 " + " ".join(f"{k} {pct(v, len(rows))}" for k, v in sorted(lb.items())))
        print(f"        深度（句数，6=≥6）" + " ".join(f"{k}:{v}" for k, v in sorted(dp.items())))
    tr_t = {r["target"] for r in data["train"]}
    for sp in ("valid", "test"):
        ov = {r["target"] for r in data[sp]} & tr_t
        print(f"  {sp} 与 train 重叠的目标：{len(ov)} 个 {sorted(ov)[:5]}")

    print("\n【2. 节点重复】")
    nodes = {}
    for sp, rows in data.items():
        cnt = defaultdict(list)
        for r in rows:
            cnt[(r["target"], tuple(r["sentences"]))].append(lab.get(r["label"], r["label"]))
        dup = {k: v for k, v in cnt.items() if len(v) > 1}
        conflict = sum(1 for v in dup.values() if len(set(v)) > 1)
        print(f"  {sp}: 唯一节点 {len(cnt)} / {len(rows)}；重复路径 {len(dup)}，其中标签不一致 {conflict}")
        nodes[sp] = {k: Counter(v).most_common(1)[0][0] for k, v in cnt.items()}

    print("\n【3. 父节点覆盖率】（父路径是否也作为带标签实例出现在同一划分）")
    for sp, nd in nodes.items():
        tot = has = 0
        by_d = defaultdict(lambda: [0, 0])
        for (t, path), y in nd.items():
            if len(path) < 2:
                continue
            tot += 1
            ok = (t, path[:-1]) in nd
            has += ok
            by_d[min(len(path), 6)][0] += ok
            by_d[min(len(path), 6)][1] += 1
        print(f"  {sp}: 有父句的节点 {tot}，父节点有标签 {has}（{pct(has, tot)}）；按深度 " +
              " ".join(f"{k}:{pct(v[0], v[1])}" for k, v in sorted(by_d.items())))

    print("\n【4. 父子立场关系】（只统计父节点有标签的节点）")
    for sp, nd in nodes.items():
        c = Counter()
        by_d = defaultdict(Counter)
        for (t, path), y in nd.items():
            if len(path) < 2 or (t, path[:-1]) not in nd:
                continue
            r = rel(nd[(t, path[:-1])], y)
            c[r] += 1
            by_d[min(len(path), 6)][r] += 1
        n = sum(c.values())
        print(f"  {sp} (n={n}): " + "  ".join(f"{k} {pct(v, n)}" for k, v in c.most_common()))
        if sp == "train":
            for dd, cc in sorted(by_d.items()):
                m = sum(cc.values())
                print(f"     深度 {dd}: " + "  ".join(f"{k} {pct(v, m)}" for k, v in cc.most_common()))
    # 父有立场时，子立场与父不同的比例（= 不能直接照抄父立场的比例）
    for sp, nd in nodes.items():
        a_ = b_ = 0
        for (t, path), y in nd.items():
            if len(path) >= 2 and (t, path[:-1]) in nd and nd[(t, path[:-1])] in ("favor", "against"):
                a_ += 1
                b_ += y != nd[(t, path[:-1])]
        print(f"  {sp}: 父有立场时子立场不同于父 {pct(b_, a_)}（n={a_}）")

    print("\n【5. 会话与目标】")
    for sp, rows in data.items():
        conv_t = defaultdict(set)
        for r in rows:
            conv_t[r["sentences"][0]].add(r["target"])
        multi = sum(1 for v in conv_t.values() if len(v) > 1)
        print(f"  {sp}: 会话（按第一句）{len(conv_t)} 个，标注多个目标的 {multi} 个；每个目标平均会话数 "
              f"{len(conv_t) / max(len({r['target'] for r in rows}), 1):.1f}")
    first = {sp: {r["sentences"][0] for r in rows} for sp, rows in data.items()}
    print(f"  会话跨划分：train∩valid {len(first['train'] & first['valid'])}，train∩test "
          f"{len(first['train'] & first['test'])}，valid∩test {len(first['valid'] & first['test'])}")

    print("\n【6. 说话人】")
    for sp, rows in data.items():
        rep = sum(1 for r in rows if len(r["speakers"]) >= 2 and r["speakers"][-1] in r["speakers"][:-1])
        n2 = sum(1 for r in rows if len(r["speakers"]) >= 2)
        print(f"  {sp}: 当前说话人在祖先中出现过 {pct(rep, n2)}（n={n2}）；"
              f"字段 {sorted(rows[0].keys())}")


if __name__ == "__main__":
    main()