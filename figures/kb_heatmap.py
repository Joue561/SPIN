# -*- coding: utf-8 -*-
"""
kb_heatmap.py
可视化语义传导知识库（只用 train，与知识库构建同一份数据），并导出可写进论文的真实知识条目。不调用 API。

图 fig_kb_heatmap.pdf：每种传导类型下，父评论有立场时子评论「继承 / 反转 / 中和」的比例，
以及父评论为 none 时子评论「表达立场」的比例；两个数据集左右并排。两边模式一致即说明知识与目标无关。
屏幕输出：
  1. 每个数据集各传导类型的比例与样本数（即热图数字）
  2. 两个数据集之间对应行的相关系数（定量说明「模式一致」）
  3. 完整键 (回复关系, 传导类型, 目标关系, 深度) 中样本数最多的若干条，供论文「设计目标」段落引用

用法（make_rag_data 目录下）：
python paper_figs/kb_heatmap.py --out_dir paper_figs/figs \
  --data C-MTCSD=processed_cmtscd_semantic_llm_causal_v4/cmtscd_semantic_instances_llm.jsonl \
         ZS-CSD=zscsd_exp/data/zscsd_semantic_instances_llm.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ("favor", "against", "none")
SP_ORDER = ["stance_inheritance", "stance_amplification", "stance_reversal", "sarcastic_reversal",
            "stance_softening", "direct_stance_expression", "target_shift", "unclear_propagation"]
SP_NAME = {"stance_inheritance": "inheritance", "stance_amplification": "amplification",
           "stance_reversal": "reversal", "sarcastic_reversal": "sarcastic reversal",
           "stance_softening": "softening", "direct_stance_expression": "direct expression",
           "target_shift": "target shift", "unclear_propagation": "unclear"}
COLS = ["inherit", "reverse", "neutralize", "express"]
INK, INK2 = "#0b0b0b", "#52514e"


def norm(x):
    x = str(x or "").strip().lower()
    return x if x in LABELS else None


def depth_bucket(d):
    try:
        d = int(d)
    except (TypeError, ValueError):
        return "?"
    return str(d) if d < 4 else "4+"


def stats(path):
    rel, exp_, full = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("split")).lower() != "train":
            continue
        y, p = norm(r.get("stance_label")), norm(r.get("parent_stance_label"))
        sp = str(r.get("llm_semantic_propagation"))
        if y is None or p is None:
            continue
        key = (str(r.get("llm_reply_relation")), sp, str(r.get("llm_target_relation")), depth_bucket(r.get("depth")))
        if p in ("favor", "against"):
            rl = "neutralize" if y == "none" else ("inherit" if y == p else "reverse")
            rel[sp][rl] += 1
            full[key][rl] += 1
        else:
            exp_[sp]["express" if y != "none" else "stay"] += 1
    return rel, exp_, full


def matrix(rel, exp_):
    M, N = [], []
    for sp in SP_ORDER:
        c, e = rel[sp], exp_[sp]
        n, ne = sum(c.values()), sum(e.values())
        M.append([c["inherit"] / n if n else np.nan, c["reverse"] / n if n else np.nan,
                  c["neutralize"] / n if n else np.nan, e["express"] / ne if ne else np.nan])
        N.append((n, ne))
    return np.array(M), N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out_dir", default="figs")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    mats = []
    for item in a.data:
        name, path = item.split("=", 1)
        rel, exp_, full = stats(path)
        M, N = matrix(rel, exp_)
        mats.append((name, M, N))
        print(f"\n===== {name}（train）")
        print(f"  {'传导类型':20s} {'继承':>6s} {'反转':>6s} {'中和':>6s} {'n(父有立场)':>11s} {'表达':>6s} {'n(父none)':>9s}")
        for sp, row, (n, ne) in zip(SP_ORDER, M, N):
            print(f"  {SP_NAME[sp]:20s} " + " ".join(f"{v:6.2f}" if v == v else "     -" for v in row[:3])
                  + f" {n:11d} " + (f"{row[3]:6.2f}" if row[3] == row[3] else "     -") + f" {ne:9d}")
        print(f"\n  完整键中样本最多的 {a.top} 条（父评论有立场）：回复关系 | 传导类型 | 目标关系 | 深度 → 继承 / 反转 / 中和 (n)")
        for key, c in sorted(full.items(), key=lambda kv: -sum(kv[1].values()))[:a.top]:
            n = sum(c.values())
            print(f"  {' | '.join(key):70s} → {c['inherit'] / n:.2f} / {c['reverse'] / n:.2f} / {c['neutralize'] / n:.2f} ({n})")

    if len(mats) == 2:
        (_, A, _), (_, B, _) = mats
        ok = ~np.isnan(A) & ~np.isnan(B)
        r = np.corrcoef(A[ok], B[ok])[0, 1]
        print(f"\n两个数据集热图对应格的皮尔逊相关系数 r = {r:.3f}（越接近 1，传导模式越一致）")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42,
                         "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    fig, axes = plt.subplots(1, len(mats), figsize=(3.25 * len(mats), 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    axes = np.atleast_1d(axes)
    for ax, (name, M, N) in zip(axes, mats):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                            color="white" if v > 0.55 else INK)
        ax.set_xticks(range(4), ["inherit", "reverse", "neutralize", "express\n(parent NONE)"], fontsize=6.5)
        ax.set_title(name, loc="left", fontsize=7.5, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axvline(2.5, color="white", linewidth=2)
    axes[0].set_yticks(range(len(SP_ORDER)), [SP_NAME[s] for s in SP_ORDER], fontsize=6.6)
    axes[0].set_ylabel("Annotated propagation type", color=INK2)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).ax.tick_params(labelsize=6)
    fig.savefig(out / "fig_kb_heatmap.pdf"); fig.savefig(out / "fig_kb_heatmap.png")
    print(f"\n已保存 {out / 'fig_kb_heatmap.pdf'} / .png")


if __name__ == "__main__":
    main()
