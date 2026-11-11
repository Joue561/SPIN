# -*- coding: utf-8 -*-
"""
plot_robustness.py —— 由三个实验的 CSV 生成论文用图（PDF）。缺哪个目录就跳过哪张图。
  python plot_robustness.py --root robustness --out robustness/figs
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = ["#2A78D6", "#EB6834", "#3BA272", "#8A5CC2", "#C9A227", "#8A8985", "#D64C7F", "#4BB3C4"]


def read(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_perturb(rows, out):
    fams = [("uniform", r"uniform mixing $\alpha$"), ("offdiag", r"off-diagonal scale $s$"),
            ("toward_none", r"mass moved to None $\delta$")]
    runs = list(dict.fromkeys(r["name"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, (fam, xl) in zip(axes, fams):
        for i, rn in enumerate(runs):
            pts = sorted((float(r["level"]), float(r["gain_vs_base"])) for r in rows if r["name"] == rn and r["family"] == fam)
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3, lw=1.2, color=COL[i % len(COL)], label=rn)
        ax.axhline(0, color="#888", lw=0.6, ls="--")
        ax.set_xlabel(xl)
    axes[0].set_ylabel(r"$\Delta$ Macro-F1 vs. base")
    axes[-1].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(out / "fig_channel_perturbation.pdf"); plt.close(fig)


def fig_shift(rows, out):
    stats = [("label_tv", "label TV"), ("pred_tv", "prediction TV"),
             ("pred_tv_given_label_shift", "pred. TV | label shift"), ("ccs_js", "class-cond. JS")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    names = [r["name"] for r in rows]
    for ax, (k, t) in zip(axes, stats):
        obs = [float(r[k]) for r in rows]
        nul = [float(r[f"null95_{k}"]) for r in rows]
        y = range(len(rows))
        ax.barh(y, obs, color=[COL[1] if float(r[f"p_{k}"]) < 0.05 else COL[0] for r in rows], height=0.6)
        ax.scatter(nul, y, marker="|", color="black", s=60, zorder=3)
        ax.set_yticks(list(y)); ax.set_yticklabels(names if ax is axes[0] else [""] * len(names), fontsize=6)
        ax.set_title(t, fontsize=7)
    fig.text(0.99, 0.01, "bar: observed (orange: p<0.05); tick: 95% of random tree re-splits", ha="right", fontsize=5.5)
    fig.tight_layout(); fig.savefig(out / "fig_channel_shift.pdf"); plt.close(fig)


def fig_symmetry(bd, out):
    sup = [r for r in bd if r["by"] == "support"]
    if not sup:
        return
    runs = list(dict.fromkeys(r["name"] for r in sup))
    variants = [k[:-4] for k in sup[0] if k.endswith("_mf1") and k not in ("base_mf1", "sym_mf1")]
    fig, axes = plt.subplots(1, len(runs), figsize=(min(7.0, 2.4 * len(runs)), 2.0), squeeze=False)
    for ax, rn in zip(axes[0], runs):
        rs = [r for r in sup if r["name"] == rn]
        x = range(len(rs))
        for i, v in enumerate(variants):
            ax.bar([j + 0.8 * i / max(len(variants), 1) - 0.3 for j in x],
                   [float(r[f"{v}_acc"]) - float(r["sym_acc"]) for r in rs],
                   width=0.8 / max(len(variants), 1), color=COL[(i + 1) % len(COL)], label=v)
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_xticks(list(x)); ax.set_xticklabels([f"{r['bucket']}\n(n={r['n']})" for r in rs], fontsize=5.5)
        ax.set_title(rn, fontsize=7); ax.set_xlabel("training support of the full key", fontsize=6)
    axes[0][0].set_ylabel(r"$\Delta$ accuracy vs. symmetric")
    axes[0][-1].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_symmetry_support.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="robustness")
    ap.add_argument("--out", default="robustness/figs")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pr = [r for r in read(root / "perturb/perturbation_results.csv") if r["family"] != "bootstrap"]
    if pr:
        fig_perturb(pr, out)
    sh = read(root / "channel/channel_shift_summary.csv")
    if sh:
        fig_shift(sh, out)
    bd = read(root / "symmetry/symmetry_breakdown.csv")
    if bd:
        fig_symmetry(bd, out)
    print("图已写入", out, sorted(p.name for p in out.glob("*.pdf")))


if __name__ == "__main__":
    main()