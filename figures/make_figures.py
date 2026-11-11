# -*- coding: utf-8 -*-
"""
make_figures.py
生成论文结果 / 分析两节的图（PDF 矢量图 + PNG 预览），数字取自已核对的实验输出。

  fig_gain_vs_base.pdf      图 3：基础模型越弱，增益越大（所有基础模型 × 两个数据集）
  fig_correction_room.pdf   图 4：修正空间——基础 / SPIN / 人工父立场上界
  fig_parent_correct.pdf    图 5：按父节点推断是否正确统计的净改对数
  fig_depth.pdf             图 6：按深度的 ΔMacro-F1（需要先在服务器上运行 depth_breakdown.py 生成 depth_breakdown.csv）

用法：python make_figures.py [--depth_csv depth_breakdown.csv] [--out_dir figs]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ 样式（数据可视化规范：细标记、淡网格、文字用墨色）
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"
COL_W = 3.25          # ACL 单栏宽度（英寸）
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  ✅ {name}.pdf / .png")


# ------------------------------------------------------------------ 图 3：增益 vs 基础模型强度
# (数据集, 名称, 基础 MF1, ΔMF1, 类型, 是否显著)
GAIN = [
    ("C", "GPT-4o-mini", 68.25, 4.25, "kNN", True), ("C", "GPT-4o", 74.62, 3.15, "kNN", True),
    ("C", "Llama-8B", 53.81, 12.49, "kNN", True), ("C", "Llama-70B", 73.41, 1.37, "kNN", True),
    ("C", "4o-mini CoT", 63.71, 7.03, "prompt", True), ("C", "4o-mini full path", 62.78, 6.89, "prompt", True),
    ("C", "GLAN", 68.02, 2.03, "supervised", True),
    ("Z", "GPT-4o-mini", 43.71, 3.21, "kNN", True), ("Z", "GPT-4o", 56.26, 1.60, "kNN", True),
    ("Z", "Llama-8B", 29.05, 14.48, "kNN", True), ("Z", "Llama-70B", 43.97, 3.15, "kNN", True),
    ("Z", "4o-mini CoT", 46.11, 5.38, "prompt", True), ("Z", "4o-mini full path", 44.08, 2.26, "prompt", True),
    ("Z", "GPT-4o CoT", 65.94, 0.44, "prompt", False), ("Z", "SITPCL", 43.40, 1.07, "supervised", False),
]
LABELLED = {("C", "Llama-8B"), ("Z", "Llama-8B"), ("Z", "GPT-4o CoT"), ("C", "GLAN"), ("Z", "SITPCL"),
            ("C", "4o-mini CoT"), ("C", "Llama-70B")}
MARK = {"kNN": "o", "prompt": "s", "supervised": "^"}


def fig_gain_vs_base(out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    grid(ax, "both")
    for ds, col, dname in (("C", BLUE, "C-MTCSD"), ("Z", ORANGE, "ZS-CSD")):
        for d, name, base, delta, kind, sig in GAIN:
            if d != ds:
                continue
            ax.scatter(base, delta, s=26, marker=MARK[kind], zorder=3,
                       facecolor=col if sig else "white", edgecolor=col if sig else col,
                       linewidth=1.0 if sig else 1.2)
            if (d, name) in LABELLED:
                dx, ha = (1.2, "left")
                if name in ("GPT-4o CoT", "Llama-70B"):
                    dx, ha = (-1.2, "right")
                ax.annotate(name, (base, delta), xytext=(base + dx, delta + 0.35), fontsize=6.3,
                            color=INK2, ha=ha, va="bottom")
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xlabel("Base model Macro-F1 (%)")
    ax.set_ylabel("Δ Macro-F1 with SPIN")
    ax.set_xlim(25, 80); ax.set_ylim(-0.8, 16)
    # 图例：颜色 = 数据集，形状 = 基础模型类型，空心 = 不显著
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="C-MTCSD", ms=4.5),
         plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="ZS-CSD", ms=4.5),
         plt.Line2D([], [], marker="o", ls="", mfc=GRAY, mec=GRAY, label="LLM, kNN", ms=4.5),
         plt.Line2D([], [], marker="s", ls="", mfc=GRAY, mec=GRAY, label="LLM, CoT / full path", ms=4.5),
         plt.Line2D([], [], marker="^", ls="", mfc=GRAY, mec=GRAY, label="supervised", ms=4.5),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, label="not significant", ms=4.5)]
    ax.legend(handles=h, loc="upper right", frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.2,
              borderaxespad=0.1)
    save(fig, out, "fig_gain_vs_base")


# ------------------------------------------------------------------ 图 4：修正空间
ROOM = [  # 设定, 基础, SPIN, 人工父立场上界
    ("C-MTCSD\n4o-mini kNN", 68.25, 72.51, 71.81),
    ("ZS-CSD\n4o-mini kNN", 43.71, 46.91, 58.30),
    ("ZS-CSD\n4o-mini CoT", 46.11, 51.49, 59.08),
    ("ZS-CSD\nGPT-4o CoT", 65.94, 66.38, 67.79),
]


def fig_correction_room(out):
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    grid(ax, "x")
    for i, (name, b, s, g) in enumerate(ROOM):
        y = len(ROOM) - 1 - i
        lo, hi = min(b, s, g), max(b, s, g)
        ax.plot([lo, hi], [y, y], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        ax.scatter([b], [y], s=30, color=GRAY, zorder=3, edgecolor="white", linewidth=1)
        ax.scatter([s], [y], s=30, color=BLUE, zorder=4, edgecolor="white", linewidth=1)
        ax.scatter([g], [y], s=34, marker="D", color=AQUA, zorder=3, edgecolor="white", linewidth=1)
        ax.annotate(f"{s:.1f}", (s, y), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=6.3, color=INK)
        ax.annotate(f"{g:.1f}", (g, y), xytext=(0, -9), textcoords="offset points", ha="center",
                    fontsize=6.3, color=INK2)
    ax.set_yticks(range(len(ROOM)), [r[0] for r in ROOM][::-1])
    ax.set_xlabel("Macro-F1 (%)")
    ax.set_xlim(40, 76); ax.set_ylim(-0.6, len(ROOM) - 0.4)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], marker="o", ls="", color=GRAY, label="base", ms=4.5),
         plt.Line2D([], [], marker="o", ls="", color=BLUE, label="SPIN", ms=4.5),
         plt.Line2D([], [], marker="D", ls="", color=AQUA, label="SPIN, gold parent", ms=4)]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    save(fig, out, "fig_correction_room")


# ------------------------------------------------------------------ 图 5：按父节点推断是否正确的净改对数
PARENT = [  # 设定, 父推断正确的净收益, 父推断错误的净收益
    ("C · 4o-mini kNN", 114, 37), ("C · Llama-70B kNN", 24, 10), ("C · GLAN", 59, 1),
    ("Z · 4o-mini kNN", 153, -90), ("Z · Llama-70B kNN", 154, -75), ("Z · SITPCL", 89, -84),
    ("Z · GPT-4o CoT", 25, -15),
]


def fig_parent_correct(out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    grid(ax, "x")
    n, h = len(PARENT), 0.36
    for i, (name, ok, bad) in enumerate(PARENT):
        y = n - 1 - i
        ax.barh(y + h / 2 + 0.01, ok, height=h, color=AQUA, zorder=3)
        ax.barh(y - h / 2 - 0.01, bad, height=h, color=VIOLET, zorder=3)
        for v, yy in ((ok, y + h / 2), (bad, y - h / 2)):
            ax.annotate(f"{v:+d}", (v, yy), xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=6.1, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.6, zorder=4)
    ax.axhline(3.5 - 0.0, color=GRID, linewidth=0.8)   # C 与 Z 的分界
    ax.set_yticks(range(n), [p[0] for p in PARENT][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Net corrected predictions (fixed − broken)")
    ax.set_xlim(-130, 190)
    hd = [plt.Rectangle((0, 0), 1, 1, color=AQUA, label="parent inferred correctly"),
          plt.Rectangle((0, 0), 1, 1, color=VIOLET, label="parent inferred wrongly")]
    ax.legend(handles=hd, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False,
              handlelength=1.0, columnspacing=1.0)
    save(fig, out, "fig_parent_correct")


# ------------------------------------------------------------------ 图 6：按深度的 ΔMacro-F1
def fig_depth(out, path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    series = defaultdict(list)
    for r in rows:
        series[r["setting"]].append((r["depth_bucket"], float(r["delta_macro_f1"]), int(r["n"])))
    order = sorted({r["depth_bucket"] for r in rows}, key=lambda x: int(x.rstrip("+")))
    cols = [BLUE, ORANGE, AQUA, VIOLET]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    grid(ax, "y")
    for (name, pts), c in zip(series.items(), cols):
        d = {b: (v, n) for b, v, n in pts}
        xs = [i for i, b in enumerate(order) if b in d]
        ys = [d[order[i]][0] for i in xs]
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", ms=4.5, mec="white", mew=1, label=name, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("Depth of reply"); ax.set_ylabel("Δ Macro-F1 with SPIN")
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)
    save(fig, out, "fig_depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_csv", default=None)
    ap.add_argument("--out_dir", default="figs")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_gain_vs_base(out)
    fig_correction_room(out)
    fig_parent_correct(out)
    if a.depth_csv:
        fig_depth(out, a.depth_csv)


if __name__ == "__main__":
    main()
